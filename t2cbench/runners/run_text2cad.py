"""Generation runner for Text2CAD v1.0.

Text2CAD is not a HuggingFace causal LM -- it is a BERT encoder feeding a custom
CAD-sequence decoder, driven through `model.test_decode(...)`. This wrapper
loads it exactly as `Cad_VLM/test_user_input.py` does and emits the predicted
`cad_vec` as JSON, so the CadVecAdapter can score it alongside everything else.

    python -m t2cbench.runners.run_text2cad \
        --text2cad-repo /content/Text2CAD \
        --checkpoint /content/weights/Text2CAD_1.0.pth \
        --split data/split_a.jsonl --out results/raw/text2cad_splitA.jsonl

Sampling note: `sampling_type: max` gives topk_index=1, which is this model's
greedy decode and therefore its pass@1. `--n-samples 5` reproduces the paper's
best-of-5 by sweeping topk_index 1..5, matching Cad_VLM/test.py.

The checkpoint lives in the gated SadilKhan/Text2CAD dataset repo -- accept the
licence on the HF page and export HF_TOKEN before downloading. It is the only
gated asset the benchmark cannot route around.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from tqdm import tqdm

# torch imported lazily -- see run_hf.py

DEFAULT_CONFIG = {
    "text_encoder": {
        "text_embedder": {"model_name": "bert_large_uncased", "max_seq_len": 512,
                          "cache_dir": None},
        "adaptive_layer": {"in_dim": 1024, "out_dim": 1024, "num_heads": 8, "dropout": 0.1},
    },
    "cad_decoder": {"tdim": 1024, "cdim": 256, "num_layers": 8, "num_heads": 8,
                    "dropout": 0.1, "ca_level_start": 2},
}


def _install_occ_stub() -> None:
    """Satisfy Text2CAD's module-scope `from OCC.Core.X import Y` without pythonocc.

    Generation needs a GPU; pythonocc needs conda. Putting both in one Colab
    runtime is a fight not worth having for a dependency the decode path does not
    use -- `Cad_VLM/models/layers/utils_decode.py` imports OCC at the top, but
    every OCC-touching function in it (brep2mesh, write_stl_file, plot, ...) is a
    geometry helper, and `model.test_decode` calls none of them.

    The stub is a poison pill, not a no-op: importing a name works, *using* one
    raises. So if that assumption is ever wrong, the run dies with a clear error
    instead of quietly producing geometry built from fake primitives.
    """
    import sys
    import types

    try:
        import OCC  # noqa: F401
        print("pythonocc is present; --stub-occ ignored")
        return
    except ImportError:
        pass

    # Dunders must behave normally. Python's own machinery (inspect, importlib,
    # pickle) walks sys.modules and reads __file__, __path__, __spec__ and
    # friends on every module it meets, so poisoning those turns an unrelated
    # `import torch` into a spurious "Text2CAD touched OCC" failure.
    def _is_dunder(name: str) -> bool:
        return name.startswith("__") and name.endswith("__")

    class _Poison:
        def __init__(self, name):
            object.__setattr__(self, "_name", name)

        def _die(self, *a, **k):
            raise RuntimeError(
                f"Text2CAD generation touched OCC ({self._name}), which --stub-occ "
                f"assumed it never does. Re-run in an environment with pythonocc-core.")

        __call__ = _die

        def __getattr__(self, k):
            if _is_dunder(k):
                raise AttributeError(k)
            return _Poison(f"{self._name}.{k}")

    class _StubModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            # Marks the stub as a package so `from OCC.Core.BRepMesh import X`
            # is allowed to descend into it; without __path__ the import system
            # rejects the submodule before the finder below is ever consulted.
            self.__path__ = []

        def __getattr__(self, name):
            if _is_dunder(name):
                raise AttributeError(name)
            return _Poison(f"{self.__name__}.{name}")

    for mod in ("OCC", "OCC.Core", "OCC.Display", "OCC.Extend"):
        sys.modules.setdefault(mod, _StubModule(mod))

    # Any OCC.Core.* / OCC.Display.* submodule resolves to a stub on demand.
    class _StubFinder:
        def find_module(self, fullname, path=None):
            return self if fullname.startswith("OCC.") else None

        def load_module(self, fullname):
            m = sys.modules.get(fullname) or _StubModule(fullname)
            sys.modules[fullname] = m
            return m

        def find_spec(self, fullname, path=None, target=None):
            if not fullname.startswith("OCC."):
                return None
            import importlib.machinery
            return importlib.machinery.ModuleSpec(fullname, _StubLoader())

    class _StubLoader:
        def create_module(self, spec):
            return _StubModule(spec.name)

        def exec_module(self, module):
            return None

        def is_package(self, fullname):
            return True

    sys.meta_path.insert(0, _StubFinder())
    print("!! pythonocc absent: installed a poison-pill OCC stub for generation only.\n"
          "   Scoring still requires a real pythonocc environment.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text2cad-repo", required=True,
                    help="checkout of https://github.com/SadilKhan/Text2CAD")
    ap.add_argument("--checkpoint", required=True, help="Text2CAD_1.0.pth")
    ap.add_argument("--name", default="text2cad")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None, help="HF cache for bert-large-uncased")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=1,
                    help="1 = greedy pass@1; 5 = the paper's best-of-5 topk sweep")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-prompt", action="store_true",
                    help="record the prompt text that was sent (Text2CAD takes it raw)")
    ap.add_argument("--stub-occ", action="store_true",
                    help="install a poison-pill OCC stub when pythonocc is absent. Text2CAD's "
                         "utils_decode imports OCC at module scope but the decode path never "
                         "calls it; the stub raises loudly if that assumption is ever wrong.")
    args = ap.parse_args()

    import torch
    if args.stub_occ:
        _install_occ_stub()
    repo = os.path.abspath(args.text2cad_repo)
    for p in (repo, os.path.join(repo, "Cad_VLM")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH
    from Cad_VLM.models.text2cad import Text2CAD

    cfg = DEFAULT_CONFIG
    if args.cache_dir:
        cfg["text_encoder"]["text_embedder"]["cache_dir"] = args.cache_dir
    cad_cfg = dict(cfg["cad_decoder"])
    cad_cfg["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Text2CAD(text_config=cfg["text_encoder"], cad_config=cad_cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = {}
    for k, v in ckpt["model_state_dict"].items():
        state[".".join(k.split(".")[1:]) if k.split(".")[0] == "module" else k] = v
    model.load_state_dict(state, strict=False)
    if "epoch" in ckpt:
        print(f"checkpoint trained for {ckpt['epoch']} epochs")
    model.eval()

    rows = [json.loads(l) for l in open(args.split) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
        print(f"resuming: {len(done)} already generated")
    todo = [r for r in rows if r["sample_id"] not in done]
    print(f"{len(todo)} prompts to generate")
    if not todo:
        return

    with open(args.out, "a") as fout, torch.no_grad():
        for i in tqdm(range(0, len(todo), args.batch_size), desc=args.name):
            batch = todo[i:i + args.batch_size]
            texts = [r["prompt"] for r in batch]
            for k in range(args.n_samples):
                pred = model.test_decode(
                    texts=texts,
                    maxlen=MAX_CAD_SEQUENCE_LENGTH,
                    nucleus_prob=0,
                    topk_index=k + 1,   # topk_index=1 is greedy
                    device=device,
                )
                vecs = pred["cad_vec"].cpu().numpy()
                for j, r in enumerate(batch):
                    rec = {
                        "sample_id": r["sample_id"],
                        "sample_idx": k,
                        "model": args.name,
                        # serialised so CadVecAdapter has the same string-in
                        # interface as every other adapter
                        "output": json.dumps(vecs[j].astype(int).tolist()),
                    }
                    if args.save_prompt:
                        rec["prompt_sent"] = texts[j]
                    fout.write(json.dumps(rec) + "\n")
            fout.flush()

    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
