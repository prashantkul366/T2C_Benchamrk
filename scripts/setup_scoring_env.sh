#!/usr/bin/env bash
# Build the environment that can score EVERY adapter, without a kernel restart.
#
# Two geometry kernels are needed and they install differently:
#   * pythonocc-core  -> Text2CAD / CADmium / CADFusion (conda-forge only, no pip wheel)
#   * cadquery (OCP)  -> cadrille / Text-to-CadQuery / general LLMs (pip or conda)
#
# micromamba rather than condacolab: condacolab replaces the runtime's Python and
# restarts the kernel, which breaks an already-working GPU session. This builds a
# side environment and leaves the main one alone.
#
#   bash scripts/setup_scoring_env.sh /content/occenv
#   /content/occenv/bin/python scripts/score_export.py --export export.json ...
set -euo pipefail

PREFIX="${1:-./occenv}"
MM_DIR="$(dirname "$PREFIX")/.micromamba"

if [ ! -x "$MM_DIR/bin/micromamba" ]; then
  mkdir -p "$MM_DIR" && cd "$MM_DIR"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
  cd - >/dev/null
fi
export MAMBA_ROOT_PREFIX="$MM_DIR/root"
MM="$MM_DIR/bin/micromamba"

# numpy is pinned <2: the conda-forge cadquery build still references np.bool8,
# which numpy 2 removed.
"$MM" create -y -q -p "$PREFIX" -c conda-forge \
  python=3.11 "numpy<2" pythonocc-core=7.7.0 cadquery \
  scipy trimesh rtree pandas tqdm pyyaml

# CadSeqProc (vendored by Text2CAD and CADmium) imports these at module scope,
# so they are required to score the sequence adapters even though none of them
# is used for geometry. torch is CPU-only here -- scoring never touches a GPU.
"$PREFIX/bin/python" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
"$PREFIX/bin/python" -m pip install -q \
  matplotlib loguru rich seaborn plotly scikit-learn joblib plyfile open3d manifold3d \
  huggingface_hub pyarrow tabulate
# embreex makes the exact point-in-solid test ~100x faster (0.5s vs 90s a sample)
"$PREFIX/bin/python" -m pip install -q --no-deps embreex

echo
echo "verifying:"
"$PREFIX/bin/python" - <<'PY'
import numpy, trimesh, cadquery, embreex
from OCC.Core.BRepCheck import BRepCheck_Analyzer
print(f"  numpy {numpy.__version__} | trimesh {trimesh.__version__} | "
      f"cadquery {cadquery.__version__} | pythonocc OK | embreex OK")
PY
echo
echo "ready: $PREFIX/bin/python"
echo
echo "Two source checkouts are needed before scoring; without them the affected"
echo "adapters are reported GEN-ONLY rather than counted against any model:"
echo "  export T2CBENCH_CADSEQ_PATH=/path/to/Text2CAD    # cadvec + minimal_json"
echo "        git clone https://github.com/SadilKhan/Text2CAD"
echo "  export T2CBENCH_CADFUSION_PATH=/path/to/CADFusion # skexgen"
echo "        git clone https://github.com/microsoft/CADFusion"
