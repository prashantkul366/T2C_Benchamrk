"""Geometric deduplication for the Text2CAD test split.

DeepCAD test carries a lot of near-identical flat plates and rectangular blocks.
Measured on a 220-mesh sample of the test split: **33% are flat** (thinnest
extent / longest < 0.15) and **32% tessellate to <=24 triangles**, i.e. are
box-like. Shapes like that stop discriminating between systems quickly, so
folding the near-identical ones frees budget for shapes that still do.

We deduplicate on a rotation-tolerant shape signature rather than on the CAD
sequence, because two different construction sequences can produce the same
solid and we care about the solid.

Signature (all computed after canonicalisation to [0,1]^3):
  - sorted eigenvalues of the point-cloud covariance   (3)  -- gross proportions
  - D2 shape distribution, 64 bins                     (64) -- Osada et al. 2002
  - volume / convex-hull volume                        (1)  -- how hollow it is
  - number of connected components                     (1)

D2 is the histogram of distances between random surface point pairs. It is
invariant to rotation and translation and cheap (~13 ms per mesh), which is what
we need for a sweep over 8k meshes.

Calibration of DEFAULT_EPS, measured on 220 real DeepCAD test meshes:

    nearest-neighbour signature distance
        p5 = 0.055   p25 = 0.086   median = 0.114   p75 = 0.164
    all pairs
        p1 = 0.128   p5  = 0.220   median = 0.598

    eps    kept
    0.05    99%      too tight -- folds essentially nothing
    0.08    89%   <- default: just above the p5 of nearest-neighbour distance,
    0.10    77%      so it folds genuine near-duplicates and little else
    0.15    56%      starts merging distinct parts
    0.20    35%      far too aggressive

The default deliberately errs toward keeping shapes: over-merging silently
shrinks the diversity of the evaluation set, which is the harder error to
notice. Re-measure with `--eps` sweeps if you change the signature.
"""

from __future__ import annotations

import numpy as np
import trimesh
from dataclasses import dataclass

from t2cbench.metrics.geometry import canonicalize

D2_BINS = 64
D2_PAIRS = 8192
SIG_POINTS = 4096
DEFAULT_EPS = 0.08


@dataclass
class Signature:
    uid: str
    vector: np.ndarray

    def distance(self, other: "Signature") -> float:
        return float(np.linalg.norm(self.vector - other.vector))


def shape_signature(mesh: trimesh.Trimesh, uid: str = "", seed: int = 0) -> Signature:
    m = canonicalize(mesh)
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(m, SIG_POINTS, seed=int(rng.integers(2**31)))
    pts = np.asarray(pts)

    # gross proportions
    cov = np.cov((pts - pts.mean(0)).T)
    eig = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
    eig = eig / (eig.sum() + 1e-12)

    # D2 distance histogram, normalised to a probability vector
    i = rng.integers(0, len(pts), D2_PAIRS)
    j = rng.integers(0, len(pts), D2_PAIRS)
    d = np.linalg.norm(pts[i] - pts[j], axis=1)
    hist, _ = np.histogram(d, bins=D2_BINS, range=(0.0, np.sqrt(3.0)), density=False)
    hist = hist / (hist.sum() + 1e-12)

    # solidity and component count
    try:
        solidity = float(abs(m.volume) / (abs(m.convex_hull.volume) + 1e-12)) if m.is_volume else 0.0
    except Exception:
        solidity = 0.0
    solidity = float(np.clip(solidity, 0.0, 1.0))
    try:
        n_comp = len(m.split(only_watertight=False))
    except Exception:
        n_comp = 1

    vec = np.concatenate([
        eig,                                    # 3
        hist * 4.0,                             # 64, upweighted: this carries the shape
        [solidity],                             # 1
        [min(n_comp, 8) / 8.0],                 # 1
    ]).astype(np.float64)
    return Signature(uid=uid, vector=vec)


def greedy_dedup(signatures: list[Signature], eps: float = DEFAULT_EPS
                 ) -> tuple[list[str], dict[str, int]]:
    """Greedy farthest-first-style dedup.

    Walks the list in order and keeps a signature only if it is at least `eps`
    from every signature already kept. Returns the kept uids and a mapping
    uid -> cluster id, so that the discarded near-duplicates stay traceable.

    O(n * k) where k is the number of kept items -- fine for n=8k, k~2-3k.
    """
    kept: list[Signature] = []
    cluster: dict[str, int] = {}
    kept_mat = np.zeros((0, len(signatures[0].vector))) if signatures else None

    for sig in signatures:
        if kept:
            d = np.linalg.norm(kept_mat - sig.vector, axis=1)
            nearest = int(np.argmin(d))
            if d[nearest] < eps:
                cluster[sig.uid] = nearest
                continue
        cluster[sig.uid] = len(kept)
        kept.append(sig)
        kept_mat = np.vstack([kept_mat, sig.vector]) if kept_mat is not None and len(kept_mat) \
            else sig.vector.reshape(1, -1)
    return [s.uid for s in kept], cluster


def complexity_bin(n_extrusions: int, n_curves: int) -> str:
    """Complexity from the ground-truth CAD sequence.

    Thresholds are heuristic, chosen so the four bins are populated rather than
    equal-width -- `score` is heavily right-skewed, so equal-width bins would put
    almost everything in "simple". Notebook 00 prints the realised bin sizes;
    check them and adjust if a bin comes out near-empty on your split.
    """
    score = n_extrusions * max(n_curves, 1)
    if n_extrusions <= 1 and n_curves <= 6:
        return "simple"
    if score <= 20:
        return "moderate"
    if score <= 60:
        return "complex"
    return "very_complex"
