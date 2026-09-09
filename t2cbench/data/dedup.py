"""Geometric deduplication for the Text2CAD test split.

DeepCAD is dominated by near-identical flat plates and rectangular blocks. A
uniform random 125 out of 8,046 would be roughly 40% simple slabs, which
flatters every system equally and wastes most of the evaluation budget on
shapes that no longer discriminate between models.

We deduplicate on a rotation-tolerant shape signature rather than on the CAD
sequence, because two different construction sequences can produce the same
solid and we care about the solid.

Signature (all computed after canonicalisation to [0,1]^3):
  - sorted eigenvalues of the point-cloud covariance   (3)  -- gross proportions
  - D2 shape distribution, 64 bins                     (64) -- Osada et al. 2002
  - volume / convex-hull volume                        (1)  -- how hollow it is
  - number of connected components                     (1)

D2 is the histogram of distances between random surface point pairs. It is
invariant to rotation and translation and cheap, which is what we need for an
all-pairs sweep over 8k meshes.
"""

from __future__ import annotations

import numpy as np
import trimesh
from dataclasses import dataclass

from t2cbench.metrics.geometry import canonicalize

D2_BINS = 64
D2_PAIRS = 8192
SIG_POINTS = 4096
DEFAULT_EPS = 0.035


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

    Thresholds chosen against the DeepCAD test distribution so the four bins are
    populated rather than equal-width; see notebooks/00_setup_data.ipynb for the
    histogram they were read off.
    """
    score = n_extrusions * max(n_curves, 1)
    if n_extrusions <= 1 and n_curves <= 6:
        return "simple"
    if score <= 20:
        return "moderate"
    if score <= 60:
        return "complex"
    return "very_complex"
