"""Data loaders: real 8x8 digits + synthetic controls. No hand-tuned files."""
import numpy as np


def load_digits600(seed=0, n_per_class=60):
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.data / 16.0  # 0..1 ink, 64 dims
    y = d.target.copy()
    rng = np.random.default_rng(seed)
    idx = []
    for k in range(10):
        pool = np.where(y == k)[0]
        sel = rng.choice(pool, size=min(n_per_class, len(pool)), replace=False)
        idx.extend(sel.tolist())
    idx = np.array(idx)
    # deterministic order shuffle
    perm = rng.permutation(len(idx))
    idx = idx[perm]
    return X[idx], y[idx]


def make_tight_loose(seed=0, n=300, tight=0.02, loose=0.2):
    """Video construction: 300 copies of ONE three + tiny noise (tight) and
    300 copies of ONE seven + 10x noise (loose). Real digit bases matter:
    sparse ink pictures give high-dim spreads ~0.13 vs ~1.11, and the map
    equalizes them (~7.3 vs ~7.3). Verified live (see tests)."""
    from sklearn.datasets import load_digits
    d = load_digits()
    pictures = d.data / 16.0
    labels = d.target
    rng = np.random.default_rng(seed)
    three = pictures[labels == 3][0]
    seven = pictures[labels == 7][0]
    T = np.clip(three + rng.normal(0, tight, (n, 64)), 0, 1)
    L = np.clip(seven + rng.normal(0, loose, (n, 64)), 0, 1)
    return T, L


def make_noise(seed=0, n=600, dim=64):
    rng = np.random.default_rng(seed)
    return rng.random((n, dim))


def make_abc(seed=0, n=150, dim=64, d_ab=1.0, d_ac=10.0, noise=0.08):
    """A/B/C groups: B is d_ab from A, C is d_ac from A along one axis.
    noise=0.08 verified live: perp30 draws B,C at ~equal distance from A
    (ratio ~1), perp450 recovers ~6x. Larger noise swamps d_ab (verified)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, noise, (n, dim))
    direction = np.zeros(dim)
    direction[0] = 1.0
    B = A.mean(0) + direction * d_ab + rng.normal(0, noise, (n, dim))
    C = A.mean(0) + direction * d_ac + rng.normal(0, noise, (n, dim))
    X = np.vstack([A, B, C])
    y = np.array([0] * n + [1] * n + [2] * n)
    return X, y
