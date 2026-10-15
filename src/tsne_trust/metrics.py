"""Trust metrics: what the video claims, measured from scratch."""
import numpy as np
from .distances import pairwise_euclidean, knn_indices


def trustworthiness(D_high, D_low, k=10):
    n = D_high.shape[0]
    hi = knn_indices(D_high, k)
    lo = knn_indices(D_low, k)
    hi_rank = np.argsort(np.argsort(D_high, axis=1), axis=1)
    total = 0.0
    for i in range(n):
        intr = [j for j in lo[i] if j not in set(hi[i])]
        for j in intr:
            total += hi_rank[i, j] - k
    return float(1 - 2 * total / (n * k * (2 * n - 3 * k - 1)))


def kept_overlap(D_high, D_low, k=10):
    hi = knn_indices(D_high, k)
    lo = knn_indices(D_low, k)
    inter = sum(len(set(hi[i]) & set(lo[i])) for i in range(len(hi))) / (len(hi) * k)
    return float(inter)


def same_label_rate(y, D, k=10):
    knn = knn_indices(D, k)
    return float(((y[knn] == y[:, None]).sum(1) / k).mean())


def island_spreads(Y, y):
    out = {}
    for k in sorted(set(y.tolist())):
        pts = Y[y == k]
        c = pts.mean(0)
        out[int(k)] = float(np.linalg.norm(pts - c, axis=1).mean())
    return out


def centroid_distances(Y, y):
    labs = sorted(set(y.tolist()))
    C = {k: Y[y == k].mean(0) for k in labs}
    return {(a, b): float(np.linalg.norm(C[a] - C[b])) for a in labs for b in labs if a < b}


def nearest_foreign_ratio(Y, y):
    D = pairwise_euclidean(Y)
    n = len(Y)
    ratios = []
    for i in range(n):
        order = np.argsort(D[i])
        d_near = D[i, order[1]]
        foreign = [j for j in order[1:] if y[j] != y[i]]
        if foreign:
            ratios.append(D[i, foreign[0]] / (d_near + 1e-12))
    return float(np.median(ratios))


def tenth_over_first(Y, sample=200, seed=0):
    rng = np.random.default_rng(seed)
    D = pairwise_euclidean(Y)
    idx = rng.choice(len(Y), min(sample, len(Y)), replace=False)
    r = []
    for i in idx:
        s = np.sort(D[i])[1:11]
        r.append(s[-1] / (s[0] + 1e-12))
    return float(np.median(r))


def clump_stats(Y):
    """10th/1st-neighbour ratio distribution. Spurious clumps from tiny
    perplexity show in the TAIL (mean/p90: isolated tight groups in empty
    space), not the median. Verified live on pure-noise maps."""
    D = pairwise_euclidean(Y)
    s = np.sort(D, axis=1)[:, 1:11]
    r = s[:, 9] / (s[:, 0] + 1e-12)
    return {"median": float(np.median(r)), "mean": float(r.mean()),
            "p90": float(np.percentile(r, 90))}


def high_dim_spread(X):
    c = X.mean(0)
    return float(np.linalg.norm(X - c, axis=1).mean())
