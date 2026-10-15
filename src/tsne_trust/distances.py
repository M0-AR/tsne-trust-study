"""Pairwise distances + kNN helpers (from scratch)."""
import numpy as np


def pairwise_euclidean(X):
    s = (X ** 2).sum(1)
    D2 = s[:, None] + s[None, :] - 2 * X @ X.T
    np.maximum(D2, 0, out=D2)
    return np.sqrt(D2)


def knn_indices(D, k, exclude_self=True):
    n = D.shape[0]
    idx = np.argsort(D, axis=1)
    if exclude_self:
        return idx[:, 1:k + 1]
    return idx[:, :k]


def knn_same_label_rate(y, knn):
    hits = (y[knn] == y[:, None]).mean()
    return float(hits)
