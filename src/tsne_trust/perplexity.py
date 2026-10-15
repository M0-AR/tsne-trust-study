"""Perplexity -> sigma binary search -> symmetric joint P. From scratch."""
import numpy as np


def _H_beta(D_row, beta):
    P = np.exp(-D_row * beta)
    P[0] = 0.0  # exclude self (assumes D_row[0-self] handled by caller mask)
    s = P.sum()
    if s == 0:
        P = np.ones_like(P)
        P[0] = 0
        s = P.sum()
    P /= s
    nz = P[P > 1e-12]
    H = -np.sum(nz * np.log2(nz))
    return H, P


def joint_P(D, perplexity=30.0, tol=1e-5, max_iter=60):
    n = D.shape[0]
    logU = np.log2(perplexity)
    cond = np.zeros((n, n))
    sigmas = np.zeros(n)
    D2 = D ** 2
    for i in range(n):
        beta_min, beta_max = -np.inf, np.inf
        beta = 1.0
        Di = D2[i].copy()
        # mask self
        for it in range(max_iter):
            H, Pi = _H_beta(Di, beta)
            # fix self index
            Pi[i] = 0.0
            Hdiff = H - logU
            if abs(Hdiff) < tol:
                break
            if Hdiff > 0:
                beta_min = beta
                beta = beta * 2.0 if beta_max == np.inf else (beta + beta_max) / 2
            else:
                beta_max = beta
                beta = beta / 2.0 if beta_min == -np.inf else (beta + beta_min) / 2
        cond[i] = Pi
        sigmas[i] = np.sqrt(1.0 / beta) if beta > 0 else 1.0
    P = (cond + cond.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    P /= P.sum()
    return P, sigmas


def effective_neighbors(cond_row):
    nz = cond_row[cond_row > 1e-12]
    H = -np.sum(nz * np.log2(nz))
    return float(2 ** H)
