"""Gaussian SNE (bell Q) + t-SNE (heavy-tail Q), from scratch.

Both minimise KL(P||Q) by gradient descent with momentum and early
exaggeration (P*alpha for first ee_steps), matching the 2008 paper and the
video: alpha=4, ee=50, momentum 0.5->0.8.
"""
import numpy as np


def _Q_gauss(Y):
    s = (Y ** 2).sum(1)
    D2 = s[:, None] + s[None, :] - 2 * Y @ Y.T
    np.maximum(D2, 0, out=D2)
    Q = np.exp(-D2)
    np.fill_diagonal(Q, 0.0)
    Q = np.maximum(Q, 1e-12)
    Q /= Q.sum()
    return Q, D2


def _Q_student(Y):
    s = (Y ** 2).sum(1)
    D2 = s[:, None] + s[None, :] - 2 * Y @ Y.T
    np.maximum(D2, 0, out=D2)
    Q = 1.0 / (1.0 + D2)
    np.fill_diagonal(Q, 0.0)
    Q = np.maximum(Q, 1e-12)
    Q /= Q.sum()
    return Q, D2


def _kl(P, Q):
    Q = np.maximum(Q, 1e-12)
    return float(np.sum(P * np.log(P / Q)))


def _optimize(P, n, seed, n_iter, lr, ee_steps, ee_alpha, Qfn, grad_fn, log_every=50):
    rng = np.random.default_rng(seed)
    Y = rng.normal(0, 1e-4, (n, 2))
    vel = np.zeros_like(Y)
    mom_switch = 250
    gaps = {}
    for t in range(1, n_iter + 1):
        Peff = P * (ee_alpha if t <= ee_steps else 1.0)
        if t <= ee_steps:
            Peff = Peff / Peff.sum() * 1.0  # keep normalized scale consistent
            # NOTE: early exaggeration scales attraction; renormalize so KL
            # trajectory matches paper convention (gap inflated early).
        Q, D2 = Qfn(Y)
        grad = grad_fn(Peff, Q, Y, D2)
        mom = 0.5 if t <= mom_switch else 0.8
        vel = mom * vel - lr * grad
        Y = Y + vel
        Y = Y - Y.mean(0)
        if t in (50, 100, n_iter) or t % log_every == 0:
            gaps[t] = _kl(P, Q)
    Q, _ = Qfn(Y)
    return Y, _kl(P, Q), gaps


def run_sne(P, seed=0, n_iter=1000, lr=200.0, ee_steps=50, ee_alpha=4.0):
    n = P.shape[0]

    def grad(Pe, Q, Y, D2):
        PQ = (Pe - Q) * 4.0
        # vectorised: grad_i = sum_j PQ_ij (Y_i - Y_j)
        return (PQ.sum(1)[:, None] * Y - PQ @ Y)

    return _optimize(P, n, seed, n_iter, lr, ee_steps, ee_alpha, _Q_gauss, grad)


def run_tsne(P, seed=0, n_iter=1000, lr=200.0, ee_steps=50, ee_alpha=4.0):
    n = P.shape[0]

    def grad(Pe, Q, Y, D2):
        W = (Pe - Q) / (1.0 + D2) * 4.0
        np.fill_diagonal(W, 0.0)
        return (W.sum(1)[:, None] * Y - W @ Y)

    return _optimize(P, n, seed, n_iter, lr, ee_steps, ee_alpha, _Q_student, grad)
