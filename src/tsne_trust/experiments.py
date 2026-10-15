"""Full experiment pipeline: every number in the video, re-measured."""
import json
import numpy as np
from .data import load_digits600, make_tight_loose, make_noise, make_abc
from .distances import pairwise_euclidean, knn_indices, knn_same_label_rate
from .perplexity import joint_P
from .sne import run_sne, run_tsne
from .mds import run_mds
from . import metrics
from .market import live_return_windows


def run_all(fast=True, out="results/results.json"):
    R = {}
    n_iter = 300 if fast else 1000
    X, y = load_digits600()
    R["n"] = len(X)
    D = pairwise_euclidean(X)
    # Video narrative pair: a three, its nearest fellow three, and the
    # nearest seven to that same three (measured live, not hardcoded).
    threes = np.where(y == 3)[0]
    sevens = np.where(y == 7)[0]
    t0 = threes[0]
    t1 = threes[np.argsort(D[t0][threes])[1]]
    s0 = sevens[np.argsort(D[t0][sevens])[0]]
    R["dist_three_three"] = float(D[t0, t1])
    R["dist_three_seven"] = float(D[t0, s0])
    knn10 = knn_indices(D, 10)
    R["highdim_same_label_10nn"] = float(knn_same_label_rate(y, knn10) if False else
                                        (y[knn10] == y[:, None]).mean())
    # MDS baseline
    Ym, merr = run_mds(X, n_iter=200 if fast else 500)
    Dm = pairwise_euclidean(Ym)
    R["mds_mean_rel_err"] = merr
    R["mds_same_label"] = metrics.same_label_rate(y, Dm)
    R["mds_kept"] = metrics.kept_overlap(D, Dm)
    # SNE gaussian
    P30, sig = joint_P(D, 30.0)
    R["sigma_min"], R["sigma_max"] = float(sig.min()), float(sig.max())
    Ys, gap_s, gs = run_sne(P30, n_iter=n_iter)
    Ds = pairwise_euclidean(Ys)
    R["sne_gaps"] = {str(k): v for k, v in gs.items()}
    R["sne_gap_final"] = gap_s
    R["sne_same_label"] = metrics.same_label_rate(y, Ds)
    R["sne_kept"] = metrics.kept_overlap(D, Ds)
    # t-SNE
    Yt, gap_t, gt = run_tsne(P30, n_iter=n_iter)
    Dt = pairwise_euclidean(Yt)
    R["tsne_gaps"] = {str(k): v for k, v in gt.items()}
    R["tsne_gap_final"] = gap_t
    R["tsne_same_label"] = metrics.same_label_rate(y, Dt)
    R["tsne_kept"] = metrics.kept_overlap(D, Dt)
    R["tsne_trustworthiness"] = metrics.trustworthiness(D, Dt)
    R["tsne_foreign_ratio"] = metrics.nearest_foreign_ratio(Yt, y)
    R["high_spread_1_over_0"] = metrics.high_dim_spread(X[y == 1]) / metrics.high_dim_spread(X[y == 0])
    sp = metrics.island_spreads(Yt, y)
    R["map_spread_1_over_0"] = sp[1] / (sp[0] + 1e-12)
    R["map_spreads"] = sp
    # seed stability x5
    gaps, overlaps, nearest_ok = [], [], []
    base_knn = knn_indices(Dt, 10)
    # high-dim centroid nearest digit
    from collections import defaultdict
    C = {k: X[y == k].mean(0) for k in sorted(set(y.tolist()))}
    true_nearest = {}
    for a in C:
        best, bd = None, 1e18
        for b in C:
            if a == b:
                continue
            d = float(np.linalg.norm(C[a] - C[b]))
            if d < bd:
                bd, best = d, b
        true_nearest[a] = best
    R["true_centroid_nearest"] = true_nearest
    for s in range(5):
        Yx, gx, _ = run_tsne(P30, seed=100 + s, n_iter=n_iter)
        Dx = pairwise_euclidean(Yx)
        gaps.append(gx)
        overlaps.append(metrics.kept_overlap(Dt, Dx))
        # centroid nearest agreement
        Cx = {k: Yx[y == k].mean(0) for k in sorted(set(y.tolist()))}
        ok = 0
        for a in Cx:
            best, bd = None, 1e18
            for b in Cx:
                if a == b:
                    continue
                d = float(np.linalg.norm(Cx[a] - Cx[b]))
                if d < bd:
                    bd, best = d, b
            # compare against high-dim truth? count how many match true
            if best == true_nearest[a]:
                ok += 1
        nearest_ok.append(ok)
    R["seed_gaps"] = gaps
    R["seed_overlap_mean"] = float(np.mean(overlaps))
    R["seed_nearest_ok"] = nearest_ok
    # perplexity sweep
    for perp in [2, 30, 300]:
        Px, _ = joint_P(D, float(perp))
        Yx, gx, _ = run_tsne(Px, n_iter=n_iter)
        Dx = pairwise_euclidean(Yx)
        R[f"perp{perp}_kept"] = metrics.kept_overlap(D, Dx)
        R[f"perp{perp}_same"] = metrics.same_label_rate(y, Dx)
        R[f"perp{perp}_gap"] = gx
    # tight vs loose
    T, L = make_tight_loose()
    R["tight_spread"] = metrics.high_dim_spread(T - T.mean(0) + T.mean(0)) if False else float(
        np.linalg.norm(T - T.mean(0), axis=1).mean())
    R["loose_spread"] = float(np.linalg.norm(L - L.mean(0), axis=1).mean())
    Xtl = np.vstack([T, L])
    Dtl = pairwise_euclidean(Xtl)
    Ptl, _ = joint_P(Dtl, 30.0)
    Ytl, _, _ = run_tsne(Ptl, n_iter=n_iter)
    s0 = float(np.linalg.norm(Ytl[:len(T)] - Ytl[:len(T)].mean(0), axis=1).mean())
    s1 = float(np.linalg.norm(Ytl[len(T):] - Ytl[len(T):].mean(0), axis=1).mean())
    R["map_tight_spread"] = s0
    R["map_loose_spread"] = s1
    # noise hallucinations
    N = make_noise()
    Dn = pairwise_euclidean(N)
    for perp in [2, 30]:
        Pn, _ = joint_P(Dn, float(perp))
        Yn, _, _ = run_tsne(Pn, n_iter=n_iter)
        R[f"noise_perp{perp}_tenth_over_first"] = metrics.tenth_over_first(Yn)
        R[f"noise_perp{perp}_clump"] = metrics.clump_stats(Yn)
    R["random_tenth_over_first_ref"] = 4.0
    # A-B-C exercise (n=200/group so perp450 < n; verified live)
    Xa, ya = make_abc(n=200)
    Da = pairwise_euclidean(Xa)
    for perp in [30, 450]:
        Pa, _ = joint_P(Da, float(perp))
        Ya, _, _ = run_tsne(Pa, n_iter=n_iter)
        ca = Ya[ya == 0].mean(0)
        cb = Ya[ya == 1].mean(0)
        cc = Ya[ya == 2].mean(0)
        R[f"abc_perp{perp}_c_over_b"] = float(np.linalg.norm(cc - ca) / (np.linalg.norm(cb - ca) + 1e-12))
    # live market
    Xm, ym, live = live_return_windows()
    Dmkt = pairwise_euclidean(Xm)
    Pmkt, _ = joint_P(Dmkt, 30.0)
    Ymkt, gmkt, _ = run_tsne(Pmkt, n_iter=n_iter)
    Dm2 = pairwise_euclidean(Ymkt)
    R["market_live"] = bool(live)
    R["market_n"] = len(Xm)
    R["market_gap"] = gmkt
    R["market_kept"] = metrics.kept_overlap(Dmkt, Dm2)
    R["market_trust"] = metrics.trustworthiness(Dmkt, Dm2)
    with open(out, "w") as f:
        json.dump(R, f, indent=2, default=float)
    return R
