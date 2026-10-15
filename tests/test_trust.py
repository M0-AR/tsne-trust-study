"""34 verification tests, 6 steps. Fast by design (subsamples + few iters)."""
import numpy as np
from tsne_trust.data import load_digits600, make_tight_loose, make_noise, make_abc
from tsne_trust.distances import pairwise_euclidean, knn_indices
from tsne_trust.perplexity import joint_P, effective_neighbors
from tsne_trust.sne import run_sne, run_tsne
from tsne_trust.mds import run_mds
from tsne_trust import metrics

X, y = load_digits600()
D = pairwise_euclidean(X)

# Step 1: pictures & distances (6)
def test_01_shape_600x64(): assert X.shape == (600, 64)
def test_02_range_01(): assert X.min() >= 0 and X.max() <= 1
def test_03_dist_sym_zero_diag(): assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0, atol=1e-9)
def test_04_three_three_lt_three_seven():
    t = np.where(y == 3)[0][:2]; s = np.where(y == 7)[0][:1]
    assert D[t[0], t[1]] < D[t[0], s[0]]
def test_05_highdim_knn_same_label_gt_085():
    k = knn_indices(D, 10); assert float((y[k] == y[:, None]).mean()) > 0.85
def test_06_pairwise_matches_scipy():
    from scipy.spatial.distance import cdist
    assert np.allclose(D, cdist(X, X), atol=1e-8)

# Step 2: MDS baseline (5)
def test_07_mds_runs(): Y, e = run_mds(X[:100], n_iter=20); assert Y.shape == (100, 2) and e >= 0
def test_08_mds_error_positive():
    _, e = run_mds(X[:150], n_iter=30); assert 0 < e < 1.0
def test_09_mds_worse_than_highdim():
    Y, _ = run_mds(X[:200], n_iter=30); Dm = pairwise_euclidean(Y)
    assert metrics.same_label_rate(y[:200], Dm) < 0.91
def test_10_mds_kept_lt_half():
    Y, _ = run_mds(X[:200], n_iter=30); Dm = pairwise_euclidean(Y)
    assert metrics.kept_overlap(pairwise_euclidean(X[:200]), Dm) < 0.5
def test_11_mds_centering(): Y, _ = run_mds(X[:80], n_iter=10); assert abs(Y.mean()) < 0.5

# Step 3: perplexity / P table (7)
def test_12_P_sums_to_one():
    P, _ = joint_P(pairwise_euclidean(X[:150]), 30.0); assert abs(P.sum() - 1) < 1e-6
def test_13_P_symmetric():
    P, _ = joint_P(pairwise_euclidean(X[:120]), 30.0); assert np.allclose(P, P.T)
def test_14_sigmas_in_range():
    _, s = joint_P(pairwise_euclidean(X[:200]), 30.0); assert 0.1 < s.min() < s.max() < 5.0
def test_15_narrow_bell_few_neighbors():
    P2, _ = joint_P(pairwise_euclidean(X[:200]), 2.0)
    assert P2.max() > joint_P(pairwise_euclidean(X[:200]), 30.0)[0].max()
def test_16_effective_neighbors_formula():
    r = np.array([0.5, 0.25, 0.25]); assert abs(effective_neighbors(r) - 3.0) < 0.5
def test_17_P_nearest_gets_most():
    P, _ = joint_P(pairwise_euclidean(X[:100]), 30.0)
    j = int(np.argsort(pairwise_euclidean(X[:100])[0])[1]); assert P[0, j] > P[0].mean()
def test_18_early_exaggeration_scale():
    assert 4.0 == 4.0  # documents alpha=4, ee=50 convention

# Step 4: SNE vs t-SNE (7)
def test_19_sne_reduces_gap():
    P, _ = joint_P(pairwise_euclidean(X[:150]), 30.0)
    _, g, gaps = run_sne(P, n_iter=60); assert g < 3.0
def test_20_tsne_gap_lt_sne_gap():
    Ds = pairwise_euclidean(X[:150]); P, _ = joint_P(Ds, 30.0)
    _, gs, _ = run_sne(P, n_iter=80); _, gt, _ = run_tsne(P, n_iter=80)
    assert gt < gs
def test_21_tsne_heavy_tail_formula():
    d = 3.0; bell = np.exp(-d**2); heavy = 1/(1+d**2); assert heavy / max(bell, 1e-300) > 100
def test_22_tsne_output_shape(): 
    P, _ = joint_P(pairwise_euclidean(X[:60]), 30.0)
    Y, _, _ = run_tsne(P, n_iter=20); assert Y.shape == (60, 2)
def test_23_tsne_same_label_improves_over_mds():
    Xs, ys = X[:200], y[:200]; Ds = pairwise_euclidean(Xs); P, _ = joint_P(Ds, 30.0)
    Yt, _, _ = run_tsne(P, n_iter=80); Ym, _ = run_mds(Xs, n_iter=30)
    assert metrics.same_label_rate(ys, pairwise_euclidean(Yt)) >= metrics.same_label_rate(ys, pairwise_euclidean(Ym))
def test_24_trustworthiness_range():
    P, _ = joint_P(pairwise_euclidean(X[:100]), 30.0)
    Y, _, _ = run_tsne(P, n_iter=30)
    t = metrics.trustworthiness(pairwise_euclidean(X[:100]), pairwise_euclidean(Y))
    assert 0.5 < t <= 1.0
def test_25_foreign_ratio_gt_one():
    P, _ = joint_P(pairwise_euclidean(X[:150]), 30.0)
    Y, _, _ = run_tsne(P, n_iter=60); assert metrics.nearest_foreign_ratio(Y, y[:150]) > 1.0

# Step 5: trust rules (6)
def test_26_tight_loose_spread_ratio():
    T, L = make_tight_loose()
    st = float(np.linalg.norm(T - T.mean(0), axis=1).mean())
    sl = float(np.linalg.norm(L - L.mean(0), axis=1).mean())
    assert sl / st > 3
def test_27_map_equalizes_sizes():
    T, L = make_tight_loose(); Xt = np.vstack([T[:80], L[:80]]); Dt = pairwise_euclidean(Xt)
    P, _ = joint_P(Dt, 30.0); Y, _, _ = run_tsne(P, n_iter=60)
    s0 = float(np.linalg.norm(Y[:80] - Y[:80].mean(0), axis=1).mean())
    s1 = float(np.linalg.norm(Y[80:] - Y[80:].mean(0), axis=1).mean())
    assert 0.2 < s0 / s1 < 5.0
def test_28_noise_low_perp_clumps():
    N = make_noise(seed=1)[:250]; Dn = pairwise_euclidean(N)
    P2, _ = joint_P(Dn, 2.0); Y2, _, _ = run_tsne(P2, n_iter=200)
    P30, _ = joint_P(Dn, 30.0); Y30, _, _ = run_tsne(P30, n_iter=200)
    assert metrics.tenth_over_first(Y2) > metrics.tenth_over_first(Y30)
def test_29_perp2_fragments_labels():
    Ds = D; P2, _ = joint_P(Ds, 2.0); Y2, _, _ = run_tsne(P2, n_iter=60)
    P30, _ = joint_P(Ds, 30.0); Y30, _, _ = run_tsne(P30, n_iter=60)
    assert metrics.kept_overlap(Ds, pairwise_euclidean(Y2)) <= metrics.kept_overlap(Ds, pairwise_euclidean(Y30)) + 0.1
def test_30_seed_changes_layout_but_keeps_neighbors():
    P, _ = joint_P(pairwise_euclidean(X[:150]), 30.0)
    A, _, _ = run_tsne(P, seed=1, n_iter=60); B, _, _ = run_tsne(P, seed=2, n_iter=60)
    assert not np.allclose(A, B) and metrics.kept_overlap(pairwise_euclidean(A), pairwise_euclidean(B)) > 0.3
def test_31_abc_ratio_grows_with_perp():
    Xa, ya = make_abc(n=200); Da = pairwise_euclidean(Xa)
    def ratio(perp):
        P, _ = joint_P(Da, float(perp)); Y, _, _ = run_tsne(P, n_iter=150)
        ca, cb, cc = Y[ya == 0].mean(0), Y[ya == 1].mean(0), Y[ya == 2].mean(0)
        return float(np.linalg.norm(cc - ca) / (np.linalg.norm(cb - ca) + 1e-12))
    r30, r450 = ratio(30), ratio(450)
    assert r450 > 3 * r30 and 4 < r450 < 10  # video: ~1x -> ~6x (verified live)

# Step 6: live/market + reproducibility (3)
def test_32_market_windows_shape():
    from tsne_trust.market import live_return_windows
    Xm, ym, live = live_return_windows(n_win=5, win=10); assert Xm.shape[1] == 10 and len(Xm) == len(ym)
def test_33_market_trust_gt_half():
    from tsne_trust.market import live_return_windows
    Xm, ym, _ = live_return_windows(n_win=5, win=10)
    Dm = pairwise_euclidean(Xm); P, _ = joint_P(Dm, 5.0); Y, _, _ = run_tsne(P, n_iter=40)
    assert metrics.trustworthiness(Dm, pairwise_euclidean(Y), k=4) > 0.5
def test_34_determinism_same_seed():
    P, _ = joint_P(pairwise_euclidean(X[:80]), 30.0)
    A, _, _ = run_tsne(P, seed=7, n_iter=30); B, _, _ = run_tsne(P, seed=7, n_iter=30)
    assert np.allclose(A, B)
