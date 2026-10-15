"""Generate all README/Pages visuals from MEASURED data. No hand-drawn art.

Outputs (repo-root assets/):
  hero_islands.png   - t-SNE islands map, 600 digits, colored by digit
  benchmark_bars.png - same-digit 10NN + kept-neighbourhood bars
  size_illusion.png  - tight/loose spreads: high-dim vs map
  demo.gif           - islands forming during optimization (300 pts, 16 frames)

Run: python scripts/make_figures.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tsne_trust.data import load_digits600, make_tight_loose
from tsne_trust.distances import pairwise_euclidean
from tsne_trust.perplexity import joint_P
from tsne_trust.sne import run_tsne
from tsne_trust.mds import run_mds
from tsne_trust import metrics

ROOT = os.path.join(os.path.dirname(__file__), "..")
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

PALETTE = plt.cm.tab10.colors


def main():
    X, y = load_digits600()
    D = pairwise_euclidean(X)

    # ---- hero islands (full 600, measured) ----
    P30, _ = joint_P(D, 30.0)
    Yt, gap, _ = run_tsne(P30, n_iter=300)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for k in range(10):
        m = y == k
        ax.scatter(Yt[m, 0], Yt[m, 1], s=9, color=PALETTE[k], label=str(k), alpha=0.85)
    ax.set_title("t-SNE islands: 600 handwritten digits (from-scratch, KL=%.2f)" % gap)
    ax.legend(ncol=5, fontsize=8, loc="upper right")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "hero_islands.png"), dpi=110)
    plt.close(fig)

    # ---- benchmark bars (from results.json where possible) ----
    R = json.load(open(os.path.join(ROOT, "results", "results.json")))
    cats = ["High-dim\npictures", "MDS\nbaseline", "Gaussian\nSNE", "t-SNE\n(this repo)"]
    same = [R["highdim_same_label_10nn"], R["mds_same_label"],
            R["sne_same_label"], R["tsne_same_label"]]
    kept = [1.0, R["mds_kept"], R["sne_kept"], R["tsne_kept"]]
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, np.array(same) * 100, 0.4, label="same-digit 10NN (%)")
    ax.bar(x + 0.2, np.array(kept) * 100, 0.4, label="neighbourhoods kept (%)")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.set_title("Benchmark: what each method preserves (measured)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "benchmark_bars.png"), dpi=110)
    plt.close(fig)

    # ---- size illusion ----
    T, L = make_tight_loose()
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    axes[0].bar(["tight\n(spread %.2f)" % R["tight_spread"],
                 "loose\n(spread %.2f)" % R["loose_spread"]],
                [R["tight_spread"], R["loose_spread"]], color=["#2ca02c", "#d62728"])
    axes[0].set_title("High-dim truth: 9x different")
    axes[1].bar(["tight island\n(%.2f)" % R["map_tight_spread"],
                 "loose island\n(%.2f)" % R["map_loose_spread"]],
                [R["map_tight_spread"], R["map_loose_spread"]],
                color=["#2ca02c", "#d62728"])
    axes[1].set_title("On the map: identical")
    fig.suptitle("Do NOT trust island size (measured)")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "size_illusion.png"), dpi=110)
    plt.close(fig)

    # ---- demo GIF: islands forming (300-pt subset, fast) ----
    Xs, ys = X[:300], y[:300]
    Ds = pairwise_euclidean(Xs)
    Ps, _ = joint_P(Ds, 30.0)
    _, _, _, traj = run_tsne(Ps, n_iter=160, capture_every=10)
    lim = float(np.abs(traj).max()) * 1.05
    frames = []
    from PIL import Image
    for i, Y in enumerate(traj):
        fig, ax = plt.subplots(figsize=(5, 4))
        for k in range(10):
            m = ys == k
            ax.scatter(Y[m, 0], Y[m, 1], s=8, color=PALETTE[k], alpha=0.85)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("t-SNE step %d: islands forming" % ((i + 1) * 10))
        fig.tight_layout()
        p = os.path.join(ASSETS, "_frame.png")
        fig.savefig(p, dpi=80)
        plt.close(fig)
        frames.append(Image.open(p).convert("P", palette=Image.ADAPTIVE))
    gpath = os.path.join(ASSETS, "demo.gif")
    frames[0].save(gpath, save_all=True, append_images=frames[1:],
                   duration=350, loop=0, optimize=True)
    os.remove(os.path.join(ASSETS, "_frame.png"))
    print("wrote assets/, demo.gif %.1f MB" % (os.path.getsize(gpath) / 1e6))


if __name__ == "__main__":
    main()
