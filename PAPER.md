# Which Parts of a t-SNE Map Can You Trust?
## A From-Scratch Replication and Extension Study on 8×8 Digits and Live Market Data

**Authors:** tsne-trust-study contributors (2026)
**Status:** manuscript draft — all empirical claims machine-verified; see `results/results.json`.
**Reproduction:** `docker compose up` (≈10 min) or `pip install -r requirements.txt && pytest -q && python scripts/run_all.py`.

---

## Abstract

t-Distributed Stochastic Neighbor Embedding (t-SNE) maps are among the most
reproduced figures in machine learning, yet readers routinely over-interpret
them. We re-implement SNE, t-SNE, classical MDS and the full perplexity
machinery **from scratch (NumPy only)** and re-measure every quantitative
claim of the classic "islands" narrative on 600 public 8×8 handwritten digits:
(1) high-dimensional 10-neighbourhoods are 91.0% same-digit; (2) a
distance-preserving baseline collapses (56.1% same-digit, 22.9% neighbourhoods
kept); (3) Gaussian SNE reaches 80.8%/44.5% at KL 1.05; (4) the Student-t fix
reaches 92.5%/65.8% at KL 0.45 with trustworthiness 0.98 and ~10×
foreign-island separation; (5) island sizes are erased by per-point bandwidth
adaptation (a 9× spread ratio maps to equal islands); (6) inter-island
placement is initialization-driven (5 seeds recover the true nearest island
for only 4–6 of 10 digits while 83% of point neighbourhoods persist);
(7) perplexity 2 fragments and 300 merges (U-shaped quality); (8) a
1-vs-10 distance construction maps to ~1× at perplexity 30 and ~6.3× at 450.
We add two novel observations: (i) hallucinated clumps in pure noise are
invisible to the standard median 10th/1st-neighbour diagnostic and appear
only in the distribution tail (p90 11.37 vs 8.87); (ii) the KL objective
cannot select perplexity (perplexity 300 attains lower KL, 0.21 vs 0.45,
while preserving fewer neighbours). Finally we replicate the trust rules on
160 **live** Yahoo Finance return windows (trustworthiness 0.88). All 34
verification tests pass.

## 1. Introduction

Since van der Maaten & Hinton (2008), t-SNE maps with well-separated
"islands" have appeared in thousands of papers. Because the algorithm adapts
nonlinearly and per-region, readers infer cluster sizes, inter-cluster
distances and global geometry that the method never promised (Wattenberg et
al., 2016; Nguyen & Holmes, 2019). Recent work formalizes unreliability via
leave-one-out map discontinuities (Liu, Ma & Zhong, 2025) and interactive
distortion diagnostics (Sankaran et al., 2026). Missing is a minimal,
fully-reproducible artifact that rebuilds the method from an empty file,
measures each classic claim, and states exactly what survives. This study is
that artifact.

## 2. Method

**Data.** 600 stratified digits from `sklearn.datasets.load_digits` (8×8,
ink ∈ [0,1], 64 dims). **Distances.** Euclidean over 64 pixels.
**Perplexity.** Per-point Gaussian bandwidth via 60-iteration binary search
(target perplexity 30 unless stated); conditionals symmetrized to a joint
distribution over ~180k pairs. **Optimizers.** Gaussian SNE and t-SNE by
gradient descent with momentum (0.5 → 0.8) and early exaggeration (×4, 50
steps), matching the 2008 paper. **Baseline.** Classical MDS by
double-centering eigendecomposition (strain-optimal linear reference).
**Metrics.** Same-label 10NN rate, kept-neighbourhood fraction, Venna's
trustworthiness, KL gap, island spreads, centroid distances, 10th/1st ratio
distribution. Core: NumPy only (`src/tsne_trust/`); scikit-learn supplies
data loading alone.

## 3. Results

All values measured; see `results/results.json` and README §3 for the
claim-by-claim table.

*Sorting without labels.* High-dim 10NN same-digit rate 91.0% — neighbourhood
geometry already encodes digit identity; t-SNE merely preserves it (map:
92.5%, i.e. the embedding tidies beyond its input).
*Why the naive map fails.* Classical MDS: 51% mean relative distance error,
56.1% same-digit, 22.9% kept. Far pairs outnumber near pairs and dominate.
*Crowding and the t fix.* Gaussian SNE: KL 1.05, groups touch. Student
`1/(1+d²)` tail (800× heavier than the bell at d=3): KL 0.45, foreign islands
at 9.8× nearest distance.
*Sizes lie.* Tight/loose copy-groups (spreads 0.131/1.174) → equal map
islands (0.637/0.645); real ones-vs-zeros (1.53×) → 2.66× map exaggeration.
*Gaps are dice.* Seed KL range 0.43–0.50; nearest-island recovery 4–6/10;
point-neighbour overlap 83%.
*Perplexity U-curve.* Kept: 60.7% (p=2) / 65.8% (p=30) / 48.4% (p=300).
*A–B–C.* Map ratio 1.05 at p=30 → 6.31 at p=450 for a true 10× construction.
*Market replication.* 160 live 30-day return windows: KL 0.44,
trustworthiness 0.88, kept 53.8% — same trust boundary on non-image data.

## 4. Novel observations

**4.1 Tail diagnostic for hallucinated clumps.** On pure-noise images the
median 10th/1st ratio cannot separate perplexity 2 from 30 (3.71 vs 3.88).
Mean (6.05 vs 5.79) and p90 (11.37 vs 8.87) can. Recommendation: reliability
dashboards should report the tail of the neighbour-ratio distribution.

**4.2 KL cannot select perplexity (measured).** Perplexity 300: KL 0.21;
perplexity 30: KL 0.45 — yet 30 preserves more neighbours (65.8% vs 48.4%).
This reproduces Nguyen & Holmes' guidance with exact numbers and warns
against KL-based model selection, still seen in practice.

**4.3 Construction sensitivity.** The size-equalization demo reproduces only
with real digit pictures as perturbation bases; uniform-random bases preserve
the 9× size ratio on the map. Synthetic controls must match data modality.

## 5. Threats to validity

Single dataset scale (600 points, 300 fast iterations; full 1000-iter profile
provided); Euclidean metric fixed; closed-form MDS baseline differs from an
iterative stress optimizer (51% vs reported 28% error — same qualitative
verdict, documented); market window is one 2-year regime sample.

## 6. Conclusion

Trust t-SNE for *who neighbours whom*; distrust sizes, gaps, and single
maps; sweep seeds and perplexities; never select perplexity by KL; check the
tail, not the median, for spurious clumps. Our artifact makes each rule a
runnable test.

## References

[1] L. van der Maaten & G. Hinton, JMLR 9:2579–2605, 2008.
[2] G. Hinton & S. Roweis, NeurIPS, 2002.
[3] M. Wattenberg et al., Distill, 2016.
[4] L. van der Maaten, JMLR, 2014 (Barnes-Hut-SNE).
[5] D. Kobak & P. Berens, Nature Communications, 2019.
[6] L. H. Nguyen & S. Holmes, PLoS Comput. Biol., 2019.
[7] Scagnostics-based t-SNE hyperparameter guidance, 2022.
[8] J. Lin et al., noise-aware DR calibration framework, 2024.
[9] Z. Liu, R. Ma & Y. Zhong, Nature Communications, 2025 (LOO-map).
[10] K. Sankaran et al., Briefings in Bioinformatics, 2026 (distortions).
[11] L. McInnes et al., UMAP, 2018.
