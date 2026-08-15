# Audited reproduction summary

This file is generated from the same gated JSON artifacts as the LaTeX report.
The canonical manuscript is `report/diffusion_smc_reproduction_report.tex`; the
verified release PDF is `output/pdf/diffusion_smc_reproduction_report.pdf`.

## Verdicts

| Work | Scope | Result |
|---|---|---|
| Null-TTA | Null-TTA nmax=55 on the SD-v1.5 backbone, 3 seeds x 50 prompts, 4 real scorers | **PASS** |
| DAS | Official GMM Figure 1, matched Python/Torch/Numba seed | **PASS_WITH_VARIANCE** |
| DAS Kaggle | Seed 42 on verified 2 x Tesla T4 allocation; official workload uses one GPU | Infrastructure PASS; paper numeric FAIL |
| FK Steering | One prompt, 3 seeds, k=4 | **PARTIAL** |
| FK Correctors | Exact Table A1, 10k samples / 1000 steps / 5 runs / 6 methods | **PASS_WITH_VARIANCE** |

## Null-TTA

Primary PickScore is 0.315612 across seeds
(paper 0.315; symmetric relative difference
0.19%; hierarchical-bootstrap 95%
CI [0.307728, 0.323986]). Paired improvement is
0.097278, with 95% CI
[0.089257, 0.105752]. Primary PASS requires both
the <=10% point match and an improvement CI strictly above zero.

Direct mapping to the original Table 1: `Paper/Repro SD-v1.5` reproduces the
first row (the unaligned backbone), while `Paper/Repro Null-TTA` reproduces the
last row (the same backbone after null-text embedding optimization).

| Source / method | PickScore | HPS v2 | Aesthetic | ImageReward |
|---|---:|---:|---:|---:|
| Paper / SD-v1.5 | 0.218 | 0.279 | 5.232 | 0.339 |
| Reproduction / SD-v1.5 | 0.218 | 0.279 | 5.230 | 0.337 |
| Paper / Null-TTA nmax=55 | 0.315 | 0.294 | 5.431 | 0.946 |
| Reproduction / Null-TTA nmax=55 | 0.316 | 0.294 | 5.471 | 0.893 |

| Metric | Repro Null-TTA [95% CI] | Paired improvement [95% CI] | Symmetric diff | Match |
|---|---:|---:|---:|---|
| PickScore | 0.316 [0.308, 0.324] | 0.097 [0.089, 0.106] | 0.2% | PASS |
| HPS v2 | 0.294 [0.289, 0.299] | 0.015 [0.012, 0.018] | 0.1% | PASS |
| Aesthetic | 5.471 [5.335, 5.603] | 0.240 [0.121, 0.355] | 0.7% | PASS |
| ImageReward | 0.893 [0.665, 1.111] | 0.556 [0.345, 0.756] | 5.8% | PASS |

| Seed | PickScore base | PickScore optimized |
|---:|---:|---:|
| 42 | 0.218593 | 0.314135 |
| 43 | 0.217488 | 0.317047 |
| 44 | 0.218922 | 0.315655 |

## DAS and Kaggle

The canonical Modal seed 42 paper-style EMD is
0.743420 versus paper 0.82
(9.80%
symmetric difference). The three-seed mean is
1.024873 +/-
0.703813; seed 43 is the bad seed at
1.825858. Kaggle r6 completes on a
verified T4x2 allocation but produces EMD
1.495862 and reward
-3.229685; this is a numeric portability failure,
not an infrastructure failure.

## FK Correctors exact Table A1

The exact six-method protocol completed on an NVIDIA L4 in
17.41 minutes for USD
0.366852. All
30/30 reproduction means lie
inside the paper mean +/- 2 reported run-SD compatibility bands;
24/30 lie inside one SD.
Corrector-versus-no-FKC directions agree in
19/20 comparisons.
The only direction disagreement is target-score systematic Energy-W2, whose
paper and reproduction distributions overlap strongly. Upstream calls the loop
index a seed but does not reseed; this report therefore labels them sequential
runs rather than claiming known seed identities.

## Modal guard

| Profile | Final reported usage (USD) | Canonical app cost | Headroom to 29 |
|---|---:|---:|---:|
| kieusontung6 | 27.68 | 21.41 | 1.32 |
| kieusontung8 | 25.05 | 20.42 | 3.95 |
| phamvanvuhoan | 27.29 | 25.07 | 1.71 |

Provider billing can lag. Canonical cost is joined by the exact 26 scientific app IDs
behind 25 artifact records (one recovered artifact uses a source-generation app plus a
remote post-scoring app). Canaries and canceled attempts are retained separately in
`results/modal_billing_final.json`.

## Evidence

- `results/null_tta_run_manifest.json`: exact 25-artifact coverage, scorer, GPU and hashes.
- `results/null_tta_table1.json`: 150-record aggregate and bootstrap.
- `results/das_rescue_multiseed.json`: matched-RNG DAS results.
- `results/das_kaggle_t4x2_rescue.json`: Kaggle T4x2 portability result.
- `results/fk_correctors_table_a1_exact.json`: exact Table A1 cells, raw runs, variance and provenance.
- `report/final_figure_table_trace.json`: hashes for report inputs.

No local GPU experiment is represented as remote evidence. Upstream repositories
remain pinned and clean; engineering retries without scientific metrics are not
counted as model failures.
