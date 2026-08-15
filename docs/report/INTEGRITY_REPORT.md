# Integrity report

Final verification: 2026-08-12 (UTC+7)

## Overall verdict

Artifact and reporting integrity: **PASS**.

Scientific verdicts remain scope-specific:

- Null-TTA SD-v1.5 Table 1: **PASS** on the pre-specified primary PickScore gate.
- DAS Figure 1 GMM: **PASS_WITH_VARIANCE**. Canonical seed 42 matches the paper point, while seed 43 is a retained bad seed.
- DAS Kaggle T4x2: infrastructure/execution **PASS**; paper numeric match and Modal-to-Kaggle numeric portability **FAIL**.
- FK Steering: **PARTIAL** because it covers one prompt and a model mirror.
- Feynman--Kac Correctors Table A1 GMM: **PASS_WITH_VARIANCE** on the exact 10k/1000/5 six-method protocol.

## Sources and environment identity

All eight bibliography entries resolve. Four official upstream clones are clean at the pinned commits:

| Work | Pinned commit |
|---|---|
| Null-TTA | `337bf73037e9f24e9f844974d3287384abb610bf` |
| DAS | `2f4b2239f29ee59f80359bdfa5ed747b6d855a1e` |
| FK Steering | `9413005dee1e79f80fb4561a4a7ac8eec704281b` |
| Feynman--Kac Correctors | `aa6f5ed4a0ebb91329d4cd5823cc7e77c5e196e6` |

Remote scientific environments use locked `uv` environments. Local execution was limited to orchestration, validation, aggregation, plotting, and document compilation.

## Null-TTA integrity and results

- `results/null_tta_run_manifest.json` validates exactly 25 canonical artifact records backed by 26 scientific Modal app IDs.
- Coverage is exactly 3 seeds x 50 official prompts = 150 non-overlapping records.
- Prompt text, upstream commit, hyperparameters, requested GPU, scoring path, and all four real scorers are checked for every artifact.
- Seeds are exactly `{42, 43, 44}`. The paper does not publish its seed IDs, so this is a disclosed protocol choice.
- The recovered seed-44 slice `[7,11)` uses four fully completed prompt pairs from the canceled L4 source app. Its CSV and eight lossless PNG files are preserved and SHA-256 checked before remote post-scoring. The unfinished suffix `[11,14)` was discarded and regenerated.
- No row from the provider-disabled retry is included.

Primary PickScore results:

| Seed | Baseline | Null-TTA |
|---:|---:|---:|
| 42 | 0.218593 | 0.314135 |
| 43 | 0.217488 | 0.317047 |
| 44 | 0.218922 | 0.315655 |

The 3-seed mean is **0.315612** (paper 0.315), with symmetric relative difference **0.194%** and hierarchical-bootstrap 95% CI **[0.307728, 0.323986]**. Paired improvement is **0.097278**, with 95% CI **[0.089257, 0.105752]**. Both pre-specified primary conditions pass.

Secondary optimized means also remain within the 10% numeric gate: HPS v2 0.293790 versus 0.294, Aesthetic 5.470581 versus 5.431, and ImageReward 0.892808 versus 0.946. These secondary metrics were not used to replace the primary gate.

## DAS and Kaggle integrity

DAS Modal runs seed Python, Torch, NumPy, and Numba consistently. All three sensitivity seeds retain 3/3 target modes:

| Seed | Paper-style EMD | Stable EMD mean +/- SD | SMC reward |
|---:|---:|---:|---:|
| 42 | 0.743420 | 0.546166 +/- 0.104628 | -0.234755 |
| 43 | 1.825858 | 1.677428 +/- 0.173691 | -0.267834 |
| 44 | 0.505342 | 0.440697 +/- 0.086391 | -0.237705 |

Canonical seed 42 differs symmetrically by 9.796% from the paper EMD 0.82 and passes the 10% point gate. The 3-seed mean is 1.024873 with SD 0.703813; seed 43 demonstrates material seed sensitivity and is not excluded.

The Kaggle r6 kernel reached `COMPLETE`, its accelerator probe verifies exactly two Tesla T4 devices, and the official workload completes on one GPU. Its seed-42 EMD is 1.495862 and SMC reward is -3.229685 with 3/3 target-mode coverage. Thus infrastructure is **PASS**, while paper numeric match (58.4% symmetric difference) and cross-backend portability versus Modal (67.2%) are **FAIL**.

Earlier Kaggle r2--r5 attempts terminated before producing `das_metrics.json`; they are engineering failures, not model results. The disabled Modal workspace retry is likewise a provider failure. None are included in scientific aggregates.

## Feynman--Kac Correctors exact Table A1

- Run `fkc-a1-exact-r3-20260812-082401`, Modal app `ap-e0xL65VL6qEKS8vsWQmmhV`, completed on one NVIDIA L4.
- Protocol is exactly 10,000 samples, 1000 integration steps, `dt=0.001`, five sequential stochastic runs and all six no-FKC/BDC/systematic methods.
- The artifact contains exactly 30 method-run rows and all 30 paper/reproduction mean comparisons.
- **24/30** reproduction means are within one paper-reported run SD and **30/30** are within two SD. The largest mean distance is target-score BDC total variation at 1.616 paper SD.
- Corrected-minus-no-FKC directions agree in **19/20** comparisons. The sole disagreement is target-score systematic Energy-W2, for which both experiments report large overlapping run dispersion.
- Bad-run evidence is retained: tempered-noise BDC run 0 has W2 29.925 and MMD 0.176; tempered-noise systematic run 1 has W2 20.849 and MMD 0.075.
- Peak CUDA allocation is 7.993 GiB; notebook runtime is 1044.865 seconds; provider cost is USD 0.366852.

The official notebook's BDC caller omitted forwarding `reset_transition_per_index=False` to a sampler whose default is `True`, despite constructing the one-dimensional accumulator required by the false branch. The runtime copy forwards this existing flag at one call site. Source, runtime and executed notebook hashes are recorded; the upstream clone remains clean. The upstream loop variable is named `seed` but does not reseed, so the report calls these sequential runs rather than inventing seed identities.

## Billing and provenance

The final provider reports expose every canonical scientific app ID. Exact raw reports are archived under `remote_artifacts/modal_billing/` with SHA-256 hashes.

| Modal profile | Final reported usage | Canonical app cost | Headroom to USD 29 |
|---|---:|---:|---:|
| kieusontung6 | 27.682628 | 21.411814 | 1.317372 |
| kieusontung8 | 25.053864 | 20.417324 | 3.946136 |
| phamvanvuhoan | 27.285695 | 25.072419 | 1.714305 |

All workspaces remain below the USD 29 hard guard. Provider billing may lag; this qualification is retained in the machine-readable artifact and manuscript.

`report/final_figure_table_trace.json` records hashes of the nine report inputs and the generated TeX/figure outputs. Figures and tables are generated from gated JSON/NPZ rather than copied from terminal text.

## PDF and package checks

- Final PDF SHA-256: `719fafde7919cf6c7231b112704b861c24261ea6d5f8ff10836584f584e6bc98`.
- `pdf_read_preflight/1.0.0`: **PASS**; declared, enumerated, and reader page counts are all 15; warnings are empty.
- pdfLaTeX/BibTeX log: no undefined citations/references and no overfull boxes.
- Final render set: `report/qa/pages_final_20260812T113529Z/` (15 PNG pages at 130 dpi).
- All 15 pages were rendered at 130 dpi and visually inspected. No clipping, overlap, unreadable table, or broken figure was found. The report now defines every metric used by Null-TTA, DAS, FK Steering and FK Correctors, including implementation-level aggregation and direction. The Null-TTA point-estimate table uses methods/sources as rows and metrics as columns; uncertainty and match gates are separated into a second table. The exact Table A1 section retains a two-row Paper/Repro layout so all 30 cells remain legible.
- Targeted secret-pattern scan covers scripts, reports, results, and text/JSON remote artifacts; credentials are excluded from the report package.

## Remaining limitations

Null-TTA seed IDs are not published by the paper. Seed 44 uses a memory-reduced sequential scorer path and mixed L4/L40S hardware; secondary scorers operate on the lossless PNGs saved by the official example after its tensor-to-uint8 conversion. DAS covers the toy GMM Figure 1, not the paper's text-to-image experiments. Kaggle provides a single matched seed and therefore identifies numeric drift without isolating its cause. FK Steering covers one prompt. FK Correctors now uses the exact Table A1 setting, but the paper's five run identities and author-side BDC code state are not published; the two-SD criterion is a descriptive compatibility band, not an equivalence test.
