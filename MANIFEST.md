# Release Manifest — PCNN 

Generated for Nature Communications open-source deposit.

## What's in this release

| Path | Files | Total size | Purpose |
|------|-------|-----------|---------|
| `LICENSE`, `README.md`, `MANIFEST.md` | 3 | — | Documentation |
| `requirements.txt`, `hyperparameters.txt`, `predict.py` | 3 | — | Environment + CLI |
| `data/` (root) | 3 | ~ 150 KB | Raw εᵣ datasets (train/test/full) |
| `data/processed/` | 3 | ~ 4 MB | Feature matrix + partition + split |
| `model_weights/` | 2 | ~ 60 MB | Trained 5-seed ensemble + scalers |
| `src/model/` | 1 | 80 KB | PyTorch model class |
| `src/physics/` | 13 | 110 KB | CM, LST, tilt physics modules |
| `src/features/` | 1 | 50 KB | Feature engineering |
| `src/training/` | 1 | 25 KB | Final training pipeline |
| `src/inference/` | 1 | 30 KB | Web-app inference module |
| `src/analysis/` | 5 | 50 KB | Per-sample data extraction |
| `scripts/` | 11 | 250 KB | Canonical analysis scripts |
| `figures/` | 24 | 200 KB | Figure-generation scripts |
| `results/` | 19 | 600 KB | Result JSONs |
| `extracted_data/` | 9 | 2 MB | Per-sample CSVs (decomposition, CV, LOFO, …) |
| `figures_output/main/` | 12 | 5 MB | 6 main figures (PDF + PNG) |
| `figures_output/supp/` | 44 | 25 MB | 22 supplementary figures (PDF + PNG) |
| `manuscript/` | 4 | ~ 8 MB | Submission deliverables: `manuscript.docx`, `supplementary.docx`, `captions.md`, `references.json` |
| **Total** | **159** | **~ 103 MB** | |

## Critical files for review

These four files together fully define the model and reproduce the headline numbers:

1. `src/model/model.py` — PyTorch model class (`CMLTRPCNNv71`)
2. `src/training/train_final_model.py` — training pipeline
3. `model_weights/cmltrv77_final.pt` — trained 5-seed state dicts
4. `results/48_cmltrv77_retrain.json` — final metrics (R² = 0.941)

## Figure inventory (28 unique figures)

### Main (6)
- `fig1_motivation` — gap in CM theory, paper motivation
- `fig2_architecture` — PCNN architecture diagram
- `fig3_performance` — parity plot + CV folds + uncertainty calibration
- `fig4_discovery` — inverse design results
- `fig5_causality` — branch ablation + permutation + counterfactual
- `fig6_limits` — honest disclosure of model boundaries

### Supplementary, numbered (15)
- `figS1_dataset_statistics`
- `figS2_error_stratification`
- `figS3_optuna_convergence` ⭐ — HP tuning trajectory (99 trials)
- `figS4_cv_folds`
- `figS5_residuals`
- `figS6_baseline_comparison`
- `figS7_conformal`
- `figS8_lofo`
- `figS9_applicability_domain`
- `figS10_feature_partition`
- `figS11_shap_importance` ⭐ — top-20 SHAP features
- `figS12_lofo_parity` ⭐ — per-sample LOFO scatter
- `figS13_counterfactual_heatmap` ⭐ — A-site swap matrix
- `figS14_permutation_null` ⭐ — null distributions
- `figS15_sigma_conf_vs_error` ⭐ — AD validation

### Inline panels (7) — drop-in subpanels
- `fig_decomposition` (CM 46.2 % / LST +54.1 % / tilt −0.68 % / res +0.44 %)
- `fig_asite_hierarchy` (Pb > Ca > La > Sr > Ba)
- `fig_law_ablation` (2³ factorial)
- `fig_branch_ablation` (ΔR² when removing branches)
- `fig_flst_by_regime`
- `fig_feature_partition`
- `fig_parity_3protocols` (Strat-GSS + Formula-split + Ensemble)

## Reproducibility checksums

These hashes verify that the canonical Script 42 result exactly reproduces
the published LOFO numbers:

| Holdout | n_test | R² (published) | R² (rerun) |
|---------|--------|----------------|------------|
| A-site Pb | 86  | −0.0971 | −0.0971 |
| A-site Ba | 436 |  0.0080 |  0.0080 |
| A-site Ca | 421 |  0.6100 |  0.6100 |
| A-site Sr | 180 |  0.4338 |  0.4338 |
| A-site La | 125 |  0.8486 |  0.8486 |
| Regime Ia | 60  | −7.3780 | −7.3780 |
| Regime Ib | 354 | −0.1577 | −0.1577 |
| Regime II | 331 |  0.6536 |  0.6536 |
| Regime III | 559 | 0.4901 |  0.4901 |

All numbers reproduce bit-perfect — script and data are frozen.
