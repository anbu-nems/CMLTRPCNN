# Figure → script → data map (PCNN v7.7)

One script per figure. Script name matches its output PNG. (Canonical scripts live in
`NC figures/scripts/`; the release ships path-adapted copies in `…/figures/` writing to
`figures_output/`. Superseded/alternate scripts are in `archive/`.)

## Main figures
**Note:** the figure *file* number ≠ the manuscript number — `build_manuscript.js` remaps them.

| Manuscript | Figure file | Script | Key data |
|---|---|---|---|
| **Fig 1** | `fig1_motivation`   | `gen_fig1_motivation.py`   | CM-baseline failure metrics |
| **Fig 2** | `fig4_discovery`    | `gen_fig4_discovery.py`    | `44_literature_validation`, decomposition |
| **Fig 3** | `fig2_architecture` | `gen_fig2_architecture.py` | schematic |
| **Fig 4** | `fig5_causality`    | `gen_fig5_causality.py`    | `53_ablation_v77` |
| **Fig 5** | `fig3_performance`  | `gen_fig3_performance.py`  | `48_retrain`, `57_baseline`, `54_conformal` |
| **Fig 6** | `fig6_limits`       | `gen_fig6_limits.py`       | `42_lofo`, `49_ad`, `61_screening`, `67_ga_*`, `68` |

## Supplementary figures (script name = output PNG)

| Fig | Script | Key data |
|---|---|---|
| S1  | `figS1_dataset_statistics.py`   | `data/raw/mixed_dataset_clean.csv` |
| S2  | `figS2_error_stratification.py` | holdout predictions |
| S3  | `figS3_optuna_convergence.py`   | `results/hp_tuning_optuna.db` (100 trials, 99 complete) |
| S4  | `figS4_cv_folds.py`             | `results/48_cmltrv77_retrain.json` |
| S5  | `figS5_residuals.py`            | `data/processed/test_holdout_predictions.csv` |
| S6  | `figS6_baseline_comparison.py`  | `results/57_baseline_comparison.json` |
| S7  | `figS7_conformal.py`            | `results/54_conformal_v77.json` + holdout |
| S8  | `figS8_lofo.py`                 | `results/42_lofo_generality.json` |
| S9  | `figS9_applicability_domain.py` | `results/49_ad_v77.json` |
| S10 | `figS10_feature_partition.py`   | `data/processed/feature_partition_v7.json` |
| S11 | `figS11_shap_importance.py`     | `results/60_shap_values.npy` + feature_partition |
| S12 | `figS12_lofo_parity.py`         | `extracted_data/lofo_predictions_v77.csv` |
| S13 | `figS13_counterfactual_heatmap.py` | `results/43_counterfactual_permutation.json` |
| S14 | `figS14_permutation_null.py`    | `extracted_data/decomposition_per_sample.csv` |
| S15 | `figS15_sigma_conf_vs_error.py` | `extracted_data/decomposition_per_sample.csv` |
| S16 | `figS16_reaney_plot.py`         | decomposition + tolerance factor |
| S17 | `figS17_asite_radius_trend.py`  | `results/44_literature_validation.json` |
| S18 | `figS18_srca_discovery_map.py`  | `results/45_composition_engineering_map.json` |
| S19 | `figS19_tolerance_histogram.py` | dataset (tolerance factor) |
| S20 | `figS20_crystal_structure.py`   | schematic |
| S21 | `figS21_descriptor_panels.py`   | `extracted_data/decomposition_per_sample.csv` |
| S22 | `figS22_element_attribution.py` | `extracted_data/decomposition_per_sample.csv` (A-site decomposition) |
| S23 | `figS23_latent_space.py`        | `extracted_data/latent_space_trunk.csv` |

**Two filenames kept for build compatibility** (PNG name ≠ current content):
`figS13_counterfactual_heatmap.*` now shows **B-site LST-routing**; `figS22_element_attribution.*`
now shows the **A-site CM/LST/tilt decomposition**. The build (`build_supplementary.js`) reads these
exact PNG names, so the script/PNG names were left unchanged.
