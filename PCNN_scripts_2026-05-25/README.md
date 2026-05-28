# PCNN v7.7 — publication pipeline scripts (2026-05-25)

The **complete, paper-only pipeline** — final model, every pipeline stage, comparison/baseline
models, and figure scripts. **No legacy/dev versions.** All files are copies; originals are untouched
in `piml_ceramic/scripts/`, `NC figures/scripts/`, and the release `src/`. The only code change was
the `CMLTRPCNNv71 → CMLTRPCNN` rename and `41_natcomm_validation → 41_validation`.

## Pipeline order

1. **`features/`** — build the v7 feature set: `02_physics_features`, `03_residual_target`, `33_feature_v7_extend`
2. **`model_training/canonical_v7.7/`** — THE published model:
   - `psrnn_mdpinn.py` — architecture (class **`CMLTRPCNN`**; alias `CMLTRPCNNv71` kept)
   - `47_optuna_formula_split.py` — hyperparameter search
   - `48_retrain_frm_best.py` — final retrain → **`models/cmltrv77_final.pt`**
3. **`data_extraction/`** — per-sample artifacts the analyses/figures read:
   `extract_per_sample_data`, `extract_lofo_predictions`, `extract_shap_per_sample`,
   `extract_counterfactual`, `gen_cv_predictions`, `extract_latent_space`
4. **`evaluation/`** — published results & robustness: `04b_cm_baseline_validcm_v77`,
   `09_shap_analysis`, `39_physics_decomposition`, `40_tilt_proof`, `41_validation`,
   `42_lofo_generality`, `43_counterfactual_permutation`, `43b_asite_flst_table_v77`,
   `43c_saturation_variance_diagnostics_v77`, `44_literature_validation`, `49_ad_v77`,
   `53_ablation_v77`, `54_conformal_prediction`, `55_cs_feature_ablation`,
   `59_physics_law_ablation`, `65_cv_protocol_comparison`, `66_stratgroupkfold_5seed`
5. **`inference_design/`** — discovery: `45_composition_engineering_map`, `46_application_relevance`,
   `61_virtual_screening`, `67_physics_ga_inverse_design`, `68_ga_candidate_ad_recheck_v77`
6. **`model_training/baselines/`** — published comparison only: `57_baseline_comparison`
   (Ridge/RF/XGBoost/CatBoost/BPNN → S6) + `56_pirnn_fair_comparison` (PIRNN)
7. **`figures/`** — 23 `figS*.py` + 6 main `gen_fig*.py` + `FIGURE_MAP.md` + `REGENERATE.md`

**"CMLTRPCNN v7.7" = the `CMLTRPCNN` architecture + best HP + final retrain → `cmltrv77_final.pt`.**
Baselines are Ridge/RF/XGBoost/CatBoost/BPNN/PIRNN. MDPINN/CMLTRPINN/CMAGPINN/PSRNN and CMLTR
v3–v76/v78/v79 are **not** in the paper and were excluded.

See `figures/FIGURE_MAP.md` for figure → script → data.
