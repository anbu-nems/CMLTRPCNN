# NMI baseline experiments (v1.2.0 additions)

Experiments added for the Nature Machine Intelligence submission, extending the
v1.1.1 release with interpretable-ML baselines, soft-penalty comparisons,
Reaney-regime out-of-distribution stress tests, and architectural-variant studies.

Run scripts from `nmi_baselines/scripts/` (paths resolve relative to the repo
root, layout as in this release: `data/processed/`, `models/`). Override the
root by editing the `ROOT` assignment at the top of each script if your layout
differs.

## Supplementary-item reproduction map

| Supplementary item | Script(s) | Result file(s) |
|---|---|---|
| Table S3 (interpretable-baseline comparison: GAM / EBM / PMN / soft-penalty) | `04_ml_literature_baselines.py`, `01_loss_penalty_pinn.py` | `04_ml_literature_baselines.json`, `01_loss_penalty_pinn.json` |
| Table S7 (performance summary — R-LOFO rows) | `03_ood_sign_violation_test.py` | `03_ood_sign_violation_test.json` |
| Fig. S25 (A-site f_LST hierarchy by family and regime) | `05_subclass_generalizability.py`, `13_standalone_figures.py` | `05_subclass_generalizability.json` |
| Fig. S26 (interpretable-baseline comparison) | `04_ml_literature_baselines.py`, `13_standalone_figures.py` | `04_ml_literature_baselines.json` |
| Fig. S27 (R-LOFO sign-preservation diagnostic) | `03_ood_sign_violation_test.py`, `13_standalone_figures.py` | `03_ood_sign_violation_test.json` |
| Fig. S28 (variant accuracy: Mono / Quant / Laplace) | `08_monotonic_lst_head.py`, `09_quantile_conformal_heads.py`, `10_laplace_last_layer.py`, `13_standalone_figures.py` | `08_*.json`, `09_*.json`, `10_*.json` |
| Fig. S29 (variant UQ coverage, Wilson 95% CI) | `09_quantile_conformal_heads.py`, `10_laplace_last_layer.py`, `12_calibration_curves.py`, `13_standalone_figures.py` | `09_*.json`, `10_*.json`, `12_calibration_curves.json` |
| Fig. S30 (variant combined comparison) | `15_combined_fig6_variants.py` | as above |
| Soft-penalty λ sweep (Discussion; λ_sign robustness) | `02_loss_penalty_lambda_sweep.py` | `02_loss_penalty_lambda_sweep.json` |

## Script index

| Script | Purpose |
|---|---|
| `01_loss_penalty_pinn.py` | Soft-penalty PINN baseline (sign constraints in the loss, not the architecture) |
| `02_loss_penalty_lambda_sweep.py` | λ_sign sweep for the soft-penalty variant |
| `03_ood_sign_violation_test.py` | Reaney-regime leave-one-out: R² collapse vs sign-violation count |
| `04_ml_literature_baselines.py` | GAM, EBM, partial-monotone NN baselines on identical splits |
| `05_subclass_generalizability.py` | Per-A-site-subclass f_LST attribution and LOFO generalizability |
| `06_generate_figures.py` / `07_combined_figure.py` / `11_combined_figure_v2.py` / `14_composite_figure.py` | Figure iterations (superseded by 13/15) |
| `08_monotonic_lst_head.py` | Variant A: monotonic (positive-weight) LST head |
| `09_quantile_conformal_heads.py` | Variant B: intrinsic quantile heads (pinball loss) |
| `10_laplace_last_layer.py` | Variant C: last-layer Laplace approximation |
| `12_calibration_curves.py` | Coverage/calibration curves across variants |
| `13_standalone_figures.py` | Final standalone supplementary figures (S25-S29 sources) |
| `15_combined_fig6_variants.py` | Variant combined comparison (S30 source) |
