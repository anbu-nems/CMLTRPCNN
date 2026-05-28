# Physics-Constrained Neural Network for ABO₃ Dielectric Constant Prediction

Code release accompanying the manuscript: **"Physics-Constrained Neural Network for Dielectric Constant Prediction of ABO₃ Microwave Dielectric Ceramics"** *(manuscript submitted, 2026; journal information to be added on acceptance)*

---

## Overview

CMLTRPCNN predicts the dielectric constant (εᵣ) of ABO₃ perovskite ceramics by decomposing every prediction into four physically interpretable contributions:

```
εᵣ = εᵣ_CM  +  δ_LST (≥0)  +  δ_tilt (≤0)  +  δ_res (confidence-gated)
```

| Branch | Physics law | Constraint | Mean contribution |
|--------|-------------|------------|-------------------|
| εᵣ_CM | Clausius-Mossotti baseline (Shannon polarizability) | parameter-free | 46.2 % |
| δ_LST | Lyddane-Sachs-Teller soft-mode enhancement | ≥ 0 (Softplus) | +54.1 % |
| δ_tilt | Goldschmidt/Reaney octahedral tilt suppression | ≤ 0 (−Softplus) | −0.68 % |
| δ_res | Confidence-gated residual correction | bounded, σ_conf-gated | +0.44 % |

### Headline performance

| Evaluation | n | R² | MAE | Notes |
|------------|---|-----|-----|-------|
| **Held-out test set** | **116** | **0.941** | **5.77** | locked split, 5-seed ensemble |
| Strat-GSS 5-fold CV | 1188 | 0.893 ± 0.044 | 5.62 | family-grouped OOD |
| Formula-split 5-fold CV | 1188 | 0.946 ± 0.008 | 3.78 | random formula folds |

Conformal 90 % prediction interval: **q̂₉₀ = 14.38**, empirical coverage = 90.5 %.

---

## Installation

```bash
git clone <repository-url>
cd PCNN-release
pip install -r requirements.txt
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0. Tested on macOS (MPS) and Linux (CUDA/CPU).

---

## Quick start — predict εᵣ for a new composition

```bash
python predict.py "BaTiO3"
python predict.py "Ba0.5Sr0.5TiO3" --st 1300
python predict.py "Sr0.9Ca0.1TiO3"   # top virtual screening candidate (εᵣ ≈ 140)
```

Returns the predicted εᵣ, a 90 % conformal interval, the four branch contributions,
the inferred Reaney regime, and the σ_conf applicability flag.

---

## Repository layout

```
PCNN-release/
├── README.md                       this file
├── LICENSE                         MIT
├── requirements.txt                Python dependencies
├── hyperparameters.txt             best HP from Optuna trial 97
├── predict.py                      CLI inference entry point
│
├── data/                           raw + processed input data
│   ├── Perovskite_dielectric_full.csv      (n = 1304)
│   ├── Perovskite_dielectric_train.csv     (n = 1188)
│   ├── Perovskite_dielectric_test.csv      (n = 116)
│   └── processed/
│       ├── feature_matrix_v7.parquet       (103 engineered features)
│       ├── feature_partition_v7.json       (LST / Tilt / Residual split)
│       └── calibration_split_idx.json      (locked 90/10 holdout indices)
│
├── model_weights/                  trained checkpoint + scalers
│   ├── cmltrv77_final.pt           5-seed ensemble state dicts
│   └── cmltrv77_scalers.pkl        StandardScaler objects fit on train
│
├── src/                            source code
│   ├── model/model.py              CMLTRPCNN PyTorch class
│   ├── physics/                    physics modules
│   │   ├── clausius_mossotti.py    CM baseline (Shannon)
│   │   ├── tolerance_factor.py     Goldschmidt t
│   │   ├── polarizability.py       Shannon + Qin tables
│   │   ├── ionic_radii.py          Shannon radii
│   │   ├── reaney_regime.py        Reaney 1994 classifier
│   │   └── ...
│   ├── features/feature_engineering.py
│   ├── training/train_final_model.py    full training pipeline
│   ├── inference/inference.py           web-app inference module
│   └── analysis/                   per-sample data extraction
│       ├── extract_per_sample_data.py    decomposition + σ_conf
│       ├── gen_cv_predictions.py         per-sample CV predictions
│       ├── extract_lofo_predictions.py   LOFO predictions
│       ├── extract_counterfactual.py     A-site swap + permutations
│       └── extract_shap_per_sample.py    SHAP from saved .npy
│
├── scripts/                        analysis scripts (canonical)
│   ├── 42_lofo_generality.py            leave-one-family-out
│   ├── 43_counterfactual_permutation.py
│   ├── 49_ad_v77.py                     applicability domain
│   ├── 53_ablation_v77.py               branch ablations
│   ├── 54_conformal_prediction.py       conformal CI
│   ├── 56_pirnn_fair_comparison.py
│   ├── 57_baseline_comparison.py        8-model leaderboard
│   ├── 59_physics_law_ablation.py       2³ factorial
│   └── 61_virtual_screening.py          inverse design (PIDR-UGS)
│
├── figures/                        figure-generation scripts (24 total)
│   ├── gen_fig1_motivation.py  …  gen_fig6_limits.py
│   ├── figS1_dataset_statistics.py  …  figS15_sigma_conf_vs_error.py
│   └── gen_remaining_supplementary.py
│
├── results/                        result JSONs (19 files)
│   ├── 39_physics_decomposition.json
│   ├── 42_lofo_generality.json
│   ├── 43_counterfactual_permutation.json
│   ├── 48_cmltrv77_retrain.json    ← model checkpoint metadata
│   ├── 49_ad_v77.json
│   ├── 53_ablation_v77.json
│   ├── 54_conformal_v77.json
│   ├── 57_baseline_comparison.json
│   ├── 59_physics_law_ablation.json
│   ├── 61_virtual_screening.json
│   ├── quick_check_8020.json       80 : 20 split robustness check
│   └── …
│
├── extracted_data/                 per-sample CSVs (9 files)
│   ├── decomposition_per_sample.csv     εᵣ_CM, δ_LST, δ_tilt, δ_res, σ_conf
│   ├── cv_predictions_v77.csv           per-sample CV predictions
│   ├── lofo_predictions_v77.csv         per-sample LOFO predictions (2 552 rows)
│   ├── counterfactual_asite_swap.csv    A-site swap effect (5 216 rows)
│   ├── permutation_asite.csv            50-shuffle A-site null
│   ├── permutation_regime.csv           50-shuffle regime null
│   ├── shap_per_sample.csv              SHAP long-format (43 200 rows)
│   ├── shap_feature_importance.csv      top features by |SHAP|
│   └── 42_lofo_generality.json          canonical LOFO summary
│
└── figures_output/                 rendered figures (28 unique, 56 files)
    ├── main/                       6 main-text figures (PDF + PNG)
    └── supp/                       15 numbered supplementary + 7 inline panels
```

---

## Reproducing the paper

### 1. Train the final model from scratch (~30 min on Apple Silicon MPS, ~2 h on CPU)
```bash
python src/training/train_final_model.py
```
Produces `model_weights/cmltrv77_final.pt` and `results/48_cmltrv77_retrain.json`.

### 2. Run a canonical analysis (LOFO generalisation)
```bash
python scripts/42_lofo_generality.py
```
Produces `results/42_lofo_generality.json` (matches the file shipped in this release exactly).

### 3. Regenerate per-sample data for the supplementary figures
```bash
python src/analysis/extract_per_sample_data.py    # ~5 s, inference only
python src/analysis/gen_cv_predictions.py         # ~17 min, retrains CV folds
python src/analysis/extract_counterfactual.py     # ~2 min, inference + 100 shuffles
```

### 4. Generate any figure
```bash
python figures/gen_fig3_performance.py
python figures/figS12_lofo_parity.py
```

---

## Key results summary

### Physics decomposition (n = 1 304 compositions)
- εᵣ_CM (46.2 %) and δ_LST (+54.1 %) together account for **~100 %** of mean εᵣ (on the 1,124 CM-computable compositions).
- The confidence-gated residual δ_res averages **+0.44 %** — evidence that the model
  is governed by known crystal-chemistry physics, not curve-fitting.

### A-site soft-mode hierarchy (Spearman = 1.000 vs literature)
**Pb (2.28×) > Ca (1.66×) > La (1.13×) > Sr (0.91×) > Ba (0.66×)**
Permutation null mean = 0.21, observed range = 1.62 → z ≈ 14σ (p ≈ 0).

### LOFO generalisation
La R² = 0.85 (STRONG) · Ca = 0.61 · Sr = 0.43 · Ba = 0.01 · Pb = −0.10 (WEAK)
Pb's 6s² lone-pair stereoactivity has no analog in Ca/Sr/Ba — model boundary
disclosed honestly.

### Top virtual-screening candidates (LCB-ranked)
| Composition | Pred εᵣ | 90 % CI | δ_LST |
|-------------|---------|---------|-------|
| Sr₀.₉Ca₀.₁TiO₃ | 140.1 | [126, 154] | 97.8 |
| Sr₀.₈₅Ca₀.₁₅TiO₃ | 140.1 | [126, 154] | 98.7 |
| Sr₀.₅Ca₀.₅TiO₃ | 138.4 | [124, 153] | 102.4 |

---

## Data sources

- Dielectric measurements: curated from peer-reviewed literature (see Methods).
- Ionic polarizabilities: Shannon, R. D. (1993) *J. Appl. Phys.* **73**, 348-366.
- Ionic radii: Shannon, R. D. (1976) *Acta Cryst.* **A32**, 751-767.
- Reaney regime boundaries: Reaney et al. (1994) *Jpn. J. Appl. Phys.* **33**, 3984.

---

## Citation

If you use this code or data, please cite:

```
[Authors] (2026) "Physics-Constrained Neural Network for Dielectric Constant
Prediction of ABO₃ Microwave Dielectric Ceramics", Nature Communications.
```

---

## License

MIT — see `LICENSE` for the full text. Free for academic and commercial use.
