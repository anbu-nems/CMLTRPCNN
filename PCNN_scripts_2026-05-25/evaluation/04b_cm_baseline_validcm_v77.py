"""
Script 04b — CM Baseline on the canonical valid-CM set (v7.7)

The original script 04 evaluated the Clausius-Mossotti baseline on an OLD corpus
(data/residual_target.csv, n=676) that predates the v7 feature expansion. The
manuscript now reports the corpus as n=1,304 and the decomposition (script 39) on
the valid-CM set n=1,124 (mask: has_sigma_CM>0.5 & er_CM>5). This script recomputes
the CM-only baseline on that SAME valid-CM mask so the manuscript is internally
consistent (same set for the CM failure in Fig 1a and the decomposition in Fig 4c).

Output: results/04b_cm_baseline_validcm_v77.json
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = os.path.join(os.path.dirname(__file__), "..")
df = pd.read_parquet(os.path.join(ROOT, "data", "processed", "feature_matrix_v7.parquet"))

def get_regime(row):
    for col, label in [("regime_Ib", "Ib"), ("regime_Ia", "Ia"),
                       ("regime_III", "III"), ("regime_II", "II")]:
        if col in row.index and row[col] > 0.5:
            return label
    return "Unknown"

df["regime"] = df.apply(get_regime, axis=1)

# Canonical valid-CM mask (identical to script 39)
valid = (df["has_sigma_CM"].fillna(0.0) > 0.5) & (df["er_CM"].fillna(0.0) > 5.0)
sub = df[valid].copy()
n_valid = int(valid.sum())

# exact vs VCA split (cm_approx_flag == 1 => VCA / approximate CM)
n_vca   = int((sub["cm_approx_flag"].fillna(0.0) > 0.5).sum())
n_exact = n_valid - n_vca

y_true = sub["epsilon_r"].values.astype(float)
y_pred = sub["er_CM"].values.astype(float)

def metrics(yt, yp):
    return {
        "n": int(len(yt)),
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(mean_squared_error(yt, yp) ** 0.5),
        "mape": float(np.mean(np.abs((yt - yp) / yt)) * 100),
    }

glob = metrics(y_true, y_pred)
print(f"Valid-CM n={n_valid} ({n_exact} exact + {n_vca} VCA)")
print(f"CM-only GLOBAL:  R2={glob['r2']:.4f}  MAE={glob['mae']:.2f}  "
      f"RMSE={glob['rmse']:.2f}  MAPE={glob['mape']:.1f}%")

per_regime = {}
print("\nPer regime:")
for reg in ["Ia", "Ib", "II", "III"]:
    s = sub[sub["regime"] == reg]
    if len(s) < 5:
        continue
    m = metrics(s["epsilon_r"].values.astype(float), s["er_CM"].values.astype(float))
    per_regime[reg] = m
    print(f"  {reg:>3s}: n={m['n']:4d}  R2={m['r2']:.3f}  MAE={m['mae']:.1f}  "
          f"RMSE={m['rmse']:.1f}  MAPE={m['mape']:.1f}%")

out = {
    "note": "CM-only baseline recomputed on canonical valid-CM set (v7.7), mask "
            "has_sigma_CM>0.5 & er_CM>5; supersedes script 04's n=676 (old corpus).",
    "n_valid_cm": n_valid, "n_exact": n_exact, "n_vca": n_vca,
    "CM_baseline_global": glob,
    "CM_baseline_per_regime": per_regime,
}
with open(os.path.join(ROOT, "results", "04b_cm_baseline_validcm_v77.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved -> results/04b_cm_baseline_validcm_v77.json")
