"""
Phase 5 — Script 09: SHAP Analysis + Four-Way Δεr Decomposition
Per-regime SHAP, global feature importance, four CM failure mode decomposition.

Uses: CatBoost+Crystal model (74 features, catboost_crystal_final.cbm)
Dataset: mixed_dataset (1054 CM-stable samples, GroupShuffleSplit by family)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, warnings
warnings.filterwarnings("ignore")

import shap
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

from src.features.feature_engineering import build_feature_matrix
from src.models.catboost_model import (
    assign_chemistry_family, get_feature_cols, get_monotone_constraints
)

ROOT     = os.path.join(os.path.dirname(__file__), "..")
RAW_CSV  = os.path.join(ROOT, "data", "raw", "mixed_dataset.csv")
RAW_XLSX = os.path.join(ROOT, "data", "raw", "mixed_dataset.xlsx")
MODEL_IN = os.path.join(ROOT, "models", "catboost_crystal_final.cbm")
PARAMS_F = os.path.join(ROOT, "results", "best_params_1054_v2.json")
FIG_DIR  = os.path.join(ROOT, "figures")
RES_DIR  = os.path.join(ROOT, "results")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load & prepare data (same pipeline as script 12) ─────────────────────────
raw = pd.read_csv(RAW_CSV) if os.path.exists(RAW_CSV) else pd.read_excel(RAW_XLSX)
raw.columns = raw.columns.str.strip()
raw.rename(columns={
    "Qf (GHz)": "Qf_GHz", "𝜏f": "tau_f",
    "crystal structure": "crystal_structure", "crystalsystem": "crystal_system",
    "Additive type ": "additive_type", "additive %": "additive_pct",
    "C-T": "CT", "C-Time": "C_time", "ST-time": "ST_time",
}, inplace=True)
raw["DC"] = pd.to_numeric(raw["DC"], errors="coerce")

feat_df = build_feature_matrix(raw)
feat_df["delta_er"]         = feat_df["DC"] - feat_df["er_CM"]
feat_df["chemistry_family"] = feat_df["formula"].apply(assign_chemistry_family)

df = feat_df[
    feat_df["delta_er"].notna() & feat_df["er_CM"].notna() & feat_df["DC"].notna()
].copy().reset_index(drop=True)
print(f"SHAP analysis on {len(df)} samples  |  families: {df['chemistry_family'].nunique()}")

feature_cols = get_feature_cols(df)
X = df[feature_cols].fillna(0)
y_er = df["DC"].values
print(f"Feature set: {len(feature_cols)} features")

# ── Load model and compute cross-validated predictions ────────────────────────
model = CatBoostRegressor()
model.load_model(MODEL_IN)
print(f"Model loaded: {MODEL_IN}")

# GroupKFold-5 CV predictions (every sample appears exactly once in test)
with open(PARAMS_F) as f:
    best_params = json.load(f)
monotone = get_monotone_constraints(feature_cols)
groups = df["chemistry_family"].values
gkf = GroupKFold(n_splits=5)
piml_preds = np.zeros(len(df))
for tr_idx, te_idx in gkf.split(X, y_er, groups):
    m = CatBoostRegressor(**best_params, monotone_constraints=monotone,
                          verbose=0, random_seed=42)
    m.fit(X.iloc[tr_idx], df["delta_er"].values[tr_idx])
    piml_preds[te_idx] = df["er_CM"].values[te_idx] + m.predict(X.iloc[te_idx])

cv_r2 = r2_score(y_er, piml_preds)
cv_mae = float(np.abs(y_er - piml_preds).mean())
print(f"CV R² (GroupKFold-5): {cv_r2:.4f}  MAE: {cv_mae:.2f}")

# ── SHAP values (full-dataset model for attribution — no data leakage in SHAP) ─
print("Computing SHAP values...")
explainer  = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap_df = pd.DataFrame(shap_values, columns=feature_cols, index=df.index)

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
top_features  = mean_abs_shap.sort_values(ascending=False).head(20)
print("\n── Top 20 features by mean |SHAP| ──")
for feat, val in top_features.items():
    print(f"  {feat:35s}  {val:.4f}")

# ── Global SHAP bar plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
top_features.sort_values().plot.barh(ax=ax, color="#2c7bb6")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Global feature importance — CatBoost+Crystal Δεr model")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "09a_shap_global.png"), dpi=300)
plt.close()
print("Saved: 09a_shap_global.png")

# ── Per-regime SHAP ───────────────────────────────────────────────────────────
# Reaney regime from feature columns (one-hot encoded)
regime_col = None
for c in ["reaney_regime", "regime"]:
    if c in df.columns:
        regime_col = c
        break

if regime_col is None:
    # Reconstruct from one-hot
    def get_regime(row):
        for r in ["Ia", "Ib", "II", "III"]:
            col = f"regime_{r}"
            if col in df.columns and row.get(col, 0) == 1:
                return r
        return "unknown"
    df["reaney_regime"] = df.apply(get_regime, axis=1)
else:
    df["reaney_regime"] = df[regime_col]

print("\n── Per-regime top features ──")
regime_shap = {}
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
regime_list = ["Ia", "Ib", "II", "III"]
colors_r = {"Ia": "#1a9641", "Ib": "#d7191c", "II": "#fdae61", "III": "#2c7bb6"}

for ax, regime in zip(axes.flat, regime_list):
    mask = df["reaney_regime"] == regime
    n = mask.sum()
    if n < 10:
        ax.set_title(f"Regime {regime} (n={n}, skipped)")
        continue
    sv = shap_values[mask.values]
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=feature_cols)
    top10 = mean_abs.sort_values(ascending=False).head(10)
    regime_shap[regime] = top10.to_dict()
    top10.sort_values().plot.barh(ax=ax, color=colors_r.get(regime, "#aaaaaa"))
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Regime {regime} (n={n})")
    print(f"  {regime} (n={n}): top feature = {top10.index[0]} ({top10.iloc[0]:.4f})")

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "09b_shap_per_regime.png"), dpi=300)
plt.close()
print("Saved: 09b_shap_per_regime.png")

# ── Four-Way Δεr Decomposition (corrected feature groups) ─────────────────────
# Mode 1 — Covalency enhancement (+Δεr)
#   Electronegativity difference and bond polarity features
covalency_feats = [c for c in [
    "covalency_index", "covalency_ratio",
    "delta_chi_A", "delta_chi_B", "chi_A", "chi_B",
    "phillips_covalency_bo", "phillips_ionicity_bo",
] if c in feature_cols]

# Mode 2 — Tilt suppression (-Δεr)
#   Octahedral tilt geometry and structural distortion
tilt_feats = [c for c in [
    "tilt_severity", "regime_II", "regime_III",
    "tolerance_factor", "tol_sq_dev",
    "continuous_tilt_strain", "charge_imbalance_proxy",
    "phase_boundary_dist_ii_iii", "phase_boundary_instability",
    "octahedral_volume", "b_size_mismatch",
] if c in feature_cols]

# Mode 3 — Soft-mode / LST enhancement (+Δεr)
#   B-site SOJT activity, LST coupling, reduced mass effects
softmode_feats = [c for c in [
    # SOJT & d0 activity
    "soft_mode_proxy", "d0_B_polarizable_A", "d0_B_fraction", "polarizable_A_fraction",
    "regime_Ib", "regime_Ia",
    # SOJT interaction features (top global features)
    "weighted_SOJT", "soft_mode_activity", "lst_enhancement_proxy",
    "soft_mode_tolerance_weighted",
    # Phonon / LST descriptors
    "b_o_reduced_mass", "d0_per_field",
    # Phase boundary proximity (precursor to ferroelectric instability)
    "phase_boundary_dist_ib_ii", "phase_boundary_min_dist",
] if c in feature_cols]

assigned = set(covalency_feats + tilt_feats + softmode_feats)
residual_feats = [c for c in feature_cols if c not in assigned]

print(f"\n── Feature group sizes ──")
print(f"  Covalency  : {len(covalency_feats)} features — {covalency_feats[:4]}...")
print(f"  Tilt       : {len(tilt_feats)} features — {tilt_feats[:4]}...")
print(f"  Soft-mode  : {len(softmode_feats)} features — {softmode_feats[:4]}...")
print(f"  Residual   : {len(residual_feats)} features")


def shap_contribution(feats):
    idx = [list(feature_cols).index(f) for f in feats if f in feature_cols]
    if not idx:
        return np.zeros(len(df))
    return shap_values[:, idx].sum(axis=1)


shap_cov  = shap_contribution(covalency_feats)
shap_tilt = shap_contribution(tilt_feats)
shap_soft = shap_contribution(softmode_feats)
shap_res  = shap_contribution(residual_feats)

delta_er = df["delta_er"].values
print(f"\n── Four-Way Δεr Decomposition (mean SHAP attribution) ──")
total_shap = (abs(shap_cov).mean() + abs(shap_tilt).mean() +
              abs(shap_soft).mean() + abs(shap_res).mean())
for name, arr in [("Covalency  (Mode 1)", shap_cov),
                  ("Tilt       (Mode 2)", shap_tilt),
                  ("Soft-mode  (Mode 3)", shap_soft),
                  ("Residual   (Mode 4)", shap_res)]:
    pct = 100 * abs(arr).mean() / total_shap
    print(f"  {name}: signed mean={arr.mean():+.2f}  |SHAP|={abs(arr).mean():.3f}  ({pct:.1f}%)")

# Bar chart of decomposition
modes = ["Covalency\n(Mode 1)", "Tilt\n(Mode 2)", "Soft-mode\n(Mode 3)", "Residual\n(Mode 4)"]
contributions = [abs(shap_cov).mean(), abs(shap_tilt).mean(),
                 abs(shap_soft).mean(), abs(shap_res).mean()]
colors_mode = ["#2c7bb6", "#1a9641", "#d7191c", "#fdae61"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(modes, contributions, color=colors_mode)
ax.set_ylabel("Mean |SHAP| attribution to Δεr")
ax.set_title("Four-way CM failure mode decomposition\n(CatBoost+Crystal, 1054 samples)")
for bar, val in zip(bars, contributions):
    pct = 100 * val / total_shap
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.2f}\n({pct:.0f}%)", ha="center", fontsize=8.5)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "09c_fourway_decomposition.png"), dpi=300)
plt.close()
print("Saved: 09c_fourway_decomposition.png")

# ── SHAP beeswarm summary (top 20) ────────────────────────────────────────────
top20_idx = [list(feature_cols).index(f) for f in top_features.index if f in feature_cols]
shap_top20 = shap_values[:, top20_idx]
top20_names = [f for f in top_features.index if f in feature_cols]

fig, ax = plt.subplots(figsize=(9, 7))
shap.summary_plot(
    shap_top20,
    X[top20_names].values,
    feature_names=top20_names,
    show=False, plot_size=None,
)
plt.title("SHAP summary — top 20 features")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "09d_shap_beeswarm.png"), dpi=300)
plt.close()
print("Saved: 09d_shap_beeswarm.png")

# ── Save ─────────────────────────────────────────────────────────────────────
output = {
    "n_samples": len(df),
    "n_features": len(feature_cols),
    "cv_r2_gkf5": float(cv_r2),
    "cv_mae_gkf5": float(cv_mae),
    "top20_global": top_features.to_dict(),
    "per_regime_top": regime_shap,
    "fourway_decomposition": {
        "covalency_mean_abs_shap":  float(abs(shap_cov).mean()),
        "covalency_signed_mean":    float(shap_cov.mean()),
        "tilt_mean_abs_shap":       float(abs(shap_tilt).mean()),
        "tilt_signed_mean":         float(shap_tilt.mean()),
        "softmode_mean_abs_shap":   float(abs(shap_soft).mean()),
        "softmode_signed_mean":     float(shap_soft.mean()),
        "residual_mean_abs_shap":   float(abs(shap_res).mean()),
        "residual_signed_mean":     float(shap_res.mean()),
        "total_mean_abs_shap":      float(total_shap),
        "covalency_pct":   float(100 * abs(shap_cov).mean() / total_shap),
        "tilt_pct":        float(100 * abs(shap_tilt).mean() / total_shap),
        "softmode_pct":    float(100 * abs(shap_soft).mean() / total_shap),
        "residual_pct":    float(100 * abs(shap_res).mean() / total_shap),
    },
    "feature_groups": {
        "covalency": covalency_feats,
        "tilt":      tilt_feats,
        "softmode":  softmode_feats,
        "residual":  residual_feats,
    },
}
with open(os.path.join(RES_DIR, "09_shap_metrics.json"), "w") as f:
    json.dump(output, f, indent=2)

print("\n✓ Script 09 complete — SHAP analysis done")
print(f"  Cov={contributions[0]:.3f}  Tilt={contributions[1]:.3f}"
      f"  Soft={contributions[2]:.3f}  Res={contributions[3]:.3f}")
pcts = [100*c/total_shap for c in contributions]
print(f"  Pct:  {pcts[0]:.1f}%  {pcts[1]:.1f}%  {pcts[2]:.1f}%  {pcts[3]:.1f}%")
