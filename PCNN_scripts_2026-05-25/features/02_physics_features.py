"""
Phase 1 — Script 02: Physics Feature Engineering
Parses compositions, computes all physics features, saves feature_matrix.csv.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import json

IN_CSV  = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_dataset.csv")
OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "feature_matrix.csv")
RES_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

from src.features.feature_engineering import build_feature_matrix

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(IN_CSV)
print(f"Loaded cleaned dataset: {len(df)} rows")

# ── Build features ───────────────────────────────────────────────────────────
feat_df = build_feature_matrix(df)
print(f"Feature matrix shape: {feat_df.shape}")

# ── Parse quality report ─────────────────────────────────────────────────────
n_parse_ok  = feat_df["parse_ok"].sum()
n_parse_fail = (feat_df["parse_ok"] == 0).sum()
print(f"\nFormula parse: {n_parse_ok} OK, {n_parse_fail} failed")
if n_parse_fail > 0:
    print("Failed formulas:")
    print(feat_df.loc[feat_df["parse_ok"] == 0, "formula"].tolist())

# ── NaN summary for key physics features ─────────────────────────────────────
key_feats = ["er_CM", "tolerance_factor", "octahedral_factor",
             "alpha_total", "covalency_index", "tilt_severity"]
print("\n── NaN counts in key physics features ──")
for col in key_feats:
    if col in feat_df.columns:
        n_nan = feat_df[col].isna().sum()
        print(f"  {col:25s}  {n_nan:4d} NaN  ({100*n_nan/len(feat_df):.1f}%)")

# ── Chemistry family for GroupShuffleSplit ────────────────────────────────────
from src.models.catboost_model import assign_chemistry_family
feat_df["chemistry_family"] = feat_df["formula"].apply(assign_chemistry_family)
n_families = feat_df["chemistry_family"].nunique()
print(f"\nChemistry families: {n_families} unique groups")

# ── Reaney regime distribution ────────────────────────────────────────────────
print("\n── Reaney regime distribution ──")
print(feat_df["reaney_regime"].value_counts().to_string())

# ── Soft-mode proxy statistics ────────────────────────────────────────────────
n_softmode = feat_df["d0_B_polarizable_A"].sum()
pct_softmode = 100 * n_softmode / len(feat_df)
print(f"\nSoft-mode proxy True: {n_softmode} ({pct_softmode:.1f}%) — expected ~46%")

# ── Save ─────────────────────────────────────────────────────────────────────
feat_df.to_csv(OUT_CSV, index=False)
print(f"\nFeature matrix saved → data/feature_matrix.csv")

metrics = {
    "n_rows": int(len(feat_df)),
    "n_features": int(feat_df.shape[1]),
    "n_parse_ok": int(n_parse_ok),
    "n_parse_fail": int(n_parse_fail),
    "n_chemistry_families": int(n_families),
    "pct_soft_mode": float(pct_softmode),
    "regime_distribution": feat_df["reaney_regime"].value_counts().to_dict(),
}
with open(os.path.join(RES_DIR, "02_feature_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("✓ Script 02 complete — feature_matrix.csv written")
print(f"  rows={len(feat_df)}  features={feat_df.shape[1]}  families={n_families}")
