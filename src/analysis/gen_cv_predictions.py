import os

# --- self-contained release root (auto-injected) ---
RELEASE_ROOT = os.path.abspath(os.path.dirname(__file__))
while RELEASE_ROOT != os.path.dirname(RELEASE_ROOT) and not os.path.isdir(os.path.join(RELEASE_ROOT, 'model_weights')):
    RELEASE_ROOT = os.path.dirname(RELEASE_ROOT)
# ----------------------------------------------------

"""
Regenerate per-sample CV predictions for the v77 model.

Runs Formula-split 5-fold CV and Strat-GSS 5-fold CV on the canonical 90:10
split. Saves per-sample predictions with regime info so we can build a
3-protocol parity plot:
  - Strat-GSS CV    (out-of-distribution, family-grouped)
  - Formula-split CV (random formula folds)
  - Ensemble holdout (final model, n=116)

Outputs:
  data/processed/cv_predictions_v77.csv
"""
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

from train_final_model import (

    HP, DEVICE, PROC_DIR,
    scale_arrays, train_one, predict, build_strat_gss,
)

import sys as _sys
for _p in (RELEASE_ROOT, os.path.join(RELEASE_ROOT,'src','training'), os.path.join(RELEASE_ROOT,'src','analysis'), os.path.join(RELEASE_ROOT,'src','model')):
    if _p not in _sys.path: _sys.path.insert(0, _p)


print("=" * 70)
print(f"Regenerate per-sample CV predictions (v77)")
print(f"  Device: {DEVICE}")
print("=" * 70)

df = pd.read_parquet(os.path.join(PROC_DIR, "feature_matrix_v7.parquet"))
with open(os.path.join(PROC_DIR, "feature_partition_v7.json")) as f:
    partition = json.load(f)
with open(os.path.join(PROC_DIR, "calibration_split_idx.json")) as f:
    calib_info = json.load(f)

def _get(cols):
    present = [c for c in cols if c in df.columns]
    return df[present].fillna(0.0).values.astype(np.float32), present

Xl, lcols = _get(partition["LST"])
Xt, _     = _get(partition["Tilt"])
Xr, _     = _get(partition["Residual"])
er_cm     = df["er_CM"].fillna(0.0).values.astype(np.float32)
has_cm    = df["has_sigma_CM"].fillna(0.0).values.astype(np.float32)
cm_approx = df["cm_approx_flag"].fillna(0.0).values.astype(np.float32)
y         = df["epsilon_r"].values.astype(np.float32)
groups    = df["chemistry_family"].values
gii       = df["GII"].fillna(0.0).values.astype(np.float32) \
            if "GII" in df.columns else np.zeros(len(df), dtype=np.float32)
phase_tr  = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
            if "phase_transition" in df.columns else np.zeros(len(df), np.float32)
formulas  = df["formula"].values if "formula" in df.columns else np.array([""] * len(df))

regime_names = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
regime_idx_list = [lcols.index(r) for r in regime_names if r in lcols]

def regime_for(i):
    for r in ["Ia", "Ib", "II", "III"]:
        col = f"regime_{r}"
        if col in df.columns and df[col].iloc[i] == 1:
            return r
    return "II"
regime_labels = np.array([regime_for(i) for i in range(len(y))])

train_idx = np.array(calib_info["train_idx"])
calib_idx = np.array(calib_info["calib_idx"])

# ── Formula-split 5-fold ─────────────────────────────────────────────────────
print("\n[1/2] Formula-split 5-fold CV ...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds_frm = np.full(len(y), np.nan, dtype=np.float32)
t0 = time.time()
for fold, (tr_rel, te_rel) in enumerate(kf.split(train_idx)):
    tr, te = train_idx[tr_rel], train_idx[te_rel]
    Xl_s, Xt_s, Xr_s, gii_s, _, _, _ = scale_arrays(Xl, Xt, Xr, gii, tr)
    models = [train_one(Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                        gii_s, phase_tr, y, regime_idx_list, tr, te, HP, s)
              for s in range(HP["n_seeds"])]
    pred = predict(models, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                   gii_s, phase_tr, te, regime_idx_list)
    preds_frm[te] = pred
    print(f"  Fold {fold+1}/5  R²={r2_score(y[te], pred):.4f}  "
          f"({time.time()-t0:.0f}s elapsed)")
print(f"  Formula-split overall R² = {r2_score(y[train_idx][~np.isnan(preds_frm[train_idx])], preds_frm[train_idx][~np.isnan(preds_frm[train_idx])]):.4f}")

# ── Strat-GSS 5-fold ─────────────────────────────────────────────────────────
print("\n[2/2] Strat-GSS 5-fold CV ...")
splits_gss = build_strat_gss(train_idx, groups)
preds_gss = np.full(len(y), np.nan, dtype=np.float32)
for fold, (tr, te) in enumerate(splits_gss):
    Xl_s, Xt_s, Xr_s, gii_s, _, _, _ = scale_arrays(Xl, Xt, Xr, gii, tr)
    models = [train_one(Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                        gii_s, phase_tr, y, regime_idx_list, tr, te, HP, s)
              for s in range(HP["n_seeds"])]
    pred = predict(models, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                   gii_s, phase_tr, te, regime_idx_list)
    preds_gss[te] = pred
    print(f"  Fold {fold+1}/5  R²={r2_score(y[te], pred):.4f}  "
          f"({time.time()-t0:.0f}s elapsed)")
mask = ~np.isnan(preds_gss)
print(f"  Strat-GSS overall R² = {r2_score(y[mask], preds_gss[mask]):.4f}")

# ── Save unified CSV ─────────────────────────────────────────────────────────
out = pd.DataFrame({
    "idx":           np.arange(len(y)),
    "formula":       formulas,
    "regime":        regime_labels,
    "er_measured":   y,
    "pred_formula":  preds_frm,
    "pred_gss":      preds_gss,
    "is_train":      np.isin(np.arange(len(y)), train_idx),
    "is_holdout":    np.isin(np.arange(len(y)), calib_idx),
})
EXTRACT_DIR = os.path.join(RELEASE_ROOT, "extracted_data")
os.makedirs(EXTRACT_DIR, exist_ok=True)
out_path = os.path.join(EXTRACT_DIR, "cv_predictions_v77.csv")
out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  (total elapsed {time.time()-t0:.0f}s)")
