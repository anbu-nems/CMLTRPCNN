import os

# --- self-contained release root (auto-injected) ---
RELEASE_ROOT = os.path.abspath(os.path.dirname(__file__))
while RELEASE_ROOT != os.path.dirname(RELEASE_ROOT) and not os.path.isdir(os.path.join(RELEASE_ROOT, 'model_weights')):
    RELEASE_ROOT = os.path.dirname(RELEASE_ROOT)
# ----------------------------------------------------

"""
LOFO per-sample predictions for v77 model.

For each A-site (Pb/Ca/Ba/Sr/La) and each Reaney regime (Ia/Ib/II/III),
hold out all samples with that feature, train on the remainder (5 seeds × 500 epochs),
and save the held-out predictions per composition.

Output: all_figures/extracted_data/lofo_predictions_v77.csv
  columns: idx, formula, regime, a_site, holdout_type, holdout_label,
           er_measured, er_predicted, ensemble_std
"""
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from train_final_model import (

    HP, DEVICE, PROC_DIR,
    scale_arrays, train_one, predict,
)

import sys as _sys
for _p in (RELEASE_ROOT, os.path.join(RELEASE_ROOT,'src','training'), os.path.join(RELEASE_ROOT,'src','analysis'), os.path.join(RELEASE_ROOT,'src','model')):
    if _p not in _sys.path: _sys.path.insert(0, _p)


OUT_DIR = os.path.join(RELEASE_ROOT, "extracted_data")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print(f"LOFO per-sample extraction (v77, 5-seed ensemble per holdout)")
print(f"  Device: {DEVICE}")
print("=" * 70)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_parquet(f"{PROC_DIR}/feature_matrix_v7.parquet")
with open(f"{PROC_DIR}/feature_partition_v7.json") as f:
    partition = json.load(f)
with open(f"{PROC_DIR}/calibration_split_idx.json") as f:
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
gii       = df["GII"].fillna(0.0).values.astype(np.float32) \
            if "GII" in df.columns else np.zeros(len(df), np.float32)
phase_tr  = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
            if "phase_transition" in df.columns else np.zeros(len(df), np.float32)
formulas  = df["formula"].values if "formula" in df.columns else np.array([""] * len(df))

regime_names = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
regime_idx_list = [lcols.index(r) for r in regime_names if r in lcols]

def regime_for(i):
    for r in ["Ia", "Ib", "II", "III"]:
        if df[f"regime_{r}"].iloc[i] == 1: return r
    return "II"
regimes = np.array([regime_for(i) for i in range(len(y))])

# Canonical Script 42 protocol: A-site from chemistry_family (NOT is_X one-hot)
a_sites = df["chemistry_family"].apply(lambda x: str(x).split("_")[0]).values

# Use canonical 90% train (not the locked holdout) for LOFO
train_idx_full = np.array(calib_info["train_idx"])

records = []
total_holdouts = 5 + 4   # 5 A-sites + 4 regimes
t0 = time.time()
ho_i = 0

def run_lofo(holdout_mask, label, htype):
    """Canonical Script 42 LOFO: test on ALL samples with the target attribute,
       train on the train_pool (1188) minus those that share the attribute.
       2 seeds, 10% of train as validation."""
    global ho_i
    ho_i += 1
    test_idx = np.where(holdout_mask)[0]                           # ALL target samples
    tr_idx   = train_idx_full[~holdout_mask[train_idx_full]]       # train_pool excluding target
    if len(test_idx) < 5:
        print(f"  [{ho_i}/{total_holdouts}] {label}: SKIP — only {len(test_idx)} samples")
        return
    t_start = time.time()
    Xl_s, Xt_s, Xr_s, gii_s, _, _, _ = scale_arrays(Xl, Xt, Xr, gii, tr_idx)
    # Hold out 10% of train as validation for early stopping (no test leakage)
    rng = np.random.RandomState(42)
    shuf = rng.permutation(tr_idx)
    va_size = max(10, len(shuf) // 10)
    va_idx = shuf[:va_size]
    tr_idx_local = shuf[va_size:]
    n_seeds = 2  # canonical Script 42 uses 2 seeds
    models = [train_one(Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                        gii_s, phase_tr, y, regime_idx_list,
                        tr_idx_local, va_idx, HP, s)
              for s in range(n_seeds)]

    # ensemble mean + per-seed for std
    per_seed = []
    for m in models:
        per_seed.append(predict([m], Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                                gii_s, phase_tr, test_idx, regime_idx_list))
    per_seed = np.array(per_seed)
    pred = per_seed.mean(axis=0)
    std  = per_seed.std(axis=0)

    from sklearn.metrics import r2_score
    r2 = r2_score(y[test_idx], pred)
    print(f"  [{ho_i}/{total_holdouts}] {label}  n_test={len(test_idx)}  "
          f"R²={r2:.4f}  ({time.time()-t_start:.0f}s, total {time.time()-t0:.0f}s)")

    for j, i in enumerate(test_idx):
        records.append({
            "idx":          int(i),
            "formula":      str(formulas[i]),
            "regime":       regimes[i],
            "a_site":       a_sites[i],
            "holdout_type": htype,
            "holdout_label": label,
            "n_test":       len(test_idx),
            "er_measured":  float(y[i]),
            "er_predicted": float(pred[j]),
            "ensemble_std": float(std[j]),
            "lofo_r2":      float(r2),
        })

# ── A-site LOFO ──────────────────────────────────────────────────────────────
print("\n[A-site LOFO]")
for a in ["Pb", "Ca", "Ba", "Sr", "La"]:
    mask = (a_sites == a)
    run_lofo(mask, f"A-site={a}", "asite")

# ── Regime LOFO ──────────────────────────────────────────────────────────────
print("\n[Regime LOFO]")
for r in ["Ia", "Ib", "II", "III"]:
    mask = (regimes == r)
    run_lofo(mask, f"Regime={r}", "regime")

# ── Save ─────────────────────────────────────────────────────────────────────
out = pd.DataFrame(records)
out_path = f"{OUT_DIR}/lofo_predictions_v77.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  ({len(out)} rows, {time.time()-t0:.0f}s total)")
