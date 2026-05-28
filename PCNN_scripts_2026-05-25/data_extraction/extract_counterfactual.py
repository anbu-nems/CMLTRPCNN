import os

# --- self-contained release root (auto-injected) ---
RELEASE_ROOT = os.path.abspath(os.path.dirname(__file__))
while RELEASE_ROOT != os.path.dirname(RELEASE_ROOT) and not os.path.isdir(os.path.join(RELEASE_ROOT, 'model_weights')):
    RELEASE_ROOT = os.path.dirname(RELEASE_ROOT)
# ----------------------------------------------------

"""
Counterfactual & permutation per-sample extraction (inference only — fast).

Uses the saved final v77 ensemble to compute:
  1. A-site swap counterfactual: for every sample, swap A-site identity and
     recompute predicted εr — quantifies A-site causal effect per composition.
  2. A-site permutation test: shuffle A-site flags, re-predict (n_perm runs).
  3. Regime permutation test: shuffle regime flags, re-predict (n_perm runs).

Output: all_figures/extracted_data/counterfactual_per_sample.csv
"""
import os, sys, json, pickle
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from model_CMLTRPCNNv77 import CMLTRPCNN
from train_final_model import HP, DEVICE, PROC_DIR

import sys as _sys
for _p in (RELEASE_ROOT, os.path.join(RELEASE_ROOT,'src','training'), os.path.join(RELEASE_ROOT,'src','analysis'), os.path.join(RELEASE_ROOT,'src','model')):
    if _p not in _sys.path: _sys.path.insert(0, _p)


OUT_DIR    = os.path.join(RELEASE_ROOT, "extracted_data")
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(RELEASE_ROOT, "model_weights/cmltrv77_final.pt")
SCAL_PATH  = os.path.join(RELEASE_ROOT, "model_weights/cmltrv77_scalers.pkl")

print("=" * 70)
print("Counterfactual & permutation per-sample extraction (inference-only)")
print("=" * 70)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_parquet(f"{PROC_DIR}/feature_matrix_v7.parquet")
with open(f"{PROC_DIR}/feature_partition_v7.json") as f:
    partition = json.load(f)

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

with open(SCAL_PATH, "rb") as f:
    sc = pickle.load(f)
Xl_s = sc["sc_lst"].transform(Xl).astype(np.float32)
Xt_s = sc["sc_tilt"].transform(Xt).astype(np.float32)
Xr_s = sc["sc_res"].transform(Xr).astype(np.float32)

train_idx = np.array(json.load(open(f"{PROC_DIR}/calibration_split_idx.json"))["train_idx"])
gii_train = gii[train_idx]
gii_max = float(np.percentile(gii_train[gii_train > 0], 95) if (gii_train > 0).sum() > 10 else 1.0)
gii_norm = np.clip(gii / gii_max, 0, 1).astype(np.float32)

# ── Load ensemble ────────────────────────────────────────────────────────────
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
models = []
for sd in ckpt["models"]:
    m = CMLTRPCNN(
        n_lst=Xl.shape[1], n_tilt=Xt.shape[1], n_res=Xr.shape[1],
        trunk_hidden=HP["trunk_hidden"], n_trunk_blocks=HP["n_trunk_blocks"],
        lst_hidden=HP["lst_hidden"], tilt_hidden=HP["tilt_hidden"],
        res_hidden=HP["res_hidden"], residual_scale=HP["residual_scale"],
        dropout=HP["dropout"],
    ).to(DEVICE)
    m.load_state_dict(sd, strict=False)
    m.eval()
    models.append(m)
print(f"Loaded {len(models)} models")

def to_t(a):
    return torch.tensor(a, dtype=torch.float32).to(DEVICE)

@torch.no_grad()
def predict_with_features(Xl_modified, Xt_modified, Xr_modified):
    """Ensemble predict given modified scaled features."""
    preds = []
    for m in models:
        out = m(to_t(Xl_modified), to_t(Xt_modified), to_t(Xr_modified),
                to_t(er_cm), to_t(has_cm), to_t(cm_approx),
                to_t(gii_norm), to_t(phase_tr), regime_idx_list)
        preds.append(out["pred"].cpu().numpy())
    return np.mean(preds, axis=0)

# Original prediction (sanity)
pred_orig = predict_with_features(Xl_s, Xt_s, Xr_s)

# Regime labels
def regime_for(i):
    for r in ["Ia", "Ib", "II", "III"]:
        if df[f"regime_{r}"].iloc[i] == 1: return r
    return "II"
regimes = np.array([regime_for(i) for i in range(len(y))])

a_cols = [c for c in df.columns if c.startswith("is_") and c[3:] in
          ("Ba","Ca","Sr","Pb","La","Nd","Sm","Ce","Y","Bi","Eu","Gd","Dy")]
def a_site_of(i):
    for c in a_cols:
        if df[c].iloc[i] == 1: return c.replace("is_", "")
    return "Other"
a_sites = np.array([a_site_of(i) for i in range(len(y))])

# A-site flags live in the Residual partition (Xr)
print("\n[1/3] A-site swap counterfactual (flags in Residual partition) ...")
res_feat_names = [c for c in partition["Residual"] if c in df.columns]
a_site_res_cols = [(c, res_feat_names.index(c))
                   for c in res_feat_names if c.startswith("is_") and len(c) <= 6]
print(f"  A-site flags in Residual: {[c for c, _ in a_site_res_cols]}")

records_cf = []
target_sites = ["Ba", "Ca", "Sr", "RE"]
for target in target_sites:
    target_col = f"is_{target}"
    if target_col not in [c for c, _ in a_site_res_cols]:
        print(f"  SKIP {target}: column not found")
        continue
    Xr_raw = Xr.copy()
    for c, idx in a_site_res_cols:
        Xr_raw[:, idx] = 1.0 if c == target_col else 0.0
    Xr_mod = sc["sc_res"].transform(Xr_raw).astype(np.float32)

    pred_cf = predict_with_features(Xl_s, Xt_s, Xr_mod)
    delta   = pred_cf - pred_orig
    print(f"  swap → {target}:  mean Δεr = {delta.mean():+.3f}, "
          f"median = {np.median(delta):+.3f}")
    for i in range(len(y)):
        records_cf.append({
            "idx": i,
            "formula": str(formulas[i]),
            "original_a_site": a_sites[i],
            "regime": regimes[i],
            "swap_to_a_site": target,
            "er_measured": float(y[i]),
            "pred_original": float(pred_orig[i]),
            "pred_swapped":  float(pred_cf[i]),
            "delta_er":      float(delta[i]),
        })

cf_path = f"{OUT_DIR}/counterfactual_asite_swap.csv"
pd.DataFrame(records_cf).to_csv(cf_path, index=False)
print(f"Saved: {cf_path}  ({len(records_cf)} rows)")

# ── A-site permutation test ──────────────────────────────────────────────────
print("\n[2/3] A-site permutation test (50 random shuffles) ...")
rng = np.random.RandomState(42)
n_perm = 50
perm_preds = np.zeros((n_perm, len(y)))
for p in range(n_perm):
    Xr_raw = Xr.copy()
    perm = rng.permutation(len(y))
    for c, idx in a_site_res_cols:
        Xr_raw[:, idx] = Xr[perm, idx]
    Xr_mod = sc["sc_res"].transform(Xr_raw).astype(np.float32)
    perm_preds[p] = predict_with_features(Xl_s, Xt_s, Xr_mod)
    if (p+1) % 10 == 0:
        print(f"  permutation {p+1}/{n_perm}")

asite_perm_df = pd.DataFrame({
    "idx":           np.arange(len(y)),
    "formula":       formulas,
    "a_site":        a_sites,
    "regime":        regimes,
    "er_measured":   y,
    "pred_original": pred_orig,
    "perm_mean":     perm_preds.mean(axis=0),
    "perm_std":      perm_preds.std(axis=0),
    "perm_delta":    perm_preds.mean(axis=0) - pred_orig,
})
asite_perm_path = f"{OUT_DIR}/permutation_asite.csv"
asite_perm_df.to_csv(asite_perm_path, index=False)
print(f"Saved: {asite_perm_path}")

# ── Regime permutation test ──────────────────────────────────────────────────
print("\n[3/3] Regime permutation test (50 random shuffles) ...")
regime_lst_idx = [lcols.index(r) for r in regime_names if r in lcols]
perm_preds_r = np.zeros((n_perm, len(y)))
for p in range(n_perm):
    Xl_raw = Xl.copy()
    perm = rng.permutation(len(y))
    for idx in regime_lst_idx:
        Xl_raw[:, idx] = Xl[perm, idx]
    Xl_mod = sc["sc_lst"].transform(Xl_raw).astype(np.float32)
    perm_preds_r[p] = predict_with_features(Xl_mod, Xt_s, Xr_s)
    if (p+1) % 10 == 0:
        print(f"  permutation {p+1}/{n_perm}")

regime_perm_df = pd.DataFrame({
    "idx":           np.arange(len(y)),
    "formula":       formulas,
    "a_site":        a_sites,
    "regime":        regimes,
    "er_measured":   y,
    "pred_original": pred_orig,
    "perm_mean":     perm_preds_r.mean(axis=0),
    "perm_std":      perm_preds_r.std(axis=0),
    "perm_delta":    perm_preds_r.mean(axis=0) - pred_orig,
})
regime_perm_path = f"{OUT_DIR}/permutation_regime.csv"
regime_perm_df.to_csv(regime_perm_path, index=False)
print(f"Saved: {regime_perm_path}")

print("\nDone — counterfactual and permutation per-sample data saved.")
