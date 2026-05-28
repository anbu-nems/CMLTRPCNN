import os

# --- self-contained release root (auto-injected) ---
RELEASE_ROOT = os.path.abspath(os.path.dirname(__file__))
while RELEASE_ROOT != os.path.dirname(RELEASE_ROOT) and not os.path.isdir(os.path.join(RELEASE_ROOT, 'model_weights')):
    RELEASE_ROOT = os.path.dirname(RELEASE_ROOT)
# ----------------------------------------------------

"""
Extract per-sample data from the trained v77 model — inference-only, fast.

For all 1,304 compositions:
  - predicted εr, ensemble mean and std (5 seeds)
  - decomposition: er_cm, δ_LST, δ_tilt, δ_res
  - σ_conf (confidence gate)
  - regime, A-site, B-site, tolerance factor
  - true εr

Outputs:
  all_figures/extracted_data/decomposition_per_sample.csv
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


OUT_DIR = os.path.join(RELEASE_ROOT, "extracted_data")
os.makedirs(OUT_DIR, exist_ok=True)

PIML_ROOT  = RELEASE_ROOT
MODEL_PATH = f"{PIML_ROOT}/model_weights/cmltrv77_final.pt"
SCAL_PATH  = f"{PIML_ROOT}/model_weights/cmltrv77_scalers.pkl"

print("Loading data ...")
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
gii       = df["GII"].fillna(0.0).values.astype(np.float32) if "GII" in df.columns else np.zeros(len(df), np.float32)
phase_tr  = df["phase_transition"].fillna(0.0).values.astype(np.float32) if "phase_transition" in df.columns else np.zeros(len(df), np.float32)
y         = df["epsilon_r"].values.astype(np.float32)

regime_names = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
regime_idx   = [lcols.index(r) for r in regime_names if r in lcols]

print(f"  Loaded {len(y)} samples")

# Apply scalers
with open(SCAL_PATH, "rb") as f:
    sc = pickle.load(f)
Xl_s = sc["sc_lst"].transform(Xl).astype(np.float32)
Xt_s = sc["sc_tilt"].transform(Xt).astype(np.float32)
Xr_s = sc["sc_res"].transform(Xr).astype(np.float32)

# GII normalization (same as training)
train_idx = np.array(json.load(open(f"{PROC_DIR}/calibration_split_idx.json"))["train_idx"])
gii_train = gii[train_idx]
gii_max = float(np.percentile(gii_train[gii_train > 0], 95)
                if (gii_train > 0).sum() > 10 else 1.0)
gii_norm = np.clip(gii / gii_max, 0, 1).astype(np.float32)

# Build models from saved state dicts
print(f"Loading models from {MODEL_PATH} ...")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
state_dicts = ckpt["models"]
print(f"  {len(state_dicts)} seeds in ensemble")

models = []
for sd in state_dicts:
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

# Run inference per seed, capture all components
print("Running inference for all 1,304 samples × 5 seeds ...")
def to_t(a):
    return torch.tensor(a, dtype=torch.float32).to(DEVICE)

all_pred, all_lst, all_tilt, all_res, all_sigma = [], [], [], [], []
with torch.no_grad():
    for m in models:
        out = m(to_t(Xl_s), to_t(Xt_s), to_t(Xr_s),
                to_t(er_cm), to_t(has_cm), to_t(cm_approx),
                to_t(gii_norm), to_t(phase_tr), regime_idx)
        all_pred.append(out["pred"].cpu().numpy())
        all_lst.append(out["delta_lst"].cpu().numpy())
        all_tilt.append(out["delta_tilt"].cpu().numpy())
        all_res.append(out["delta_res"].cpu().numpy())
        all_sigma.append(out["sigma_conf"].cpu().numpy())

all_pred  = np.array(all_pred)
all_lst   = np.array(all_lst)
all_tilt  = np.array(all_tilt)
all_res   = np.array(all_res)
all_sigma = np.array(all_sigma)

pred_mean = all_pred.mean(axis=0)
pred_std  = all_pred.std(axis=0)
lst_mean  = all_lst.mean(axis=0)
tilt_mean = all_tilt.mean(axis=0)
res_mean  = all_res.mean(axis=0)
sigma_mean = all_sigma.mean(axis=0)

# Regime labels
def regime_for(i):
    for r in ["Ia", "Ib", "II", "III"]:
        col = f"regime_{r}"
        if col in df.columns and df[col].iloc[i] == 1: return r
    return "II"
regimes = [regime_for(i) for i in range(len(y))]

# Split flags
calib_info = json.load(open(f"{PROC_DIR}/calibration_split_idx.json"))
is_holdout = np.zeros(len(y), bool)
is_holdout[calib_info["calib_idx"]] = True
is_train = ~is_holdout

# A-site & B-site
a_cols = [c for c in df.columns if c.startswith("is_") and c[3:] in
          ("Ba","Ca","Sr","Pb","La","Nd","Sm","Ce","Y","Bi","Eu","Gd","Dy","K","Na")]
def a_site_of(i):
    for c in a_cols:
        if df[c].iloc[i] == 1:
            return c.replace("is_", "")
    return "Other"
a_sites = [a_site_of(i) for i in range(len(y))]

out = pd.DataFrame({
    "idx":         np.arange(len(y)),
    "formula":     df["formula"].values if "formula" in df.columns else "",
    "regime":      regimes,
    "a_site":      a_sites,
    "is_train":    is_train,
    "is_holdout":  is_holdout,
    "er_measured": y,
    "er_predicted": pred_mean,
    "ensemble_std": pred_std,
    "er_cm":       er_cm,
    "delta_lst":   lst_mean,
    "delta_tilt":  tilt_mean,
    "delta_res":   res_mean,
    "sigma_conf":  sigma_mean,
    "has_cm":      has_cm.astype(bool),
    "cm_approx":   cm_approx.astype(bool),
    "gii":         gii,
    "gii_norm":    gii_norm,
})

# Add tolerance factor if computable
if "Goldschmidt_t" in df.columns:
    out["tolerance_factor"] = df["Goldschmidt_t"].values
elif "tolerance_factor" in df.columns:
    out["tolerance_factor"] = df["tolerance_factor"].values

out_path = f"{OUT_DIR}/decomposition_per_sample.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"  Columns: {len(out.columns)}, Rows: {len(out)}")
print(f"\nQuick check (train set only):")
tr = out[out["is_train"]]
print(f"  Mean predicted εr  = {tr['er_predicted'].mean():.3f}")
print(f"  Mean er_cm         = {tr['er_cm'].mean():.3f}")
print(f"  Mean δ_LST         = {tr['delta_lst'].mean():.3f}")
print(f"  Mean δ_tilt        = {tr['delta_tilt'].mean():.3f}")
print(f"  Mean δ_res         = {tr['delta_res'].mean():.3f}")
print(f"  Mean σ_conf        = {tr['sigma_conf'].mean():.3f}")
