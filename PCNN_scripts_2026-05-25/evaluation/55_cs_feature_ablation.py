"""
Script 55 — Crystal system feature ablation
Does zeroing all cs_ features in the Residual branch change model performance?

Outputs:
  results/55_cs_ablation.json
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error

from src.models.psrnn_mdpinn import CMLTRPCNNv71

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results")

# ── Load data (same pipeline as script 48) ────────────────────────────────────
df        = pd.read_parquet(os.path.join(PROC_DIR, "feature_matrix_v7.parquet"))
partition = json.load(open(os.path.join(PROC_DIR, "feature_partition_v7.json")))
calib     = json.load(open(os.path.join(PROC_DIR, "calibration_split_idx.json")))

def get(cols):
    present = [c for c in cols if c in df.columns]
    return df[present].fillna(0.0).values.astype(np.float32), present

Xl, lcols = get(partition["LST"])
Xt, _     = get(partition["Tilt"])
Xr, rcols = get(partition["Residual"])
er_cm     = df["er_CM"].fillna(0.0).values.astype(np.float32)
has_cm    = df["has_sigma_CM"].fillna(0.0).values.astype(np.float32)
cm_approx = df["cm_approx_flag"].fillna(0.0).values.astype(np.float32)
y         = df["epsilon_r"].values.astype(np.float32)
gii       = np.zeros(len(df), dtype=np.float32)
phase_tr  = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
            if "phase_transition" in df.columns else np.zeros(len(df), np.float32)
regime_names = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
regime_idx   = [lcols.index(r) for r in regime_names if r in lcols]

calib_idx = np.array(calib["calib_idx"])

# ── Identify cs_ column indices in Xr ─────────────────────────────────────────
cs_cols = [c for c in rcols if c.startswith("cs_")]
cs_idx  = [rcols.index(c) for c in cs_cols]
print(f"Residual branch: {len(rcols)} features")
print(f"cs_ features ({len(cs_cols)}): {cs_cols}")
print(f"cs_ indices in Xr: {cs_idx}")

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(os.path.join(ROOT, "models", "cmltrv77_final.pt"),
                  map_location="cpu", weights_only=False)
hp   = ckpt["hp"]

# Build & load each seed model
def load_model(state_dict):
    m = CMLTRPCNNv71(
        n_lst=Xl.shape[1], n_tilt=Xt.shape[1], n_res=Xr.shape[1],
        trunk_hidden=hp["trunk_hidden"], n_trunk_blocks=hp["n_trunk_blocks"],
        lst_hidden=hp["lst_hidden"], tilt_hidden=hp["tilt_hidden"],
        res_hidden=hp["res_hidden"], dropout=0.0,
        residual_scale=hp["residual_scale"]
    )
    m.load_state_dict(state_dict)
    m.eval()
    return m.to(DEVICE)

models = [load_model(sd) for sd in ckpt["models"]]
print(f"Loaded {len(models)} seed models")

# ── Inference helper ──────────────────────────────────────────────────────────
def predict(Xl_input, Xt_input, Xr_input):
    idx  = calib_idx
    preds_all = []
    with torch.no_grad():
        xl  = torch.tensor(Xl_input[idx]).to(DEVICE)
        xt  = torch.tensor(Xt_input[idx]).to(DEVICE)
        xr  = torch.tensor(Xr_input[idx]).to(DEVICE)
        ecm = torch.tensor(er_cm[idx]).to(DEVICE)
        hcm = torch.tensor(has_cm[idx]).to(DEVICE)
        cma = torch.tensor(cm_approx[idx]).to(DEVICE)
        g   = torch.tensor(gii[idx]).to(DEVICE)
        pt  = torch.tensor(phase_tr[idx]).to(DEVICE)
        ri  = regime_idx

        for m in models:
            out = m(xl, xt, xr, ecm, hcm, cma, g, pt, ri)
            if isinstance(out, dict):
                er_hat = out["pred"]
            elif isinstance(out, tuple):
                er_hat = out[0]
            else:
                er_hat = out
            preds_all.append(er_hat.cpu().numpy())

    return np.mean(preds_all, axis=0)

# ── Scaler: reproduce training StandardScaler on train_idx ────────────────────
from sklearn.preprocessing import StandardScaler

train_idx = np.array(calib["train_idx"])

def scale_features(Xl_, Xt_, Xr_):
    scl = StandardScaler()
    X_all_tr = np.concatenate([Xl_[train_idx], Xt_[train_idx], Xr_[train_idx]], axis=1)
    scl.fit(X_all_tr)
    n_l, n_t, n_r = Xl_.shape[1], Xt_.shape[1], Xr_.shape[1]
    X_all = np.concatenate([Xl_, Xt_, Xr_], axis=1)
    X_sc  = scl.transform(X_all)
    return X_sc[:, :n_l], X_sc[:, n_l:n_l+n_t], X_sc[:, n_l+n_t:]

# Scale original
Xl_s, Xt_s, Xr_s = scale_features(Xl, Xt, Xr)

# cs_-zeroed version: zero BEFORE scaling (set raw values to 0 before fit/transform)
Xr_no_cs        = Xr.copy()
Xr_no_cs[:, cs_idx] = 0.0
_, _, Xr_no_cs_s = scale_features(Xl, Xt, Xr_no_cs)

# ── Run both predictions ───────────────────────────────────────────────────────
print("\nRunning inference — original features...")
pred_orig  = predict(Xl_s, Xt_s, Xr_s)

print("Running inference — cs_ features zeroed...")
pred_no_cs = predict(Xl_s, Xt_s, Xr_no_cs_s)

y_cal = y[calib_idx]

r2_orig  = r2_score(y_cal, pred_orig)
r2_nocs  = r2_score(y_cal, pred_no_cs)
mae_orig = mean_absolute_error(y_cal, pred_orig)
mae_nocs = mean_absolute_error(y_cal, pred_no_cs)

delta_r2  = r2_nocs  - r2_orig
delta_mae = mae_nocs - mae_orig

print(f"\n{'='*50}")
print(f"  Original (with cs_):   R² = {r2_orig:.4f},  MAE = {mae_orig:.3f}")
print(f"  cs_ features zeroed:   R² = {r2_nocs:.4f},  MAE = {mae_nocs:.3f}")
print(f"  ΔR²  = {delta_r2:+.4f}")
print(f"  ΔMAE = {delta_mae:+.3f}")
print(f"{'='*50}")

if abs(delta_r2) < 0.005:
    verdict = "NEGLIGIBLE — crystal system features have no measurable impact"
elif abs(delta_r2) < 0.02:
    verdict = "MINOR — crystal system features have small but non-zero impact"
else:
    verdict = "SIGNIFICANT — crystal system features matter"
print(f"\nVerdict: {verdict}")

# Also compare predictions directly
max_diff  = float(np.max(np.abs(pred_orig - pred_no_cs)))
mean_diff = float(np.mean(np.abs(pred_orig - pred_no_cs)))
print(f"\nPrediction change: mean |Δ| = {mean_diff:.4f},  max |Δ| = {max_diff:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    "cs_features": cs_cols,
    "n_cs_features": len(cs_cols),
    "n_residual_features": len(rcols),
    "n_calib_samples": len(calib_idx),
    "original": {"r2": round(r2_orig, 4), "mae": round(mae_orig, 3)},
    "cs_zeroed": {"r2": round(r2_nocs, 4), "mae": round(mae_nocs, 3)},
    "delta_r2":  round(delta_r2,  4),
    "delta_mae": round(delta_mae, 3),
    "mean_pred_change": round(mean_diff, 4),
    "max_pred_change":  round(max_diff,  4),
    "verdict": verdict,
}

out_path = os.path.join(RES_DIR, "55_cs_ablation.json")
json.dump(results, open(out_path, "w"), indent=2)
print(f"\nSaved: {out_path}")
