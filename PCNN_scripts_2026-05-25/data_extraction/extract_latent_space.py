"""
Extract the 192-dim trunk hidden state from the trained v77 ensemble, then
project to 2D via PCA. Saves both raw latent and projected coordinates.

The trunk hidden state is the model's internal representation BEFORE
splitting into the 3 physics branches — it captures the model's view
of each composition independent of any specific branch.

Outputs:
  all_figures/extracted_data/latent_space_trunk.csv    (1304 × ~196 cols)
"""
import os, sys, json, pickle
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from model_CMLTRPCNNv77 import CMLTRPCNNv71
from train_final_model import HP, DEVICE, PROC_DIR

OUT_DIR    = "/Users/anbu/Desktop/NC figures/all_figures/extracted_data"
os.makedirs(OUT_DIR, exist_ok=True)
PIML       = "/Users/anbu/Desktop/PIML/piml_ceramic"
MODEL_PATH = f"{PIML}/models/cmltrv77_final.pt"
SCAL_PATH  = f"{PIML}/models/cmltrv77_scalers.pkl"

print("=" * 70)
print("Extracting 192-dim trunk hidden state for all 1,304 compositions")
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
gii       = df["GII"].fillna(0.0).values.astype(np.float32) \
            if "GII" in df.columns else np.zeros(len(df), np.float32)
phase_tr  = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
            if "phase_transition" in df.columns else np.zeros(len(df), np.float32)
y         = df["epsilon_r"].values.astype(np.float32)

with open(SCAL_PATH, "rb") as f:
    sc = pickle.load(f)
Xl_s = sc["sc_lst"].transform(Xl).astype(np.float32)
Xt_s = sc["sc_tilt"].transform(Xt).astype(np.float32)
Xr_s = sc["sc_res"].transform(Xr).astype(np.float32)

train_idx = np.array(json.load(open(f"{PROC_DIR}/calibration_split_idx.json"))["train_idx"])
gii_train = gii[train_idx]
gii_max = float(np.percentile(gii_train[gii_train > 0], 95)
                if (gii_train > 0).sum() > 10 else 1.0)
gii_norm = np.clip(gii / gii_max, 0, 1).astype(np.float32)

# ── Load ensemble ────────────────────────────────────────────────────────────
print("Loading 5-seed ensemble ...")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
models = []
for sd in ckpt["models"]:
    m = CMLTRPCNNv71(
        n_lst=Xl.shape[1], n_tilt=Xt.shape[1], n_res=Xr.shape[1],
        trunk_hidden=HP["trunk_hidden"], n_trunk_blocks=HP["n_trunk_blocks"],
        lst_hidden=HP["lst_hidden"], tilt_hidden=HP["tilt_hidden"],
        res_hidden=HP["res_hidden"], residual_scale=HP["residual_scale"],
        dropout=HP["dropout"],
    ).to(DEVICE)
    m.load_state_dict(sd, strict=False)
    m.eval()
    models.append(m)

# ── Extract trunk hidden state per model, then average across seeds ─────────
def to_t(a):
    return torch.tensor(a, dtype=torch.float32).to(DEVICE)

print("Running trunk forward for all 1,304 samples × 5 seeds ...")
all_trunk = []
with torch.no_grad():
    for m in models:
        t_out = m._trunk(to_t(Xl_s), to_t(Xt_s), to_t(Xr_s))
        all_trunk.append(t_out.cpu().numpy())

all_trunk = np.stack(all_trunk, axis=0)   # (5, 1304, 192)
trunk_mean = all_trunk.mean(axis=0)        # (1304, 192) — average across seeds
print(f"Trunk hidden state shape: {trunk_mean.shape}")

# ── PCA to 2D ───────────────────────────────────────────────────────────────
pca = PCA(n_components=4)
trunk_2d = pca.fit_transform(trunk_mean)
print(f"PCA explained variance: {pca.explained_variance_ratio_.cumsum()}")

# ── Regime + A-site for coloring ────────────────────────────────────────────
def regime_for(i):
    for r in ["Ia", "Ib", "II", "III"]:
        if df[f"regime_{r}"].iloc[i] == 1: return r
    return "II"
regimes  = [regime_for(i) for i in range(len(y))]
a_sites  = df["chemistry_family"].apply(lambda x: str(x).split("_")[0]).values

# ── Save extracted CSV ──────────────────────────────────────────────────────
out = pd.DataFrame({
    "idx":     np.arange(len(y)),
    "formula": df["formula"].values if "formula" in df.columns else "",
    "regime":  regimes,
    "a_site":  a_sites,
    "er_measured":  y,
    "PC1":     trunk_2d[:, 0],
    "PC2":     trunk_2d[:, 1],
    "PC3":     trunk_2d[:, 2],
    "PC4":     trunk_2d[:, 3],
})
out_path = f"{OUT_DIR}/latent_space_trunk.csv"
out.to_csv(out_path, index=False)
print(f"Saved: {out_path}")

# Also save explained variance for documentation
with open(f"{OUT_DIR}/latent_space_pca_meta.json", "w") as f:
    json.dump({
        "trunk_dim": int(trunk_mean.shape[1]),
        "n_samples": int(len(y)),
        "pca_components": 4,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": pca.explained_variance_ratio_.cumsum().tolist(),
    }, f, indent=2)
print(f"Saved: {OUT_DIR}/latent_space_pca_meta.json")
