"""
Script 56 — PIRNN fair comparison against CMLTRPCNNv7.7
Trains PIRNN on feature_matrix_v7 with identical Strat-GSS CV protocol.

Key difference from PCNN:
  PIRNN: all 103 features → single shared encoder → sign-constrained heads
  PCNN:  strict 3-way feature partition + regime scaling + confidence gate

Outputs: results/56_pirnn_comparison.json
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from collections import Counter

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results")

# ── Load data ─────────────────────────────────────────────────────────────────
df        = pd.read_parquet(os.path.join(PROC_DIR, "feature_matrix_v7.parquet"))
partition = json.load(open(os.path.join(PROC_DIR, "feature_partition_v7.json")))
calib     = json.load(open(os.path.join(PROC_DIR, "calibration_split_idx.json")))

def get(cols):
    present = [c for c in cols if c in df.columns]
    return df[present].fillna(0.0).values.astype(np.float32), present

Xl, lcols = get(partition["LST"])
Xt, _     = get(partition["Tilt"])
Xr, _     = get(partition["Residual"])

# All 103 features concatenated — PIRNN sees everything at once (no partition)
X_all = np.concatenate([Xl, Xt, Xr], axis=1).astype(np.float32)

er_cm     = df["er_CM"].fillna(0.0).values.astype(np.float32)
has_cm    = df["has_sigma_CM"].fillna(0.0).values.astype(np.float32)
cm_approx = df["cm_approx_flag"].fillna(0.0).values.astype(np.float32)
y         = df["epsilon_r"].values.astype(np.float32)
gii       = np.zeros(len(df), dtype=np.float32)

# d0pf and tilt for PIRNN modulation (from LST/Tilt features)
d0pf_col = "d0_B_polarizable_A"
tilt_col  = "tilt_severity"
d0pf = df[d0pf_col].fillna(0.0).values.astype(np.float32) if d0pf_col in df.columns \
       else np.zeros(len(df), np.float32)
tilt = df[tilt_col].fillna(0.0).values.astype(np.float32) if tilt_col in df.columns \
       else np.zeros(len(df), np.float32)

groups     = df["chemistry_family"].values
train_idx  = np.array(calib["train_idx"])
calib_idx  = np.array(calib["calib_idx"])

# ── PIRNN definition (fair: same hidden size scale as PCNN branches) ──────────
class PIRNNFair(nn.Module):
    """
    Physics-Input Residual Neural Network — fair comparison version.
    All 103 features enter a single shared encoder (no branch partition).
    Sign constraints via Softplus but NO regime scaling, NO confidence gate.
    Hidden=256 (comparable total params to PCNN branches combined).
    """
    def __init__(self, n_in, hidden=256, dropout=0.15):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
        )
        h2 = hidden // 2
        self.lst_head  = nn.Sequential(nn.Linear(h2, 64), nn.SiLU(), nn.Linear(64, 1))
        self.tilt_head = nn.Sequential(nn.Linear(h2, 32), nn.SiLU(), nn.Linear(32, 1))
        self.res_head  = nn.Sequential(nn.Linear(h2, 64), nn.SiLU(), nn.Linear(64, 1))
        self.fallback  = nn.Sequential(nn.Linear(n_in, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x, er_cm, has_cm, d0pf, tilt):
        h       = self.enc(x)
        delta_l = F.softplus(self.lst_head(h).squeeze(-1))          # ≥ 0
        delta_t = -F.softplus(self.tilt_head(h).squeeze(-1))        # ≤ 0
        delta_r = torch.tanh(self.res_head(h).squeeze(-1)) * 30.0   # bounded ±
        er_phys = (er_cm + delta_l + delta_t + delta_r).clamp(1, 600)
        er_fall = (self.fallback(x).squeeze(-1) + 8.0).clamp(1, 600)
        return has_cm * er_phys + (1 - has_cm) * er_fall

# ── Training helpers ──────────────────────────────────────────────────────────
def pirnn_loss(pred, y_batch):
    return torch.sqrt(F.mse_loss(pred, y_batch))

def scale(X, train_idx):
    sc = StandardScaler().fit(X[train_idx])
    return sc.transform(X).astype(np.float32)

def train_pirnn(X_s, er_cm, has_cm, d0pf, tilt, y,
                tr_idx, va_idx, seed=0, n_epochs=400, lr=0.001,
                batch=128, wd=0.0005):
    torch.manual_seed(seed)
    model = PIRNNFair(n_in=X_s.shape[1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    arrays = [X_s, er_cm, has_cm, d0pf, tilt, y]
    best_va, best_sd, patience, wait = 1e9, None, 60, 0

    for ep in range(n_epochs):
        model.train()
        idx = tr_idx[torch.randperm(len(tr_idx), generator=torch.Generator().manual_seed(ep))]
        for i in range(0, len(idx), batch):
            sl = idx[i:i+batch]
            bx, bc, bh, bd, bt, by = [torch.tensor(a[sl]).to(DEVICE) for a in arrays]
            opt.zero_grad()
            pirnn_loss(model(bx, bc, bh, bd, bt), by).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vx, vc, vh, vd, vt, vy = [torch.tensor(a[va_idx]).to(DEVICE) for a in arrays]
            va_loss = float(pirnn_loss(model(vx, vc, vh, vd, vt), vy))
        if va_loss < best_va:
            best_va, best_sd, wait = va_loss, {k: v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_sd)
    return model

def predict_pirnn(model, X_s, er_cm, has_cm, d0pf, tilt, idx):
    model.eval()
    with torch.no_grad():
        out = model(
            torch.tensor(X_s[idx]).to(DEVICE),
            torch.tensor(er_cm[idx]).to(DEVICE),
            torch.tensor(has_cm[idx]).to(DEVICE),
            torch.tensor(d0pf[idx]).to(DEVICE),
            torch.tensor(tilt[idx]).to(DEVICE),
        )
    return out.cpu().numpy()

# ── Strat-GSS CV (same protocol as PCNN) ─────────────────────────────────────
MIN_FAMILY_SIZE = 8
N_SPLITS = 5

def build_strat_gss(train_idx, groups, n_splits=5, random_state=42):
    rng = np.random.RandomState(random_state)
    family_counts = Counter(groups[train_idx])
    large = sorted([f for f, n in family_counts.items() if n >= MIN_FAMILY_SIZE])
    ftr, fte = [], []
    for fam in large:
        idx = train_idx[groups[train_idx] == fam]
        rng.shuffle(idx)
        split = max(1, int(0.2 * len(idx)))
        fte.append(idx[:split])
        ftr.append(idx[split:])
    all_tr = np.concatenate(ftr)
    all_te = np.concatenate(fte)
    fold_size = len(all_te) // n_splits
    folds = []
    for i in range(n_splits):
        va = all_te[i*fold_size:(i+1)*fold_size]
        tr = np.concatenate([all_tr, np.setdiff1d(all_te, va)])
        folds.append((tr, va))
    return folds

folds = build_strat_gss(train_idx, groups)
print(f"Strat-GSS folds: {len(folds)}, fold sizes: {[len(f[1]) for f in folds]}")

# ── Run CV ────────────────────────────────────────────────────────────────────
N_SEEDS = 3
gss_r2_per_fold = []

for fold_i, (tr, va) in enumerate(folds):
    print(f"\nFold {fold_i+1}/{N_SPLITS} (tr={len(tr)}, va={len(va)})")

    X_s   = scale(X_all, tr)
    preds = []
    for seed in range(N_SEEDS):
        m = train_pirnn(X_s, er_cm, has_cm, d0pf, tilt, y, tr, va,
                        seed=seed, n_epochs=400)
        preds.append(predict_pirnn(m, X_s, er_cm, has_cm, d0pf, tilt, va))

    p_mean = np.mean(preds, axis=0)
    r2 = r2_score(y[va], p_mean)
    mae = mean_absolute_error(y[va], p_mean)
    print(f"  Fold R² = {r2:.4f}  MAE = {mae:.3f}")
    gss_r2_per_fold.append(r2)

gss_r2_mean = float(np.mean(gss_r2_per_fold))
gss_r2_std  = float(np.std(gss_r2_per_fold))
print(f"\nStrat-GSS R² = {gss_r2_mean:.4f} ± {gss_r2_std:.4f}")

# ── Held-out calibration set evaluation ──────────────────────────────────────
print("\nTraining final model on full train_idx for held-out evaluation...")
X_s_final = scale(X_all, train_idx)
final_preds = []
for seed in range(N_SEEDS):
    m = train_pirnn(X_s_final, er_cm, has_cm, d0pf, tilt, y,
                    train_idx, calib_idx, seed=seed, n_epochs=400)
    final_preds.append(predict_pirnn(m, X_s_final, er_cm, has_cm, d0pf, tilt, calib_idx))

p_final = np.mean(final_preds, axis=0)
r2_holdout  = float(r2_score(y[calib_idx], p_final))
mae_holdout = float(mean_absolute_error(y[calib_idx], p_final))
print(f"Held-out R² = {r2_holdout:.4f}  MAE = {mae_holdout:.3f}")

# ── Compare with PCNN ─────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Model comparison (same data, same splits)")
print(f"{'='*55}")
print(f"  {'Model':<25} {'Strat-GSS R²':<18} {'Holdout R²'}")
print(f"  {'-'*52}")
_retrain = json.load(open(os.path.join(RES_DIR, "48_cmltrv77_retrain.json")))
_conf    = json.load(open(os.path.join(RES_DIR, "54_conformal_v77.json")))
PCNN_GSS_R2  = round(_retrain["strat_gss"]["r2_mean"], 4)
PCNN_GSS_STD = round(_retrain["strat_gss"]["r2_std"],  4)
PCNN_HO_R2   = round(_conf["r2_calibration"], 4)
PCNN_HO_MAE  = round(_conf["mae_calibration"], 3)

print(f"  {'PIRNN (this script)':<25} {gss_r2_mean:.4f} ± {gss_r2_std:.4f}   {r2_holdout:.4f}")
print(f"  {'CMLTRPCNNv7.7 (PCNN)':<25} {PCNN_GSS_R2:.4f} ± {PCNN_GSS_STD:.4f}   {PCNN_HO_R2:.4f}")
print(f"  {'Delta (PCNN - PIRNN)':<25} {PCNN_GSS_R2 - gss_r2_mean:+.4f}{'':13} {PCNN_HO_R2 - r2_holdout:+.4f}")
print(f"{'='*55}")

delta_gss     = round(PCNN_GSS_R2 - gss_r2_mean, 4)
delta_holdout = round(PCNN_HO_R2  - r2_holdout, 4)

if delta_gss > 0.05:
    verdict = "PCNN architectural constraints provide substantial improvement over physics-input baseline"
elif delta_gss > 0.02:
    verdict = "PCNN architectural constraints provide moderate improvement over physics-input baseline"
else:
    verdict = "Marginal difference — both architectures perform similarly"
print(f"\nVerdict: {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    "comparison": "PIRNN vs CMLTRPCNNv7.7 (PCNN)",
    "protocol": f"Strat-GSS 5-fold CV + held-out set (n={len(calib_idx)})",
    "PIRNN": {
        "description": "Physics-Input Residual NN — all 103 features in single encoder, sign constraints only, no branch partition",
        "n_seeds": N_SEEDS,
        "strat_gss_r2_mean": round(gss_r2_mean, 4),
        "strat_gss_r2_std":  round(gss_r2_std, 4),
        "strat_gss_r2_per_fold": [round(r, 4) for r in gss_r2_per_fold],
        "holdout_r2":  round(r2_holdout, 4),
        "holdout_mae": round(mae_holdout, 3),
    },
    "PCNN": {
        "description": "CMLTRPCNNv7.7 — strict 3-way feature partition, regime scaling, confidence gate",
        "strat_gss_r2_mean": PCNN_GSS_R2,
        "strat_gss_r2_std":  PCNN_GSS_STD,
        "holdout_r2":  PCNN_HO_R2,
        "holdout_mae": PCNN_HO_MAE,
    },
    "delta_strat_gss_r2":  delta_gss,
    "delta_holdout_r2":    delta_holdout,
    "verdict": verdict,
}

out_path = os.path.join(RES_DIR, "56_pirnn_comparison.json")
json.dump(results, open(out_path, "w"), indent=2)
print(f"\nSaved: {out_path}")
