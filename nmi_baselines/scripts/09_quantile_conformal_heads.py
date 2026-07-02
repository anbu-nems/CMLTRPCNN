"""
09_quantile_conformal_heads.py — Architectural conformal prediction (Option B)

Replaces PCNN's post-hoc conformal calibration with INTRINSIC quantile-regression
heads trained via pinball loss. Per-prediction prediction intervals come out of
the architecture by construction rather than as a wrapper applied after training.

Architecture:
  Same trunk + 4-branch decomposition as canonical PCNN, but the FINAL prediction
  εr is replaced by three quantile outputs:
     q05_pred,  q50_pred,  q95_pred
  where:
     point estimate = q50_pred
     90% prediction interval = [q05_pred, q95_pred]   (asymmetric in general)

  Each branch (CM, LST, tilt, residual) still produces a single point estimate;
  the quantile heads are added as a final layer on top of εr_pred = sum-of-branches.
  Each quantile head is a residual offset:
     q05_pred = εr_pred - softplus(off_05)    (offset ≤ 0 → q05 ≤ εr_pred)
     q50_pred = εr_pred
     q95_pred = εr_pred + softplus(off_95)    (offset ≥ 0 → q95 ≥ εr_pred)
  This enforces q05 ≤ q50 ≤ q95 by construction (no quantile crossings).

Training loss:
  L = pinball(y, q05, 0.05) + pinball(y, q50, 0.50) + pinball(y, q95, 0.95)
  + small RMSE-anchor on q50 for stable optimisation

Reportable metrics (compared against canonical PCNN's post-hoc conformal):
  • Coverage on holdout: % of samples with y ∈ [q05, q95] (target = 90%)
  • Mean PI width = mean(q95 - q05)  vs PCNN's post-hoc q̂₉₀ × 2 = 28.76
  • Sharpness ratio: smaller width at same coverage → sharper prediction

Output: results/09_quantile_conformal_heads.json
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

ROOT     = "../.."  # repo root (run from nmi_baselines/scripts/)
PROC_DIR = os.path.join(ROOT, "data", "processed")
OUT_DIR  = os.path.join(ROOT, "nmi_baselines", "results")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                     else "cuda" if torch.cuda.is_available()
                     else "cpu")
print(f"[init] device: {DEVICE}")

# ── load data ─────────────────────────────────────────────────────────────────
df        = pd.read_parquet(os.path.join(PROC_DIR, "feature_matrix_v7.parquet"))
partition = json.load(open(os.path.join(PROC_DIR, "feature_partition_v7.json")))
calib     = json.load(open(os.path.join(PROC_DIR, "calibration_split_idx.json")))


def get(cols):
    present = [c for c in cols if c in df.columns]
    return df[present].fillna(0.0).values.astype(np.float32), present

Xl, _ = get(partition["LST"])
Xt, _ = get(partition["Tilt"])
Xr, _ = get(partition["Residual"])
X_all = np.concatenate([Xl, Xt, Xr], axis=1).astype(np.float32)
y       = df["epsilon_r"].values.astype(np.float32)
er_cm   = df["er_CM"].fillna(0.0).values.astype(np.float32)
lst_triplet  = df[["soft_mode_activity", "b_o_reduced_mass", "lst_enhancement_proxy"]].fillna(0.0).values.astype(np.float32)
tilt_triplet = df[["tilt_severity", "charge_imbalance_proxy", "continuous_tilt_strain"]].fillna(0.0).values.astype(np.float32)
is_ib   = df["d0_B_polarizable_A"].fillna(0.0).values.astype(np.float32)
groups  = df["chemistry_family"].values
train_idx = np.array(calib["train_idx"])
calib_idx = np.array(calib["calib_idx"])

MIN_FAMILY_SIZE = 8; N_SPLITS = 5
def build_strat_gss(train_idx, groups, n_splits=5, random_state=42):
    rng = np.random.RandomState(random_state)
    family_counts = Counter(groups[train_idx])
    large = sorted([f for f, n in family_counts.items() if n >= MIN_FAMILY_SIZE])
    ftr, fte = [], []
    for fam in large:
        idx = train_idx[groups[train_idx] == fam]
        rng.shuffle(idx)
        split = max(1, int(0.2 * len(idx)))
        fte.append(idx[:split]); ftr.append(idx[split:])
    all_tr = np.concatenate(ftr); all_te = np.concatenate(fte)
    fold_size = len(all_te) // n_splits
    folds = []
    for i in range(n_splits):
        va = all_te[i*fold_size:(i+1)*fold_size]
        tr = np.concatenate([all_tr, np.setdiff1d(all_te, va)])
        folds.append((tr, va))
    return folds
folds = build_strat_gss(train_idx, groups)


# ── PCNN with intrinsic quantile heads ──────────────────────────────────────
class PCNN_QuantileHeads(nn.Module):
    """
    Canonical PCNN architecture (sign-bounded 4-branch decomposition) +
    three quantile output heads with non-crossing parameterisation.
    """
    def __init__(self, n_in, n_lst_feats=3, n_tilt_feats=3,
                 hidden=128, n_layers=4, residual_scale=80.0):
        super().__init__()
        self.residual_scale = residual_scale

        layers = [nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        self.encoder = nn.Sequential(*layers)
        self.lst_head    = nn.Sequential(nn.Linear(hidden + n_lst_feats, 64), nn.SiLU(), nn.Linear(64, 1), nn.Softplus())
        self.lst_head_ib = nn.Sequential(nn.Linear(hidden + n_lst_feats, 128), nn.SiLU(),
                                          nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1), nn.Softplus())
        self.tilt_head   = nn.Sequential(nn.Linear(hidden + n_tilt_feats, 64), nn.SiLU(), nn.Linear(64, 1), nn.Softplus())
        self.res_head    = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1), nn.Tanh())

        # Quantile-offset heads: take trunk encoding, output a non-crossing offset
        # Architecture-level guarantee: q05 = q50 - softplus(off_lo), q95 = q50 + softplus(off_hi)
        self.q_lo_head = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1), nn.Softplus())
        self.q_hi_head = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, x, er_cm, lst_feats, tilt_feats, is_ib):
        h = self.encoder(x)
        h_lst = torch.cat([h, lst_feats], dim=1)
        d_lst_std = self.lst_head(h_lst).squeeze(-1)
        d_lst_ib  = self.lst_head_ib(h_lst).squeeze(-1)
        delta_lst = (1 - is_ib) * d_lst_std + is_ib * d_lst_ib
        delta_tilt = -self.tilt_head(torch.cat([h, tilt_feats], dim=1)).squeeze(-1)
        delta_res  = self.res_head(h).squeeze(-1) * self.residual_scale
        q50 = er_cm + delta_lst + delta_tilt + delta_res
        off_lo = self.q_lo_head(h).squeeze(-1)   # ≥ 0
        off_hi = self.q_hi_head(h).squeeze(-1)   # ≥ 0
        q05 = q50 - off_lo                       # ≤ q50 by construction
        q95 = q50 + off_hi                       # ≥ q50 by construction
        return {"q05": q05, "q50": q50, "q95": q95,
                "delta_lst": delta_lst, "delta_tilt": delta_tilt,
                "delta_res": delta_res, "er_cm": er_cm}


def pinball(y_true, y_pred, q):
    err = y_true - y_pred
    return torch.maximum(q * err, (q - 1) * err).mean()


def train_one(Xtr, ytr, er_cm_tr, lst_tr, tilt_tr, ib_tr,
              Xva, yva, er_cm_va, lst_va, tilt_va, ib_va,
              n_in, epochs=300, lr=7.4e-4, batch_size=256, seed=42,
              w_q50=1.0, w_q05=1.0, w_q95=1.0, w_anchor=0.5):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PCNN_QuantileHeads(n_in=n_in).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xtr_t   = torch.from_numpy(Xtr).to(DEVICE)
    ytr_t   = torch.from_numpy(ytr).to(DEVICE)
    er_cm_t = torch.from_numpy(er_cm_tr).to(DEVICE)
    lst_t   = torch.from_numpy(lst_tr).to(DEVICE)
    tilt_t  = torch.from_numpy(tilt_tr).to(DEVICE)
    ib_t    = torch.from_numpy(ib_tr).to(DEVICE)
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            b = perm[i:i+batch_size]
            out = model(Xtr_t[b], er_cm_t[b], lst_t[b], tilt_t[b], ib_t[b])
            l_q05 = pinball(ytr_t[b], out["q05"], 0.05)
            l_q50 = pinball(ytr_t[b], out["q50"], 0.50)
            l_q95 = pinball(ytr_t[b], out["q95"], 0.95)
            l_anchor = torch.sqrt(F.mse_loss(out["q50"], ytr_t[b]))
            loss = w_q05*l_q05 + w_q50*l_q50 + w_q95*l_q95 + w_anchor*l_anchor
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(Xva).to(DEVICE),
                    torch.from_numpy(er_cm_va).to(DEVICE),
                    torch.from_numpy(lst_va).to(DEVICE),
                    torch.from_numpy(tilt_va).to(DEVICE),
                    torch.from_numpy(ib_va).to(DEVICE))
    return {k: v.cpu().numpy() for k, v in out.items()}


# CV loop
fold_r2, fold_cov, fold_width = [], [], []
print("[train] running 5-fold Strat-GSS CV (PCNN_QuantileHeads)…")
t0 = time.time()

for fi, (tr, va) in enumerate(folds):
    sc = StandardScaler().fit(X_all[tr])
    Xtr_s = sc.transform(X_all[tr]).astype(np.float32)
    Xva_s = sc.transform(X_all[va]).astype(np.float32)
    out = train_one(Xtr_s, y[tr], er_cm[tr], lst_triplet[tr], tilt_triplet[tr], is_ib[tr],
                    Xva_s, y[va], er_cm[va], lst_triplet[va], tilt_triplet[va], is_ib[va],
                    n_in=X_all.shape[1])
    r2 = r2_score(y[va], out["q50"])
    in_pi = ((y[va] >= out["q05"]) & (y[va] <= out["q95"])).mean()
    width = (out["q95"] - out["q05"]).mean()
    fold_r2.append(r2); fold_cov.append(float(in_pi)); fold_width.append(float(width))
    print(f"  fold {fi}: R²={r2:.4f} | coverage={100*in_pi:.1f}% | mean PI width={width:.2f}")

# Holdout
print("\n[train] running final holdout pass…")
sc = StandardScaler().fit(X_all[train_idx])
Xtr_s = sc.transform(X_all[train_idx]).astype(np.float32)
Xho_s = sc.transform(X_all[calib_idx]).astype(np.float32)
out_ho = train_one(Xtr_s, y[train_idx], er_cm[train_idx], lst_triplet[train_idx], tilt_triplet[train_idx], is_ib[train_idx],
                   Xho_s, y[calib_idx], er_cm[calib_idx], lst_triplet[calib_idx], tilt_triplet[calib_idx], is_ib[calib_idx],
                   n_in=X_all.shape[1])
r2_ho  = r2_score(y[calib_idx], out_ho["q50"])
mae_ho = mean_absolute_error(y[calib_idx], out_ho["q50"])
cov_ho = float(((y[calib_idx] >= out_ho["q05"]) & (y[calib_idx] <= out_ho["q95"])).mean())
width_ho = float((out_ho["q95"] - out_ho["q05"]).mean())
ho_lst_v = int((out_ho["delta_lst"] < 0).sum())

elapsed = time.time() - t0

print(f"\n[result] PCNN_QuantileHeads:")
print(f"  Strat-GSS R² = {np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}")
print(f"  Holdout R² = {r2_ho:.4f} | MAE = {mae_ho:.2f}")
print(f"  90% PI coverage on holdout = {100*cov_ho:.1f}%  (target = 90%)")
print(f"  Mean PI width on holdout = {width_ho:.2f}  (PCNN post-hoc conformal: 28.76)")
print(f"  LST sign violations on holdout: {ho_lst_v}/{len(calib_idx)}  (architectural ≥ 0)")
print(f"\n[reference] PCNN + post-hoc conformal: Strat-GSS R²=0.893±0.044, "
      f"Holdout R²=0.941, MAE=5.77, 90%-coverage=90.5%, PI width=28.76")

result = {
    "model": "PCNN_QuantileHeads",
    "description": "Intrinsic quantile-regression heads (5%, 50%, 95%) trained via pinball loss; replaces post-hoc conformal calibration",
    "architecture": "Canonical PCNN + non-crossing-parameterised quantile-offset heads (q05 = q50 - softplus, q95 = q50 + softplus)",
    "strat_gss_r2_mean":   round(float(np.mean(fold_r2)), 4),
    "strat_gss_r2_std":    round(float(np.std(fold_r2)), 4),
    "strat_gss_r2_per_fold": [round(r, 4) for r in fold_r2],
    "strat_gss_coverage_mean":  round(float(np.mean(fold_cov)), 4),
    "strat_gss_pi_width_mean":  round(float(np.mean(fold_width)), 3),
    "holdout_r2":               round(r2_ho, 4),
    "holdout_mae":              round(mae_ho, 3),
    "holdout_90pct_coverage":   round(cov_ho, 4),
    "holdout_pi_width_mean":    round(width_ho, 3),
    "holdout_lst_violations":   ho_lst_v,
    "n_holdout":                int(len(calib_idx)),
    "training_time_seconds":    round(elapsed, 1),
    "pcnn_post_hoc_conformal_reference": {
        "strat_gss_r2": "0.893 ± 0.044", "holdout_r2": 0.941, "holdout_mae": 5.77,
        "holdout_90pct_coverage": 0.905, "pi_width_2qhat90": 28.76,
        "note": "Canonical PCNN with post-hoc conformal calibration (q̂₉₀ = 14.38)",
    },
}
out_path = os.path.join(OUT_DIR, "09_quantile_conformal_heads.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n[done] saved → {out_path}")
