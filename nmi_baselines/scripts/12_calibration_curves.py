"""
12_calibration_curves.py — compute empirical-vs-nominal calibration data for
each PCNN architectural variant so we can replace panel (e) of the combined
figure with a calibration curve (gold-standard UQ plot).

For each variant, compute empirical coverage on the holdout (n=116) at nominal
levels α ∈ {0.5, 0.3, 0.2, 0.1, 0.05} (target coverage = 1−α).

Variants & methodology:
  PCNN canonical (post-hoc conformal):
    - Train PCNN-equivalent on 80% of train_idx
    - Compute residuals on remaining 20% (calibration set)
    - q̂_α = (1-α)-quantile of |residuals|
    - On holdout: empirical coverage = fraction of |y_true - pred| ≤ q̂_α

  PCNN + Laplace (Bayesian):
    - Train PCNN-equivalent on full train_idx
    - Fit Laplace last-layer posterior
    - At each α: PI = mean ± z(α/2) × std; empirical coverage on holdout

  PCNN + Quantile heads (intrinsic):
    - Already trained; only gives 90% PI by construction
    - Plot only the single (nominal=90%, empirical=57.8%) point

Output: results/12_calibration_curves.json (calibration data for plotting)
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

ROOT     = "../.."  # repo root (run from nmi_baselines/scripts/)
PROC_DIR = os.path.join(ROOT, "data", "processed")
OUT_DIR  = os.path.join(ROOT, "nmi_baselines", "results")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cpu")   # cpu for Laplace compat
print(f"[init] device: {DEVICE}")

# ── load data ─────────────────────────────────────────────────────────────────
df        = pd.read_parquet(os.path.join(PROC_DIR, "feature_matrix_v7.parquet"))
partition = json.load(open(os.path.join(PROC_DIR, "feature_partition_v7.json")))
calib     = json.load(open(os.path.join(PROC_DIR, "calibration_split_idx.json")))

def get(cols):
    present = [c for c in cols if c in df.columns]
    return df[present].fillna(0.0).values.astype(np.float32), present

Xl, _ = get(partition["LST"]); Xt, _ = get(partition["Tilt"]); Xr, _ = get(partition["Residual"])
X_all = np.concatenate([Xl, Xt, Xr], axis=1).astype(np.float32)
y       = df["epsilon_r"].values.astype(np.float32)
er_cm   = df["er_CM"].fillna(0.0).values.astype(np.float32)
lst_triplet  = df[["soft_mode_activity", "b_o_reduced_mass", "lst_enhancement_proxy"]].fillna(0.0).values.astype(np.float32)
tilt_triplet = df[["tilt_severity", "charge_imbalance_proxy", "continuous_tilt_strain"]].fillna(0.0).values.astype(np.float32)
is_ib   = df["d0_B_polarizable_A"].fillna(0.0).values.astype(np.float32)
train_idx = np.array(calib["train_idx"])
calib_idx = np.array(calib["calib_idx"])


# ── Flat PCNN-equivalent (same as script 10) ─────────────────────────────
class FlatPCNN(nn.Module):
    def __init__(self, n_in_flat, hidden=128, n_layers=4, residual_scale=80.0):
        super().__init__()
        self.n_main = n_in_flat - 8
        self.residual_scale = residual_scale
        layers = [nn.Linear(self.n_main, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        self.encoder = nn.Sequential(*layers)
        self.lst_head    = nn.Sequential(nn.Linear(hidden + 3, 64), nn.SiLU(), nn.Linear(64, 1), nn.Softplus())
        self.lst_head_ib = nn.Sequential(nn.Linear(hidden + 3, 128), nn.SiLU(),
                                          nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1), nn.Softplus())
        self.tilt_head   = nn.Sequential(nn.Linear(hidden + 3, 64), nn.SiLU(), nn.Linear(64, 1), nn.Softplus())
        self.res_head    = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, x_flat):
        x      = x_flat[:, :self.n_main]
        er_cm_ = x_flat[:, self.n_main + 0]
        is_ib_ = x_flat[:, self.n_main + 1]
        lst_f  = x_flat[:, self.n_main + 2 : self.n_main + 5]
        tilt_f = x_flat[:, self.n_main + 5 : self.n_main + 8]
        h = self.encoder(x)
        d_lst_std = self.lst_head(torch.cat([h, lst_f], dim=1)).squeeze(-1)
        d_lst_ib  = self.lst_head_ib(torch.cat([h, lst_f], dim=1)).squeeze(-1)
        delta_lst = (1 - is_ib_) * d_lst_std + is_ib_ * d_lst_ib
        delta_tilt = -self.tilt_head(torch.cat([h, tilt_f], dim=1)).squeeze(-1)
        delta_res  = self.res_head(h).squeeze(-1) * self.residual_scale
        return (er_cm_ + delta_lst + delta_tilt + delta_res).unsqueeze(-1)


def make_flat(X, er_cm_v, lst_v, tilt_v, ib_v):
    return np.concatenate([
        X, er_cm_v[:, None].astype(np.float32), ib_v[:, None].astype(np.float32),
        lst_v.astype(np.float32), tilt_v.astype(np.float32)
    ], axis=1)


def train_flat(Xtr_flat, ytr, epochs=300, lr=7.4e-4, batch_size=256, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = FlatPCNN(n_in_flat=Xtr_flat.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xtr_t = torch.from_numpy(Xtr_flat).to(DEVICE)
    ytr_t = torch.from_numpy(ytr).to(DEVICE).unsqueeze(-1)
    n = len(Xtr_flat)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            b = perm[i:i+batch_size]
            pred = model(Xtr_t[b])
            loss = torch.sqrt(F.mse_loss(pred, ytr_t[b]))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
    return model


# ── Standardize, build flat features ────────────────────────────────────
sc = StandardScaler().fit(X_all[train_idx])
Xtr_s = sc.transform(X_all[train_idx]).astype(np.float32)
Xho_s = sc.transform(X_all[calib_idx]).astype(np.float32)
Xtr_flat = make_flat(Xtr_s, er_cm[train_idx], lst_triplet[train_idx],
                     tilt_triplet[train_idx], is_ib[train_idx])
Xho_flat = make_flat(Xho_s, er_cm[calib_idx], lst_triplet[calib_idx],
                     tilt_triplet[calib_idx], is_ib[calib_idx])

# Nominal alpha levels for the calibration curve
ALPHAS = [0.50, 0.30, 0.20, 0.10, 0.05]
NOMINAL_COVERAGE = [1.0 - a for a in ALPHAS]   # 0.50, 0.70, 0.80, 0.90, 0.95


# ════════════════════════════════════════════════════════════════════════
# 1) PCNN canonical + post-hoc conformal (split-conformal)
# ════════════════════════════════════════════════════════════════════════
print("\n[1/2] PCNN canonical (post-hoc split-conformal)…")
rng = np.random.RandomState(0)
n_train = len(train_idx)
perm = rng.permutation(n_train)
n_calib_subset = int(0.20 * n_train)
calib_subset_pos  = perm[:n_calib_subset]
proper_train_pos  = perm[n_calib_subset:]

# Train on the 80% proper-train subset
print(f"  train on {len(proper_train_pos)} samples, calibrate on {len(calib_subset_pos)}…")
t0 = time.time()
model_pcnn = train_flat(Xtr_flat[proper_train_pos], y[train_idx][proper_train_pos])
print(f"  trained in {time.time()-t0:.1f}s")

# Compute residuals on the calibration set
model_pcnn.eval()
with torch.no_grad():
    pred_calib = model_pcnn(torch.from_numpy(Xtr_flat[calib_subset_pos])).cpu().numpy().squeeze(-1)
    pred_ho    = model_pcnn(torch.from_numpy(Xho_flat)).cpu().numpy().squeeze(-1)
residuals_calib = np.abs(y[train_idx][calib_subset_pos] - pred_calib)
residuals_holdout = np.abs(y[calib_idx] - pred_ho)

# For each α: q̂_α = (1-α)-empirical-quantile of calibration residuals (split-conformal)
# Empirical coverage on holdout = fraction of |y_test - pred| ≤ q̂_α
pcnn_emp_coverage = []
pcnn_qhats = []
for a in ALPHAS:
    # Split-conformal: take ⌈(n+1)(1-α)⌉/n quantile to ensure marginal coverage ≥ 1-α
    n_c = len(residuals_calib)
    q_level = np.ceil((n_c + 1) * (1 - a)) / n_c
    q_level = min(q_level, 1.0)
    qhat = np.quantile(residuals_calib, q_level, method="higher")
    pcnn_qhats.append(float(qhat))
    cov = float((residuals_holdout <= qhat).mean())
    pcnn_emp_coverage.append(cov)
    print(f"  α={a:.2f} (nominal={1-a:.2f}): q̂={qhat:.2f}, empirical coverage={cov:.4f}")


# ════════════════════════════════════════════════════════════════════════
# 2) PCNN + Last-Layer Laplace
# ════════════════════════════════════════════════════════════════════════
print("\n[2/2] PCNN + Last-Layer Laplace (Daxberger 2021)…")
print(f"  train on full {len(train_idx)} samples…")
t0 = time.time()
model_lap_base = train_flat(Xtr_flat, y[train_idx])
print(f"  trained in {time.time()-t0:.1f}s")

from laplace import Laplace
print("  fitting Laplace last-layer posterior (diagonal Hessian)…")
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(Xtr_flat), torch.from_numpy(y[train_idx]).unsqueeze(-1)),
    batch_size=64, shuffle=False)
la = Laplace(model_lap_base, "regression", subset_of_weights="all",
             hessian_structure="diag")
t1 = time.time()
la.fit(train_loader)
la.optimize_prior_precision()
print(f"  fitted in {time.time()-t1:.1f}s | sigma_noise={la.sigma_noise.item():.4f}")

# Predict on holdout
pred_mean, pred_var = la(torch.from_numpy(Xho_flat))
pred_mean = pred_mean.detach().cpu().numpy().squeeze(-1)
pred_std  = pred_var.detach().cpu().numpy().squeeze(-1) ** 0.5

# Empirical coverage at each α: PI = mean ± z(α/2) × std
lap_emp_coverage = []
for a in ALPHAS:
    z = norm.ppf(1 - a/2)                           # two-sided z-score
    ci_lo = pred_mean - z * pred_std
    ci_hi = pred_mean + z * pred_std
    cov   = float(((y[calib_idx] >= ci_lo) & (y[calib_idx] <= ci_hi)).mean())
    lap_emp_coverage.append(cov)
    print(f"  α={a:.2f} (nominal={1-a:.2f}): z={z:.3f}, empirical coverage={cov:.4f}")


# ════════════════════════════════════════════════════════════════════════
# 3) PCNN + Quantile heads (only 90% PI available; load from script 09)
# ════════════════════════════════════════════════════════════════════════
print("\n[3/2] PCNN + Quantile heads — only 90% PI by construction…")
with open(os.path.join(OUT_DIR, "09_quantile_conformal_heads.json")) as f:
    r09 = json.load(f)
quant_single_point = {
    "nominal_coverage":   0.90,
    "empirical_coverage": float(r09["holdout_90pct_coverage"]),
}
print(f"  single point: nominal=0.90, empirical={quant_single_point['empirical_coverage']:.4f}")


# ── Save ────────────────────────────────────────────────────────────────────
result = {
    "description": "Calibration curve data for empirical vs nominal coverage across α levels",
    "alphas":              ALPHAS,
    "nominal_coverage":    NOMINAL_COVERAGE,
    "pcnn_post_hoc_conformal": {
        "empirical_coverage":  [round(v, 4) for v in pcnn_emp_coverage],
        "qhat_per_alpha":      [round(v, 3) for v in pcnn_qhats],
        "calibration_set_size": int(n_calib_subset),
        "n_proper_train":       int(len(proper_train_pos)),
    },
    "pcnn_laplace": {
        "empirical_coverage":  [round(v, 4) for v in lap_emp_coverage],
        "predictive_std_mean": round(float(pred_std.mean()), 3),
        "laplace_sigma_noise": round(float(la.sigma_noise.item()), 4),
    },
    "pcnn_quantile_heads_single_point": quant_single_point,
    "n_holdout": int(len(calib_idx)),
}
out_path = os.path.join(OUT_DIR, "12_calibration_curves.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n[done] saved → {out_path}\n")
print("=" * 72)
print(f"{'nominal':<10} {'PCNN+conf.':<14} {'PCNN+Laplace':<14} {'Quant heads':<12}")
print("-" * 72)
for i, (a, nom) in enumerate(zip(ALPHAS, NOMINAL_COVERAGE)):
    q_emp = quant_single_point['empirical_coverage'] if abs(nom - 0.90) < 1e-6 else "—"
    if isinstance(q_emp, float):
        q_str = f"{q_emp:.3f}"
    else:
        q_str = q_emp
    print(f"{nom*100:.0f}%        {100*pcnn_emp_coverage[i]:6.1f}%         "
          f"{100*lap_emp_coverage[i]:6.1f}%         {q_str}")
print("=" * 72)
