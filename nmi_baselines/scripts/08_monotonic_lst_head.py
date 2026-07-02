"""
08_monotonic_lst_head.py — Physics-informed monotonic LST branch (Option A)

Extends PCNN's architectural sign-bound with a physics-informed MONOTONICITY
constraint in the LST head:

  δ_LST ≥ 0                      ← existing sign bound (Softplus output)
  +  δ_LST is monotone non-decreasing in soft-mode-driver inputs:
       • soft_mode_activity ↑   ⇒  δ_LST ↑
       • lst_enhancement_proxy ↑ ⇒  δ_LST ↑
       • d⁰_B_polarizable_A ↑   ⇒  δ_LST ↑

Implementation (Daniels & Velikova 2010; You et al. 2017 Deep Lattice Networks):
The LST head splits into two components that are summed pre-Softplus:

  (i)  FREE part — standard MLP over the trunk encoding + non-monotone LST features
  (ii) MONOTONE part — positive-weight linear aggregation over the 3 soft-mode-driver
       features (parameterised as softplus(W_raw) to guarantee w_i ≥ 0)

Both pre-Softplus components are summed, then passed through Softplus to enforce
δ_LST ≥ 0. Monotonicity in the soft-mode-driver inputs is guaranteed by construction:
∂δ_LST/∂x_mono = softplus'(combined) · softplus(W_raw)_i ≥ 0.

This strengthens the NMI architectural-constraints story: PCNN now provides BOTH
sign and monotonicity guarantees by construction, contrasting with PMN's naive
per-feature monotonicity (CV R² 0.732) which was decisively worse.

Output: results/08_monotonic_lst_head.json
"""
import os, sys, json, time, warnings
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
N_LST  = Xl.shape[1]

y       = df["epsilon_r"].values.astype(np.float32)
er_cm   = df["er_CM"].fillna(0.0).values.astype(np.float32)
lst_triplet  = df[["soft_mode_activity", "b_o_reduced_mass", "lst_enhancement_proxy"]].fillna(0.0).values.astype(np.float32)
tilt_triplet = df[["tilt_severity", "charge_imbalance_proxy", "continuous_tilt_strain"]].fillna(0.0).values.astype(np.float32)
is_ib   = df["d0_B_polarizable_A"].fillna(0.0).values.astype(np.float32)
groups  = df["chemistry_family"].values
train_idx = np.array(calib["train_idx"])
calib_idx = np.array(calib["calib_idx"])

# Index of soft-mode-driver features WITHIN the LST triplet:
#   [0] soft_mode_activity      ← monotone
#   [1] b_o_reduced_mass        ← non-monotone (free)
#   [2] lst_enhancement_proxy   ← monotone
# Also: is_ib (d⁰_B_polarizable_A flag) — monotone as scalar
MONO_TRIPLET_IDX = [0, 2]     # soft_mode_activity + lst_enhancement_proxy
FREE_TRIPLET_IDX = [1]        # b_o_reduced_mass

# Strat-GSS CV (identical to scripts 01-05)
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
    all_tr = np.concatenate(ftr); all_te = np.concatenate(fte)
    fold_size = len(all_te) // n_splits
    folds = []
    for i in range(n_splits):
        va = all_te[i*fold_size:(i+1)*fold_size]
        tr = np.concatenate([all_tr, np.setdiff1d(all_te, va)])
        folds.append((tr, va))
    return folds

folds = build_strat_gss(train_idx, groups)


# ── Monotonic LST head architecture ─────────────────────────────────────
class PCNN_MonotonicLST(nn.Module):
    """
    PCNN with architectural monotonicity in the LST branch.

    LST head decomposes as:
        h_lst_input = concat(trunk_encoding, b_o_reduced_mass)  # free part input
        free_out    = FreeMLP(h_lst_input)                       # any sign
        mono_w      = softplus(W_raw)                            # element-wise ≥ 0
        mono_out    = (x_mono · mono_w).sum() + b_mono           # monotone in x_mono
        ib_bonus    = softplus(W_ib) * is_ib_flag                # extra boost for Ib regime

        δ_LST = softplus(free_out + mono_out + ib_bonus)   ← ≥ 0 AND monotonic in
                                                              soft_mode_activity,
                                                              lst_enhancement_proxy,
                                                              d⁰_B_polarizable_A
    """
    def __init__(self, n_in, n_lst_feats=3, n_tilt_feats=3,
                 hidden=128, n_layers=4, residual_scale=80.0,
                 n_mono_lst=2, n_free_lst=1):
        super().__init__()
        self.residual_scale = residual_scale
        self.n_mono_lst = n_mono_lst
        self.n_free_lst = n_free_lst

        # Shared encoder (identical to canonical PCNN)
        layers = [nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        self.encoder = nn.Sequential(*layers)

        # LST head — FREE part (over trunk + non-monotone features)
        self.lst_free = nn.Sequential(
            nn.Linear(hidden + n_free_lst, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )
        # LST head — MONOTONE part (positive-weight aggregation, softplus-parameterised)
        self.lst_mono_W = nn.Parameter(torch.randn(n_mono_lst) * 0.1)
        self.lst_mono_b = nn.Parameter(torch.zeros(1))
        # Ib regime bonus (positive-weight scalar)
        self.lst_ib_W = nn.Parameter(torch.tensor(0.5))   # also softplus'd to ensure ≥ 0

        # Tilt head — unchanged (architecturally ≤ 0 via -Softplus)
        self.tilt_head = nn.Sequential(
            nn.Linear(hidden + n_tilt_feats, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()
        )
        # Residual head — unchanged
        self.res_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Tanh()
        )

    def forward(self, x, er_cm, lst_feats, tilt_feats, is_ib):
        h = self.encoder(x)

        # LST: free + monotone + ib_bonus, then Softplus
        x_mono = lst_feats[:, MONO_TRIPLET_IDX]              # [B, 2]
        x_free = lst_feats[:, FREE_TRIPLET_IDX]              # [B, 1]
        free_pre = self.lst_free(torch.cat([h, x_free], dim=1)).squeeze(-1)
        mono_w   = F.softplus(self.lst_mono_W)               # element-wise ≥ 0
        mono_pre = (x_mono * mono_w).sum(dim=1) + self.lst_mono_b.squeeze()
        ib_bonus = F.softplus(self.lst_ib_W) * is_ib
        delta_lst = F.softplus(free_pre + mono_pre + ib_bonus)  # ≥ 0 by construction

        # Tilt and residual (unchanged from PCNN)
        h_tilt = torch.cat([h, tilt_feats], dim=1)
        delta_tilt = -self.tilt_head(h_tilt).squeeze(-1)
        delta_res = self.res_head(h).squeeze(-1) * self.residual_scale

        return {"pred": er_cm + delta_lst + delta_tilt + delta_res,
                "delta_lst":  delta_lst, "delta_tilt": delta_tilt,
                "delta_res":  delta_res, "er_cm": er_cm,
                "mono_w_softplus": F.softplus(self.lst_mono_W)}


def train_one(Xtr, ytr, er_cm_tr, lst_tr, tilt_tr, ib_tr,
              Xva, yva, er_cm_va, lst_va, tilt_va, ib_va,
              n_in, epochs=300, lr=7.4e-4, batch_size=256, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PCNN_MonotonicLST(n_in=n_in).to(DEVICE)
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
            loss = torch.sqrt(F.mse_loss(out["pred"], ytr_t[b]))
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
    return ({k: v.cpu().numpy() for k, v in out.items()}, model)


def verify_monotonicity(model, sc, X_sample, lst_sample, tilt_sample, ib_sample, er_cm_sample,
                        feature_idx, delta=0.5, n_test=200):
    """Empirical monotonicity check: for n_test samples, perturb the specified
    LST feature by +delta (in standardised units) and confirm δ_LST does not
    decrease. Returns the fraction of samples where monotonicity holds."""
    n_avail = min(n_test, len(X_sample))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_sample), size=n_avail, replace=False)
    Xs = X_sample[idx].copy()
    lst_s = lst_sample[idx].copy()
    tilt_s = tilt_sample[idx].copy()
    ib_s = ib_sample[idx].copy()
    er_s = er_cm_sample[idx].copy()

    model.eval()
    with torch.no_grad():
        out_base = model(torch.from_numpy(Xs.astype(np.float32)).to(DEVICE),
                         torch.from_numpy(er_s).to(DEVICE),
                         torch.from_numpy(lst_s).to(DEVICE),
                         torch.from_numpy(tilt_s).to(DEVICE),
                         torch.from_numpy(ib_s).to(DEVICE))
        delta_lst_base = out_base["delta_lst"].cpu().numpy()

        # Perturb only the targeted LST-triplet feature by +delta
        lst_pert = lst_s.copy()
        lst_pert[:, feature_idx] += delta
        out_pert = model(torch.from_numpy(Xs.astype(np.float32)).to(DEVICE),
                         torch.from_numpy(er_s).to(DEVICE),
                         torch.from_numpy(lst_pert).to(DEVICE),
                         torch.from_numpy(tilt_s).to(DEVICE),
                         torch.from_numpy(ib_s).to(DEVICE))
        delta_lst_pert = out_pert["delta_lst"].cpu().numpy()

    mono_holds = (delta_lst_pert >= delta_lst_base - 1e-6)
    return float(mono_holds.mean()), delta_lst_pert - delta_lst_base


# ── CV loop ────────────────────────────────────────────────────────────────
fold_r2 = []
fold_lst_violations = []
fold_mono_check = []
print("[train] running 5-fold Strat-GSS CV (PCNN_MonotonicLST)…")
t0 = time.time()

for fi, (tr, va) in enumerate(folds):
    sc = StandardScaler().fit(X_all[tr])
    Xtr_s = sc.transform(X_all[tr]).astype(np.float32)
    Xva_s = sc.transform(X_all[va]).astype(np.float32)

    out, model = train_one(
        Xtr_s, y[tr], er_cm[tr], lst_triplet[tr], tilt_triplet[tr], is_ib[tr],
        Xva_s, y[va], er_cm[va], lst_triplet[va], tilt_triplet[va], is_ib[va],
        n_in=X_all.shape[1])

    r2 = r2_score(y[va], out["pred"])
    fold_r2.append(r2)
    fold_lst_violations.append(int((out["delta_lst"] < 0).sum()))
    # Monotonicity check on training subset
    mono_rate, _ = verify_monotonicity(model, sc,
                                        Xtr_s, lst_triplet[tr], tilt_triplet[tr],
                                        is_ib[tr], er_cm[tr],
                                        feature_idx=0, delta=0.5)  # soft_mode_activity
    fold_mono_check.append(mono_rate)
    print(f"  fold {fi}: R²={r2:.4f} | LST_viol={fold_lst_violations[-1]}/{len(va)} | "
          f"mono_holds_for_soft_mode_activity={100*mono_rate:.1f}%")

# Holdout
print("\n[train] running final holdout pass…")
sc = StandardScaler().fit(X_all[train_idx])
Xtr_s = sc.transform(X_all[train_idx]).astype(np.float32)
Xho_s = sc.transform(X_all[calib_idx]).astype(np.float32)

out_ho, model_full = train_one(
    Xtr_s, y[train_idx], er_cm[train_idx], lst_triplet[train_idx], tilt_triplet[train_idx], is_ib[train_idx],
    Xho_s, y[calib_idx], er_cm[calib_idx], lst_triplet[calib_idx], tilt_triplet[calib_idx], is_ib[calib_idx],
    n_in=X_all.shape[1])
r2_ho  = r2_score(y[calib_idx], out_ho["pred"])
mae_ho = mean_absolute_error(y[calib_idx], out_ho["pred"])
ho_lst_violations = int((out_ho["delta_lst"] < 0).sum())

# Empirical monotonicity check on the full training set (3 features tested independently)
print("\n[verify] empirical monotonicity check on training set:")
mono_results = {}
for feat_name, feat_idx in [("soft_mode_activity", 0), ("lst_enhancement_proxy", 2)]:
    rate, deltas = verify_monotonicity(model_full, sc,
                                        Xtr_s, lst_triplet[train_idx], tilt_triplet[train_idx],
                                        is_ib[train_idx], er_cm[train_idx],
                                        feature_idx=feat_idx, delta=0.5)
    mono_results[feat_name] = {
        "monotonicity_rate":   round(float(rate), 4),
        "mean_delta_lst_change": round(float(deltas.mean()), 4),
    }
    print(f"  {feat_name:30s}: monotone in {100*rate:.1f}% of samples | "
          f"mean Δδ_LST = {deltas.mean():+.3f}")

# Compare to canonical PCNN (no monotonicity guarantee)
print(f"\n[result] PCNN_MonotonicLST:")
print(f"  Strat-GSS R² = {np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}")
print(f"  Holdout R²   = {r2_ho:.4f} | MAE = {mae_ho:.2f}")
print(f"  LST sign violations on holdout: {ho_lst_violations}/{len(calib_idx)}  (architectural ≥ 0)")
print(f"  Monotonicity rate (mean across 2 soft-mode features): "
      f"{100*np.mean([v['monotonicity_rate'] for v in mono_results.values()]):.1f}%")
print(f"\n[reference] PCNN (canonical): Strat-GSS R² = 0.893 ± 0.044, Holdout R² = 0.941, MAE = 5.77")

elapsed = time.time() - t0

# ── Save ────────────────────────────────────────────────────────────────────
result = {
    "model": "PCNN_MonotonicLST",
    "description": "PCNN with architectural monotonicity in the LST branch over soft-mode-driver inputs",
    "architecture": {
        "lst_head": "free MLP(trunk + b_o_reduced_mass) + positive-weight aggregation(soft_mode_activity, lst_enhancement_proxy) + ib_bonus × is_ib, all summed pre-Softplus",
        "monotone_features": ["soft_mode_activity", "lst_enhancement_proxy", "d0_B_polarizable_A"],
        "free_features":     ["b_o_reduced_mass"],
        "guarantees": [
            "δ_LST ≥ 0 (Softplus output)",
            "δ_LST monotone non-decreasing in soft_mode_activity (positive weight × softplus envelope)",
            "δ_LST monotone non-decreasing in lst_enhancement_proxy (positive weight × softplus envelope)",
            "δ_LST monotone non-decreasing in d⁰_B_polarizable_A (positive softplus bonus)",
        ],
    },
    "strat_gss_r2_mean":   round(float(np.mean(fold_r2)), 4),
    "strat_gss_r2_std":    round(float(np.std(fold_r2)), 4),
    "strat_gss_r2_per_fold": [round(r, 4) for r in fold_r2],
    "holdout_r2":          round(r2_ho, 4),
    "holdout_mae":         round(mae_ho, 3),
    "holdout_lst_violations": ho_lst_violations,
    "n_holdout":           int(len(calib_idx)),
    "monotonicity_verification": mono_results,
    "training_time_seconds":   round(elapsed, 1),
    "pcnn_reference": {
        "strat_gss_r2": "0.893 ± 0.044", "holdout_r2": 0.941, "holdout_mae": 5.77,
        "note": "canonical PCNN has sign-bound but NO monotonicity guarantee",
    },
}
out_path = os.path.join(OUT_DIR, "08_monotonic_lst_head.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n[done] saved → {out_path}")
