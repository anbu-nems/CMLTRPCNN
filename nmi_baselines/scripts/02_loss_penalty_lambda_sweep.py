"""
02_loss_penalty_lambda_sweep.py — λ_sign sensitivity sweep

Defensive sweep of the loss-penalty PINN at four sign-penalty weights:
    λ_sign ∈ {0.1, 1.0, 10.0, 100.0}

The single-shot run in 01_loss_penalty_pinn.py showed 3.4% LST sign violations
on the holdout at λ_sign = 1.0. The NMI argument ("architecture-level constraints
provide guarantees that loss penalties cannot") is much stronger if we can show
that even at very high penalty weights, the loss-penalty PINN continues to
produce sign-violating predictions on out-of-distribution data.

Hypothesis to test:
  • Increasing λ_sign reduces violation RATE but never reaches 0 on OOD samples
  • Increasing λ_sign degrades RMSE accuracy (the soft constraint distorts fit
    in directions the architecture would have absorbed structurally)
  • PCNN's architecture-level constraint dominates the Pareto frontier
    (accuracy at zero violations) at any λ_sign

For each λ, runs the full 5-fold Strat-GSS CV + holdout pass, mirroring
01_loss_penalty_pinn.py exactly (same model class, same data, same protocol).

Output: results/02_loss_penalty_lambda_sweep.json
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

# Reuse the same classes/functions from 01
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
mod = import_module("01_loss_penalty_pinn".replace("01_", "_01_") if False else "_lpp_helpers", package=None) if False else None

# ── Inline copies (to keep this script self-contained; matches script 01 exactly) ──
ROOT     = "../.."  # repo root (run from nmi_baselines/scripts/)
PROC_DIR = os.path.join(ROOT, "data", "processed")
OUT_DIR  = os.path.join(ROOT, "nmi_baselines", "results")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                     else "cuda" if torch.cuda.is_available()
                     else "cpu")

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


# ── Strat-GSS CV (identical to 01) ──────────────────────────────────────────
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


# ── Loss-Penalty PINN architecture (identical to 01) ────────────────────────
class LossPenaltyPINN(nn.Module):
    def __init__(self, n_in, n_lst_feats=3, n_tilt_feats=3,
                 hidden=128, n_layers=4, residual_scale=80.0):
        super().__init__()
        self.residual_scale = residual_scale
        layers = [nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        self.encoder = nn.Sequential(*layers)
        self.lst_head    = nn.Sequential(nn.Linear(hidden + n_lst_feats, 64), nn.SiLU(), nn.Linear(64, 1))
        self.lst_head_ib = nn.Sequential(nn.Linear(hidden + n_lst_feats, 128), nn.SiLU(),
                                          nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.tilt_head   = nn.Sequential(nn.Linear(hidden + n_tilt_feats, 64), nn.SiLU(), nn.Linear(64, 1))
        self.res_head    = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, x, er_cm, lst_feats, tilt_feats, is_ib):
        h = self.encoder(x)
        h_lst = torch.cat([h, lst_feats], dim=1)
        delta_lst_std = self.lst_head(h_lst).squeeze(-1)
        delta_lst_ib  = self.lst_head_ib(h_lst).squeeze(-1)
        delta_lst = (1 - is_ib) * delta_lst_std + is_ib * delta_lst_ib
        h_tilt = torch.cat([h, tilt_feats], dim=1)
        delta_tilt = self.tilt_head(h_tilt).squeeze(-1)
        delta_res = self.res_head(h).squeeze(-1) * self.residual_scale
        return {"pred": er_cm + delta_lst + delta_tilt + delta_res,
                "delta_lst": delta_lst, "delta_tilt": delta_tilt, "delta_res": delta_res}


def loss_penalty_loss(out, y, lambda_sign, lambda_cap=0.324, er_cm=None,
                      lst_capacity_scalar=2.081):
    pred, dlst, dtlt = out["pred"], out["delta_lst"], out["delta_tilt"]
    rmse = torch.sqrt(F.mse_loss(pred, y))
    sign_pen = F.relu(-dlst).mean() + F.relu(dtlt).mean()
    if er_cm is not None:
        cap_pen = (F.relu(dlst - lst_capacity_scalar * er_cm.abs()) ** 2).mean()
    else:
        cap_pen = torch.tensor(0.0, device=pred.device)
    return rmse + lambda_sign * sign_pen + lambda_cap * cap_pen, rmse, sign_pen, cap_pen


def train_one(Xtr, ytr, er_cm_tr, lst_tr, tilt_tr, ib_tr,
              Xva, yva, er_cm_va, lst_va, tilt_va, ib_va,
              n_in, lambda_sign, epochs=300, lr=7.4e-4, batch_size=256, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LossPenaltyPINN(n_in=n_in).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xtr_t   = torch.from_numpy(Xtr).to(DEVICE)
    ytr_t   = torch.from_numpy(ytr).to(DEVICE)
    er_cm_tr_t = torch.from_numpy(er_cm_tr).to(DEVICE)
    lst_tr_t   = torch.from_numpy(lst_tr).to(DEVICE)
    tilt_tr_t  = torch.from_numpy(tilt_tr).to(DEVICE)
    ib_tr_t    = torch.from_numpy(ib_tr).to(DEVICE)

    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            b = perm[i:i+batch_size]
            out = model(Xtr_t[b], er_cm_tr_t[b], lst_tr_t[b], tilt_tr_t[b], ib_tr_t[b])
            loss, *_ = loss_penalty_loss(out, ytr_t[b], lambda_sign=lambda_sign, er_cm=er_cm_tr_t[b])
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
    return out["pred"].cpu().numpy(), out["delta_lst"].cpu().numpy(), out["delta_tilt"].cpu().numpy()


# ── λ sweep ─────────────────────────────────────────────────────────────────
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
results = []

print(f"[init] device: {DEVICE}")
print(f"[init] sweeping λ_sign ∈ {LAMBDAS} | 5-fold Strat-GSS + holdout each\n")

t_all = time.time()
for lam in LAMBDAS:
    t0 = time.time()
    print(f"== λ_sign = {lam} ==")
    fold_r2, fold_lst_v, fold_tilt_v = [], [], []
    for fi, (tr, va) in enumerate(folds):
        sc = StandardScaler().fit(X_all[tr])
        Xtr_s = sc.transform(X_all[tr]).astype(np.float32)
        Xva_s = sc.transform(X_all[va]).astype(np.float32)
        pred, dlst, dtlt = train_one(
            Xtr_s, y[tr], er_cm[tr], lst_triplet[tr], tilt_triplet[tr], is_ib[tr],
            Xva_s, y[va], er_cm[va], lst_triplet[va], tilt_triplet[va], is_ib[va],
            n_in=X_all.shape[1], lambda_sign=lam)
        r2 = r2_score(y[va], pred)
        fold_r2.append(r2)
        fold_lst_v.append(int((dlst < 0).sum()))
        fold_tilt_v.append(int((dtlt > 0).sum()))
        print(f"  fold {fi}: R²={r2:.3f} | LST_viol={fold_lst_v[-1]}/{len(va)} | tilt_viol={fold_tilt_v[-1]}/{len(va)}")

    # Holdout
    sc = StandardScaler().fit(X_all[train_idx])
    Xtr_s = sc.transform(X_all[train_idx]).astype(np.float32)
    Xho_s = sc.transform(X_all[calib_idx]).astype(np.float32)
    pred_ho, dlst_ho, dtlt_ho = train_one(
        Xtr_s, y[train_idx], er_cm[train_idx], lst_triplet[train_idx], tilt_triplet[train_idx], is_ib[train_idx],
        Xho_s, y[calib_idx], er_cm[calib_idx], lst_triplet[calib_idx], tilt_triplet[calib_idx], is_ib[calib_idx],
        n_in=X_all.shape[1], lambda_sign=lam)
    r2_ho  = r2_score(y[calib_idx], pred_ho)
    mae_ho = mean_absolute_error(y[calib_idx], pred_ho)
    ho_lst_v  = int((dlst_ho < 0).sum())
    ho_tilt_v = int((dtlt_ho > 0).sum())
    print(f"  HOLDOUT: R²={r2_ho:.3f} MAE={mae_ho:.2f} | LST_viol={ho_lst_v}/{len(calib_idx)} | tilt_viol={ho_tilt_v}/{len(calib_idx)}")

    elapsed = time.time() - t0
    print(f"  ({elapsed:.1f}s)\n")

    results.append({
        "lambda_sign":              lam,
        "strat_gss_r2_mean":        round(float(np.mean(fold_r2)), 4),
        "strat_gss_r2_std":         round(float(np.std(fold_r2)), 4),
        "strat_gss_r2_per_fold":    [round(r, 4) for r in fold_r2],
        "holdout_r2":               round(r2_ho, 4),
        "holdout_mae":              round(mae_ho, 3),
        "ho_lst_violations":        ho_lst_v,
        "ho_tilt_violations":       ho_tilt_v,
        "ho_lst_violation_rate":    round(ho_lst_v  / len(calib_idx), 4),
        "ho_tilt_violation_rate":   round(ho_tilt_v / len(calib_idx), 4),
        "fold_lst_violations":      fold_lst_v,
        "fold_tilt_violations":     fold_tilt_v,
        "training_time_seconds":    round(elapsed, 1),
    })

elapsed_all = time.time() - t_all
print(f"\n=== λ sweep complete ({elapsed_all:.1f}s) ===")
print(f"{'λ_sign':<8} {'CV R²':<18} {'Holdout R²':<11} {'LST_viol%':<10} {'tilt_viol%':<10}")
for r in results:
    cvr = f"{r['strat_gss_r2_mean']:.3f} ± {r['strat_gss_r2_std']:.3f}"
    print(f"{r['lambda_sign']:<8} {cvr:<18} {r['holdout_r2']:<11.3f} "
          f"{100*r['ho_lst_violation_rate']:<10.1f} {100*r['ho_tilt_violation_rate']:<10.1f}")
print(f"\nPCNN (architecture):                                  {0.0:<10.1f} {0.0:<10.1f}  ← guaranteed")

out = {
    "model": "LossPenaltyPINN_lambda_sweep",
    "description": "λ_sign sensitivity sweep — robustness of the 'architecture > loss-penalty' finding",
    "lambdas":   LAMBDAS,
    "results":   results,
    "pcnn_reference": {
        "strat_gss_r2": "0.893 ± 0.044",
        "holdout_r2":   0.941,
        "holdout_mae":  5.77,
        "ho_lst_violation_rate":  0.0,
        "ho_tilt_violation_rate": 0.0,
        "note": "architectural sign bounds (Softplus / -Softplus) → 0% violations by construction",
    },
}
out_path = os.path.join(OUT_DIR, "02_loss_penalty_lambda_sweep.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[done] saved → {out_path}")
