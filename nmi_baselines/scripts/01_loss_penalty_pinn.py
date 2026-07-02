"""
01_loss_penalty_pinn.py — Loss-Penalty PINN baseline

Mirrors the canonical CMLTRPCNN architecture but replaces the architecture-level
sign constraints (Softplus on δ_LST, -Softplus on δ_tilt) with a SOFT loss
penalty for sign violations:

    L = RMSE(y, ŷ) + λ_sign · [ ReLU(-δ_LST).mean() + ReLU(+δ_tilt).mean() ]
                   + λ_cap   · capacity_violation²

Everything else (encoder/heads/features/CV protocol) is identical to PCNN so the
comparison is fair. The key reportable quantities:

  1. Per-fold R² + holdout R²/MAE — does loss-penalty match PCNN's accuracy?
  2. **Sign violation counts** on the holdout — does the soft penalty actually
     prevent inference-time sign violations, or does it just discourage them?

This is the critical evidence for the NMI claim:
  "Architecture-level constraints provide guarantees that loss penalties cannot."

Output: results/01_loss_penalty_pinn.json
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

# ── paths ──────────────────────────────────────────────────────────────────────
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

# Physics anchors (same as PCNN)
y       = df["epsilon_r"].values.astype(np.float32)
er_cm   = df["er_CM"].fillna(0.0).values.astype(np.float32)
has_cm  = df["has_sigma_CM"].fillna(0.0).values.astype(np.float32)

# Mechanism feature triplets used by PCNN's branch heads
lst_triplet = df[["soft_mode_activity", "b_o_reduced_mass", "lst_enhancement_proxy"]].fillna(0.0).values.astype(np.float32)
tilt_triplet = df[["tilt_severity", "charge_imbalance_proxy", "continuous_tilt_strain"]].fillna(0.0).values.astype(np.float32)
is_ib = df["d0_B_polarizable_A"].fillna(0.0).values.astype(np.float32)

groups    = df["chemistry_family"].values
train_idx = np.array(calib["train_idx"])
calib_idx = np.array(calib["calib_idx"])

print(f"[data] rows={len(df)} | features={X_all.shape[1]} | train={len(train_idx)} | holdout={len(calib_idx)}")
print(f"[data] regime-Ib fraction in train: {is_ib[train_idx].mean():.3f}")

# ── Strat-GSS CV (identical to PCNN's 57_baseline_comparison.py) ────────────
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
print(f"[cv] Strat-GSS {len(folds)} folds, sizes={[len(f[1]) for f in folds]}\n")


# ── Loss-Penalty PINN architecture ─────────────────────────────────────────
class LossPenaltyPINN(nn.Module):
    """
    Mirrors the canonical CMLTRPCNN architecture but WITHOUT the
    Softplus/-Softplus sign-bounding output activations. δ_LST and δ_tilt
    can be ANY real value at inference; sign compliance is encouraged only
    by an additive loss penalty during training.

    The δ_LST capacity bound is also removed (was architectural in PCNN) —
    replaced with a soft capacity loss for fair "all constraints as loss" test.
    """
    def __init__(self, n_in, n_lst_feats=3, n_tilt_feats=3,
                 hidden=128, n_layers=4, residual_scale=80.0):
        super().__init__()
        self.residual_scale = residual_scale

        # Shared encoder (identical to PCNN)
        layers = [nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        self.encoder = nn.Sequential(*layers)

        # LST head (standard) — NO Softplus at output (linear)
        self.lst_head = nn.Sequential(
            nn.Linear(hidden + n_lst_feats, 64), nn.SiLU(),
            nn.Linear(64, 1)                       # linear: can be ANY real value
        )

        # LST head (Regime Ib specialized) — NO Softplus
        self.lst_head_ib = nn.Sequential(
            nn.Linear(hidden + n_lst_feats, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1)                       # linear
        )

        # Tilt head — NO -Softplus (no architectural sign bound)
        self.tilt_head = nn.Sequential(
            nn.Linear(hidden + n_tilt_feats, 64), nn.SiLU(),
            nn.Linear(64, 1)                       # linear: can be ANY real value
        )

        # Residual head — bounded ±residual_scale (kept as in PCNN — neutral)
        self.res_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Tanh()
        )

    def forward(self, x, er_cm, lst_feats, tilt_feats, is_ib):
        h = self.encoder(x)

        # LST branch — blend standard vs Ib head (NO sign constraint applied)
        h_lst = torch.cat([h, lst_feats], dim=1)
        delta_lst_std = self.lst_head(h_lst).squeeze(-1)
        delta_lst_ib  = self.lst_head_ib(h_lst).squeeze(-1)
        delta_lst = (1 - is_ib) * delta_lst_std + is_ib * delta_lst_ib

        # Tilt branch — NO negation, NO Softplus
        h_tilt = torch.cat([h, tilt_feats], dim=1)
        delta_tilt = self.tilt_head(h_tilt).squeeze(-1)

        # Residual (bounded)
        delta_res = self.res_head(h).squeeze(-1) * self.residual_scale

        er_pred = er_cm + delta_lst + delta_tilt + delta_res
        return {
            "pred":       er_pred,
            "delta_lst":  delta_lst,
            "delta_tilt": delta_tilt,
            "delta_res":  delta_res,
        }


def loss_penalty_loss(out, y, lambda_sign=1.0, lambda_cap=0.324, er_cm=None,
                      lst_capacity_scalar=2.081):
    """
    Standard RMSE + soft sign-violation penalty + soft capacity violation
    penalty. All physics that PCNN enforces architecturally is here as a loss.

    lambda_sign : weight on sign-violation penalty (tune as needed)
    lambda_cap  : weight on capacity-violation penalty (matches PCNN's value)
    """
    pred = out["pred"]
    dlst = out["delta_lst"]
    dtlt = out["delta_tilt"]

    rmse = torch.sqrt(F.mse_loss(pred, y))

    # Sign-violation penalties
    sign_pen_lst  = F.relu(-dlst).mean()   # penalises δ_LST < 0
    sign_pen_tilt = F.relu( dtlt).mean()   # penalises δ_tilt > 0
    sign_pen = sign_pen_lst + sign_pen_tilt

    # Capacity-violation penalty (LST cannot exceed capacity × |er_CM|)
    if er_cm is not None:
        cap_violation = F.relu(dlst - lst_capacity_scalar * er_cm.abs())
        cap_pen = (cap_violation ** 2).mean()
    else:
        cap_pen = torch.tensor(0.0, device=pred.device)

    total = rmse + lambda_sign * sign_pen + lambda_cap * cap_pen
    return total, rmse, sign_pen, cap_pen


# ── Training & evaluation ─────────────────────────────────────────────────
def train_one(Xtr, ytr, er_cm_tr, lst_tr, tilt_tr, ib_tr,
              Xva, yva, er_cm_va, lst_va, tilt_va, ib_va,
              n_in, epochs=300, lr=7.4e-4, batch_size=256,
              lambda_sign=1.0, lambda_cap=0.324, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LossPenaltyPINN(n_in=n_in).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    ytr_t = torch.from_numpy(ytr).to(DEVICE)
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
            loss, rmse, sign_p, cap_p = loss_penalty_loss(
                out, ytr_t[b], lambda_sign=lambda_sign, lambda_cap=lambda_cap,
                er_cm=er_cm_tr_t[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

    # eval on validation fold
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(Xva).to(DEVICE),
                    torch.from_numpy(er_cm_va).to(DEVICE),
                    torch.from_numpy(lst_va).to(DEVICE),
                    torch.from_numpy(tilt_va).to(DEVICE),
                    torch.from_numpy(ib_va).to(DEVICE))
        pred = out["pred"].cpu().numpy()
        dlst = out["delta_lst"].cpu().numpy()
        dtlt = out["delta_tilt"].cpu().numpy()

    return pred, dlst, dtlt


# Standardize features
def fit_scaler(idx):
    return StandardScaler().fit(X_all[idx])

# CV loop
fold_r2 = []
fold_violations = []  # sign violation counts per fold
print("[train] running 5-fold Strat-GSS CV (Loss-Penalty PINN)…")
t0 = time.time()

for fi, (tr, va) in enumerate(folds):
    sc = fit_scaler(tr)
    Xtr_s = sc.transform(X_all[tr]).astype(np.float32)
    Xva_s = sc.transform(X_all[va]).astype(np.float32)

    pred, dlst, dtlt = train_one(
        Xtr_s, y[tr], er_cm[tr], lst_triplet[tr], tilt_triplet[tr], is_ib[tr],
        Xva_s, y[va], er_cm[va], lst_triplet[va], tilt_triplet[va], is_ib[va],
        n_in=X_all.shape[1])

    r2 = r2_score(y[va], pred)
    fold_r2.append(r2)
    n_lst_violations  = int((dlst < 0).sum())
    n_tilt_violations = int((dtlt > 0).sum())
    fold_violations.append({
        "fold": fi,
        "n_total":          int(len(va)),
        "n_lst_violations":  n_lst_violations,
        "n_tilt_violations": n_tilt_violations,
        "lst_violation_rate":  round(n_lst_violations / len(va), 4),
        "tilt_violation_rate": round(n_tilt_violations / len(va), 4),
    })
    print(f"  fold {fi}: R²={r2:.4f} | LST violations={n_lst_violations}/{len(va)} | "
          f"tilt violations={n_tilt_violations}/{len(va)}")

# Holdout pass: train on full train, test on calibration
print("\n[train] running final holdout pass…")
sc_full = fit_scaler(train_idx)
Xtr_s = sc_full.transform(X_all[train_idx]).astype(np.float32)
Xho_s = sc_full.transform(X_all[calib_idx]).astype(np.float32)

pred_ho, dlst_ho, dtlt_ho = train_one(
    Xtr_s, y[train_idx], er_cm[train_idx], lst_triplet[train_idx], tilt_triplet[train_idx], is_ib[train_idx],
    Xho_s, y[calib_idx], er_cm[calib_idx], lst_triplet[calib_idx], tilt_triplet[calib_idx], is_ib[calib_idx],
    n_in=X_all.shape[1])

r2_ho  = r2_score(y[calib_idx], pred_ho)
mae_ho = mean_absolute_error(y[calib_idx], pred_ho)
ho_lst_violations  = int((dlst_ho < 0).sum())
ho_tilt_violations = int((dtlt_ho > 0).sum())

print(f"  holdout: R²={r2_ho:.4f} | MAE={mae_ho:.2f}")
print(f"  holdout violations: LST={ho_lst_violations}/{len(calib_idx)} | "
      f"tilt={ho_tilt_violations}/{len(calib_idx)}")

elapsed = time.time() - t0

# ── Save ────────────────────────────────────────────────────────────────────
result = {
    "model": "LossPenaltyPINN",
    "description": "Same architecture as PCNN but sign constraints as soft loss penalty (no Softplus / -Softplus output activations)",
    "hyperparameters": {
        "hidden":      128,
        "n_layers":    4,
        "lr":          7.4e-4,
        "batch_size":  256,
        "epochs":      300,
        "lambda_sign": 1.0,
        "lambda_cap":  0.324,
        "residual_scale": 80.0,
    },
    "strat_gss_r2_mean":   round(float(np.mean(fold_r2)), 4),
    "strat_gss_r2_std":    round(float(np.std(fold_r2)), 4),
    "strat_gss_r2_per_fold": [round(r, 4) for r in fold_r2],
    "holdout_r2":          round(r2_ho, 4),
    "holdout_mae":         round(mae_ho, 3),
    "sign_violations_holdout": {
        "n_total":            int(len(calib_idx)),
        "n_lst_violations":   ho_lst_violations,
        "n_tilt_violations":  ho_tilt_violations,
        "lst_violation_rate":  round(ho_lst_violations  / len(calib_idx), 4),
        "tilt_violation_rate": round(ho_tilt_violations / len(calib_idx), 4),
    },
    "sign_violations_per_fold": fold_violations,
    "training_time_seconds":   round(elapsed, 1),
    "comparison_target_pcnn": {
        "strat_gss_r2": "0.893 ± 0.044",
        "holdout_r2":   0.941,
        "holdout_mae":  5.77,
        "lst_violation_rate":  0.0,
        "tilt_violation_rate": 0.0,
        "note": "PCNN enforces sign constraints architecturally → 0% violations guaranteed",
    },
}

out_path = os.path.join(OUT_DIR, "01_loss_penalty_pinn.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n=== RESULT ({elapsed:.1f}s) ===")
print(f"Loss-Penalty PINN:  Strat-GSS R² = {np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}  "
      f"| Holdout R² = {r2_ho:.3f} | MAE = {mae_ho:.2f}")
print(f"PCNN (architecture): Strat-GSS R² = 0.893 ± 0.044  | Holdout R² = 0.941 | MAE = 5.77")
print()
print(f"Sign violations on holdout (Loss-Penalty PINN):")
print(f"  LST  (δ_LST < 0):  {ho_lst_violations}/{len(calib_idx)} ({100*ho_lst_violations/len(calib_idx):.1f}%)")
print(f"  tilt (δ_tilt > 0): {ho_tilt_violations}/{len(calib_idx)} ({100*ho_tilt_violations/len(calib_idx):.1f}%)")
print(f"  PCNN (architecture):  0/{len(calib_idx)} ({0.0:.1f}%) — guaranteed by construction")
print(f"\n[done] saved → {out_path}")
