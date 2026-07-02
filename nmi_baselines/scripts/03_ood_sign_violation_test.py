"""
03_ood_sign_violation_test.py — controlled distribution-shift test

The λ_sign sweep (script 02) showed loss-penalty PINN matches PCNN's 0% violation
rate on the in-distribution holdout at λ ≥ 10. This script asks the harder
question that GA inverse design actually depends on: when inputs drift FAR from
the training distribution (as happens during compositional extrapolation),
which models stay sign-compliant?

DESIGN — clean controlled comparison

Train TWO models on identical data with identical hyperparameters:
  A) "PCNN-equivalent"     — Softplus on δ_LST, -Softplus on δ_tilt  (architectural sign bound)
  B) "LossPenaltyPINN"     — linear output, λ_sign = 10 (well-tuned soft penalty)

Then evaluate both on the 116 holdout samples under progressive feature-space
distribution shift. We perturb the LST-related features (the ones the LST
branch consumes most directly) at multipliers {1.0, 1.5, 2.0, 3.0, 5.0}×.

The 1.0× case is the in-distribution baseline (should match script 02). Higher
multipliers simulate progressively more OOD inputs — analogous to what the GA
encounters when proposing novel compositions far from the training set.

The reportable comparison:
  • Sign violation rates at each OOD level for each model
  • Does the architectural model stay 0% across all OOD levels (it MUST — by
    construction)? Does the loss-penalty model stay 0% only on in-distribution
    inputs and fail at higher OOD levels?

Output: results/03_ood_sign_violation_test.json
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

Xl, lst_cols  = get(partition["LST"])
Xt, tilt_cols = get(partition["Tilt"])
Xr, res_cols  = get(partition["Residual"])
X_all = np.concatenate([Xl, Xt, Xr], axis=1).astype(np.float32)

# Index ranges in the concatenated feature vector for LST / Tilt blocks (so we
# can perturb them per-block for the OOD test)
N_LST   = Xl.shape[1]
N_TILT  = Xt.shape[1]
N_RES   = Xr.shape[1]
IDX_LST_END  = N_LST
IDX_TILT_END = N_LST + N_TILT
print(f"[data] feature block sizes: LST={N_LST}, Tilt={N_TILT}, Residual={N_RES} (total={N_LST+N_TILT+N_RES})")

y       = df["epsilon_r"].values.astype(np.float32)
er_cm   = df["er_CM"].fillna(0.0).values.astype(np.float32)
lst_triplet  = df[["soft_mode_activity", "b_o_reduced_mass", "lst_enhancement_proxy"]].fillna(0.0).values.astype(np.float32)
tilt_triplet = df[["tilt_severity", "charge_imbalance_proxy", "continuous_tilt_strain"]].fillna(0.0).values.astype(np.float32)
is_ib   = df["d0_B_polarizable_A"].fillna(0.0).values.astype(np.float32)

train_idx = np.array(calib["train_idx"])
calib_idx = np.array(calib["calib_idx"])


# ── Generic architecture with toggleable sign constraint ─────────────────
class PINN_Configurable(nn.Module):
    """
    One architecture, two modes:
      mode='architectural'  → Softplus on δ_LST, -Softplus on δ_tilt (hard sign bound)
      mode='loss_penalty'   → linear output, sign penalty added at loss
    Everything else identical so the comparison is clean.
    """
    def __init__(self, n_in, mode, n_lst_feats=3, n_tilt_feats=3,
                 hidden=128, n_layers=4, residual_scale=80.0):
        super().__init__()
        assert mode in ("architectural", "loss_penalty")
        self.mode = mode
        self.residual_scale = residual_scale
        layers = [nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        self.encoder = nn.Sequential(*layers)
        # output activation chosen per-mode
        out_act_lst = nn.Softplus() if mode == "architectural" else nn.Identity()
        out_act_tilt = nn.Softplus() if mode == "architectural" else nn.Identity()
        self.lst_head    = nn.Sequential(nn.Linear(hidden + n_lst_feats, 64), nn.SiLU(), nn.Linear(64, 1), out_act_lst)
        self.lst_head_ib = nn.Sequential(nn.Linear(hidden + n_lst_feats, 128), nn.SiLU(),
                                          nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1), out_act_lst)
        self.tilt_head   = nn.Sequential(nn.Linear(hidden + n_tilt_feats, 64), nn.SiLU(), nn.Linear(64, 1), out_act_tilt)
        self.res_head    = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, x, er_cm, lst_feats, tilt_feats, is_ib):
        h = self.encoder(x)
        h_lst = torch.cat([h, lst_feats], dim=1)
        delta_lst_std = self.lst_head(h_lst).squeeze(-1)
        delta_lst_ib  = self.lst_head_ib(h_lst).squeeze(-1)
        delta_lst = (1 - is_ib) * delta_lst_std + is_ib * delta_lst_ib
        h_tilt = torch.cat([h, tilt_feats], dim=1)
        delta_tilt = self.tilt_head(h_tilt).squeeze(-1)
        if self.mode == "architectural":
            delta_tilt = -delta_tilt          # negate to enforce ≤ 0 (PCNN convention)
        delta_res = self.res_head(h).squeeze(-1) * self.residual_scale
        return {"pred": er_cm + delta_lst + delta_tilt + delta_res,
                "delta_lst": delta_lst, "delta_tilt": delta_tilt, "delta_res": delta_res}


def standard_loss(out, y):
    return torch.sqrt(F.mse_loss(out["pred"], y))


def loss_penalty_loss(out, y, lambda_sign, lambda_cap=0.324, er_cm=None,
                      lst_capacity_scalar=2.081):
    rmse = torch.sqrt(F.mse_loss(out["pred"], y))
    sign_pen = F.relu(-out["delta_lst"]).mean() + F.relu(out["delta_tilt"]).mean()
    if er_cm is not None:
        cap_pen = (F.relu(out["delta_lst"] - lst_capacity_scalar * er_cm.abs()) ** 2).mean()
    else:
        cap_pen = torch.tensor(0.0, device=out["pred"].device)
    return rmse + lambda_sign * sign_pen + lambda_cap * cap_pen


def train_model(mode, lambda_sign, Xtr, ytr, er_cm_tr, lst_tr, tilt_tr, ib_tr,
                n_in, epochs=300, lr=7.4e-4, batch_size=256, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PINN_Configurable(n_in=n_in, mode=mode).to(DEVICE)
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
            if mode == "architectural":
                loss = standard_loss(out, ytr_t[b])
            else:
                loss = loss_penalty_loss(out, ytr_t[b], lambda_sign=lambda_sign,
                                          er_cm=er_cm_t[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
    return model


def evaluate(model, X, er_cm_va, lst_va, tilt_va, ib_va):
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X.astype(np.float32)).to(DEVICE),
                    torch.from_numpy(er_cm_va).to(DEVICE),
                    torch.from_numpy(lst_va).to(DEVICE),
                    torch.from_numpy(tilt_va).to(DEVICE),
                    torch.from_numpy(ib_va).to(DEVICE))
    return {k: v.cpu().numpy() for k, v in out.items()}


# ── Train both models on identical data ─────────────────────────────────────
print("\n[train] standardize features (fit on training set only)…")
sc = StandardScaler().fit(X_all[train_idx])
Xtr_s = sc.transform(X_all[train_idx]).astype(np.float32)
Xho_s = sc.transform(X_all[calib_idx]).astype(np.float32)

print("[train] model A: PCNN-equivalent (architectural sign bound)…")
t0 = time.time()
model_arch = train_model("architectural", lambda_sign=0.0,
    Xtr=Xtr_s, ytr=y[train_idx],
    er_cm_tr=er_cm[train_idx], lst_tr=lst_triplet[train_idx],
    tilt_tr=tilt_triplet[train_idx], ib_tr=is_ib[train_idx],
    n_in=X_all.shape[1])
print(f"  ({time.time()-t0:.1f}s)")

print("[train] model B: LossPenaltyPINN at λ_sign = 10…")
t0 = time.time()
model_lp = train_model("loss_penalty", lambda_sign=10.0,
    Xtr=Xtr_s, ytr=y[train_idx],
    er_cm_tr=er_cm[train_idx], lst_tr=lst_triplet[train_idx],
    tilt_tr=tilt_triplet[train_idx], ib_tr=is_ib[train_idx],
    n_in=X_all.shape[1])
print(f"  ({time.time()-t0:.1f}s)")


# ── OOD perturbation evaluation ────────────────────────────────────────────
SCALES = [1.0, 1.5, 2.0, 3.0, 5.0]
results_arch = []
results_lp   = []

print("\n[eval] running OOD perturbation test on holdout…")
print(f"{'scale':<6} {'model':<14} {'R²':<8} {'MAE':<7} {'LST_viol':<10} {'tilt_viol':<10}")
print("-" * 60)

for scale in SCALES:
    # Build perturbed holdout features:
    #   scale LST-block columns by `scale`; leave Tilt + Residual blocks intact
    Xho_perturbed = Xho_s.copy()
    Xho_perturbed[:, :IDX_LST_END] = Xho_perturbed[:, :IDX_LST_END] * scale

    # Architectural model
    out_a = evaluate(model_arch, Xho_perturbed,
                     er_cm[calib_idx], lst_triplet[calib_idx],
                     tilt_triplet[calib_idx], is_ib[calib_idx])
    r2_a  = r2_score(y[calib_idx], out_a["pred"]) if scale == 1.0 else None
    mae_a = mean_absolute_error(y[calib_idx], out_a["pred"]) if scale == 1.0 else None
    lst_v_a  = int((out_a["delta_lst"]  < 0).sum())
    tilt_v_a = int((out_a["delta_tilt"] > 0).sum())
    results_arch.append({"scale": scale, "lst_violations": lst_v_a,
                          "tilt_violations": tilt_v_a, "r2_at_1x": r2_a,
                          "mae_at_1x": mae_a,
                          "n": int(len(calib_idx))})

    # Loss-penalty model
    out_b = evaluate(model_lp, Xho_perturbed,
                     er_cm[calib_idx], lst_triplet[calib_idx],
                     tilt_triplet[calib_idx], is_ib[calib_idx])
    r2_b  = r2_score(y[calib_idx], out_b["pred"]) if scale == 1.0 else None
    mae_b = mean_absolute_error(y[calib_idx], out_b["pred"]) if scale == 1.0 else None
    lst_v_b  = int((out_b["delta_lst"]  < 0).sum())
    tilt_v_b = int((out_b["delta_tilt"] > 0).sum())
    results_lp.append({"scale": scale, "lst_violations": lst_v_b,
                        "tilt_violations": tilt_v_b, "r2_at_1x": r2_b,
                        "mae_at_1x": mae_b,
                        "n": int(len(calib_idx))})

    print(f"{scale:<6} {'architectural':<14} {r2_a if r2_a is not None else '-':<8} "
          f"{mae_a if mae_a is not None else '-':<7} "
          f"{lst_v_a:>3}/{len(calib_idx)}    {tilt_v_a:>3}/{len(calib_idx)}")
    print(f"{scale:<6} {'loss_penalty':<14} {r2_b if r2_b is not None else '-':<8} "
          f"{mae_b if mae_b is not None else '-':<7} "
          f"{lst_v_b:>3}/{len(calib_idx)}    {tilt_v_b:>3}/{len(calib_idx)}")

# Save
result = {
    "experiment": "OOD distribution-shift test via LST feature scaling",
    "design": "Identical architecture trained two ways: (A) Softplus/-Softplus sign bound; (B) linear output + λ_sign=10 soft penalty. Holdout features perturbed by scaling LST-block by {1.0, 1.5, 2.0, 3.0, 5.0}×. Counts sign violations per OOD level.",
    "n_holdout": int(len(calib_idx)),
    "n_lst_feats_scaled": N_LST,
    "scales": SCALES,
    "architectural":  results_arch,
    "loss_penalty":   results_lp,
    "lambda_sign_loss_penalty": 10.0,
}
out_path = os.path.join(OUT_DIR, "03_ood_sign_violation_test.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n[done] saved → {out_path}")

# Punchline summary
print("\n" + "=" * 70)
print("PUNCHLINE — sign violation rate vs OOD perturbation magnitude:")
print(f"{'scale':<6} {'architectural':<24} {'loss_penalty (λ=10)':<24}")
print(f"{'':6} {'LST viol % | tilt viol %':<24} {'LST viol % | tilt viol %':<24}")
for ra, rb in zip(results_arch, results_lp):
    s = ra["scale"]; n = ra["n"]
    print(f"{s:<6.1f} "
          f"{100*ra['lst_violations']/n:>5.1f}%      | {100*ra['tilt_violations']/n:>5.1f}%          "
          f"{100*rb['lst_violations']/n:>5.1f}%      | {100*rb['tilt_violations']/n:>5.1f}%")
