"""
04_ml_literature_baselines.py — three ML-literature additive / interpretable
baselines as comparators for PCNN on identical CV protocol.

Baselines added:
  1. GAM  (Generalized Additive Model, pyGAM)     — additive splines per feature
  2. EBM  (Explainable Boosting Machine, interpret) — cyclic-gradient-boosted shape functions
  3. PMN  (Partial-Monotone Neural Net)            — MLP with positive-weight output heads
                                                       enforcing monotonicity per sign-relevant feature

All use the same Strat-GSS 5-fold + held-out protocol as 01-03 and the existing
57_baseline_comparison.py. Targets the same 103 features, same train/holdout
split, same standard scaler.

This is the "ML-literature alternative" baseline batch for the NMI submission.
Existing materials-ML baselines (Ridge, RF, XGBoost, CatBoost, BPNN, PIRNN)
are already covered by 57_baseline_comparison.py and will be referenced.

Output: results/04_ml_literature_baselines.json
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
N_TILT = Xt.shape[1]
N_RES  = Xr.shape[1]

y       = df["epsilon_r"].values.astype(np.float32)
groups  = df["chemistry_family"].values
train_idx = np.array(calib["train_idx"])
calib_idx = np.array(calib["calib_idx"])

# ── Strat-GSS CV ───────────────────────────────────────────────────────────
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
print(f"[init] device: {DEVICE} | {N_LST+N_TILT+N_RES} features | {len(folds)} CV folds")


def eval_sklearn(model_fn, X, y, folds, calib_idx, train_idx, scale=True):
    """Standard fold-eval loop matching the pattern in 57_baseline_comparison.py."""
    fold_r2 = []
    for tr, va in folds:
        if scale:
            sc = StandardScaler().fit(X[tr])
            Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        else:
            Xtr, Xva = X[tr], X[va]
        m = model_fn()
        m.fit(Xtr, y[tr])
        fold_r2.append(r2_score(y[va], m.predict(Xva)))
    # holdout
    if scale:
        sc = StandardScaler().fit(X[train_idx])
        Xtr_f = sc.transform(X[train_idx]); Xca_f = sc.transform(X[calib_idx])
    else:
        Xtr_f = X[train_idx]; Xca_f = X[calib_idx]
    mf = model_fn(); mf.fit(Xtr_f, y[train_idx])
    p_ho = mf.predict(Xca_f)
    return {
        "strat_gss_r2_mean": round(float(np.mean(fold_r2)), 4),
        "strat_gss_r2_std":  round(float(np.std(fold_r2)),  4),
        "strat_gss_r2_per_fold": [round(r, 4) for r in fold_r2],
        "holdout_r2":  round(r2_score(y[calib_idx], p_ho),     4),
        "holdout_mae": round(mean_absolute_error(y[calib_idx], p_ho), 3),
    }


# ── 1. GAM (Generalized Additive Model, pyGAM) ─────────────────────────────
print("\n[1/3] GAM (pyGAM)…")
from pygam import LinearGAM, s
t0 = time.time()

def make_gam():
    # one spline per feature, default n_splines=20, lambda regularisation auto
    terms = s(0)
    for i in range(1, X_all.shape[1]):
        terms = terms + s(i)
    return LinearGAM(terms, max_iter=200, tol=1e-4)

gam_res = eval_sklearn(make_gam, X_all, y, folds, calib_idx, train_idx, scale=True)
gam_res["training_time_s"] = round(time.time() - t0, 1)
print(f"   GAM: CV R²={gam_res['strat_gss_r2_mean']:.3f} ± {gam_res['strat_gss_r2_std']:.3f} | "
      f"Holdout R²={gam_res['holdout_r2']:.3f} MAE={gam_res['holdout_mae']:.2f}  "
      f"({gam_res['training_time_s']}s)")


# ── 2. EBM (Explainable Boosting Machine, interpret-ml) ────────────────────
print("\n[2/3] EBM (interpret-ml)…")
from interpret.glassbox import ExplainableBoostingRegressor
t0 = time.time()

def make_ebm():
    return ExplainableBoostingRegressor(random_state=42, max_bins=256, learning_rate=0.01)

ebm_res = eval_sklearn(make_ebm, X_all, y, folds, calib_idx, train_idx, scale=False)
ebm_res["training_time_s"] = round(time.time() - t0, 1)
print(f"   EBM: CV R²={ebm_res['strat_gss_r2_mean']:.3f} ± {ebm_res['strat_gss_r2_std']:.3f} | "
      f"Holdout R²={ebm_res['holdout_r2']:.3f} MAE={ebm_res['holdout_mae']:.2f}  "
      f"({ebm_res['training_time_s']}s)")


# ── 3. Partial-Monotone NN (custom) ──────────────────────────────────────
class PartialMonotoneNN(nn.Module):
    """
    Shared MLP encoder + TWO output heads with sign-enforced weights:
      LST head : output = sum(softplus(W_lst) · x_lst)   → always ≥ 0
      Tilt head: output = -sum(softplus(W_tilt) · x_tilt) → always ≤ 0
      Free head: standard linear on Residual features    → unconstrained

    Final prediction = LST_term + Tilt_term + Free_term.
    Mirrors the "monotonicity by positive-weight construction" classic from the
    ML literature (see e.g. Daniels & Velikova 2010, "Monotone and Partially
    Monotone Neural Networks").
    """
    def __init__(self, n_lst, n_tilt, n_res, hidden=128):
        super().__init__()
        # Shared encoder for residual features (free)
        self.res_encoder = nn.Sequential(
            nn.Linear(n_res, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(hidden, 1))
        # Monotone weights for LST and tilt (parameterised via softplus to ensure ≥ 0)
        self.W_lst  = nn.Parameter(torch.randn(n_lst)  * 0.1)
        self.W_tilt = nn.Parameter(torch.randn(n_tilt) * 0.1)
        self.b_lst  = nn.Parameter(torch.zeros(1))
        self.b_tilt = nn.Parameter(torch.zeros(1))
        self.b_res  = nn.Parameter(torch.zeros(1))
        self.n_lst = n_lst; self.n_tilt = n_tilt; self.n_res = n_res

    def forward(self, x):
        x_lst  = x[:, :self.n_lst]
        x_tilt = x[:, self.n_lst:self.n_lst + self.n_tilt]
        x_res  = x[:, self.n_lst + self.n_tilt:]
        # Monotone heads
        w_lst  = F.softplus(self.W_lst)
        w_tilt = F.softplus(self.W_tilt)
        out_lst  =  (x_lst  * w_lst ).sum(dim=1) + self.b_lst.squeeze()
        out_tilt = -(x_tilt * w_tilt).sum(dim=1) + self.b_tilt.squeeze()
        out_res  = self.res_encoder(x_res).squeeze(-1) + self.b_res.squeeze()
        return out_lst + out_tilt + out_res

print("\n[3/3] Partial-Monotone NN…")
t0 = time.time()

def train_pmn(Xtr, ytr, Xva, yva, epochs=300, lr=1e-3, batch_size=256, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PartialMonotoneNN(N_LST, N_TILT, N_RES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xtr_t = torch.from_numpy(Xtr.astype(np.float32)).to(DEVICE)
    ytr_t = torch.from_numpy(ytr).to(DEVICE)
    n = len(Xtr)
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
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xva.astype(np.float32)).to(DEVICE)).cpu().numpy()
    return pred

fold_r2 = []
for tr, va in folds:
    sc = StandardScaler().fit(X_all[tr])
    Xtr_s = sc.transform(X_all[tr]).astype(np.float32)
    Xva_s = sc.transform(X_all[va]).astype(np.float32)
    pred = train_pmn(Xtr_s, y[tr], Xva_s, y[va])
    fold_r2.append(r2_score(y[va], pred))
sc = StandardScaler().fit(X_all[train_idx])
Xtr_s = sc.transform(X_all[train_idx]).astype(np.float32)
Xho_s = sc.transform(X_all[calib_idx]).astype(np.float32)
p_ho = train_pmn(Xtr_s, y[train_idx], Xho_s, y[calib_idx])
pmn_res = {
    "strat_gss_r2_mean": round(float(np.mean(fold_r2)), 4),
    "strat_gss_r2_std":  round(float(np.std(fold_r2)), 4),
    "strat_gss_r2_per_fold": [round(r, 4) for r in fold_r2],
    "holdout_r2":  round(r2_score(y[calib_idx], p_ho), 4),
    "holdout_mae": round(mean_absolute_error(y[calib_idx], p_ho), 3),
    "training_time_s": round(time.time() - t0, 1),
}
print(f"   PMN: CV R²={pmn_res['strat_gss_r2_mean']:.3f} ± {pmn_res['strat_gss_r2_std']:.3f} | "
      f"Holdout R²={pmn_res['holdout_r2']:.3f} MAE={pmn_res['holdout_mae']:.2f}  "
      f"({pmn_res['training_time_s']}s)")


# ── Save combined result ─────────────────────────────────────────────────
results = {
    "GAM":  {**gam_res, "description": "Generalized Additive Model (pyGAM): one cubic spline per feature, additive"},
    "EBM":  {**ebm_res, "description": "Explainable Boosting Machine (interpret-ml): cyclic gradient-boosted shape functions, additive"},
    "PMN":  {**pmn_res, "description": "Partial-Monotone NN: positive-weight LST + tilt heads + free residual MLP head; monotonicity by construction"},
    "PCNN_reference": {
        "strat_gss_r2_mean": 0.893, "strat_gss_r2_std": 0.044,
        "holdout_r2":        0.941, "holdout_mae":      5.77,
        "description":       "Architecture-level physics-constrained NN with 4-branch decomposition (paper canonical)",
    },
}
out_path = os.path.join(OUT_DIR, "04_ml_literature_baselines.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

# ── Print summary table ──────────────────────────────────────────────────
print("\n" + "=" * 78)
print(f"{'Model':<14} {'CV R²':<18} {'Holdout R²':<11} {'Holdout MAE':<12} {'Time (s)':<10}")
print("-" * 78)
for name in ["GAM", "EBM", "PMN", "PCNN_reference"]:
    r = results[name]
    cvr = f"{r['strat_gss_r2_mean']:.3f} ± {r['strat_gss_r2_std']:.3f}"
    t = r.get('training_time_s', '-')
    print(f"{name:<14} {cvr:<18} {r['holdout_r2']:<11.3f} {r['holdout_mae']:<12.2f} {t}")
print("=" * 78)
print(f"\n[done] saved → {out_path}")
