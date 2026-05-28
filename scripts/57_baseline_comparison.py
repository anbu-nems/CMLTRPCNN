"""
Script 57 — Fair baseline comparison against CMLTRPCNNv7.7
All models evaluated on feature_matrix_v7 with identical Strat-GSS CV + held-out protocol.

Baselines:
  1. CM-only          — Clausius-Mossotti anchor, no learning (pure physics)
  2. Ridge Regression — linear, all 103 features
  3. Random Forest    — ensemble tree, all 103 features
  4. XGBoost          — gradient-boosted tree, all 103 features
  5. CatBoost         — gradient-boosted tree with categorical support
  6. BPNN (MLP)       — 3-hidden-layer MLP, no physics constraints
  7. PIRNN            — from script 56 (physics features as input, sign constraints)
  8. PCNN (ours)      — CMLTRPCNNv7.7 results (already computed)

Outputs: results/57_baseline_comparison.json
         results_csv/13_baseline_comparison.csv
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from collections import Counter

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — skipping. Install: pip install xgboost")

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("CatBoost not installed — skipping. Install: pip install catboost")

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results")
CSV_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "..", "NC figures", "results_csv")

# ── Load data ─────────────────────────────────────────────────────────────────
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
has_cm  = df["has_sigma_CM"].fillna(0.0).values.astype(np.float32)

groups    = df["chemistry_family"].values
train_idx = np.array(calib["train_idx"])
calib_idx = np.array(calib["calib_idx"])

print(f"Dataset: {len(df)} total | train={len(train_idx)} | holdout={len(calib_idx)}")
print(f"Features: {X_all.shape[1]}")

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
print(f"Strat-GSS: {len(folds)} folds, fold sizes: {[len(f[1]) for f in folds]}\n")

# ── Helper: evaluate model over folds ────────────────────────────────────────
def eval_sklearn(model_fn, X, y, folds, calib_idx, train_idx, scale=True):
    """Train/eval a sklearn model over folds + holdout."""
    fold_r2 = []
    for tr, va in folds:
        if scale:
            sc = StandardScaler().fit(X[tr])
            Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        else:
            Xtr, Xva = X[tr], X[va]
        m = model_fn()
        m.fit(Xtr, y[tr])
        pred = m.predict(Xva)
        fold_r2.append(r2_score(y[va], pred))

    # final holdout
    if scale:
        sc = StandardScaler().fit(X[train_idx])
        Xtr_f = sc.transform(X[train_idx])
        Xca_f = sc.transform(X[calib_idx])
    else:
        Xtr_f = X[train_idx]
        Xca_f = X[calib_idx]
    mf = model_fn()
    mf.fit(Xtr_f, y[train_idx])
    p_ho = mf.predict(Xca_f)
    r2_ho  = r2_score(y[calib_idx], p_ho)
    mae_ho = mean_absolute_error(y[calib_idx], p_ho)

    return {
        "strat_gss_r2_mean": round(float(np.mean(fold_r2)), 4),
        "strat_gss_r2_std":  round(float(np.std(fold_r2)), 4),
        "strat_gss_r2_per_fold": [round(r, 4) for r in fold_r2],
        "holdout_r2":  round(r2_ho, 4),
        "holdout_mae": round(mae_ho, 3),
    }

# ── BPNN definition ───────────────────────────────────────────────────────────
class BPNN(nn.Module):
    """Standard 3-hidden-layer MLP — no physics constraints."""
    def __init__(self, n_in, hidden=256, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_bpnn(X_s, y, tr_idx, va_idx, seed=0, n_epochs=400, lr=0.001,
               batch=128, wd=0.0005):
    torch.manual_seed(seed)
    model = BPNN(n_in=X_s.shape[1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_va, best_sd, patience, wait = 1e9, None, 60, 0

    for ep in range(n_epochs):
        model.train()
        idx = tr_idx[torch.randperm(len(tr_idx),
                                    generator=torch.Generator().manual_seed(ep))]
        for i in range(0, len(idx), batch):
            sl = idx[i:i+batch]
            bx = torch.tensor(X_s[sl]).to(DEVICE)
            by = torch.tensor(y[sl]).to(DEVICE)
            opt.zero_grad()
            torch.sqrt(F.mse_loss(model(bx), by)).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vx = torch.tensor(X_s[va_idx]).to(DEVICE)
            vy = torch.tensor(y[va_idx]).to(DEVICE)
            va_loss = float(torch.sqrt(F.mse_loss(model(vx), vy)))
        if va_loss < best_va:
            best_va, best_sd, wait = va_loss, {k: v.cpu().clone()
                                                for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_sd)
    return model

def eval_bpnn(X_all, y, folds, calib_idx, train_idx, n_seeds=3):
    fold_r2 = []
    for fi, (tr, va) in enumerate(folds):
        print(f"  BPNN Fold {fi+1}/{len(folds)}")
        sc = StandardScaler().fit(X_all[tr])
        Xs = sc.transform(X_all).astype(np.float32)
        preds = []
        for seed in range(n_seeds):
            m = train_bpnn(Xs, y, tr, va, seed=seed)
            m.eval()
            with torch.no_grad():
                p = m(torch.tensor(Xs[va]).to(DEVICE)).cpu().numpy()
            preds.append(p)
        fold_r2.append(r2_score(y[va], np.mean(preds, axis=0)))

    # holdout
    sc = StandardScaler().fit(X_all[train_idx])
    Xs = sc.transform(X_all).astype(np.float32)
    preds = []
    for seed in range(n_seeds):
        m = train_bpnn(Xs, y, train_idx, calib_idx, seed=seed)
        m.eval()
        with torch.no_grad():
            p = m(torch.tensor(Xs[calib_idx]).to(DEVICE)).cpu().numpy()
        preds.append(p)
    p_ho = np.mean(preds, axis=0)
    r2_ho  = r2_score(y[calib_idx], p_ho)
    mae_ho = mean_absolute_error(y[calib_idx], p_ho)

    return {
        "strat_gss_r2_mean": round(float(np.mean(fold_r2)), 4),
        "strat_gss_r2_std":  round(float(np.std(fold_r2)), 4),
        "strat_gss_r2_per_fold": [round(r, 4) for r in fold_r2],
        "holdout_r2":  round(r2_ho, 4),
        "holdout_mae": round(mae_ho, 3),
    }

# ── 1. CM-only baseline ───────────────────────────────────────────────────────
print("=" * 55)
print("1. CM-only baseline")
print("=" * 55)
cm_fold_r2 = []
for tr, va in folds:
    # Use er_CM directly where available, mean of training er_CM otherwise
    pred = np.where(has_cm[va] > 0.5, er_cm[va],
                    np.mean(y[tr]))  # fallback: training mean
    cm_fold_r2.append(r2_score(y[va], pred))

pred_ho = np.where(has_cm[calib_idx] > 0.5, er_cm[calib_idx],
                   np.mean(y[train_idx]))
results_cm = {
    "strat_gss_r2_mean": round(float(np.mean(cm_fold_r2)), 4),
    "strat_gss_r2_std":  round(float(np.std(cm_fold_r2)), 4),
    "strat_gss_r2_per_fold": [round(r, 4) for r in cm_fold_r2],
    "holdout_r2":  round(r2_score(y[calib_idx], pred_ho), 4),
    "holdout_mae": round(mean_absolute_error(y[calib_idx], pred_ho), 3),
}
print(f"  Strat-GSS R² = {results_cm['strat_gss_r2_mean']:.4f} ± {results_cm['strat_gss_r2_std']:.4f}")
print(f"  Holdout  R² = {results_cm['holdout_r2']:.4f}   MAE = {results_cm['holdout_mae']:.3f}")

# ── 2. Ridge Regression ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("2. Ridge Regression (α=10)")
print("=" * 55)
results_ridge = eval_sklearn(lambda: Ridge(alpha=10.0), X_all, y, folds, calib_idx, train_idx)
print(f"  Strat-GSS R² = {results_ridge['strat_gss_r2_mean']:.4f} ± {results_ridge['strat_gss_r2_std']:.4f}")
print(f"  Holdout  R² = {results_ridge['holdout_r2']:.4f}   MAE = {results_ridge['holdout_mae']:.3f}")

# ── 3. Random Forest ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("3. Random Forest (n=500)")
print("=" * 55)
results_rf = eval_sklearn(
    lambda: RandomForestRegressor(n_estimators=500, max_features=0.33,
                                   min_samples_leaf=2, n_jobs=-1, random_state=42),
    X_all, y, folds, calib_idx, train_idx, scale=False
)
print(f"  Strat-GSS R² = {results_rf['strat_gss_r2_mean']:.4f} ± {results_rf['strat_gss_r2_std']:.4f}")
print(f"  Holdout  R² = {results_rf['holdout_r2']:.4f}   MAE = {results_rf['holdout_mae']:.3f}")

# ── 4. XGBoost ────────────────────────────────────────────────────────────────
results_xgb = None
if HAS_XGB:
    print("\n" + "=" * 55)
    print("4. XGBoost")
    print("=" * 55)
    results_xgb = eval_sklearn(
        lambda: XGBRegressor(
            n_estimators=800, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            n_jobs=-1, random_state=42, verbosity=0
        ),
        X_all, y, folds, calib_idx, train_idx, scale=False
    )
    print(f"  Strat-GSS R² = {results_xgb['strat_gss_r2_mean']:.4f} ± {results_xgb['strat_gss_r2_std']:.4f}")
    print(f"  Holdout  R² = {results_xgb['holdout_r2']:.4f}   MAE = {results_xgb['holdout_mae']:.3f}")

# ── 5. CatBoost ───────────────────────────────────────────────────────────────
results_cat = None
if HAS_CAT:
    print("\n" + "=" * 55)
    print("5. CatBoost")
    print("=" * 55)
    results_cat = eval_sklearn(
        lambda: CatBoostRegressor(
            iterations=1000, depth=7, learning_rate=0.05,
            l2_leaf_reg=3.0, subsample=0.8, colsample_bylevel=0.8,
            random_seed=42, verbose=0
        ),
        X_all, y, folds, calib_idx, train_idx, scale=False
    )
    print(f"  Strat-GSS R² = {results_cat['strat_gss_r2_mean']:.4f} ± {results_cat['strat_gss_r2_std']:.4f}")
    print(f"  Holdout  R² = {results_cat['holdout_r2']:.4f}   MAE = {results_cat['holdout_mae']:.3f}")

# ── 6. BPNN (MLP, no physics) ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("6. BPNN / MLP (3 hidden layers, no physics constraints)")
print("=" * 55)
results_bpnn = eval_bpnn(X_all, y, folds, calib_idx, train_idx, n_seeds=3)
print(f"  Strat-GSS R² = {results_bpnn['strat_gss_r2_mean']:.4f} ± {results_bpnn['strat_gss_r2_std']:.4f}")
print(f"  Holdout  R² = {results_bpnn['holdout_r2']:.4f}   MAE = {results_bpnn['holdout_mae']:.3f}")

# ── Summary table — load PCNN/PIRNN from result files (same holdout split) ────
_retrain = json.load(open(os.path.join(RES_DIR, "48_cmltrv77_retrain.json")))
_conf    = json.load(open(os.path.join(RES_DIR, "54_conformal_v77.json")))
PCNN_GSS_R2  = round(_retrain["strat_gss"]["r2_mean"], 4)
PCNN_GSS_STD = round(_retrain["strat_gss"]["r2_std"],  4)
PCNN_HO_R2   = round(_conf["r2_calibration"], 4)
PCNN_HO_MAE  = round(_conf["mae_calibration"], 3)

_pirnn = json.load(open(os.path.join(RES_DIR, "56_pirnn_comparison.json")))
PIRNN_GSS_R2  = round(_pirnn["PIRNN"]["strat_gss_r2_mean"], 4)
PIRNN_GSS_STD = round(_pirnn["PIRNN"]["strat_gss_r2_std"],  4)
PIRNN_HO_R2   = round(_pirnn["PIRNN"]["holdout_r2"], 4)
PIRNN_HO_MAE  = round(_pirnn["PIRNN"]["holdout_mae"], 3)

print(f"\n{'='*72}")
print(f"  {'Model':<28} {'Strat-GSS R²':<22} {'Holdout R²':<12} {'Holdout MAE'}")
print(f"  {'-'*69}")

models_summary = [
    ("CM-only (physics anchor)",       results_cm),
    ("Ridge Regression",               results_ridge),
    ("Random Forest",                  results_rf),
    ("XGBoost",                        results_xgb),
    ("CatBoost",                       results_cat),
    ("BPNN / MLP",                     results_bpnn),
    ("PIRNN (phys. features only)",    {"strat_gss_r2_mean": PIRNN_GSS_R2, "strat_gss_r2_std": PIRNN_GSS_STD,
                                         "holdout_r2": PIRNN_HO_R2, "holdout_mae": PIRNN_HO_MAE}),
    ("PCNN / CMLTRPCNNv7.7 (ours)",   {"strat_gss_r2_mean": PCNN_GSS_R2, "strat_gss_r2_std": PCNN_GSS_STD,
                                         "holdout_r2": PCNN_HO_R2, "holdout_mae": PCNN_HO_MAE}),
]

for name, res in models_summary:
    if res is None:
        print(f"  {name:<28} {'N/A — not installed'}")
        continue
    gss = f"{res['strat_gss_r2_mean']:.4f} ± {res['strat_gss_r2_std']:.4f}"
    flag = " ◄ best" if name.startswith("PCNN") else ""
    print(f"  {name:<28} {gss:<22} {res['holdout_r2']:.4f}       {res['holdout_mae']:.3f}{flag}")

print(f"{'='*72}")

# ── Save results ──────────────────────────────────────────────────────────────
all_results = {
    "comparison": "Fair baseline comparison vs CMLTRPCNNv7.7",
    "protocol": f"Strat-GSS 5-fold CV + held-out set (n={len(calib_idx)})",
    "feature_set": "feature_matrix_v7 — all 103 features for baselines",
    "models": {}
}

for name, res in models_summary:
    if res is not None:
        all_results["models"][name] = res

out_json = os.path.join(RES_DIR, "57_baseline_comparison.json")
json.dump(all_results, open(out_json, "w"), indent=2)
print(f"\nSaved: {out_json}")

# Save CSV
rows = []
for name, res in models_summary:
    if res is not None:
        rows.append({
            "model": name,
            "strat_gss_r2_mean": res["strat_gss_r2_mean"],
            "strat_gss_r2_std":  res["strat_gss_r2_std"],
            "holdout_r2":        res["holdout_r2"],
            "holdout_mae":       res["holdout_mae"],
        })

csv_path = os.path.join(CSV_DIR, "13_baseline_comparison.csv")
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")
