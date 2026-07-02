"""
05_subclass_generalizability.py — Reaney-regime LOFO + per-subclass branch
attribution audit.

T2 from the NMI strengthening plan: demonstrate that PCNN's 4-branch additive
decomposition is robust across natural data variations (Reaney regimes,
dominant A-site cation) and that the per-branch contributions track known
physics across these subclasses.

TWO EXPERIMENTS:

  Exp A — Reaney-regime Leave-One-Out (LOFO)
    For each Reaney regime r ∈ {Ia, Ib, II, III}:
      • Train PCNN-equivalent on samples from the OTHER three regimes
      • Test on the held-out regime r
      • Report R², MAE, sign-violation rate
    Shows whether PCNN's architecture generalizes to a regime it has never seen.

  Exp B — Per-subclass branch attribution audit
    Train PCNN-equivalent once on the full training set.
    For each subclass (Reaney regime + dominant A-site), report:
      • Mean predicted ε_r
      • Mean per-branch contribution: ⟨δ_CM⟩, ⟨δ_LST⟩, ⟨δ_tilt⟩, ⟨δ_res⟩
      • f_LST = mean δ_LST / mean δ_CM   (the canonical PCNN summary statistic)
    Shows whether the architectural decomposition tracks established physics:
      • Regime Ib (d0 B-site, polarizable A): expect HIGH δ_LST (soft-mode amp)
      • Regime II/III (octahedral tilting):    expect more-negative δ_tilt
      • Pb-A-site: expect HIGH δ_LST (lone-pair amplification)
      • Ba-A-site: expect LOW  δ_LST (no lone pair, weak amplification)

Both experiments use the same PCNN-equivalent architecture as script 03 with
identical training hyperparameters, so the result is comparable.

Output: results/05_subclass_generalizability.json
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
N_TILT = Xt.shape[1]

y       = df["epsilon_r"].values.astype(np.float32)
er_cm   = df["er_CM"].fillna(0.0).values.astype(np.float32)
lst_triplet  = df[["soft_mode_activity", "b_o_reduced_mass", "lst_enhancement_proxy"]].fillna(0.0).values.astype(np.float32)
tilt_triplet = df[["tilt_severity", "charge_imbalance_proxy", "continuous_tilt_strain"]].fillna(0.0).values.astype(np.float32)
is_ib   = df["d0_B_polarizable_A"].fillna(0.0).values.astype(np.float32)

train_idx = np.array(calib["train_idx"])
calib_idx = np.array(calib["calib_idx"])

# Reaney regime — one-hot columns regime_Ia, regime_Ib, regime_II, regime_III
regime_cols = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
regime = np.full(len(df), "Unknown", dtype="<U10")
for col in regime_cols:
    if col in df.columns:
        regime[df[col].fillna(0.0).values.astype(bool)] = col.replace("regime_", "")

print(f"[data] regime counts: {dict(Counter(regime))}")

# Dominant A-site cation — extract from chemistry_family (e.g., 'Ba_Mg' → 'Ba')
asite_dominant = df["chemistry_family"].apply(lambda x: x.split("_")[0] if isinstance(x, str) else "Unknown").values
asite_counts = Counter(asite_dominant)
print(f"[data] dominant A-site counts (top 6): {dict(sorted(asite_counts.items(), key=lambda kv: -kv[1])[:6])}")


# ── PCNN-equivalent architecture (matches script 03) ────────────────────
class PINN_Arch(nn.Module):
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

    def forward(self, x, er_cm, lst_feats, tilt_feats, is_ib):
        h = self.encoder(x)
        h_lst = torch.cat([h, lst_feats], dim=1)
        delta_lst_std = self.lst_head(h_lst).squeeze(-1)
        delta_lst_ib  = self.lst_head_ib(h_lst).squeeze(-1)
        delta_lst = (1 - is_ib) * delta_lst_std + is_ib * delta_lst_ib
        h_tilt = torch.cat([h, tilt_feats], dim=1)
        delta_tilt = -self.tilt_head(h_tilt).squeeze(-1)   # enforce ≤ 0
        delta_res = self.res_head(h).squeeze(-1) * self.residual_scale
        return {"pred": er_cm + delta_lst + delta_tilt + delta_res,
                "delta_lst": delta_lst, "delta_tilt": delta_tilt, "delta_res": delta_res,
                "er_cm": er_cm}


def train_model(tr_indices, epochs=300, lr=7.4e-4, batch_size=256, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    sc = StandardScaler().fit(X_all[tr_indices])
    Xtr_s = sc.transform(X_all[tr_indices]).astype(np.float32)
    model = PINN_Arch(n_in=X_all.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xtr_t   = torch.from_numpy(Xtr_s).to(DEVICE)
    ytr_t   = torch.from_numpy(y[tr_indices]).to(DEVICE)
    er_cm_t = torch.from_numpy(er_cm[tr_indices]).to(DEVICE)
    lst_t   = torch.from_numpy(lst_triplet[tr_indices]).to(DEVICE)
    tilt_t  = torch.from_numpy(tilt_triplet[tr_indices]).to(DEVICE)
    ib_t    = torch.from_numpy(is_ib[tr_indices]).to(DEVICE)
    n = len(tr_indices)
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
    return model, sc


def evaluate(model, sc, eval_indices):
    model.eval()
    Xe_s = sc.transform(X_all[eval_indices]).astype(np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(Xe_s).to(DEVICE),
                    torch.from_numpy(er_cm[eval_indices]).to(DEVICE),
                    torch.from_numpy(lst_triplet[eval_indices]).to(DEVICE),
                    torch.from_numpy(tilt_triplet[eval_indices]).to(DEVICE),
                    torch.from_numpy(is_ib[eval_indices]).to(DEVICE))
    return {k: v.cpu().numpy() for k, v in out.items()}


# ── Exp A: Reaney-regime LOFO ─────────────────────────────────────────────
print("\n=== Exp A: Reaney-regime LOFO ===")
print("For each regime r: train on the OTHER three regimes (within the training partition), test on r")
regimes_to_test = ["Ia", "Ib", "II", "III"]
lofo_results = []
for held_out in regimes_to_test:
    # Train on training samples NOT in held_out regime
    train_mask  = np.array([(i in train_idx) and (regime[i] != held_out) for i in range(len(df))])
    test_mask   = np.array([(regime[i] == held_out) for i in range(len(df))])
    tr_ix = np.where(train_mask)[0]
    te_ix = np.where(test_mask)[0]
    if len(te_ix) < 5:
        print(f"  {held_out}: SKIP (only {len(te_ix)} test samples)")
        continue
    t0 = time.time()
    model, sc = train_model(tr_ix)
    out = evaluate(model, sc, te_ix)
    r2  = r2_score(y[te_ix], out["pred"])
    mae = mean_absolute_error(y[te_ix], out["pred"])
    lst_v  = int((out["delta_lst"]  < 0).sum())
    tilt_v = int((out["delta_tilt"] > 0).sum())
    elapsed = time.time() - t0
    lofo_results.append({
        "held_out_regime": held_out,
        "n_train": int(len(tr_ix)), "n_test": int(len(te_ix)),
        "r2":  round(float(r2), 4), "mae": round(float(mae), 3),
        "lst_violations":  lst_v, "tilt_violations": tilt_v,
        "time_s": round(elapsed, 1),
    })
    print(f"  HOLD-OUT {held_out:<4}: n_train={len(tr_ix)} | n_test={len(te_ix)} | "
          f"R²={r2:>6.3f} | MAE={mae:>5.2f} | violations={lst_v}/{tilt_v}  ({elapsed:.1f}s)")


# ── Exp B: Per-subclass branch attribution audit ─────────────────────────
print("\n=== Exp B: Per-subclass branch attribution audit ===")
print("Train PCNN-equivalent on full training set; report mean per-branch contribution by subclass\n")

t0 = time.time()
model_full, sc_full = train_model(train_idx)
elapsed = time.time() - t0
print(f"(model trained in {elapsed:.1f}s)")

# Evaluate on ALL data (for attribution analysis)
out_full = evaluate(model_full, sc_full, np.arange(len(df)))


def attribution(mask, label):
    """Return mean per-branch contribution + sample count for the given mask."""
    if mask.sum() == 0:
        return None
    er_cm_mean    = float(np.mean(out_full["er_cm"][mask]))
    delta_lst_m   = float(np.mean(out_full["delta_lst"][mask]))
    delta_tilt_m  = float(np.mean(out_full["delta_tilt"][mask]))
    delta_res_m   = float(np.mean(out_full["delta_res"][mask]))
    pred_mean     = float(np.mean(out_full["pred"][mask]))
    y_mean        = float(np.mean(y[mask]))
    return {
        "label": label,
        "n":     int(mask.sum()),
        "y_mean":      round(y_mean, 2),
        "pred_mean":   round(pred_mean, 2),
        "er_CM_mean":  round(er_cm_mean, 2),
        "delta_LST_mean":  round(delta_lst_m, 2),
        "delta_tilt_mean": round(delta_tilt_m, 2),
        "delta_res_mean":  round(delta_res_m, 2),
        "f_LST": round(delta_lst_m / er_cm_mean, 3) if er_cm_mean > 0 else None,
    }


# By Reaney regime
print("\n--- by Reaney regime ---")
print(f"{'regime':<8} {'n':<5} {'y_mean':<8} {'CM':<8} {'δ_LST':<8} {'δ_tilt':<8} {'δ_res':<8} {'f_LST':<7}")
print("-" * 70)
by_regime = []
for r in regimes_to_test:
    mask = (regime == r)
    a = attribution(mask, f"regime_{r}")
    if a:
        by_regime.append(a)
        print(f"{r:<8} {a['n']:<5} {a['y_mean']:<8.2f} {a['er_CM_mean']:<8.2f} "
              f"{a['delta_LST_mean']:<8.2f} {a['delta_tilt_mean']:<8.2f} "
              f"{a['delta_res_mean']:<8.2f} {a['f_LST']:<7.3f}" if a['f_LST'] else "—")

# By dominant A-site
print("\n--- by dominant A-site cation ---")
print(f"{'A-site':<8} {'n':<5} {'y_mean':<8} {'CM':<8} {'δ_LST':<8} {'δ_tilt':<8} {'δ_res':<8} {'f_LST':<7}")
print("-" * 70)
by_asite = []
top_asites = [k for k, _ in sorted(asite_counts.items(), key=lambda kv: -kv[1])[:8]]
for a_site in top_asites:
    mask = (asite_dominant == a_site)
    a = attribution(mask, f"asite_{a_site}")
    if a and a["n"] >= 10:
        by_asite.append(a)
        print(f"{a_site:<8} {a['n']:<5} {a['y_mean']:<8.2f} {a['er_CM_mean']:<8.2f} "
              f"{a['delta_LST_mean']:<8.2f} {a['delta_tilt_mean']:<8.2f} "
              f"{a['delta_res_mean']:<8.2f} {a['f_LST']:<7.3f}" if a['f_LST'] else "—")


# ── Save ────────────────────────────────────────────────────────────────────
result = {
    "experiment_A_reaney_lofo": {
        "description": "Leave-one-Reaney-regime-out: train on 3 regimes, test on the held-out 4th regime",
        "results": lofo_results,
    },
    "experiment_B_subclass_attribution": {
        "description": "Per-subclass mean per-branch attribution from PCNN-equivalent trained on full training set",
        "by_regime": by_regime,
        "by_asite":  by_asite,
    },
    "model": "PINN_Arch (PCNN-equivalent, identical to script 03 model A)",
    "n_total": int(len(df)),
}
out_path = os.path.join(OUT_DIR, "05_subclass_generalizability.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n[done] saved → {out_path}")
