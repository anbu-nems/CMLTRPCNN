import os

# --- self-contained release root (auto-injected) ---
RELEASE_ROOT = os.path.abspath(os.path.dirname(__file__))
while RELEASE_ROOT != os.path.dirname(RELEASE_ROOT) and not os.path.isdir(os.path.join(RELEASE_ROOT, 'model_weights')):
    RELEASE_ROOT = os.path.dirname(RELEASE_ROOT)
# ----------------------------------------------------

"""
Script 42 — Leave-One-Family-Out (LOFO) Generality Tests

Priority 1 (submission-blocking): LOFO A-site + Leave-One-Reaney-Regime-Out.
Trains a model on all data EXCEPT the held-out group, then tests on that group.

Tests:
  A. LOFO A-site: hold out {Pb, Ba, Ca, Sr, La} one at a time
  B. Leave-one-Reaney-regime-out: hold out {Ia, Ib, II, III} one at a time

Decision table (from user brief):
  LOFO strong (R²≥0.80) + counterfactual strong → "generalizable mechanism"
  LOFO moderate (0.60–0.80) + counterfactual strong → "robust trend, limited extrapolation"
  LOFO weak (<0.60) + counterfactual strong → "within-family mechanistic trend"
  LOFO weak + counterfactual weak → not publication-ready

Outputs:
  results/42_lofo_generality.json
  figures/42a_lofo_asite.png
  figures/42b_lofo_regime.png
"""

import sys, os, warnings, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, TensorDataset

from model_CMLTRPCNNv77 import CMLTRPCNNv71

import sys as _sys
for _p in (RELEASE_ROOT, os.path.join(RELEASE_ROOT,'src','training'), os.path.join(RELEASE_ROOT,'src','analysis'), os.path.join(RELEASE_ROOT,'src','model')):
    if _p not in _sys.path: _sys.path.insert(0, _p)


DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)
FIG_DIR  = os.path.join(ROOT, "figures");  os.makedirs(FIG_DIR, exist_ok=True)
MODEL_PT = os.path.join(ROOT, "model_weights", "cmltrv77_final.pt")
LOG_FLOOR = 5.0

# Strength thresholds for decision table
STRONG   = 0.80
MODERATE = 0.60

HP = {
    "trunk_hidden": 256, "n_trunk_blocks": 2,
    "lst_hidden": 128, "tilt_hidden": 96, "res_hidden": 256,
    "residual_scale": 40.0, "dropout": 0.134, "weight_decay": 0.0002,
    "mixup_alpha": 0.158, "lr": 0.000876, "swa_start_frac": 0.65,
    "swa_lr": 1.35e-05, "lambda_lst": 0.00187, "lambda_gii": 0.0572,
    "lambda_vca": 0.00426, "log_loss_alpha": 0.536,
    "batch_size": 128, "n_epochs": 300, "early_stop": 60,
    "n_seeds": 2,  # 2 seeds for LOFO — single train/test split, no CV
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.grid": False, "axes.linewidth": 1.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 5, "ytick.major.size": 5,
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.fontsize": 9,
})

def style4(ax):
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(1.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


# ── Regime label for model ─────────────────────────────────────────────────
def load_regime_idx():
    ck = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
    return ck["regime_idx"]


# ── Loss ───────────────────────────────────────────────────────────────────
def v75_loss(out, y, gii, cm_approx, sample_weights=None,
             log_alpha=0.5, lambda_lst=0.01, lambda_gii=0.05,
             lambda_vca=0.01, er_cm=None):
    pred = out["pred"]
    w = sample_weights if sample_weights is not None else torch.ones_like(y)
    loss_log = (w*(torch.log(pred.clamp(min=LOG_FLOOR))-torch.log(y.clamp(min=LOG_FLOOR))).pow(2)).mean()
    loss_mse = (w*(pred-y).pow(2)).mean()
    loss_data = log_alpha*loss_log + (1-log_alpha)*loss_mse
    if er_cm is not None:
        loss_lst = lambda_lst * F.relu(out["delta_lst"]-5.0*er_cm.abs()).pow(2).mean()
        loss_gii = lambda_gii * (F.relu(er_cm-pred)*(gii<0.3).float()).pow(2).mean()
    else:
        loss_lst = loss_gii = torch.tensor(0.0, device=pred.device)
    loss_vca = lambda_vca * (out["delta_res"].pow(2)*(cm_approx>0.5).float()).mean()
    return loss_data+loss_lst+loss_gii+loss_vca, loss_data


def make_loader(arrays, idx, batch_size, shuffle):
    ds = TensorDataset(*[torch.tensor(a[idx], dtype=torch.float32) for a in arrays])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=False)


def scale_arrays(Xl, Xt, Xr, gii, train_idx):
    sc_lst  = StandardScaler().fit(Xl[train_idx])
    sc_tilt = StandardScaler().fit(Xt[train_idx])
    sc_res  = StandardScaler().fit(Xr[train_idx])
    gii_max = float(np.percentile(gii[train_idx][gii[train_idx]>0], 95)
                    if (gii[train_idx]>0).sum()>10 else 1.0)
    return (sc_lst.transform(Xl).astype(np.float32),
            sc_tilt.transform(Xt).astype(np.float32),
            sc_res.transform(Xr).astype(np.float32),
            np.clip(gii/gii_max, 0, 1).astype(np.float32))


def train_one(Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx, gii_s, phase_tr,
              y, regime_idx, tr_idx, va_idx, hp, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CMLTRPCNNv71(
        n_lst=Xl_s.shape[1], n_tilt=Xt_s.shape[1], n_res=Xr_s.shape[1],
        trunk_hidden=hp["trunk_hidden"], n_trunk_blocks=hp["n_trunk_blocks"],
        lst_hidden=hp["lst_hidden"], tilt_hidden=hp["tilt_hidden"],
        res_hidden=hp["res_hidden"], residual_scale=hp["residual_scale"],
        dropout=hp["dropout"]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["n_epochs"], eta_min=1e-6)
    swa_m = AveragedModel(model)
    swa_s = SWALR(opt, swa_lr=hp["swa_lr"], anneal_epochs=10)
    swa_start = int(hp["swa_start_frac"]*hp["n_epochs"]); swa_started = False
    lk    = {k: hp[k] for k in ("lambda_lst", "lambda_gii", "lambda_vca")}
    # uniform sample weights for LOFO (no family reweighting)
    fw_ones = np.ones(len(y), dtype=np.float32)
    arrays  = [Xl_s,Xt_s,Xr_s,er_cm,has_cm,cm_approx,gii_s,phase_tr,y,fw_ones]
    tr_ldr  = make_loader(arrays, tr_idx, hp["batch_size"], shuffle=True)
    va_ldr  = make_loader(arrays, va_idx, hp["batch_size"], shuffle=False)
    best_val, best_state, patience = 1e9, None, 0
    for epoch in range(hp["n_epochs"]):
        model.train()
        for batch in tr_ldr:
            bl,bt,br,b_cm,b_hcm,b_cap,b_gii,b_pt,by,_ = [x.to(DEVICE) for x in batch]
            if hp.get("mixup_alpha",0)>0:
                lam = float(np.random.beta(hp["mixup_alpha"], hp["mixup_alpha"]))
                idx2 = torch.randperm(bl.size(0), device=DEVICE)
                bl=lam*bl+(1-lam)*bl[idx2]; bt=lam*bt+(1-lam)*bt[idx2]
                br=lam*br+(1-lam)*br[idx2]; by=lam*by+(1-lam)*by[idx2]
                b_cm=lam*b_cm+(1-lam)*b_cm[idx2]
            out = model(bl,bt,br,b_cm,b_hcm,b_cap,b_gii,b_pt,regime_idx)
            loss,_ = v75_loss(out,by,b_gii,b_cap,log_alpha=hp["log_loss_alpha"],er_cm=b_cm,**lk)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if epoch >= swa_start: swa_m.update_parameters(model); swa_s.step(); swa_started=True
        else: sched.step()
        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for batch in va_ldr:
                bl,bt,br,b_cm,b_hcm,b_cap,b_gii,b_pt,by,_ = [x.to(DEVICE) for x in batch]
                out = model(bl,bt,br,b_cm,b_hcm,b_cap,b_gii,b_pt,regime_idx)
                l,_ = v75_loss(out,by,b_gii,b_cap,log_alpha=hp["log_loss_alpha"],er_cm=b_cm,**lk)
                va_loss += l.item()*len(by)
        va_loss /= max(len(va_idx), 1)
        if not swa_started:
            if va_loss < best_val-1e-5: best_val=va_loss; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; patience=0
            else:
                patience += 1
                if patience >= hp["early_stop"]: break
    if swa_started: return swa_m.module
    if best_state: model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(models_list, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
            gii_s, phase_tr, idx, regime_idx, batch=512):
    """Average predictions from multiple seed models."""
    preds = []
    for m in models_list:
        m.eval(); p = []
        for i in range(0, len(idx), batch):
            sl = idx[i:i+batch]
            out = m(torch.tensor(Xl_s[sl]).to(DEVICE), torch.tensor(Xt_s[sl]).to(DEVICE),
                    torch.tensor(Xr_s[sl]).to(DEVICE), torch.tensor(er_cm[sl]).to(DEVICE),
                    torch.tensor(has_cm[sl]).to(DEVICE), torch.tensor(cm_approx[sl]).to(DEVICE),
                    torch.tensor(gii_s[sl]).to(DEVICE), torch.tensor(phase_tr[sl]).to(DEVICE),
                    regime_idx)
            p.append(out["pred"].cpu().numpy())
        preds.append(np.concatenate(p))
    return np.mean(preds, 0)


def lofo_run(label, train_idx, test_idx, Xl, Xt, Xr, er_cm, has_cm,
             cm_approx, gii_norm, phase_tr, y, regime_idx, hp):
    """Train on train_idx, evaluate on test_idx. Returns R²."""
    if len(test_idx) == 0:
        print(f"    [{label}] SKIP — no test samples")
        return None, None
    # Fit scalers on training data only
    Xl_s, Xt_s, Xr_s, gii_s = scale_arrays(Xl, Xt, Xr, gii_norm, train_idx)
    # Use 10% of train as validation for early stopping
    rng = np.random.RandomState(42)
    shuf = rng.permutation(train_idx)
    va_size = max(10, len(shuf)//10)
    va_idx_local = shuf[:va_size]
    tr_idx_local = shuf[va_size:]
    # Train
    ms = [train_one(Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                    gii_s, phase_tr, y, regime_idx,
                    tr_idx_local, va_idx_local, hp, seed=s)
          for s in range(hp["n_seeds"])]
    # Evaluate on test
    pred = predict(ms, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                   gii_s, phase_tr, test_idx, regime_idx)
    r2 = float(r2_score(y[test_idx], pred))
    mae = float(np.mean(np.abs(y[test_idx] - pred)))
    return r2, mae


def strength_label(r2):
    if r2 is None: return "N/A"
    if r2 >= STRONG:   return "STRONG"
    if r2 >= MODERATE: return "MODERATE"
    return "WEAK"


def final_claim(lofo_r2s, cf_strong):
    """Determine final publishable claim from LOFO results."""
    valid = [r for r in lofo_r2s if r is not None]
    if not valid: return "insufficient data"
    mean_lofo = np.mean(valid)
    if mean_lofo >= STRONG and cf_strong:
        return "generalizable mechanism"
    elif mean_lofo >= MODERATE and cf_strong:
        return "robust trend with limited extrapolation"
    elif mean_lofo >= MODERATE:
        return "robust trend (awaiting counterfactual)"
    elif cf_strong:
        return "within-family mechanistic trend"
    return "not publication-ready — strengthen validation"


def main():
    print("="*70)
    print("Script 42 — LOFO Generality Tests")
    print("="*70)

    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_parquet(os.path.join(PROC_DIR, "feature_matrix_v7.parquet"))
    with open(os.path.join(PROC_DIR, "feature_partition_v7.json")) as f:
        partition = json.load(f)
    with open(os.path.join(PROC_DIR, "calibration_split_idx.json")) as f:
        calib_info = json.load(f)

    def _get(cols):
        present = [c for c in cols if c in df.columns]
        return df[present].fillna(0.0).values.astype(np.float32)

    Xl = _get(partition["LST"])
    Xt = _get(partition["Tilt"])
    Xr = _get(partition["Residual"])
    er_cm    = df["er_CM"].fillna(0.0).values.astype(np.float32)
    has_cm   = df["has_sigma_CM"].fillna(0.0).values.astype(np.float32)
    cm_approx= df["cm_approx_flag"].fillna(0.0).values.astype(np.float32)
    phase_tr = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
               if "phase_transition" in df.columns else np.zeros(len(df), np.float32)
    y        = df["epsilon_r"].values.astype(np.float32)
    gii_norm = np.zeros(len(df), np.float32)
    groups   = df["chemistry_family"].values
    train_pool = np.array(calib_info["train_idx"])

    regime_idx = load_regime_idx()

    # A-site and regime labels
    df["a_site"] = df["chemistry_family"].apply(lambda x: str(x).split("_")[0])
    # Dominant regime per sample (priority: III > II > Ib > Ia)
    reg_cols = {"Ia": "regime_Ia", "Ib": "regime_Ib", "II": "regime_II", "III": "regime_III"}
    dom_regime = np.full(len(df), "Ia", dtype=object)
    for rname, rcol in reg_cols.items():
        if rcol in df.columns:
            dom_regime[df[rcol].values > 0.5] = rname

    print(f"\n  {len(df)} total samples | train pool: {len(train_pool)}")
    print(f"  A-site counts: {dict(pd.Series(df['a_site'].values).value_counts().head(8))}")
    print(f"  Regime counts: {dict(pd.Series(dom_regime).value_counts())}")

    results = {"lofo_asite": {}, "lofo_regime": {}, "decision": {}}

    # ══════════════════════════════════════════════════════════════════════
    # PART A — LOFO A-site
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("PART A — Leave-One-A-site-Out (LOFO A-site)")
    print("─"*60)
    print("  Train pool → exclude target A-site; test on ALL target samples")
    print()

    TARGET_ASITES = ["Pb", "Ba", "Ca", "Sr", "La"]
    lofo_r2_list  = []

    for asite in TARGET_ASITES:
        mask  = df["a_site"].values == asite
        t_idx = np.where(mask)[0]                          # ALL asite samples → test
        tr_idx = train_pool[df["a_site"].values[train_pool] != asite]  # train without asite

        print(f"  A-site={asite}  train={len(tr_idx)}  test={len(t_idx)}")
        r2, mae = lofo_run(
            f"LOFO-{asite}", tr_idx, t_idx,
            Xl, Xt, Xr, er_cm, has_cm, cm_approx, gii_norm, phase_tr, y, regime_idx, HP)
        lofo_r2_list.append(r2)
        strength = strength_label(r2)
        print(f"    R²={r2:.4f}  MAE={mae:.2f}  [{strength}]")
        results["lofo_asite"][asite] = {
            "n_train": int(len(tr_idx)), "n_test": int(len(t_idx)),
            "r2": float(r2) if r2 is not None else None,
            "mae": float(mae) if mae is not None else None,
            "strength": strength
        }

    # Summary
    valid_r2 = [r for r in lofo_r2_list if r is not None]
    mean_lofo_asite = float(np.mean(valid_r2)) if valid_r2 else 0.0
    print(f"\n  Mean LOFO A-site R² = {mean_lofo_asite:.4f}")

    # Critical Pb check
    pb_r2 = results["lofo_asite"].get("Pb", {}).get("r2")
    if pb_r2 is not None:
        if pb_r2 >= STRONG:
            print(f"  Pb LOFO R²={pb_r2:.4f} — STRONG: f_LST rule generalizes ✓")
        elif pb_r2 >= MODERATE:
            print(f"  Pb LOFO R²={pb_r2:.4f} — MODERATE: partial generalization")
        else:
            print(f"  Pb LOFO R²={pb_r2:.4f} — WEAK: Pb finding is interpolation, not general law")

    # ══════════════════════════════════════════════════════════════════════
    # PART B — Leave-One-Reaney-Regime-Out
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("PART B — Leave-One-Reaney-Regime-Out")
    print("─"*60)
    print("  Train pool → exclude target regime; test on ALL target-regime samples")
    print()

    TARGET_REGIMES = ["Ia", "Ib", "II", "III"]
    regime_r2_list = []

    for reg in TARGET_REGIMES:
        mask   = dom_regime == reg
        t_idx  = np.where(mask)[0]                              # all target-regime samples
        tr_idx = train_pool[dom_regime[train_pool] != reg]     # train without target regime

        print(f"  Regime={reg}  train={len(tr_idx)}  test={len(t_idx)}")
        if len(t_idx) < 5:
            print(f"    SKIP — too few test samples ({len(t_idx)})")
            results["lofo_regime"][reg] = {"n_test": int(len(t_idx)), "r2": None, "strength": "N/A"}
            regime_r2_list.append(None)
            continue
        r2, mae = lofo_run(
            f"LOFO-{reg}", tr_idx, t_idx,
            Xl, Xt, Xr, er_cm, has_cm, cm_approx, gii_norm, phase_tr, y, regime_idx, HP)
        regime_r2_list.append(r2)
        strength = strength_label(r2)
        print(f"    R²={r2:.4f}  MAE={mae:.2f}  [{strength}]")
        results["lofo_regime"][reg] = {
            "n_train": int(len(tr_idx)), "n_test": int(len(t_idx)),
            "r2": float(r2) if r2 is not None else None,
            "mae": float(mae) if mae is not None else None,
            "strength": strength
        }

    valid_regime_r2 = [r for r in regime_r2_list if r is not None]
    mean_lofo_regime = float(np.mean(valid_regime_r2)) if valid_regime_r2 else 0.0
    print(f"\n  Mean LOFO Regime R² = {mean_lofo_regime:.4f}")

    # Critical Ib check (soft-mode regime — the core of Finding 2)
    ib_r2 = results["lofo_regime"].get("Ib", {}).get("r2")
    if ib_r2 is not None:
        strength = strength_label(ib_r2)
        print(f"  Regime Ib LOFO R²={ib_r2:.4f} — {strength}")
        if ib_r2 >= STRONG:
            print("    → Ib regime pattern learned from Ia/II/III structure ✓")
        else:
            print("    → Ib soft-mode regime requires seen examples — limited extrapolation")

    # ══════════════════════════════════════════════════════════════════════
    # DECISION TABLE
    # ══════════════════════════════════════════════════════════════════════
    # Placeholder for counterfactual result (script 43 will fill this)
    # We assume "moderate" counterfactual here; script 43 updates the claim
    cf_strong_placeholder = True  # will be confirmed by script 43
    claim = final_claim(lofo_r2_list, cf_strong_placeholder)
    results["decision"] = {
        "mean_lofo_asite_r2": mean_lofo_asite,
        "mean_lofo_regime_r2": mean_lofo_regime,
        "pb_lofo_r2": pb_r2,
        "ib_regime_lofo_r2": ib_r2,
        "lofo_strength": strength_label(mean_lofo_asite),
        "preliminary_claim": claim,
        "note": "counterfactual_strong assumed True; run script 43 to confirm"
    }

    print("\n" + "="*60)
    print("DECISION TABLE")
    print("="*60)
    print(f"  Mean LOFO A-site R²  : {mean_lofo_asite:.4f}  [{strength_label(mean_lofo_asite)}]")
    print(f"  Mean LOFO Regime R²  : {mean_lofo_regime:.4f}  [{strength_label(mean_lofo_regime)}]")
    print(f"  Pb LOFO R²           : {pb_r2}")
    print(f"  Ib Regime LOFO R²    : {ib_r2}")
    print(f"  Preliminary claim    : {claim}")
    print("  (Final claim confirmed after script 43 counterfactual)")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════
    print("\n[Generating figures...]")

    # Figure A — LOFO A-site bar chart
    fig, ax = plt.subplots(figsize=(6, 3.8))
    asites  = [a for a in TARGET_ASITES if results["lofo_asite"].get(a, {}).get("r2") is not None]
    r2vals  = [results["lofo_asite"][a]["r2"] for a in asites]
    colors  = []
    for r2 in r2vals:
        if r2 >= STRONG:   colors.append("#2A9D8F")
        elif r2 >= MODERATE: colors.append("#E9C46A")
        else:              colors.append("#E76F51")
    bars = ax.bar(asites, r2vals, color=colors, edgecolor="white", linewidth=0.8, width=0.55)
    ax.axhline(STRONG,   color="#2A9D8F", linestyle="--", lw=1.2, label=f"Strong (R²≥{STRONG})")
    ax.axhline(MODERATE, color="#E9C46A", linestyle="--", lw=1.2, label=f"Moderate (R²≥{MODERATE})")
    ax.axhline(0, color="black", lw=0.8)
    for bar, r2 in zip(bars, r2vals):
        ax.text(bar.get_x()+bar.get_width()/2, max(r2+0.02, 0.05),
                f"{r2:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("R² on held-out A-site")
    ax.set_title("LOFO A-site: OOD generalization by A-site")
    ax.set_ylim(min(0, min(r2vals)-0.1), 1.05)
    ax.legend(loc="lower right", fontsize=8)
    style4(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "42a_lofo_asite.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure B — LOFO Regime bar chart
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    regs_ok = [r for r in TARGET_REGIMES if results["lofo_regime"].get(r, {}).get("r2") is not None]
    r2regs  = [results["lofo_regime"][r]["r2"] for r in regs_ok]
    colors2 = []
    for r2 in r2regs:
        if r2 >= STRONG:   colors2.append("#2A9D8F")
        elif r2 >= MODERATE: colors2.append("#E9C46A")
        else:              colors2.append("#E76F51")
    bars2 = ax.bar(regs_ok, r2regs, color=colors2, edgecolor="white", linewidth=0.8, width=0.5)
    ax.axhline(STRONG,   color="#2A9D8F", linestyle="--", lw=1.2, label=f"Strong (R²≥{STRONG})")
    ax.axhline(MODERATE, color="#E9C46A", linestyle="--", lw=1.2, label=f"Moderate (R²≥{MODERATE})")
    ax.axhline(0, color="black", lw=0.8)
    for bar, r2 in zip(bars2, r2regs):
        ax.text(bar.get_x()+bar.get_width()/2, max(r2+0.02, 0.05),
                f"{r2:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("R² on held-out regime")
    ax.set_title("LOFO Regime: OOD generalization by Reaney regime")
    ax.set_ylim(min(0, min(r2regs)-0.1), 1.05)
    ax.legend(loc="lower right", fontsize=8)
    style4(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "42b_lofo_regime.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save results
    with open(os.path.join(RES_DIR, "42_lofo_generality.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → figures/42a_lofo_asite.png, 42b_lofo_regime.png")
    print(f"  Results → results/42_lofo_generality.json")

    print("\n" + "="*70)
    print("LOFO GENERALITY TESTS COMPLETE")
    print("="*70)
    asite_summary = {a: "%.3f [%s]" % (results["lofo_asite"][a]["r2"], results["lofo_asite"][a]["strength"]) for a in asites}
    regime_summary= {r: "%.3f [%s]" % (results["lofo_regime"][r]["r2"], results["lofo_regime"][r]["strength"]) for r in regs_ok}
    print(f"  LOFO A-site : {asite_summary}")
    print(f"  LOFO Regime : {regime_summary}")
    print(f"  Preliminary claim: {claim}")
    print("="*70)


if __name__ == "__main__":
    main()
