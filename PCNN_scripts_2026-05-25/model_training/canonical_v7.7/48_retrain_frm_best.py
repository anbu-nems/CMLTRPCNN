"""
Script 48 — Final retrain: best HP from script 47 Optuna (formula-split objective)

HP source: results/47_optuna_frm.db Trial 34 (frm R²=0.9281, 1 seed)
Key differences vs v76 (script 37):
  trunk_hidden   : 256 → 512
  lst_hidden     : 128 → 96
  tilt_hidden    :  96 → 32
  res_hidden     : 256 → 192
  residual_scale :  40 → 80
  lr             : 8.76e-4 → 1.72e-3
  swa_start_frac : 0.65 → 0.75
  log_loss_alpha : 0.536 → 0.757
  mixup_alpha    : 0.158 → 0.005
  n_seeds        : 5 (same)
  n_epochs       : 500 (same)

Outputs:
  models/cmltrv77_final.pt
  results/48_cmltrv77_retrain.json
  figures/48_v77_parity.png
"""

import sys, os, warnings, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.psrnn_mdpinn import CMLTRPCNN

DEVICE    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT      = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR  = os.path.join(ROOT, "data", "processed")
MODEL_OUT = os.path.join(ROOT, "models", "cmltrv77_final.pt")
RES_DIR   = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)
FIG_DIR   = os.path.join(ROOT, "figures"); os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

LOG_FLOOR       = 5.0
MIN_FAMILY_SIZE = 50

# Best HP from script 47 Optuna Trial 34
HP = {
    "trunk_hidden":      512,
    "n_trunk_blocks":    2,
    "lst_hidden":        96,
    "tilt_hidden":       32,
    "res_hidden":        192,
    "residual_scale":    80.0,
    "dropout":           0.1681982132911167,
    "weight_decay":      0.0008312905717274681,
    "mixup_alpha":       0.005005005146664418,
    "lr":                0.0017216692708802786,
    "swa_start_frac":    0.75,
    "swa_lr":            2.9047556806477378e-05,
    "lambda_lst":        0.0075683080032818454,
    "lambda_gii":        0.01171762763214901,
    "lambda_vca":        0.009401738824421016,
    "log_loss_alpha":    0.7572570965399923,
    "batch_size":        128,
    "n_epochs":          500,
    "early_stop":        100,
    "n_seeds":           5,
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_parquet(os.path.join(PROC_DIR, "feature_matrix_v7.parquet"))
    with open(os.path.join(PROC_DIR, "feature_partition_v7.json")) as f:
        partition = json.load(f)
    with open(os.path.join(PROC_DIR, "calibration_split_idx.json")) as f:
        calib_info = json.load(f)

    def _get(cols):
        present = [c for c in cols if c in df.columns]
        return df[present].fillna(0.0).values.astype(np.float32), present

    Xl, lcols = _get(partition["LST"])
    Xt, _     = _get(partition["Tilt"])
    Xr, _     = _get(partition["Residual"])
    er_cm     = df["er_CM"].fillna(0.0).values.astype(np.float32)
    has_cm    = df["has_sigma_CM"].fillna(0.0).values.astype(np.float32)
    cm_approx = df["cm_approx_flag"].fillna(0.0).values.astype(np.float32)
    y         = df["DC"].values.astype(np.float32)
    groups    = df["chemistry_family"].values
    gii       = np.zeros(len(df), dtype=np.float32)
    phase_tr  = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
                if "phase_transition" in df.columns else np.zeros(len(df), np.float32)
    formulas  = df["formula"].values if "formula" in df.columns else np.array([""] * len(df))

    regime_names = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
    regime_idx   = [lcols.index(r) for r in regime_names if r in lcols]

    train_idx = np.array(calib_info["train_idx"])
    calib_idx = np.array(calib_info["calib_idx"])

    return (Xl, Xt, Xr, er_cm, has_cm, cm_approx,
            gii, phase_tr, y, groups, regime_idx,
            train_idx, calib_idx, formulas)


# ── Loss ──────────────────────────────────────────────────────────────────────
def v75_loss(out, y, gii, cm_approx_flag, sample_weights=None,
             log_alpha=0.5, lambda_lst=0.01, lambda_gii=0.05,
             lambda_vca=0.01, er_cm=None):
    pred      = out["pred"]
    w         = sample_weights if sample_weights is not None else torch.ones_like(y)
    loss_log  = (w * (torch.log(pred.clamp(min=LOG_FLOOR)) -
                      torch.log(y.clamp(min=LOG_FLOOR))).pow(2)).mean()
    loss_mse  = (w * (pred - y).pow(2)).mean()
    loss_data = log_alpha * loss_log + (1.0 - log_alpha) * loss_mse
    if er_cm is not None:
        loss_lst = lambda_lst * F.relu(out["delta_lst"] - 5.0 * er_cm.abs()).pow(2).mean()
        loss_gii = lambda_gii * (F.relu(er_cm - pred) * (gii < 0.3).float()).pow(2).mean()
    else:
        loss_lst = loss_gii = torch.tensor(0.0, device=pred.device)
    loss_vca = lambda_vca * (out["delta_res"].pow(2) * (cm_approx_flag > 0.5).float()).mean()
    return loss_data + loss_lst + loss_gii + loss_vca, loss_data


def scale_arrays(Xl, Xt, Xr, gii, train_idx):
    sc_lst  = StandardScaler().fit(Xl[train_idx])
    sc_tilt = StandardScaler().fit(Xt[train_idx])
    sc_res  = StandardScaler().fit(Xr[train_idx])
    gii_max = float(np.percentile(gii[train_idx][gii[train_idx] > 0], 95)
                    if (gii[train_idx] > 0).sum() > 10 else 1.0)
    return (sc_lst.transform(Xl).astype(np.float32),
            sc_tilt.transform(Xt).astype(np.float32),
            sc_res.transform(Xr).astype(np.float32),
            np.clip(gii / gii_max, 0, 1).astype(np.float32))


def make_loader(arrays, idx, batch_size, shuffle):
    ds = TensorDataset(*[torch.tensor(a[idx], dtype=torch.float32) for a in arrays])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=False)


def train_one(Xl, Xt, Xr, er_cm, has_cm, cm_approx, gii, phase_tr,
              y, regime_idx, tr_idx, va_idx, hp, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    fam_w = np.ones(len(y), dtype=np.float32)

    model = CMLTRPCNN(
        n_lst=Xl.shape[1], n_tilt=Xt.shape[1], n_res=Xr.shape[1],
        trunk_hidden=hp["trunk_hidden"], n_trunk_blocks=hp["n_trunk_blocks"],
        lst_hidden=hp["lst_hidden"], tilt_hidden=hp["tilt_hidden"],
        res_hidden=hp["res_hidden"], residual_scale=hp["residual_scale"],
        dropout=hp["dropout"],
    ).to(DEVICE)

    opt   = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                               weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=hp["n_epochs"], eta_min=1e-6)
    swa_model = AveragedModel(model)
    swa_sched = SWALR(opt, swa_lr=hp["swa_lr"], anneal_epochs=10)
    swa_start = int(hp["swa_start_frac"] * hp["n_epochs"])
    swa_started = False

    arrays = [Xl, Xt, Xr, er_cm, has_cm, cm_approx, gii, phase_tr, y, fam_w]
    tr_ldr = make_loader(arrays, tr_idx, hp["batch_size"], shuffle=True)
    va_ldr = make_loader(arrays, va_idx, hp["batch_size"], shuffle=False)
    lk     = {k: hp[k] for k in ("lambda_lst", "lambda_gii", "lambda_vca")}

    best_val, best_state, patience = 1e9, None, 0

    for epoch in range(hp["n_epochs"]):
        model.train()
        for batch in tr_ldr:
            bl, bt, br, b_cm, b_hcm, b_cap, b_gii, b_pt, by, b_fw = \
                [x.to(DEVICE) for x in batch]
            if hp.get("mixup_alpha", 0.0) > 0:
                lam  = float(np.random.beta(hp["mixup_alpha"], hp["mixup_alpha"]))
                idx2 = torch.randperm(bl.size(0), device=DEVICE)
                bl   = lam * bl  + (1 - lam) * bl[idx2]
                bt   = lam * bt  + (1 - lam) * bt[idx2]
                br   = lam * br  + (1 - lam) * br[idx2]
                by   = lam * by  + (1 - lam) * by[idx2]
                b_cm = lam * b_cm+ (1 - lam) * b_cm[idx2]
            w   = 1.0 + 2.0 * by / 143.0
            w   = w / w.mean()
            out = model(bl, bt, br, b_cm, b_hcm, b_cap, b_gii, b_pt, regime_idx)
            loss, _ = v75_loss(out, by, b_gii, b_cap, sample_weights=w,
                               log_alpha=hp["log_loss_alpha"], er_cm=b_cm, **lk)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if epoch >= swa_start:
            swa_model.update_parameters(model); swa_sched.step(); swa_started = True
        else:
            sched.step()

        if not swa_started:
            model.eval(); va_loss = 0.0
            with torch.no_grad():
                for batch in va_ldr:
                    bl, bt, br, b_cm, b_hcm, b_cap, b_gii, b_pt, by, b_fw = \
                        [x.to(DEVICE) for x in batch]
                    out_loss = v75_loss(
                        model(bl, bt, br, b_cm, b_hcm, b_cap, b_gii, b_pt, regime_idx),
                        by, b_gii, b_cap,
                        log_alpha=hp["log_loss_alpha"], er_cm=b_cm, **lk)
                    va_loss += out_loss[0].item() * len(by)
            va_loss /= len(va_idx)
            if va_loss < best_val - 1e-5:
                best_val   = va_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience += 1
                if patience >= hp["early_stop"]:
                    break

    if swa_started:
        return swa_model.module
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(models, Xl, Xt, Xr, er_cm, has_cm, cm_approx,
            gii, phase_tr, idx, regime_idx, batch=512):
    preds = []
    for m in models:
        m.eval(); p = []
        for i in range(0, len(idx), batch):
            sl  = idx[i:i+batch]
            out = m(
                torch.tensor(Xl[sl],        dtype=torch.float32).to(DEVICE),
                torch.tensor(Xt[sl],        dtype=torch.float32).to(DEVICE),
                torch.tensor(Xr[sl],        dtype=torch.float32).to(DEVICE),
                torch.tensor(er_cm[sl],     dtype=torch.float32).to(DEVICE),
                torch.tensor(has_cm[sl],    dtype=torch.float32).to(DEVICE),
                torch.tensor(cm_approx[sl], dtype=torch.float32).to(DEVICE),
                torch.tensor(gii[sl],       dtype=torch.float32).to(DEVICE),
                torch.tensor(phase_tr[sl],  dtype=torch.float32).to(DEVICE),
                regime_idx,
            )
            p.append(out["pred"].cpu().numpy())
        preds.append(np.concatenate(p))
    return np.mean(preds, axis=0)


def build_strat_gss(train_idx, groups, n_splits=5, random_state=42):
    rng = np.random.RandomState(random_state)
    family_counts = Counter(groups[train_idx])
    large = sorted([f for f, n in family_counts.items() if n >= MIN_FAMILY_SIZE])
    ftr, fte = [], []
    for fam in large:
        idx  = train_idx[groups[train_idx] == fam]
        shuf = rng.permutation(idx)
        half = len(shuf) // 2
        ftr.append(shuf[:half]); fte.append(shuf[half:])
    forced_tr = np.concatenate(ftr) if ftr else np.array([], dtype=int)
    forced_te = np.concatenate(fte) if fte else np.array([], dtype=int)
    forced_all = np.concatenate([forced_tr, forced_te])
    gss_idx    = train_idx[~np.isin(train_idx, forced_all)]
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.20, random_state=random_state)
    splits = []
    for tr_r, te_r in gss.split(gss_idx, groups=groups[gss_idx]):
        splits.append((
            np.concatenate([gss_idx[tr_r], forced_tr]),
            np.concatenate([gss_idx[te_r], forced_te]),
        ))
    return splits


def style4(ax):
    for s in ax.spines.values():
        s.set_linewidth(1.0); s.set_visible(True)
    ax.tick_params(direction="in", length=4, width=1.0, top=True, right=True, labelsize=12)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Script 48 — CMLTRPCNNv77: Final retrain (formula-split best HP)")
    print(f"  trunk={HP['trunk_hidden']}  blk={HP['n_trunk_blocks']}  "
          f"lr={HP['lr']:.2e}  seeds={HP['n_seeds']}  epochs={HP['n_epochs']}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    (Xl, Xt, Xr, er_cm, has_cm, cm_approx,
     gii, phase_tr, y, groups, regime_idx,
     train_idx, calib_idx, formulas) = load_data()

    # ── Formula-split 5-fold ──────────────────────────────────────────────────
    print("\n[1/3] Formula-split 5-fold CV ...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s_frm, maes_frm, preds_frm = [], [], np.full(len(y), np.nan)

    for fold, (tr_rel, te_rel) in enumerate(kf.split(train_idx)):
        tr, te = train_idx[tr_rel], train_idx[te_rel]
        Xl_s, Xt_s, Xr_s, gii_s = scale_arrays(Xl, Xt, Xr, gii, tr)
        models = [train_one(Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                            gii_s, phase_tr, y, regime_idx, tr, te, HP, s)
                  for s in range(HP["n_seeds"])]
        pred = predict(models, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                       gii_s, phase_tr, te, regime_idx)
        preds_frm[te] = pred
        r2  = float(r2_score(y[te], pred))
        mae = float(mean_absolute_error(y[te], pred))
        r2s_frm.append(r2); maes_frm.append(mae)
        print(f"  Fold {fold+1}/5  R²={r2:.4f}  MAE={mae:.2f}")

    frm_r2_mean = float(np.mean(r2s_frm))
    frm_r2_std  = float(np.std(r2s_frm))
    frm_mae     = float(np.mean(maes_frm))
    print(f"  Formula-split: R² = {frm_r2_mean:.4f} ± {frm_r2_std:.4f}  MAE = {frm_mae:.2f}")

    # ── Strat-GSS 5-fold ─────────────────────────────────────────────────────
    print("\n[2/3] Stratified-GSS 5-fold CV ...")
    splits_gss = build_strat_gss(train_idx, groups)
    r2s_gss, maes_gss = [], []

    for fold, (tr, te) in enumerate(splits_gss):
        Xl_s, Xt_s, Xr_s, gii_s = scale_arrays(Xl, Xt, Xr, gii, tr)
        models = [train_one(Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                            gii_s, phase_tr, y, regime_idx, tr, te, HP, s)
                  for s in range(HP["n_seeds"])]
        pred = predict(models, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                       gii_s, phase_tr, te, regime_idx)
        r2  = float(r2_score(y[te], pred))
        mae = float(mean_absolute_error(y[te], pred))
        r2s_gss.append(r2); maes_gss.append(mae)
        print(f"  Fold {fold+1}/5  R²={r2:.4f}  MAE={mae:.2f}")

    gss_r2_mean = float(np.mean(r2s_gss))
    gss_r2_std  = float(np.std(r2s_gss))
    gss_mae     = float(np.mean(maes_gss))
    print(f"  Strat-GSS:     R² = {gss_r2_mean:.4f} ± {gss_r2_std:.4f}  MAE = {gss_mae:.2f}")

    # ── Full retrain on all train data ────────────────────────────────────────
    print("\n[3/3] Full retrain on all training data ...")
    Xl_s, Xt_s, Xr_s, gii_s = scale_arrays(Xl, Xt, Xr, gii, train_idx)
    final_models = [train_one(Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                              gii_s, phase_tr, y, regime_idx,
                              train_idx, calib_idx, HP, s)
                    for s in range(HP["n_seeds"])]
    pred_insample = predict(final_models, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                            gii_s, phase_tr, train_idx, regime_idx)
    r2_insample = float(r2_score(y[train_idx], pred_insample))
    print(f"  In-sample R² = {r2_insample:.4f}")

    torch.save({"models": [m.state_dict() for m in final_models],
                "hp": HP, "scaler_info": "StandardScaler on train_idx"},
               MODEL_OUT)
    print(f"  Model saved → {MODEL_OUT}")

    # ── Parity figure ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.grid": False, "axes.labelsize": 14, "axes.titlesize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })
    valid = ~np.isnan(preds_frm)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y[valid], preds_frm[valid], s=6, alpha=0.45,
               color="#2874A6", edgecolors="none")
    lim = (0, 145)
    ax.plot(lim, lim, "k--", linewidth=1.0, alpha=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"Measured $\varepsilon_r$")
    ax.set_ylabel(r"Predicted $\varepsilon_r$")
    ax.set_title(f"CMLTRPCNNv77  Formula-split  R²={frm_r2_mean:.4f} ± {frm_r2_std:.4f}",
                 loc="left", fontweight="bold")
    style4(ax)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "48_v77_parity.png")
    fig.savefig(fig_path, dpi=300); plt.close(fig)
    print(f"  Figure → {fig_path}")

    # ── JSON output ───────────────────────────────────────────────────────────
    out = {
        "model":        "CMLTRPCNNv77",
        "optuna_trial": 34,
        "hp":           HP,
        "formula_split": {
            "r2_mean":    frm_r2_mean,
            "r2_std":     frm_r2_std,
            "mae_mean":   frm_mae,
            "r2_per_fold": r2s_frm,
        },
        "strat_gss": {
            "r2_mean":    gss_r2_mean,
            "r2_std":     gss_r2_std,
            "mae_mean":   gss_mae,
            "r2_per_fold": r2s_gss,
        },
        "r2_insample":  r2_insample,
        "vs_v76": {
            "formula_split_delta": round(frm_r2_mean - 0.9288, 4),
            "strat_gss_delta":     round(gss_r2_mean - 0.8878, 4),
        },
    }
    res_path = os.path.join(RES_DIR, "48_cmltrv77_retrain.json")
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results → {res_path}")

    print("\n" + "=" * 70)
    print("SUMMARY — CMLTRPCNNv77")
    print("=" * 70)
    print(f"  Formula-split R² : {frm_r2_mean:.4f} ± {frm_r2_std:.4f}  "
          f"(v76: 0.9288, Δ={frm_r2_mean-0.9288:+.4f})")
    print(f"  Strat-GSS    R² : {gss_r2_mean:.4f} ± {gss_r2_std:.4f}  "
          f"(v76: 0.8878, Δ={gss_r2_mean-0.8878:+.4f})")
    print(f"  In-sample    R² : {r2_insample:.4f}")
    print(f"  Per-fold frm    : {[round(r,4) for r in r2s_frm]}")
    print(f"  Per-fold gss    : {[round(r,4) for r in r2s_gss]}")


if __name__ == "__main__":
    main()
