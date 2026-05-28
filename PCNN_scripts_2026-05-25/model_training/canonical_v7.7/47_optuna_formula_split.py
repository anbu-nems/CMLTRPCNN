"""
Script 47 — Optuna HP Search: Pure Formula-Split objective

Goal: push formula-split R² above 0.93 (currently 0.929 on v76).

Key differences from script 36:
  - ALPHA_FRM = 1.0  (pure formula-split, no strat-GSS in objective)
  - Architecture expanded: trunk 256-512, blocks 2-4
  - Strat-GSS loop dropped during search → ~40% faster per trial
  - Warm-started with v76 best HP (Trial 52 from script 36)

Best HP → results/47_optuna_frm_best.json
Study DB → results/47_optuna_frm.db  (resume-able)

Usage:
    cd /Users/anbu/Desktop/PIML/piml_ceramic
    python scripts/47_optuna_formula_split.py
    python scripts/47_optuna_formula_split.py --trials 80
"""
import sys, os, argparse, warnings, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, TensorDataset

from src.models.psrnn_mdpinn import CMLTRPCNN

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)

N_TRIALS   = 60
N_SEEDS    = 1       # 1 seed for Optuna speed; final retraining (script 48) uses 5+
N_EPOCHS   = 350
EARLY_STOP = 70
BATCH_SIZE = 128
N_CV_FOLDS = 5      # 5-fold formula-split (matches final eval protocol)

LOG_FLOOR  = 5.0

DB_PATH  = os.path.join(RES_DIR, "47_optuna_frm.db")
BEST_OUT = os.path.join(RES_DIR, "47_optuna_frm_best.json")

# ── Data cache ────────────────────────────────────────────────────────────────
_DATA = None

def get_data():
    global _DATA
    if _DATA is not None:
        return _DATA

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
    groups    = df["chemistry_family"].values if "chemistry_family" in df.columns \
                else np.zeros(len(df), dtype=object)
    gii       = np.zeros(len(df), dtype=np.float32)
    phase_tr  = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
                if "phase_transition" in df.columns else np.zeros(len(df), np.float32)

    regime_names = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
    regime_idx   = [lcols.index(r) for r in regime_names if r in lcols]

    train_idx = np.array(calib_info["train_idx"])

    _DATA = (Xl, Xt, Xr, er_cm, has_cm, cm_approx,
             gii, phase_tr, y, groups, regime_idx, train_idx)
    return _DATA


# ── Loss (identical to script 36/37) ─────────────────────────────────────────
def v75_loss(out, y, gii, cm_approx_flag, sample_weights=None,
             log_alpha=0.5, lambda_lst=0.01, lambda_gii=0.05,
             lambda_vca=0.01, er_cm=None):
    pred     = out["pred"]
    w        = sample_weights if sample_weights is not None else torch.ones_like(y)
    loss_log = (w * (torch.log(pred.clamp(min=LOG_FLOOR)) -
                     torch.log(y.clamp(min=LOG_FLOOR))).pow(2)).mean()
    loss_mse = (w * (pred - y).pow(2)).mean()
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


def train_fold(Xl, Xt, Xr, er_cm, has_cm, cm_approx, gii, phase_tr,
               y, regime_idx, tr_idx, va_idx, hp, seed=0):
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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=1e-6)
    swa_model = AveragedModel(model)
    swa_sched = SWALR(opt, swa_lr=hp["swa_lr"], anneal_epochs=10)
    swa_start = int(hp["swa_start_frac"] * N_EPOCHS)
    swa_started = False

    arrays  = [Xl, Xt, Xr, er_cm, has_cm, cm_approx, gii, phase_tr, y, fam_w]
    tr_ldr  = make_loader(arrays, tr_idx, BATCH_SIZE, shuffle=True)
    va_ldr  = make_loader(arrays, va_idx, BATCH_SIZE, shuffle=False)
    lk      = {k: hp[k] for k in ("lambda_lst", "lambda_gii", "lambda_vca")}

    best_val, best_state, patience = 1e9, None, 0

    for epoch in range(N_EPOCHS):
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
                               log_alpha=hp.get("log_loss_alpha", 0.5), er_cm=b_cm, **lk)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if epoch >= swa_start:
            swa_model.update_parameters(model); swa_sched.step(); swa_started = True
        else:
            sched.step()

        if not swa_started:
            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for batch in va_ldr:
                    bl, bt, br, b_cm, b_hcm, b_cap, b_gii, b_pt, by, b_fw = \
                        [x.to(DEVICE) for x in batch]
                    out_loss = v75_loss(
                        model(bl, bt, br, b_cm, b_hcm, b_cap, b_gii, b_pt, regime_idx),
                        by, b_gii, b_cap,
                        log_alpha=hp.get("log_loss_alpha", 0.5), er_cm=b_cm, **lk)
                    va_loss += out_loss[0].item() * len(by)
            va_loss /= len(va_idx)
            if va_loss < best_val - 1e-5:
                best_val   = va_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience += 1
                if patience >= EARLY_STOP:
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


# ── Optuna objective — pure formula-split ─────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    hp = {
        # Architecture — expanded upper bound vs script 36
        "trunk_hidden":   trial.suggest_categorical("trunk_hidden",   [256, 384, 512]),
        "n_trunk_blocks": trial.suggest_int("n_trunk_blocks",         2, 4),
        "lst_hidden":     trial.suggest_categorical("lst_hidden",     [64, 96, 128, 192]),
        "tilt_hidden":    trial.suggest_categorical("tilt_hidden",    [32, 64, 96]),
        "res_hidden":     trial.suggest_categorical("res_hidden",     [128, 192, 256, 384]),
        "residual_scale": trial.suggest_float("residual_scale",       20.0, 80.0, step=10.0),
        # Regularisation
        "dropout":        trial.suggest_float("dropout",              0.05, 0.35),
        "weight_decay":   trial.suggest_float("weight_decay",         1e-5, 5e-3, log=True),
        "mixup_alpha":    trial.suggest_float("mixup_alpha",          0.0,  0.40),
        # Optimisation
        "lr":             trial.suggest_float("lr",                   2e-4, 2e-3, log=True),
        # SWA
        "swa_start_frac": trial.suggest_float("swa_start_frac",       0.55, 0.80, step=0.05),
        "swa_lr":         trial.suggest_float("swa_lr",               5e-6, 1e-4, log=True),
        # Loss weights
        "lambda_lst":     trial.suggest_float("lambda_lst",           5e-4, 0.05, log=True),
        "lambda_gii":     trial.suggest_float("lambda_gii",           0.01, 0.20, log=True),
        "lambda_vca":     trial.suggest_float("lambda_vca",           5e-4, 0.02, log=True),
        "log_loss_alpha": trial.suggest_float("log_loss_alpha",       0.2,  0.8),
    }

    (Xl, Xt, Xr, er_cm, has_cm, cm_approx,
     gii, phase_tr, y, groups, regime_idx, train_idx) = get_data()

    kf   = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
    r2s  = []
    for tr_rel, te_rel in kf.split(train_idx):
        tr, te = train_idx[tr_rel], train_idx[te_rel]
        Xl_s, Xt_s, Xr_s, gii_s = scale_arrays(Xl, Xt, Xr, gii, tr)
        mdls = [train_fold(Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                           gii_s, phase_tr, y, regime_idx, tr, te, hp, s)
                for s in range(N_SEEDS)]
        pred = predict(mdls, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                       gii_s, phase_tr, te, regime_idx)
        r2s.append(float(r2_score(y[te], pred)))

    r2_frm = float(np.mean(r2s))
    trial.set_user_attr("r2_frm_folds", r2s)
    trial.set_user_attr("r2_frm_std",   float(np.std(r2s)))

    print(f"  Trial {trial.number:3d}  frm={r2_frm:.4f} ± {np.std(r2s):.4f}  "
          f"folds={[round(r,4) for r in r2s]}  "
          f"th={hp['trunk_hidden']}  blk={hp['n_trunk_blocks']}  "
          f"lr={hp['lr']:.1e}  do={hp['dropout']:.2f}")
    return r2_frm


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int, default=N_TRIALS)
    parser.add_argument("--study",   type=str, default="cmltrv7_frm_optuna")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Wall-clock timeout in seconds (optional)")
    args = parser.parse_args()

    print("=" * 70)
    print("Optuna HP Search — Pure Formula-Split objective (script 47)")
    print(f"  Target   : formula-split R² > 0.93  (v76 baseline = 0.9288)")
    print(f"  Trials   : {args.trials}   Seeds/fold : {N_SEEDS}   Epochs : {N_EPOCHS}")
    print(f"  Device   : {DEVICE}")
    print(f"  DB       : {DB_PATH}")
    print(f"  Arch     : trunk [256,384,512]  blocks [2-4]  (expanded vs script 36)")
    print("=" * 70)

    get_data()

    study = optuna.create_study(
        study_name=args.study,
        storage=f"sqlite:///{DB_PATH}",
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=12),
        load_if_exists=True,
    )

    # Warm-start: v76 best HP (trunk=256, blocks=2) as first trial
    if len(study.trials) == 0:
        study.enqueue_trial({
            "trunk_hidden": 256, "n_trunk_blocks": 2,
            "lst_hidden": 128, "tilt_hidden": 96, "res_hidden": 256,
            "residual_scale": 40.0,
            "dropout": 0.135, "weight_decay": 2e-4,
            "mixup_alpha": 0.158, "lr": 8.76e-4,
            "swa_start_frac": 0.65, "swa_lr": 1.35e-5,
            "lambda_lst": 1.87e-3, "lambda_gii": 5.72e-2, "lambda_vca": 4.26e-3,
            "log_loss_alpha": 0.536,
        })
        # Also try larger architecture variants up front
        study.enqueue_trial({
            "trunk_hidden": 384, "n_trunk_blocks": 2,
            "lst_hidden": 128, "tilt_hidden": 96, "res_hidden": 256,
            "residual_scale": 40.0,
            "dropout": 0.135, "weight_decay": 2e-4,
            "mixup_alpha": 0.158, "lr": 8.76e-4,
            "swa_start_frac": 0.65, "swa_lr": 1.35e-5,
            "lambda_lst": 1.87e-3, "lambda_gii": 5.72e-2, "lambda_vca": 4.26e-3,
            "log_loss_alpha": 0.536,
        })
        study.enqueue_trial({
            "trunk_hidden": 512, "n_trunk_blocks": 2,
            "lst_hidden": 128, "tilt_hidden": 96, "res_hidden": 256,
            "residual_scale": 40.0,
            "dropout": 0.135, "weight_decay": 2e-4,
            "mixup_alpha": 0.158, "lr": 8.76e-4,
            "swa_start_frac": 0.65, "swa_lr": 1.35e-5,
            "lambda_lst": 1.87e-3, "lambda_gii": 5.72e-2, "lambda_vca": 4.26e-3,
            "log_loss_alpha": 0.536,
        })

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout,
                   show_progress_bar=False)

    best = study.best_trial
    print("\n" + "=" * 70)
    print("OPTUNA COMPLETE")
    print(f"  Best formula-split R²: {best.value:.4f}")
    print(f"  Per-fold R²          : {best.user_attrs.get('r2_frm_folds')}")
    print(f"  Std                  : {best.user_attrs.get('r2_frm_std', 0):.4f}")
    print(f"  Best params          : {best.params}")
    print("=" * 70)

    out = {
        "best_frm_r2":   best.value,
        "best_params":   best.params,
        "r2_frm_folds":  best.user_attrs.get("r2_frm_folds"),
        "r2_frm_std":    best.user_attrs.get("r2_frm_std"),
        "n_trials":      len(study.trials),
        "objective":     "pure formula-split",
        "baseline_v76":  0.9288,
    }
    with open(BEST_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nBest HP → {BEST_OUT}")
    print("Next step: python scripts/48_retrain_frm_best.py")

    print("\nTop 5 trials:")
    top5 = sorted(study.trials, key=lambda t: t.value or -1, reverse=True)[:5]
    for t in top5:
        print(f"  #{t.number:3d}  frm={t.value:.4f} ± {t.user_attrs.get('r2_frm_std',0):.4f}  "
              f"th={t.params.get('trunk_hidden')}  blk={t.params.get('n_trunk_blocks')}  "
              f"lr={t.params.get('lr'):.1e}  do={t.params.get('dropout'):.2f}")


if __name__ == "__main__":
    main()
