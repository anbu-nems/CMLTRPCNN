"""
Script 65 — CV protocol comparison: GroupKFold vs StratifiedGroupKFold
======================================================================

Re-runs PCNN cross-validation under two new splitters that give 100% test-set
coverage (vs current Strat-GSS which only covers 734/1188).

Strategy A: 1 seed per fold (vs final model's 5) to fit the 90-min budget.
Hyperparameters: identical to v77 Trial 97 (per CLAUDE.md / 48_cmltrv77_retrain.json).

Outputs:
  results/65_cv_groupkfold.json
  results/65_cv_groupkfold_predictions.csv
  results/65_cv_stratifiedgroupkfold.json
  results/65_cv_stratifiedgroupkfold_predictions.csv
"""
import sys, os, json, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from src.models.psrnn_mdpinn import CMLTRPCNNv71

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)

LOG_FLOOR = 5.0

# Hard wall-clock guard — fall back to Strategy B if we are about to overshoot.
HARD_STOP_SECONDS = 75 * 60   # leave 15 min for plotting

# Trial 97 HPs (CLAUDE.md / 48_cmltrv77_retrain.json) ─────────────────────────
HP = {
    "trunk_hidden":      384,
    "n_trunk_blocks":    1,
    "lst_hidden":        96,
    "tilt_hidden":       64,
    "res_hidden":        256,
    "residual_scale":    44.89677799261463,
    "dropout":           0.1671791544255202,
    "weight_decay":      2.3014840266014856e-05,
    "mixup_alpha":       0.025183675402199783,
    "lr":                0.001107319855148978,
    "swa_start_frac":    0.7645722154519098,
    "swa_lr":            2.5129733678594355e-05,
    "lambda_lst":        0.0014836181196749463,
    "lambda_gii":        0.03,                       # re-tuned with real GII
    "lambda_vca":        0.0446992472658469,
    "log_loss_alpha":    0.505198011404557,
    "batch_size":        128,
    "n_epochs":          500,
    "early_stop":        100,
    "n_seeds":           1,                          # Strategy A
}


# ── Data loading ────────────────────────────────────────────────────────────
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
    y         = df["epsilon_r"].values.astype(np.float32) \
                if "epsilon_r" in df.columns else df["DC"].values.astype(np.float32)
    groups    = df["chemistry_family"].values
    # real GII if present, else zeros
    gii       = df["GII"].fillna(0.0).values.astype(np.float32) \
                if "GII" in df.columns else np.zeros(len(df), np.float32)
    phase_tr  = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
                if "phase_transition" in df.columns else np.zeros(len(df), np.float32)
    formulas  = df["formula"].values if "formula" in df.columns else np.array([""] * len(df))

    regime_names = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
    regime_idx   = [lcols.index(r) for r in regime_names if r in lcols]

    # Decode regime label per sample for StratifiedGroupKFold stratification key
    reg_labels = np.full(len(df), "II", dtype=object)
    for name in regime_names:
        if name in df.columns:
            mask = df[name].fillna(0).values.astype(int) == 1
            reg_labels[mask] = name.replace("regime_", "")

    train_idx = np.array(calib_info["train_idx"])

    return dict(
        Xl=Xl, Xt=Xt, Xr=Xr, er_cm=er_cm, has_cm=has_cm, cm_approx=cm_approx,
        gii=gii, phase_tr=phase_tr, y=y, groups=groups,
        regime_idx=regime_idx, regime_labels=reg_labels,
        train_idx=train_idx, formulas=formulas,
    )


# ── Loss ────────────────────────────────────────────────────────────────────
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


def train_one(D, tr_idx, va_idx, hp, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xl, Xt, Xr = D["Xl"], D["Xt"], D["Xr"]
    er_cm, has_cm, cm_approx = D["er_cm"], D["has_cm"], D["cm_approx"]
    gii, phase_tr, y = D["gii"], D["phase_tr"], D["y"]
    regime_idx = D["regime_idx"]
    fam_w = np.ones(len(y), dtype=np.float32)

    model = CMLTRPCNNv71(
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
                b_cm = lam * b_cm + (1 - lam) * b_cm[idx2]
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
def predict(models, D_scaled, idx, batch=512):
    Xl, Xt, Xr = D_scaled["Xl"], D_scaled["Xt"], D_scaled["Xr"]
    er_cm, has_cm, cm_approx = D_scaled["er_cm"], D_scaled["has_cm"], D_scaled["cm_approx"]
    gii, phase_tr = D_scaled["gii"], D_scaled["phase_tr"]
    regime_idx = D_scaled["regime_idx"]
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


def run_protocol(D, splits, protocol_name, t0):
    """Run 5-fold CV for one splitter; return dict + per-sample preds vector."""
    y          = D["y"]
    train_idx  = D["train_idx"]
    formulas   = D["formulas"]
    regime_lab = D["regime_labels"]

    preds_full = np.full(len(y), np.nan)
    folds_meta = []
    r2s, maes  = [], []
    n_train_l, n_test_l = [], []

    for fold, (tr, te) in enumerate(splits, 1):
        elapsed = time.time() - t0
        print(f"[{protocol_name}] Fold {fold}/5  elapsed={elapsed/60:.1f} min  "
              f"n_train={len(tr)}  n_test={len(te)}", flush=True)
        if elapsed > HARD_STOP_SECONDS:
            print(f"[{protocol_name}] HARD STOP at fold {fold} — partial result", flush=True)
            break

        # Re-scale per fold using training data only
        Xl_s, Xt_s, Xr_s, gii_s = scale_arrays(D["Xl"], D["Xt"], D["Xr"], D["gii"], tr)
        D_s = dict(D); D_s.update(Xl=Xl_s, Xt=Xt_s, Xr=Xr_s, gii=gii_s)

        models = [train_one(D_s, tr, te, HP, seed=s) for s in range(HP["n_seeds"])]
        pred   = predict(models, D_s, te)
        preds_full[te] = pred

        r2  = float(r2_score(y[te], pred))
        mae = float(mean_absolute_error(y[te], pred))
        r2s.append(r2); maes.append(mae)
        n_train_l.append(int(len(tr))); n_test_l.append(int(len(te)))
        folds_meta.append({"fold": fold, "n_train": int(len(tr)),
                           "n_test": int(len(te)), "r2": r2, "mae": mae})
        print(f"[{protocol_name}] Fold {fold}/5  R²={r2:.4f}  MAE={mae:.2f}  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    # Pooled metrics over samples that have a prediction
    covered = ~np.isnan(preds_full)
    n_cov   = int(covered.sum())
    r2_pool = float(r2_score(y[covered], preds_full[covered])) if n_cov > 1 else float("nan")
    mae_pool= float(mean_absolute_error(y[covered], preds_full[covered])) if n_cov > 1 else float("nan")

    out = {
        "protocol":      protocol_name,
        "n_folds":       len(r2s),
        "n_seeds":       HP["n_seeds"],
        "r2_per_fold":   r2s,
        "mae_per_fold":  maes,
        "r2_mean":       float(np.mean(r2s)) if r2s else float("nan"),
        "r2_std":        float(np.std(r2s))  if r2s else float("nan"),
        "mae_mean":      float(np.mean(maes)) if maes else float("nan"),
        "r2_pooled":     r2_pool,
        "mae_pooled":    mae_pool,
        "n_test_pooled": n_cov,
        "fold_meta":     folds_meta,
        "hp":            HP,
    }
    # Per-sample predictions CSV (only training-pool rows)
    rows = []
    for i in train_idx:
        if covered[i]:
            rows.append({
                "idx":          int(i),
                "formula":      str(formulas[i]),
                "regime":       str(regime_lab[i]),
                "er_measured":  float(y[i]),
                "er_predicted": float(preds_full[i]),
            })
    df_preds = pd.DataFrame(rows)
    return out, df_preds


def main():
    t0 = time.time()
    print("=" * 72)
    print("Script 65 — GroupKFold vs StratifiedGroupKFold CV  (Strategy A)")
    print(f"  Device: {DEVICE}  n_seeds={HP['n_seeds']}  n_epochs={HP['n_epochs']}")
    print(f"  HP source: Trial 97 (CLAUDE.md / 48_cmltrv77_retrain.json)")
    print(f"  Hard wall-clock limit: {HARD_STOP_SECONDS/60:.0f} min")
    print("=" * 72, flush=True)

    D = load_data()
    train_idx     = D["train_idx"]
    groups_tr     = D["groups"][train_idx]
    reg_labels_tr = D["regime_labels"][train_idx]
    y_tr_int      = (D["y"][train_idx] // 20).astype(int)  # crude stratification helper
    print(f"  n_train={len(train_idx)}  unique_families={len(set(groups_tr))}",
          flush=True)

    # ── GroupKFold ────────────────────────────────────────────────────────────
    print("\n[1/2] GroupKFold (5 folds, family-grouped) ...", flush=True)
    gkf = GroupKFold(n_splits=5)
    splits_gk = [(train_idx[tr], train_idx[te])
                 for tr, te in gkf.split(train_idx, groups=groups_tr)]
    res_gk, df_gk = run_protocol(D, splits_gk, "GroupKFold", t0)

    # Save now in case StratifiedGroupKFold runs over budget
    p1 = os.path.join(RES_DIR, "65_cv_groupkfold.json")
    p2 = os.path.join(RES_DIR, "65_cv_groupkfold_predictions.csv")
    with open(p1, "w") as f: json.dump(res_gk, f, indent=2)
    df_gk.to_csv(p2, index=False)
    print(f"  saved {p1}\n  saved {p2}", flush=True)

    # ── StratifiedGroupKFold ─────────────────────────────────────────────────
    print("\n[2/2] StratifiedGroupKFold (5 folds, family-grouped, regime-stratified) ...",
          flush=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits_sgk = [(train_idx[tr], train_idx[te])
                  for tr, te in sgkf.split(train_idx, y=reg_labels_tr, groups=groups_tr)]
    res_sgk, df_sgk = run_protocol(D, splits_sgk, "StratifiedGroupKFold", t0)

    p3 = os.path.join(RES_DIR, "65_cv_stratifiedgroupkfold.json")
    p4 = os.path.join(RES_DIR, "65_cv_stratifiedgroupkfold_predictions.csv")
    with open(p3, "w") as f: json.dump(res_sgk, f, indent=2)
    df_sgk.to_csv(p4, index=False)
    print(f"  saved {p3}\n  saved {p4}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Wall clock: {(time.time()-t0)/60:.1f} min")
    for r in (res_gk, res_sgk):
        print(f"\n  {r['protocol']}: fold-R² = {r['r2_mean']:.4f} ± {r['r2_std']:.4f}")
        print(f"    pooled R² = {r['r2_pooled']:.4f}  pooled MAE = {r['mae_pooled']:.2f}  "
              f"n_test={r['n_test_pooled']}")
        print(f"    per-fold R²: {[round(x,4) for x in r['r2_per_fold']]}")
        print(f"    per-fold n_test: {[m['n_test'] for m in r['fold_meta']]}")


if __name__ == "__main__":
    main()
