"""
Script 66 — StratifiedGroupKFold CV with 5-seed ensemble (v7.7 protocol)
========================================================================

Re-runs the StratifiedGroupKFold protocol from script 65 with the full
5-seed ensemble used in the published PCNN v7.7. Per fold: 5 PCNN models
are trained with different seeds (0..4); per-sample prediction = mean
across seeds, σ_conf (seed_std) = std across seeds.

Expected vs the 1-seed run (R² = 0.819 ± 0.060):
  • fold-mean R² → 0.84–0.86 range
  • fold-std    → ≈ 0.04 (≈30% tighter)

Outputs:
  results/66_stratgroupkfold_5seed.json
  results/66_stratgroupkfold_5seed_predictions.csv
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from src.models.psrnn_mdpinn import CMLTRPCNNv71

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)

LOG_FLOOR = 5.0

# Hard wall-clock cap (task constraint: 45 min)
HARD_STOP_SECONDS = 45 * 60

# Trial 97 HPs (CLAUDE.md / 48_cmltrv77_retrain.json) — same as script 65
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
    "lambda_gii":        0.03,
    "lambda_vca":        0.0446992472658469,
    "log_loss_alpha":    0.505198011404557,
    "batch_size":        128,
    "n_epochs":          500,
    "early_stop":        100,
    "n_seeds":           5,                          # ← v7.7 ensemble
}


# ── Data loading (identical to script 65) ───────────────────────────────────
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
    gii       = df["GII"].fillna(0.0).values.astype(np.float32) \
                if "GII" in df.columns else np.zeros(len(df), np.float32)
    phase_tr  = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
                if "phase_transition" in df.columns else np.zeros(len(df), np.float32)
    formulas  = df["formula"].values if "formula" in df.columns else np.array([""] * len(df))

    regime_names = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
    regime_idx   = [lcols.index(r) for r in regime_names if r in lcols]

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


# ── Loss (identical to script 65) ───────────────────────────────────────────
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
def predict_per_seed(models, D_scaled, idx, batch=512):
    """Return (n_seeds, n_idx) array of predictions — one row per seed."""
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
    return np.stack(preds, axis=0)        # (n_seeds, n_idx)


def run_protocol_5seed(D, splits, protocol_name, t0):
    y          = D["y"]
    train_idx  = D["train_idx"]
    formulas   = D["formulas"]
    regime_lab = D["regime_labels"]

    preds_mean  = np.full(len(y), np.nan)
    preds_std   = np.full(len(y), np.nan)
    fold_id     = np.full(len(y), -1, dtype=int)
    folds_meta  = []
    r2s, maes   = [], []
    seed_std_mean_per_fold = []

    for fold, (tr, te) in enumerate(splits, 1):
        elapsed = time.time() - t0
        print(f"[{protocol_name}] Fold {fold}/5  elapsed={elapsed/60:.1f} min  "
              f"n_train={len(tr)}  n_test={len(te)}", flush=True)
        if elapsed > HARD_STOP_SECONDS:
            print(f"[{protocol_name}] HARD STOP at fold {fold} — partial result",
                  flush=True)
            break

        # Re-scale per fold using training data only
        Xl_s, Xt_s, Xr_s, gii_s = scale_arrays(D["Xl"], D["Xt"], D["Xr"], D["gii"], tr)
        D_s = dict(D); D_s.update(Xl=Xl_s, Xt=Xt_s, Xr=Xr_s, gii=gii_s)

        # Train 5 PCNN models with seeds 0..4
        models = []
        for s in range(HP["n_seeds"]):
            t_s = time.time()
            m = train_one(D_s, tr, te, HP, seed=s)
            models.append(m)
            print(f"  [{protocol_name}] Fold {fold} seed {s}/{HP['n_seeds']-1} "
                  f"trained ({(time.time()-t_s)/60:.2f} min, total "
                  f"{(time.time()-t0)/60:.1f} min)", flush=True)

        # Per-seed predictions on test fold
        per_seed = predict_per_seed(models, D_s, te)          # (5, n_te)
        pred     = per_seed.mean(axis=0)
        sd       = per_seed.std(axis=0)

        preds_mean[te] = pred
        preds_std[te]  = sd
        fold_id[te]    = fold

        r2  = float(r2_score(y[te], pred))
        mae = float(mean_absolute_error(y[te], pred))
        r2s.append(r2); maes.append(mae)
        seed_std_mean_per_fold.append(float(sd.mean()))
        folds_meta.append({
            "fold":              fold,
            "n_train":           int(len(tr)),
            "n_test":            int(len(te)),
            "r2":                r2,
            "mae":               mae,
            "seed_std_mean":     float(sd.mean()),
            "seed_std_median":   float(np.median(sd)),
            "seed_std_max":      float(sd.max()),
        })
        print(f"[{protocol_name}] Fold {fold}/5  R²={r2:.4f}  MAE={mae:.2f}  "
              f"seed_std_mean={sd.mean():.2f}  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    covered = ~np.isnan(preds_mean)
    n_cov   = int(covered.sum())
    r2_pool = float(r2_score(y[covered], preds_mean[covered])) if n_cov > 1 else float("nan")
    mae_pool= float(mean_absolute_error(y[covered], preds_mean[covered])) if n_cov > 1 else float("nan")

    out = {
        "protocol":              protocol_name,
        "n_folds":               len(r2s),
        "n_seeds":               HP["n_seeds"],
        "r2_per_fold":           r2s,
        "mae_per_fold":          maes,
        "r2_mean":               float(np.mean(r2s)) if r2s else float("nan"),
        "r2_std":                float(np.std(r2s))  if r2s else float("nan"),
        "mae_mean":              float(np.mean(maes)) if maes else float("nan"),
        "r2_pooled":             r2_pool,
        "mae_pooled":            mae_pool,
        "n_test_pooled":         n_cov,
        "seed_std_mean_per_fold": seed_std_mean_per_fold,
        "seed_std_mean_overall": float(np.nanmean(preds_std)),
        "seed_std_median_overall": float(np.nanmedian(preds_std)),
        "fold_meta":             folds_meta,
        "hp":                    HP,
    }

    rows = []
    for i in train_idx:
        if covered[i]:
            rows.append({
                "idx":           int(i),
                "formula":       str(formulas[i]),
                "regime":        str(regime_lab[i]),
                "fold":          int(fold_id[i]),
                "er_measured":   float(y[i]),
                "er_predicted":  float(preds_mean[i]),
                "seed_std":      float(preds_std[i]),
            })
    df_preds = pd.DataFrame(rows)
    return out, df_preds


def main():
    t0 = time.time()
    print("=" * 72)
    print("Script 66 — StratifiedGroupKFold CV (5-seed ensemble, v7.7 protocol)")
    print(f"  Device: {DEVICE}  n_seeds={HP['n_seeds']}  n_epochs={HP['n_epochs']}")
    print(f"  HP source: Trial 97 (CLAUDE.md / 48_cmltrv77_retrain.json)")
    print(f"  Hard wall-clock cap: {HARD_STOP_SECONDS/60:.0f} min")
    print("=" * 72, flush=True)

    D = load_data()
    train_idx     = D["train_idx"]
    groups_tr     = D["groups"][train_idx]
    reg_labels_tr = D["regime_labels"][train_idx]
    print(f"  n_train={len(train_idx)}  unique_families={len(set(groups_tr))}",
          flush=True)

    # Same splits as script 65 (random_state=42 reproduces them exactly)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = [(train_idx[tr], train_idx[te])
              for tr, te in sgkf.split(train_idx, y=reg_labels_tr, groups=groups_tr)]

    res, df_pred = run_protocol_5seed(D, splits, "StratifiedGroupKFold_5seed", t0)

    p_json = os.path.join(RES_DIR, "66_stratgroupkfold_5seed.json")
    p_csv  = os.path.join(RES_DIR, "66_stratgroupkfold_5seed_predictions.csv")
    with open(p_json, "w") as f: json.dump(res, f, indent=2)
    df_pred.to_csv(p_csv, index=False)

    print("\n" + "=" * 72)
    print("SUMMARY  (StratifiedGroupKFold · 5-seed ensemble)")
    print("=" * 72)
    print(f"  Wall clock:      {(time.time()-t0)/60:.1f} min")
    print(f"  fold-R²:         {res['r2_mean']:.4f} ± {res['r2_std']:.4f}")
    print(f"  pooled R²/MAE:   {res['r2_pooled']:.4f} / {res['mae_pooled']:.2f}")
    print(f"  per-fold R²:     {[round(x,4) for x in res['r2_per_fold']]}")
    print(f"  per-fold MAE:    {[round(x,2) for x in res['mae_per_fold']]}")
    print(f"  per-fold seed_std_mean: "
          f"{[round(x,2) for x in res['seed_std_mean_per_fold']]}")
    print(f"  saved {p_json}")
    print(f"  saved {p_csv}")


if __name__ == "__main__":
    main()
