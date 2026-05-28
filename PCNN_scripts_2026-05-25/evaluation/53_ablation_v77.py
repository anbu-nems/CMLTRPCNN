"""
Script 53 — Ablation study on CMLTRPCNNv77

Proper ablation: each variant is RETRAINED from scratch without its branch
so remaining branches can compensate. Uses 3-fold Strat-GSS, 3 seeds.

Variants:
  full     — baseline v77
  no_cm    — er_cm=0, has_cm=0 (no physics anchor)
  no_lst   — delta_lst zeroed during training
  no_tilt  — delta_tilt zeroed during training
  no_res   — delta_res zeroed (sigma_conf=0) during training
  cm_only  — predict er_cm directly, no NN

Outputs:
  results/53_ablation_v77.json
  figures/53_ablation_v77.png
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.psrnn_mdpinn import CMLTRPCNNv71

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)
FIG_DIR  = os.path.join(ROOT, "figures"); os.makedirs(FIG_DIR, exist_ok=True)

LOG_FLOOR       = 5.0
MIN_FAMILY_SIZE = 50

HP = {
    "trunk_hidden":   512, "n_trunk_blocks": 2,
    "lst_hidden":     96,  "tilt_hidden":    32,
    "res_hidden":     192, "residual_scale": 80.0,
    "dropout":        0.1681982132911167,
    "weight_decay":   0.0008312905717274681,
    "mixup_alpha":    0.005005005146664418,
    "lr":             0.0017216692708802786,
    "swa_start_frac": 0.75,
    "swa_lr":         2.9047556806477378e-05,
    "lambda_lst":     0.0075683080032818454,
    "lambda_gii":     0.01171762763214901,
    "lambda_vca":     0.009401738824421016,
    "log_loss_alpha": 0.7572570965399923,
    "batch_size":     128,
    "n_epochs":       500,
    "early_stop":     100,
    "n_seeds":        3,   # faster for ablation
    "n_folds":        3,
}


# ── Ablation-aware model wrapper ───────────────────────────────────────────────
class AblationModel(nn.Module):
    """Wraps CMLTRPCNNv71 and zeros out the specified branch during forward."""

    def __init__(self, base: CMLTRPCNNv71, ablation: str):
        super().__init__()
        self.base     = base
        self.ablation = ablation

    def forward(self, lst_feats, tilt_feats, res_feats,
                er_cm, has_cm, cm_approx, gii_norm, phase_tr, regime_idx):
        # Input modifications for no_cm
        if self.ablation == "no_cm":
            er_cm  = torch.zeros_like(er_cm)
            has_cm = torch.zeros_like(has_cm)

        out = self.base(lst_feats, tilt_feats, res_feats,
                        er_cm, has_cm, cm_approx, gii_norm, phase_tr, regime_idx)
        out = dict(out)

        if self.ablation == "no_lst":
            out["delta_lst"] = torch.zeros_like(out["delta_lst"])
            er_ph = (er_cm + out["delta_tilt"] + out["delta_res"]).clamp(1.0, 600.0)
            er_fb = (self.base.fallback_mlp(res_feats).squeeze(-1) + 8.0).clamp(1.0, 600.0)
            out["pred"] = has_cm * er_ph + (1.0 - has_cm) * er_fb

        elif self.ablation == "no_tilt":
            out["delta_tilt"] = torch.zeros_like(out["delta_tilt"])
            er_ph = (er_cm + out["delta_lst"] + out["delta_res"]).clamp(1.0, 600.0)
            er_fb = (self.base.fallback_mlp(res_feats).squeeze(-1) + 8.0).clamp(1.0, 600.0)
            out["pred"] = has_cm * er_ph + (1.0 - has_cm) * er_fb

        elif self.ablation == "no_res":
            out["delta_res"] = torch.zeros_like(out["delta_res"])
            er_ph = (er_cm + out["delta_lst"] + out["delta_tilt"]).clamp(1.0, 600.0)
            er_fb = (self.base.fallback_mlp(res_feats).squeeze(-1) + 8.0).clamp(1.0, 600.0)
            out["pred"] = has_cm * er_ph + (1.0 - has_cm) * er_fb

        return out

    # Expose parameters from base for optimizer
    def parameters(self, recurse=True):
        return self.base.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.base.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.base.load_state_dict(*args, **kwargs)

    def train(self, mode=True):
        self.base.train(mode); return self

    def eval(self):
        self.base.eval(); return self


# ── Data ───────────────────────────────────────────────────────────────────────
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
    y         = df["epsilon_r"].values.astype(np.float32)
    groups    = df["chemistry_family"].values
    gii       = np.zeros(len(df), dtype=np.float32)
    phase_tr  = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
                if "phase_transition" in df.columns else np.zeros(len(df), np.float32)

    regime_names = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
    regime_idx   = [lcols.index(r) for r in regime_names if r in lcols]
    train_idx = np.array(calib_info["train_idx"])
    calib_idx = np.array(calib_info["calib_idx"])
    return (Xl, Xt, Xr, er_cm, has_cm, cm_approx,
            gii, phase_tr, y, groups, regime_idx, train_idx, calib_idx)


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
              y, regime_idx, tr_idx, va_idx, hp, seed, ablation):
    torch.manual_seed(seed); np.random.seed(seed)
    fam_w = np.ones(len(y), dtype=np.float32)

    base = CMLTRPCNNv71(
        n_lst=Xl.shape[1], n_tilt=Xt.shape[1], n_res=Xr.shape[1],
        trunk_hidden=hp["trunk_hidden"], n_trunk_blocks=hp["n_trunk_blocks"],
        lst_hidden=hp["lst_hidden"], tilt_hidden=hp["tilt_hidden"],
        res_hidden=hp["res_hidden"], residual_scale=hp["residual_scale"],
        dropout=hp["dropout"],
    ).to(DEVICE)
    model = AblationModel(base, ablation).to(DEVICE)

    opt   = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                               weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=hp["n_epochs"], eta_min=1e-6)
    swa_model = AveragedModel(base)
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
            bl, bt, br, b_cm, b_hcm, b_cap, b_gii, b_pt, by, _ = \
                [x.to(DEVICE) for x in batch]
            if hp.get("mixup_alpha", 0.0) > 0:
                lam  = float(np.random.beta(hp["mixup_alpha"], hp["mixup_alpha"]))
                idx2 = torch.randperm(bl.size(0), device=DEVICE)
                bl   = lam*bl  + (1-lam)*bl[idx2]
                bt   = lam*bt  + (1-lam)*bt[idx2]
                br   = lam*br  + (1-lam)*br[idx2]
                by   = lam*by  + (1-lam)*by[idx2]
                b_cm = lam*b_cm+ (1-lam)*b_cm[idx2]
            w   = 1.0 + 2.0 * by / 143.0; w = w / w.mean()
            out = model(bl, bt, br, b_cm, b_hcm, b_cap, b_gii, b_pt, regime_idx)
            loss, _ = v75_loss(out, by, b_gii, b_cap, sample_weights=w,
                               log_alpha=hp["log_loss_alpha"], er_cm=b_cm, **lk)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if epoch >= swa_start:
            swa_model.update_parameters(base); swa_sched.step(); swa_started = True
        else:
            sched.step()

        if not swa_started:
            model.eval(); va_loss = 0.0
            with torch.no_grad():
                for batch in va_ldr:
                    bl, bt, br, b_cm, b_hcm, b_cap, b_gii, b_pt, by, _ = \
                        [x.to(DEVICE) for x in batch]
                    out_loss = v75_loss(
                        model(bl, bt, br, b_cm, b_hcm, b_cap, b_gii, b_pt, regime_idx),
                        by, b_gii, b_cap,
                        log_alpha=hp["log_loss_alpha"], er_cm=b_cm, **lk)
                    va_loss += out_loss[0].item() * len(by)
            va_loss /= len(va_idx)
            if va_loss < best_val - 1e-5:
                best_val   = va_loss
                best_state = {k: v.cpu().clone() for k, v in base.state_dict().items()}
                patience   = 0
            else:
                patience += 1
                if patience >= hp["early_stop"]:
                    break

    if swa_started:
        wrapped = AblationModel(swa_model.module, ablation)
        return wrapped
    if best_state:
        base.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(models, Xl, Xt, Xr, er_cm, has_cm, cm_approx,
            gii, phase_tr, idx, regime_idx, ablation, batch=512):
    if ablation == "cm_only":
        return er_cm[idx].copy()
    preds = []
    for m in models:
        m.eval(); p = []
        for i in range(0, len(idx), batch):
            sl  = idx[i:i+batch]
            out = m(torch.tensor(Xl[sl], dtype=torch.float32).to(DEVICE),
                    torch.tensor(Xt[sl], dtype=torch.float32).to(DEVICE),
                    torch.tensor(Xr[sl], dtype=torch.float32).to(DEVICE),
                    torch.tensor(er_cm[sl], dtype=torch.float32).to(DEVICE),
                    torch.tensor(has_cm[sl], dtype=torch.float32).to(DEVICE),
                    torch.tensor(cm_approx[sl], dtype=torch.float32).to(DEVICE),
                    torch.tensor(gii[sl], dtype=torch.float32).to(DEVICE),
                    torch.tensor(phase_tr[sl], dtype=torch.float32).to(DEVICE),
                    regime_idx)
            p.append(out["pred"].cpu().numpy())
        preds.append(np.concatenate(p))
    return np.mean(preds, axis=0)


def build_strat_gss(train_idx, groups, n_splits=3, random_state=42):
    rng = np.random.RandomState(random_state)
    family_counts = Counter(groups[train_idx])
    large = sorted([f for f, n in family_counts.items() if n >= MIN_FAMILY_SIZE])
    ftr, fte = [], []
    for fam in large:
        idx  = train_idx[groups[train_idx] == fam]
        shuf = rng.permutation(idx)
        half = len(shuf) // 2
        ftr.append(shuf[:half]); fte.append(shuf[half:])
    forced_tr  = np.concatenate(ftr) if ftr else np.array([], dtype=int)
    forced_te  = np.concatenate(fte) if fte else np.array([], dtype=int)
    forced_all = np.concatenate([forced_tr, forced_te])
    gss_idx    = train_idx[~np.isin(train_idx, forced_all)]
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.20, random_state=random_state)
    splits = []
    for tr_r, te_r in gss.split(gss_idx, groups=groups[gss_idx]):
        splits.append((np.concatenate([gss_idx[tr_r], forced_tr]),
                       np.concatenate([gss_idx[te_r], forced_te])))
    return splits


def run_ablation(ablation, Xl, Xt, Xr, er_cm, has_cm, cm_approx,
                 gii, phase_tr, y, groups, regime_idx, train_idx, splits, hp):
    r2s = []
    for fold, (tr, te) in enumerate(splits):
        Xl_s, Xt_s, Xr_s, gii_s = scale_arrays(Xl, Xt, Xr, gii, tr)

        if ablation == "cm_only":
            pred = er_cm[te].copy()
        else:
            models = [train_one(Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                                gii_s, phase_tr, y, regime_idx, tr, te, hp, s, ablation)
                      for s in range(hp["n_seeds"])]
            pred = predict(models, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                           gii_s, phase_tr, te, regime_idx, ablation)

        r2 = float(r2_score(y[te], pred))
        mae = float(mean_absolute_error(y[te], pred))
        r2s.append(r2)
        print(f"    Fold {fold+1}/{len(splits)}  R²={r2:.4f}  MAE={mae:.2f}")
    return float(np.mean(r2s)), float(np.std(r2s))


def main():
    print("=" * 70)
    print("Script 53 — Ablation study on CMLTRPCNNv77 (3-fold GSS, 3 seeds)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    (Xl, Xt, Xr, er_cm, has_cm, cm_approx,
     gii, phase_tr, y, groups, regime_idx, train_idx, calib_idx) = load_data()

    splits = build_strat_gss(train_idx, groups, n_splits=HP["n_folds"])

    ablations = ["full", "no_cm", "no_lst", "no_tilt", "no_res", "cm_only"]
    results   = {}

    for abl in ablations:
        print(f"\n  [{abl}]")
        r2_mean, r2_std = run_ablation(
            abl, Xl, Xt, Xr, er_cm, has_cm, cm_approx,
            gii, phase_tr, y, groups, regime_idx, train_idx, splits, HP)
        results[abl] = {"r2_mean": r2_mean, "r2_std": r2_std}
        print(f"    → R² = {r2_mean:.4f} ± {r2_std:.4f}")

    full_r2 = results["full"]["r2_mean"]
    r2_drops = {k: round(full_r2 - v["r2_mean"], 4)
                for k, v in results.items() if k != "full"}

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.grid": False, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
    })
    labels = {
        "full":    "Full model",
        "no_cm":   "No CM anchor",
        "no_lst":  "No LST branch",
        "no_tilt": "No Tilt branch",
        "no_res":  "No Residual",
        "cm_only": "CM only (baseline)",
    }
    keys   = ["full", "no_cm", "no_lst", "no_tilt", "no_res", "cm_only"]
    r2vals = [results[k]["r2_mean"] for k in keys]
    r2errs = [results[k]["r2_std"]  for k in keys]
    colors = ["#2874A6" if k == "full" else
              "#E74C3C" if results[k]["r2_mean"] < 0 else
              "#E67E22" for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    ypos = np.arange(len(keys))
    bars = ax.barh(ypos, r2vals, xerr=r2errs, color=colors, height=0.55,
                   edgecolor="white", linewidth=0.5, capsize=3)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(full_r2, color="#2874A6", linewidth=1.0, linestyle=":", alpha=0.7)
    for i, (k, v, e) in enumerate(zip(keys, r2vals, r2errs)):
        label = f"{v:.3f}"
        if k != "full":
            drop = full_r2 - v
            label += f"  (Δ={-drop:+.3f})"
        ax.text(max(v, 0) + 0.01, i, label, va="center", fontsize=9, color="#333")
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[k] for k in keys])
    ax.set_xlabel("Strat-GSS R²")
    ax.set_title("CMLTRPCNNv77 — Branch ablation (3-fold GSS, 3 seeds)",
                 loc="left", fontweight="bold")
    ax.set_xlim(-1.1, 1.15)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    ax.tick_params(direction="in", length=3)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "53_ablation_v77.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Figure → {fig_path}")

    # ── JSON ──────────────────────────────────────────────────────────────────
    out = {
        "model":    "CMLTRPCNNv77",
        "protocol": f"{HP['n_folds']}-fold Strat-GSS, {HP['n_seeds']} seeds, {HP['n_epochs']} epochs",
        "full_model_r2": full_r2,
        "ablations": results,
        "r2_drops":  r2_drops,
    }
    res_path = os.path.join(RES_DIR, "53_ablation_v77.json")
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results → {res_path}")

    print()
    print("=" * 70)
    print("SUMMARY — Ablation (v77)")
    print("=" * 70)
    print(f"  Full model R²  : {full_r2:.4f}")
    for k, drop in r2_drops.items():
        print(f"  {k:10s}  ΔR² = {-drop:+.4f}  (R²={results[k]['r2_mean']:.4f})")


if __name__ == "__main__":
    main()
