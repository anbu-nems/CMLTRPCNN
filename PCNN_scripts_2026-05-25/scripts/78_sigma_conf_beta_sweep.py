"""
σ_conf β ±50% full sensitivity sweep.

9 perturbations × 5 seeds = 45 training runs at v77 HP (500 epochs each).
Override the model's class-level GATE_BETA before each model instantiation;
canonical checkpoint cmltrv77_final.pt is NEVER touched.

For each perturbation: train 5 seeds → ensemble inference on the full 1,304-
composition corpus → per-A-site mean f_LST, σ_conf, |δ_res|, in-AD%.

Output: /Users/anbu/Desktop/sigma_conf_sweep/results/sweep_full.json
"""
import sys, os, time, json, re, warnings
sys.path.insert(0, "/Users/anbu/Desktop/PIML/piml_ceramic")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from src.models.psrnn_mdpinn import CMLTRPCNN

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PROC_DIR = "/Users/anbu/Desktop/PIML/piml_ceramic/data/processed"
OUT_DIR  = "/Users/anbu/Desktop/sigma_conf_sweep/results"
os.makedirs(OUT_DIR, exist_ok=True)

# Full v77 HP (script 48)
HP = {
    "trunk_hidden":     512, "n_trunk_blocks": 2,
    "lst_hidden":       96,  "tilt_hidden":   32,  "res_hidden": 192,
    "residual_scale":   80.0, "dropout":        0.16819821329,
    "weight_decay":     8.31290571e-4,
    "mixup_alpha":      5.00500514e-3,
    "lr":               1.72166927e-3,
    "swa_start_frac":   0.75, "swa_lr": 2.90475568e-5,
    "lambda_lst":       7.5683080e-3,
    "lambda_gii":       1.17176276e-2,
    "lambda_vca":       9.40173882e-3,
    "log_loss_alpha":   0.7572570965,
    "batch_size":       128,
    "n_epochs":         500,
    "early_stop":       100,
}
LOG_FLOOR = 1e-6

# 9 β perturbation configurations
BETA_BASE = [-2.0, 3.0, 2.0, 2.0]
PERTURBATIONS = {
    "baseline":  [-2.0, 3.0, 2.0, 2.0],
    "b0_low":    [-1.0, 3.0, 2.0, 2.0],   # β₀ × 0.5
    "b0_high":   [-3.0, 3.0, 2.0, 2.0],   # β₀ × 1.5
    "b1_low":    [-2.0, 1.5, 2.0, 2.0],   # β₁ × 0.5
    "b1_high":   [-2.0, 4.5, 2.0, 2.0],   # β₁ × 1.5
    "b2_low":    [-2.0, 3.0, 1.0, 2.0],   # β₂ × 0.5
    "b2_high":   [-2.0, 3.0, 3.0, 2.0],   # β₂ × 1.5
    "b3_low":    [-2.0, 3.0, 2.0, 1.0],   # β₃ × 0.5
    "b3_high":   [-2.0, 3.0, 2.0, 3.0],   # β₃ × 1.5
}
N_SEEDS = 5

FAMILIES = ["Pb", "Ca", "La", "Sr", "Ba"]


def primary_a_site(formula):
    m = re.findall(r"[A-Z][a-z]?", str(formula))
    return m[0] if m else "?"


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
    gii       = np.zeros(len(df), dtype=np.float32)
    phase_tr  = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
                if "phase_transition" in df.columns else np.zeros(len(df), np.float32)
    formulas  = df["formula"].values if "formula" in df.columns else np.array([""]*len(df))

    regime_names = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
    regime_idx   = [lcols.index(r) for r in regime_names if r in lcols]

    train_idx = np.array(calib_info["train_idx"])
    calib_idx = np.array(calib_info["calib_idx"])
    return (Xl, Xt, Xr, er_cm, has_cm, cm_approx, gii, phase_tr,
            y, regime_idx, train_idx, calib_idx, formulas)


def scale(Xl, Xt, Xr, train_idx):
    return (StandardScaler().fit(Xl[train_idx]).transform(Xl).astype(np.float32),
            StandardScaler().fit(Xt[train_idx]).transform(Xt).astype(np.float32),
            StandardScaler().fit(Xr[train_idx]).transform(Xr).astype(np.float32))


def loss_fn(out, y, gii, cm_approx_flag, w, log_alpha, lst_l, gii_l, vca_l, er_cm):
    pred = out["pred"]
    loss_log = (w * (torch.log(pred.clamp(min=LOG_FLOOR)) -
                     torch.log(y.clamp(min=LOG_FLOOR))).pow(2)).mean()
    loss_mse = (w * (pred - y).pow(2)).mean()
    loss_data = log_alpha * loss_log + (1.0 - log_alpha) * loss_mse
    loss_lst = lst_l * F.relu(out["delta_lst"] - 5.0 * er_cm.abs()).pow(2).mean()
    loss_gii = gii_l * (F.relu(er_cm - pred) * (gii < 0.3).float()).pow(2).mean()
    loss_vca = vca_l * (out["delta_res"].pow(2) * (cm_approx_flag > 0.5).float()).mean()
    return loss_data + loss_lst + loss_gii + loss_vca


def train_one_seed(arrays, regime_idx, tr_idx, beta_vec, seed):
    Xl, Xt, Xr, er_cm, has_cm, cm_approx, gii, phase_tr, y = arrays
    torch.manual_seed(seed); np.random.seed(seed)

    # ── Override GATE_BETA on the class BEFORE instantiation ───────────────────
    CMLTRPCNN.GATE_BETA = torch.tensor(beta_vec, dtype=torch.float32)

    model = CMLTRPCNN(
        n_lst=Xl.shape[1], n_tilt=Xt.shape[1], n_res=Xr.shape[1],
        trunk_hidden=HP["trunk_hidden"], n_trunk_blocks=HP["n_trunk_blocks"],
        lst_hidden=HP["lst_hidden"], tilt_hidden=HP["tilt_hidden"],
        res_hidden=HP["res_hidden"], residual_scale=HP["residual_scale"],
        dropout=HP["dropout"],
    ).to(DEVICE)
    # Ensure the buffer/tensor on the model carries the perturbed β
    model.GATE_BETA = CMLTRPCNN.GATE_BETA.to(DEVICE)

    opt   = torch.optim.AdamW(model.parameters(), lr=HP["lr"],
                              weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HP["n_epochs"], eta_min=1e-6)
    swa   = AveragedModel(model)
    swa_sched = SWALR(opt, swa_lr=HP["swa_lr"], anneal_epochs=10)
    swa_start = int(HP["swa_start_frac"] * HP["n_epochs"])
    swa_started = False

    tr_arr = [torch.tensor(a[tr_idx]) for a in arrays]
    tr_ldr = DataLoader(TensorDataset(*tr_arr), batch_size=HP["batch_size"], shuffle=True)

    for epoch in range(HP["n_epochs"]):
        model.train()
        for batch in tr_ldr:
            bl, bt, br, b_cm, b_hcm, b_cap, b_gii, b_pt, by = [x.to(DEVICE) for x in batch]
            if HP["mixup_alpha"] > 0:
                lam = float(np.random.beta(HP["mixup_alpha"], HP["mixup_alpha"]))
                idx2 = torch.randperm(bl.size(0), device=DEVICE)
                bl, bt, br = lam*bl + (1-lam)*bl[idx2], lam*bt + (1-lam)*bt[idx2], lam*br + (1-lam)*br[idx2]
                by, b_cm  = lam*by + (1-lam)*by[idx2], lam*b_cm + (1-lam)*b_cm[idx2]
            w = 1.0 + 2.0 * by / 143.0; w = w / w.mean()
            out = model(bl, bt, br, b_cm, b_hcm, b_cap, b_gii, b_pt, regime_idx)
            loss = loss_fn(out, by, b_gii, b_cap, w, HP["log_loss_alpha"],
                           HP["lambda_lst"], HP["lambda_gii"], HP["lambda_vca"], b_cm)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if epoch >= swa_start:
            swa.update_parameters(model); swa_sched.step(); swa_started = True
        else:
            sched.step()

    final_model = swa.module if swa_started else model
    # Make sure SWA model carries the right GATE_BETA
    final_model.GATE_BETA = torch.tensor(beta_vec, dtype=torch.float32).to(DEVICE)
    return final_model


def inference_all(model, arrays, regime_idx):
    """Inference over the full corpus, returning per-composition outputs."""
    Xl, Xt, Xr, er_cm, has_cm, cm_approx, gii, phase_tr, y = arrays
    model.eval()
    with torch.no_grad():
        out = model(
            torch.tensor(Xl).to(DEVICE),
            torch.tensor(Xt).to(DEVICE),
            torch.tensor(Xr).to(DEVICE),
            torch.tensor(er_cm).to(DEVICE),
            torch.tensor(has_cm).to(DEVICE),
            torch.tensor(cm_approx).to(DEVICE),
            torch.tensor(gii).to(DEVICE),
            torch.tensor(phase_tr).to(DEVICE),
            regime_idx,
        )
    return {k: v.cpu().numpy() for k, v in out.items()}


def aggregate_per_asite(pred_ens, sigma_conf_ens, delta_lst_ens, delta_res_ens,
                       er_cm, has_cm, formulas, y, calib_idx):
    """Per-A-site statistics on valid-CM compositions (where f_LST is defined)."""
    asite = np.array([primary_a_site(f) for f in formulas])
    has_valid_cm = (has_cm > 0.5) & (er_cm > 0)
    per_asite = {}
    for fam in FAMILIES:
        mask = (asite == fam) & has_valid_cm
        n = int(mask.sum())
        if n == 0:
            per_asite[fam] = {"n": 0}
            continue
        f_lst = delta_lst_ens[mask] / np.clip(er_cm[mask], 1e-3, None)
        per_asite[fam] = {
            "n":              n,
            "mean_f_lst":     float(f_lst.mean()),
            "mean_sigma_conf": float(sigma_conf_ens[mask].mean()),
            "mean_abs_delta_res": float(np.abs(delta_res_ens[mask]).mean()),
            "in_ad_fraction": float((sigma_conf_ens[mask] < 0.35).mean()),
        }
    # holdout R²
    r2 = float(r2_score(y[calib_idx], pred_ens[calib_idx]))
    return per_asite, r2


def main():
    t_main = time.time()
    print(f"=== σ_conf β ±50% full sweep — device={DEVICE} ===")
    print(f"  {len(PERTURBATIONS)} perturbations × {N_SEEDS} seeds = "
          f"{len(PERTURBATIONS)*N_SEEDS} runs")
    print(f"  Expected wall-clock: ~{len(PERTURBATIONS)*N_SEEDS*1.1:.0f} min "
          f"(at 1.1 min/seed extrapolated from 50-epoch test)")
    print()

    (Xl, Xt, Xr, er_cm, has_cm, cm_approx, gii, phase_tr,
     y, regime_idx, tr_idx, va_idx, formulas) = load_data()
    Xl, Xt, Xr = scale(Xl, Xt, Xr, tr_idx)
    arrays = [Xl, Xt, Xr, er_cm, has_cm, cm_approx, gii, phase_tr, y]
    print(f"  Data: n={len(y)}, train={len(tr_idx)}, calib={len(va_idx)}\n")

    all_results = {
        "definition": "σ_conf gate β coefficient ±50% sensitivity sweep (v77 HP, 5-seed ensembles)",
        "baseline_beta": BETA_BASE,
        "perturbations":  PERTURBATIONS,
        "n_seeds":        N_SEEDS,
        "n_epochs":       HP["n_epochs"],
        "runs":           {},
    }

    for pname, beta_vec in PERTURBATIONS.items():
        t_p = time.time()
        print(f"━━━ Perturbation '{pname}' β = {beta_vec} ━━━")
        seed_outputs = []
        for s in range(N_SEEDS):
            t_s = time.time()
            model = train_one_seed(arrays, regime_idx, tr_idx, beta_vec, seed=s)
            out = inference_all(model, arrays, regime_idx)
            seed_outputs.append(out)
            print(f"    seed {s}: trained in {time.time()-t_s:.0f}s")
        # Ensemble = mean across seeds for pred / sigma_conf / delta_lst / delta_res
        pred_ens       = np.mean([o["pred"]        for o in seed_outputs], axis=0)
        sigma_conf_ens = np.mean([o["sigma_conf"]  for o in seed_outputs], axis=0)
        delta_lst_ens  = np.mean([o["delta_lst"]   for o in seed_outputs], axis=0)
        delta_res_ens  = np.mean([o["delta_res"]   for o in seed_outputs], axis=0)

        per_asite, holdout_r2 = aggregate_per_asite(
            pred_ens, sigma_conf_ens, delta_lst_ens, delta_res_ens,
            er_cm, has_cm, formulas, y, va_idx)

        all_results["runs"][pname] = {
            "beta":        beta_vec,
            "holdout_r2":  holdout_r2,
            "per_asite":   per_asite,
        }
        print(f"    Holdout R² = {holdout_r2:.3f}")
        print(f"    Per-A-site f_LST: " + ", ".join(
            [f"{a}={per_asite[a].get('mean_f_lst', float('nan')):.2f}" for a in FAMILIES if per_asite[a].get('n', 0) > 0]
        ))
        print(f"    Time: {time.time()-t_p:.0f}s\n")

        # Save incremental progress (don't lose work if interrupted)
        with open(os.path.join(OUT_DIR, "sweep_full.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    total = time.time() - t_main
    print(f"━━━ ALL DONE — total {total/60:.1f} min ━━━")
    print(f"  Output: {OUT_DIR}/sweep_full.json")


if __name__ == "__main__":
    main()
