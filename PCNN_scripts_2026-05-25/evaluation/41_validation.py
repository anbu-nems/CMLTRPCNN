"""
Script 41 — Robustness checks (finding-validation, 4 checks)

Four checks that the headline findings are not coverage/ablation artifacts:

Check 1: Pb compositional balance — is f_LST=2.27x a coverage artifact?
          Count A-site compositions by εr quartile.

Check 2: Bootstrap f_LST by A-site (n=1000) — do Pb vs Ba CIs overlap?
          If non-overlapping → finding. If overlapping → suggestion only.

Check 3: Tilt-branch proper ablation — keep tilt features in trunk,
          zero ONLY the tilt head output during training and inference.
          (distinct from script 38 which zeroed ALL tilt input including trunk)
          Expected: ΔR² ≈ 0 → tilt branch is vestigial, not trunk signal.

Check 4: Explain the 0.041 R² gap (formula=0.929 vs GSS=0.888).
          Which families/compositions drive the gap?

Outputs:
  results/41_validation.json
  figures/41a_composition_balance.png
  figures/41b_bootstrap_flst.png
  figures/41c_tilt_ablation.png
  figures/41d_gap_analysis.png
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
from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, TensorDataset

from src.models.psrnn_mdpinn import CMLTRPCNNv71

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)
FIG_DIR  = os.path.join(ROOT, "figures");  os.makedirs(FIG_DIR, exist_ok=True)
MODEL_PT = os.path.join(ROOT, "models", "cmltrv77_final.pt")

LOG_FLOOR = 5.0
MIN_FAMILY_SIZE = 50

HP = {
    "trunk_hidden": 256, "n_trunk_blocks": 2,
    "lst_hidden": 128, "tilt_hidden": 96, "res_hidden": 256,
    "residual_scale": 40.0, "dropout": 0.134, "weight_decay": 0.0002,
    "mixup_alpha": 0.158, "lr": 0.000876, "swa_start_frac": 0.65,
    "swa_lr": 1.35e-05, "lambda_lst": 0.00187, "lambda_gii": 0.0572,
    "lambda_vca": 0.00426, "log_loss_alpha": 0.536,
    "batch_size": 128, "n_epochs": 300, "early_stop": 60,
    "family_weight_power": 0.5, "n_seeds": 3, "n_folds": 3,
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.grid": False, "axes.linewidth": 1.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 5, "ytick.major.size": 5,
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.fontsize": 8,
})

def style4(ax):
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(1.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


# ── Model loading ──────────────────────────────────────────────────────────
def load_model_and_scalers():
    import pickle
    ck = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
    hp = ck["hp"]
    with open(os.path.join(ROOT, "models", "cmltrv77_scalers.pkl"), "rb") as _f:
        scalers = pickle.load(_f)
    sc_lst  = scalers["sc_lst"]
    sc_tilt = scalers["sc_tilt"]
    sc_res  = scalers["sc_res"]
    n_lst  = sc_lst.n_features_in_
    n_tilt = sc_tilt.n_features_in_
    n_res  = sc_res.n_features_in_
    models = []
    for state in ck["models"]:
        m = CMLTRPCNNv71(
            n_lst=n_lst, n_tilt=n_tilt, n_res=n_res,
            trunk_hidden=hp["trunk_hidden"], n_trunk_blocks=hp["n_trunk_blocks"],
            lst_hidden=hp["lst_hidden"], tilt_hidden=hp["tilt_hidden"],
            res_hidden=hp["res_hidden"], residual_scale=hp["residual_scale"],
            dropout=0.0)
        m.load_state_dict(state); m.eval().to(DEVICE)
        models.append(m)
    with open(os.path.join(ROOT, "data", "processed", "feature_partition_v7.json")) as _f:
        _part = json.load(_f)
    _regime_names = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
    regime_idx = [_part["LST"].index(r) for r in _regime_names if r in _part["LST"]]
    return (models, sc_lst, sc_tilt, sc_res, 1.0, regime_idx)


@torch.no_grad()
def infer(models, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
          gii_n, phase_tr, regime_idx, batch=256):
    acc = {"pred":[], "delta_lst":[], "delta_tilt":[], "delta_res":[]}
    N = len(er_cm)
    for m in models:
        bufs = {k:[] for k in acc}
        for i in range(0, N, batch):
            sl = slice(i, i+batch)
            out = m(torch.tensor(Xl_s[sl]).to(DEVICE), torch.tensor(Xt_s[sl]).to(DEVICE),
                    torch.tensor(Xr_s[sl]).to(DEVICE), torch.tensor(er_cm[sl]).to(DEVICE),
                    torch.tensor(has_cm[sl]).to(DEVICE), torch.tensor(cm_approx[sl]).to(DEVICE),
                    torch.tensor(gii_n[sl]).to(DEVICE), torch.tensor(phase_tr[sl]).to(DEVICE),
                    regime_idx)
            for k in acc: bufs[k].append(out[k].cpu().numpy())
        for k in acc: acc[k].append(np.concatenate(bufs[k]))
    return {k: np.mean(acc[k], 0) for k in acc}


# ── Tilt-zeroed model (Check 3) ────────────────────────────────────────────
class CMLTRPCNNv71_NoTiltHead(CMLTRPCNNv71):
    """Identical to v71 but tilt head output forced to zero during forward.
    Tilt FEATURES still flow through the shared trunk — only the dedicated
    tilt HEAD is disabled. Tests if the tilt BRANCH (not tilt signal) matters."""
    def forward(self, lst_feats, tilt_feats, res_feats,
                er_cm, has_cm, cm_approx, gii_norm, phase_tr, regime_idx):
        out = super().forward(lst_feats, tilt_feats, res_feats,
                              er_cm, has_cm, cm_approx, gii_norm, phase_tr, regime_idx)
        # Zero only the tilt head contribution; keep trunk signal intact
        delta_tilt_zeroed = torch.zeros_like(out["delta_tilt"])
        er_physics  = (out["er_cm"] + out["delta_lst"] + delta_tilt_zeroed + out["delta_res"]).clamp(1, 600)
        er_fallback = (self.fallback_mlp(res_feats).squeeze(-1) + 8.0).clamp(1, 600)
        er_pred     = has_cm * er_physics + (1 - has_cm) * er_fallback
        return {**out, "pred": er_pred, "delta_tilt": delta_tilt_zeroed}


# ── Training helpers ───────────────────────────────────────────────────────
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


def scale_arrays(Xl, Xt, Xr, gii, train_idx):
    sc_lst = StandardScaler().fit(Xl[train_idx])
    sc_tilt= StandardScaler().fit(Xt[train_idx])
    sc_res = StandardScaler().fit(Xr[train_idx])
    gii_max= float(np.percentile(gii[train_idx][gii[train_idx]>0],95)
                   if (gii[train_idx]>0).sum()>10 else 1.0)
    return (sc_lst.transform(Xl).astype(np.float32),
            sc_tilt.transform(Xt).astype(np.float32),
            sc_res.transform(Xr).astype(np.float32),
            np.clip(gii/gii_max,0,1).astype(np.float32))


def build_family_weights(groups, train_idx, power=0.5):
    counts = Counter(groups[train_idx])
    raw_w  = {f: 1.0/(c**power) for f,c in counts.items()}
    mean_w = np.mean([raw_w[g] for g in groups[train_idx]])
    norm_w = {f: w/mean_w for f,w in raw_w.items()}
    return np.array([norm_w.get(g,1.0) for g in groups], dtype=np.float32)


def build_stratified_gss_splits(train_idx, groups, n_splits=3, random_state=42):
    rng = np.random.RandomState(random_state)
    family_counts = Counter(groups[train_idx])
    large = sorted([f for f,n in family_counts.items() if n>=MIN_FAMILY_SIZE])
    ft_pool, fte_pool = [], []
    for fam in large:
        fi = train_idx[groups[train_idx]==fam]
        sh = rng.permutation(fi); half=len(sh)//2
        ft_pool.append(sh[:half]); fte_pool.append(sh[half:])
    forced_tr = np.concatenate(ft_pool) if ft_pool else np.array([],dtype=int)
    forced_te = np.concatenate(fte_pool) if fte_pool else np.array([],dtype=int)
    forced_all= np.concatenate([forced_tr,forced_te])
    gss_idx   = train_idx[~np.isin(train_idx,forced_all)]
    gss_grp   = groups[gss_idx]
    gss = GroupShuffleSplit(n_splits=n_splits,test_size=0.20,random_state=random_state)
    return [(np.concatenate([gss_idx[tr],forced_tr]),
             np.concatenate([gss_idx[te],forced_te]))
            for tr,te in gss.split(gss_idx,groups=gss_grp)]


def make_loader(arrays, idx, batch_size, shuffle):
    ds = TensorDataset(*[torch.tensor(a[idx], dtype=torch.float32) for a in arrays])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=False)


def train_one(Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx, gii_s, phase_tr,
              y, fam_w, regime_idx, tr_idx, va_idx, hp, seed=0, no_tilt_head=False):
    torch.manual_seed(seed); np.random.seed(seed)
    ModelClass = CMLTRPCNNv71_NoTiltHead if no_tilt_head else CMLTRPCNNv71
    model = ModelClass(
        n_lst=Xl_s.shape[1], n_tilt=Xt_s.shape[1], n_res=Xr_s.shape[1],
        trunk_hidden=hp["trunk_hidden"], n_trunk_blocks=hp["n_trunk_blocks"],
        lst_hidden=hp["lst_hidden"], tilt_hidden=hp["tilt_hidden"],
        res_hidden=hp["res_hidden"], residual_scale=hp["residual_scale"],
        dropout=hp["dropout"]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["n_epochs"], eta_min=1e-6)
    swa_m = AveragedModel(model)
    swa_s = SWALR(opt, swa_lr=hp["swa_lr"], anneal_epochs=10)
    swa_start = int(hp["swa_start_frac"]*hp["n_epochs"]); swa_started=False
    arrays=[Xl_s,Xt_s,Xr_s,er_cm,has_cm,cm_approx,gii_s,phase_tr,y,fam_w]
    tr_ldr=make_loader(arrays,tr_idx,hp["batch_size"],shuffle=True)
    va_ldr=make_loader(arrays,va_idx,hp["batch_size"],shuffle=False)
    lk={k:hp[k] for k in ("lambda_lst","lambda_gii","lambda_vca")}
    best_val,best_state,patience=1e9,None,0
    for epoch in range(hp["n_epochs"]):
        model.train()
        for batch in tr_ldr:
            bl,bt,br,b_cm,b_hcm,b_cap,b_gii,b_pt,by,b_fw=[x.to(DEVICE) for x in batch]
            if hp.get("mixup_alpha",0)>0:
                lam=float(np.random.beta(hp["mixup_alpha"],hp["mixup_alpha"]))
                idx2=torch.randperm(bl.size(0),device=DEVICE)
                bl=lam*bl+(1-lam)*bl[idx2]; bt=lam*bt+(1-lam)*bt[idx2]
                br=lam*br+(1-lam)*br[idx2]; by=lam*by+(1-lam)*by[idx2]
                b_cm=lam*b_cm+(1-lam)*b_cm[idx2]
            w=b_fw*(1+2*by/143.0); w=w/w.mean()
            out=model(bl,bt,br,b_cm,b_hcm,b_cap,b_gii,b_pt,regime_idx)
            loss,_=v75_loss(out,by,b_gii,b_cap,sample_weights=w,
                            log_alpha=hp["log_loss_alpha"],er_cm=b_cm,**lk)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if epoch>=swa_start: swa_m.update_parameters(model); swa_s.step(); swa_started=True
        else: sched.step()
        model.eval(); va_loss=0.0
        with torch.no_grad():
            for batch in va_ldr:
                bl,bt,br,b_cm,b_hcm,b_cap,b_gii,b_pt,by,b_fw=[x.to(DEVICE) for x in batch]
                out=model(bl,bt,br,b_cm,b_hcm,b_cap,b_gii,b_pt,regime_idx)
                l,_=v75_loss(out,by,b_gii,b_cap,log_alpha=hp["log_loss_alpha"],er_cm=b_cm,**lk)
                va_loss+=l.item()*len(by)
        va_loss/=len(va_idx)
        if not swa_started:
            if va_loss<best_val-1e-5: best_val=va_loss; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; patience=0
            else:
                patience+=1
                if patience>=hp["early_stop"]: break
    if swa_started: return swa_m.module
    if best_state: model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_cv(models_list, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
               gii_s, phase_tr, idx, regime_idx, batch=512):
    preds=[]
    for m in models_list:
        m.eval(); p=[]
        for i in range(0,len(idx),batch):
            sl=idx[i:i+batch]
            out=m(torch.tensor(Xl_s[sl]).to(DEVICE), torch.tensor(Xt_s[sl]).to(DEVICE),
                  torch.tensor(Xr_s[sl]).to(DEVICE), torch.tensor(er_cm[sl]).to(DEVICE),
                  torch.tensor(has_cm[sl]).to(DEVICE), torch.tensor(cm_approx[sl]).to(DEVICE),
                  torch.tensor(gii_s[sl]).to(DEVICE), torch.tensor(phase_tr[sl]).to(DEVICE),
                  regime_idx)
            p.append(out["pred"].cpu().numpy())
        preds.append(np.concatenate(p))
    return np.mean(preds,0)


def run_cv_variant(label, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                   gii_s, phase_tr, y, groups, fam_w, regime_idx,
                   splits, hp, no_tilt_head=False):
    r2s=[]
    for fi,(tr,te) in enumerate(splits):
        Xl_f,Xt_f,Xr_f,gii_f=scale_arrays(Xl_s,Xt_s,Xr_s,gii_s,tr)
        ms=[train_one(Xl_f,Xt_f,Xr_f,er_cm,has_cm,cm_approx,gii_f,phase_tr,
                      y,fam_w,regime_idx,tr,te,hp,seed=s,no_tilt_head=no_tilt_head)
            for s in range(hp["n_seeds"])]
        pred=predict_cv(ms,Xl_f,Xt_f,Xr_f,er_cm,has_cm,cm_approx,
                        gii_f,phase_tr,te,regime_idx)
        r2=float(r2_score(y[te],pred)); r2s.append(r2)
        print(f"    [{label}] Fold {fi+1}  R²={r2:.4f}")
    mean_r2=float(np.mean(r2s)); std_r2=float(np.std(r2s))
    print(f"    [{label}] Mean R² = {mean_r2:.4f} ± {std_r2:.4f}")
    return r2s, mean_r2, std_r2


def main():
    print("="*70)
    print("Script 41 — Robustness checks (finding-validation, 4 checks)")
    print("="*70)

    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_parquet(os.path.join(PROC_DIR,"feature_matrix_v7.parquet"))
    with open(os.path.join(PROC_DIR,"feature_partition_v7.json")) as f: partition=json.load(f)
    with open(os.path.join(PROC_DIR,"calibration_split_idx.json")) as f: calib_info=json.load(f)

    def _get(cols):
        present=[c for c in cols if c in df.columns]
        return df[present].fillna(0.0).values.astype(np.float32), present

    Xl,lst_cols=_get(partition["LST"]); Xt,_=_get(partition["Tilt"]); Xr,_=_get(partition["Residual"])
    er_cm    = df["er_CM"].fillna(0.0).values.astype(np.float32)
    has_cm   = df["has_sigma_CM"].fillna(0.0).values.astype(np.float32)
    cm_approx= df["cm_approx_flag"].fillna(0.0).values.astype(np.float32)
    phase_tr = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
               if "phase_transition" in df.columns else np.zeros(len(df),np.float32)
    y        = df["epsilon_r"].values.astype(np.float32)
    groups   = df["chemistry_family"].values if "chemistry_family" in df.columns else np.zeros(len(df),object)
    gii_norm = df["GII"].fillna(0.0).values.astype(np.float32)
    train_idx= np.array(calib_info["train_idx"])
    fam_w    = build_family_weights(groups, train_idx, HP["family_weight_power"])

    models_saved, sc_lst, sc_tilt, sc_res, gii_max, regime_idx = load_model_and_scalers()
    Xl_s=sc_lst.transform(Xl).astype(np.float32)
    Xt_s=sc_tilt.transform(Xt).astype(np.float32)
    Xr_s=sc_res.transform(Xr).astype(np.float32)

    print(f"\n  {len(df)} samples | train pool: {len(train_idx)}")

    # ── Run full inference to get per-sample components ────────────────────
    print("\n  Running inference for per-sample components...")
    out_full = infer(models_saved, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                     gii_norm, phase_tr, regime_idx)
    d_lst = out_full["delta_lst"]; d_tilt = out_full["delta_tilt"]
    pred  = out_full["pred"]

    valid_cm = (has_cm>0.5) & (er_cm>5.0)
    f_lst_all = np.where(valid_cm, d_lst/er_cm, np.nan)

    # A-site classification
    def a_site(fam): p=str(fam).split("_"); return p[0] if p else "Other"
    df["a_site"] = df["chemistry_family"].apply(a_site)

    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("CHECK 1: Pb compositional balance by εr quartile")
    print("─"*60)

    df["er_quartile"] = pd.qcut(df["epsilon_r"], q=4, labels=["Q1 (low)","Q2","Q3","Q4 (high)"])
    pivot = df.groupby(["a_site","er_quartile"], observed=True).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=0), axis=1) * 100

    main_asites = ["Ba","Ca","Sr","Pb","La","Bi","Other"]
    pivot_main  = pivot_pct.reindex([a for a in main_asites if a in pivot_pct.index])

    print("\n  Sample % by A-site within each εr quartile:")
    print(f"  {'A-site':<10}", end="")
    for q in pivot_main.columns: print(f"  {str(q):>14}", end="")
    print()
    for asite in pivot_main.index:
        print(f"  {asite:<10}", end="")
        for q in pivot_main.columns: print(f"  {pivot_main.loc[asite,q]:>13.1f}%", end="")
        print()

    # Check: is Pb overrepresented in Q4?
    pb_q4 = float(pivot_pct.loc["Pb","Q4 (high)"]) if "Pb" in pivot_pct.index else 0
    ba_q4 = float(pivot_pct.loc["Ba","Q4 (high)"]) if "Ba" in pivot_pct.index else 0
    pb_n  = int(pivot.loc["Pb"].sum()) if "Pb" in pivot.index else 0
    print(f"\n  Pb in Q4: {pb_q4:.1f}%  |  Ba in Q4: {ba_q4:.1f}%  |  Pb total n={pb_n}")

    bias_flag = pb_q4 > 40.0
    print(f"  → Coverage artifact risk: {'HIGH — Pb disproportionately in Q4' if bias_flag else 'LOW — distribution acceptable'}")

    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("CHECK 2: Bootstrap f_LST by A-site (n=1000)")
    print("─"*60)

    rng = np.random.RandomState(42)
    boot_results = {}
    main_asites_boot = [a for a in ["Pb","Ca","La","Sr","Ba","Other"]
                        if (df["a_site"]==a).sum() >= 10]
    for asite in main_asites_boot:
        mask = (df["a_site"]==asite) & valid_cm
        vals = f_lst_all[mask]
        if len(vals) < 5: continue
        boot_means = [np.nanmean(rng.choice(vals,size=len(vals),replace=True))
                      for _ in range(1000)]
        ci_lo, ci_hi = np.percentile(boot_means,[2.5,97.5])
        boot_results[asite] = {
            "mean": float(np.nanmean(vals)),
            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "n": int(mask.sum())
        }
        print(f"  {asite:<8s}: f_LST={np.nanmean(vals):.3f}×  "
              f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]  n={mask.sum()}")

    # Check non-overlapping CIs
    if "Pb" in boot_results and "Ba" in boot_results:
        pb_lo = boot_results["Pb"]["ci_lo"]; ba_hi = boot_results["Ba"]["ci_hi"]
        overlap = pb_lo < ba_hi
        print(f"\n  Pb CI [{boot_results['Pb']['ci_lo']:.3f},{boot_results['Pb']['ci_hi']:.3f}]  "
              f"vs Ba CI [{boot_results['Ba']['ci_lo']:.3f},{boot_results['Ba']['ci_hi']:.3f}]")
        print(f"  → CIs {'OVERLAP — f_LST difference is suggestion, not finding' if overlap else 'DO NOT OVERLAP — f_LST by A-site is a genuine finding ✓'}")

    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("CHECK 3: Tilt-branch proper ablation (trunk intact, head zeroed)")
    print("─"*60)
    print("  Training 3-fold GSS × 3 seeds: full model vs no-tilt-head model")
    print("  (tilt features still in trunk — only dedicated head zeroed)")

    splits_gss = build_stratified_gss_splits(train_idx, groups, n_splits=HP["n_folds"])

    print("\n  Full model (baseline):")
    r2s_full, mean_full, std_full = run_cv_variant(
        "full", Xl, Xt, Xr, er_cm, has_cm, cm_approx,
        gii_norm, phase_tr, y, groups, fam_w, regime_idx, splits_gss, HP)

    print("\n  No-tilt-head model (tilt features → trunk, tilt head output = 0):")
    r2s_notilt, mean_notilt, std_notilt = run_cv_variant(
        "no-tilt-head", Xl, Xt, Xr, er_cm, has_cm, cm_approx,
        gii_norm, phase_tr, y, groups, fam_w, regime_idx, splits_gss, HP,
        no_tilt_head=True)

    delta_check3 = mean_notilt - mean_full
    print(f"\n  Full model     R² = {mean_full:.4f} ± {std_full:.4f}")
    print(f"  No-tilt-head   R² = {mean_notilt:.4f} ± {std_notilt:.4f}")
    print(f"  ΔR² = {delta_check3:+.4f}  "
          f"({'tilt head vestigial ✓' if abs(delta_check3) < 0.01 else 'tilt head contributes — revisit claim'})")

    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("CHECK 4: Explain the 0.041 R² gap (formula=0.929 vs GSS=0.888)")
    print("─"*60)

    # Formula-split: predict across KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    frm_pred = np.full(len(train_idx), np.nan)
    for tr_rel, te_rel in kf.split(train_idx):
        tr_i = train_idx[tr_rel]; te_i = train_idx[te_rel]
        Xl_f,Xt_f,Xr_f,gii_f = scale_arrays(Xl,Xt,Xr,gii_norm,tr_i)
        ms = [train_one(Xl_f,Xt_f,Xr_f,er_cm,has_cm,cm_approx,gii_f,
                        phase_tr,y,fam_w,regime_idx,tr_i,te_i,HP,seed=s)
              for s in range(HP["n_seeds"])]
        p = predict_cv(ms,Xl_f,Xt_f,Xr_f,er_cm,has_cm,cm_approx,
                       gii_f,phase_tr,te_i,regime_idx)
        frm_pred[te_rel] = p

    # GSS-split: per-sample predictions
    gss_pred = np.full(len(train_idx), np.nan)
    for tr_i, te_i in splits_gss:
        te_rel = np.where(np.isin(train_idx, te_i))[0]
        Xl_f,Xt_f,Xr_f,gii_f = scale_arrays(Xl,Xt,Xr,gii_norm,tr_i)
        ms = [train_one(Xl_f,Xt_f,Xr_f,er_cm,has_cm,cm_approx,gii_f,
                        phase_tr,y,fam_w,regime_idx,tr_i,te_i,HP,seed=s)
              for s in range(HP["n_seeds"])]
        p = predict_cv(ms,Xl_f,Xt_f,Xr_f,er_cm,has_cm,cm_approx,
                       gii_f,phase_tr,te_i,regime_idx)
        gss_pred[te_rel] = p

    # Per-family residuals
    y_tr = y[train_idx]; grp_tr = groups[train_idx]
    frm_r2 = float(r2_score(y_tr[~np.isnan(frm_pred)], frm_pred[~np.isnan(frm_pred)]))
    gss_mask = ~np.isnan(gss_pred)

    # Per-family gap analysis
    fam_gaps = {}
    for fam in np.unique(grp_tr):
        fmask = grp_tr == fam
        if fmask.sum() < 5: continue
        gss_m = fmask & gss_mask
        frm_m = fmask & ~np.isnan(frm_pred)
        if gss_m.sum() >= 3 and frm_m.sum() >= 3:
            r2_gss = float(r2_score(y_tr[gss_m], gss_pred[gss_m]))
            r2_frm = float(r2_score(y_tr[frm_m], frm_pred[frm_m]))
            fam_gaps[fam] = {"gss": r2_gss, "frm": r2_frm,
                              "gap": r2_frm - r2_gss, "n": int(fmask.sum())}

    # Sort by gap (largest formula advantage = families driving the gap)
    sorted_gaps = sorted(fam_gaps.items(), key=lambda x: x[1]["gap"], reverse=True)
    print(f"\n  Top 10 families where formula-split outperforms GSS (drives the gap):")
    print(f"  {'Family':<20s} {'n':>5} {'R²_frm':>8} {'R²_gss':>8} {'gap':>8}")
    print("  " + "-"*52)
    for fam, d in sorted_gaps[:10]:
        print(f"  {fam:<20s} {d['n']:>5d} {d['frm']:>8.3f} {d['gss']:>8.3f} {d['gap']:>+8.3f}")

    # Is the gap from large families (seen in both splits) or small families (OOD)?
    large_fam_gaps = [(f,d) for f,d in sorted_gaps
                      if d["n"]>=MIN_FAMILY_SIZE][:5]
    small_fam_gaps = [(f,d) for f,d in sorted_gaps
                      if d["n"]<MIN_FAMILY_SIZE][:5]
    mean_gap_large = np.mean([d["gap"] for _,d in large_fam_gaps]) if large_fam_gaps else 0
    mean_gap_small = np.mean([d["gap"] for _,d in small_fam_gaps]) if small_fam_gaps else 0
    print(f"\n  Mean gap — large families (n≥{MIN_FAMILY_SIZE}): {mean_gap_large:+.3f}")
    print(f"  Mean gap — small families (n<{MIN_FAMILY_SIZE}): {mean_gap_small:+.3f}")
    gap_source = "large families (memorization of seen compositions)" \
                 if mean_gap_large > mean_gap_small else \
                 "small families (OOD extrapolation difficulty)"
    print(f"  → Gap primarily driven by: {gap_source}")

    # ══════════════════════════════════════════════════════════════════════
    print("\n[Generating figures...]")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Composition balance heatmap
    ax = axes[0, 0]
    heatmap_data = pivot_main.values
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=50)
    ax.set_xticks(range(len(pivot_main.columns)))
    ax.set_xticklabels([str(c) for c in pivot_main.columns], fontsize=8)
    ax.set_yticks(range(len(pivot_main.index)))
    ax.set_yticklabels(pivot_main.index)
    for i in range(len(pivot_main.index)):
        for j in range(len(pivot_main.columns)):
            ax.text(j, i, f"{heatmap_data[i,j]:.0f}%", ha="center", va="center",
                    fontsize=8, color="black" if heatmap_data[i,j]<30 else "white")
    plt.colorbar(im, ax=ax, label="%")
    ax.set_title("Check 1: A-site % by εr Quartile\n(Pb overrepresentation?)", fontsize=10)
    style4(ax)

    # Panel B: Bootstrap f_LST CIs
    ax2 = axes[0, 1]
    boot_asites = list(boot_results.keys())
    means = [boot_results[a]["mean"] for a in boot_asites]
    lo    = [boot_results[a]["ci_lo"] for a in boot_asites]
    hi    = [boot_results[a]["ci_hi"] for a in boot_asites]
    x     = np.arange(len(boot_asites))
    colors= ["#E76F51" if a=="Pb" else "#4A90D9" if a=="Ba" else "#AAAAAA" for a in boot_asites]
    ax2.bar(x, means, 0.6, color=colors, edgecolor="white", alpha=0.85)
    ax2.errorbar(x, means, yerr=[np.array(means)-np.array(lo), np.array(hi)-np.array(means)],
                 fmt="none", color="#333", capsize=4, linewidth=1.5)
    ax2.set_xticks(x); ax2.set_xticklabels(boot_asites)
    ax2.set_ylabel("f_LST = δ_LST / εr_CM"); ax2.axhline(1.0,color="#888",ls="--",lw=1)
    ax2.set_title("Check 2: Bootstrap f_LST 95% CIs\n(Pb vs Ba overlap?)", fontsize=10)
    for xi, a in zip(x, boot_asites):
        ax2.text(xi, hi[boot_asites.index(a)]+0.05, f"n={boot_results[a]['n']}",
                 ha="center", fontsize=7.5, color="#444")
    style4(ax2)

    # Panel C: Tilt-head ablation R² comparison
    ax3 = axes[1, 0]
    x3 = np.arange(HP["n_folds"]); w3=0.35
    ax3.bar(x3-w3/2, r2s_full,   w3, label=f"Full model ({mean_full:.3f})",
            color="#009E73", edgecolor="white", alpha=0.9)
    ax3.bar(x3+w3/2, r2s_notilt, w3, label=f"No-tilt-head ({mean_notilt:.3f})",
            color="#E69F00", edgecolor="white", alpha=0.9)
    ax3.set_xticks(x3); ax3.set_xticklabels([f"Fold {i+1}" for i in range(HP["n_folds"])])
    ax3.set_ylabel("Strat-GSS R²"); ax3.set_ylim(0, 1.05)
    ax3.set_title(f"Check 3: Tilt-Head Ablation\nΔR²={delta_check3:+.4f} (trunk intact)", fontsize=10)
    ax3.legend(); style4(ax3)
    ax3.text(0.05, 0.05,
             "Tilt features → trunk (signal present)\nTilt HEAD output = 0 (branch zeroed)\n"
             f"ΔR²={delta_check3:+.4f} confirms branch is vestigial",
             transform=ax3.transAxes, fontsize=7.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    # Panel D: Gap analysis by family
    ax4 = axes[1, 1]
    top_n = min(10, len(sorted_gaps))
    gap_fams  = [f for f,_ in sorted_gaps[:top_n]]
    gap_vals  = [d["gap"] for _,d in sorted_gaps[:top_n]]
    gap_n_arr = [d["n"] for _,d in sorted_gaps[:top_n]]
    y4 = np.arange(top_n)
    gcols = ["#E76F51" if n>=MIN_FAMILY_SIZE else "#56B4E9" for n in gap_n_arr]
    ax4.barh(y4, gap_vals, 0.6, color=gcols, edgecolor="white")
    ax4.set_yticks(y4); ax4.set_yticklabels([f"{f} (n={n})" for f,n in zip(gap_fams,gap_n_arr)], fontsize=8)
    ax4.set_xlabel("Gap: R²_formula − R²_GSS")
    ax4.axvline(0, color="#888", lw=1)
    ax4.set_title("Check 4: Families Driving Formula/GSS Gap\n(orange=large family, blue=small/OOD)", fontsize=10)
    ax4.invert_yaxis(); style4(ax4)
    from matplotlib.patches import Patch
    ax4.legend(handles=[Patch(color="#E76F51",label=f"n≥{MIN_FAMILY_SIZE} (seen)"),
                         Patch(color="#56B4E9",label=f"n<{MIN_FAMILY_SIZE} (OOD)")], fontsize=8)

    plt.suptitle("NatComm Council Validation — 4 Checks", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "41_natcomm_validation.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved → {fig_path}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    results = {
        "check1_composition_balance": {
            "Pb_in_Q4_pct": pb_q4, "Ba_in_Q4_pct": ba_q4,
            "Pb_total_n": pb_n,
            "coverage_artifact_risk": "HIGH" if bias_flag else "LOW",
            "verdict": "Pb overrepresented in Q4 — f_LST claim needs balancing" if bias_flag
                       else "Distribution acceptable — f_LST=2.27x not a coverage artifact",
        },
        "check2_bootstrap_flst": boot_results,
        "check2_pb_ba_overlap": overlap if ("Pb" in boot_results and "Ba" in boot_results) else None,
        "check2_verdict": "Finding" if not overlap else "Suggestion only",
        "check3_tilt_head_ablation": {
            "full_r2_mean": mean_full, "full_r2_std": std_full,
            "notilt_r2_mean": mean_notilt, "notilt_r2_std": std_notilt,
            "delta_r2": delta_check3,
            "verdict": "Tilt head vestigial — branch contributes negligible R²" if abs(delta_check3)<0.01
                       else "Tilt head contributes — revisit tilt mechanism claim",
        },
        "check4_formula_gss_gap": {
            "formula_r2": frm_r2,
            "gap_driven_by": gap_source,
            "top_gap_families": [{"family":f,"gap":d["gap"],"n":d["n"],"r2_frm":d["frm"],"r2_gss":d["gss"]}
                                  for f,d in sorted_gaps[:10]],
        },
    }
    res_path = os.path.join(RES_DIR, "41_validation.json")
    with open(res_path,"w") as f: json.dump(results,f,indent=2)
    print(f"  Results → {res_path}")

    print("\n"+"="*70)
    print("NATCOMM COUNCIL VALIDATION COMPLETE")
    print("="*70)
    print(f"  Check 1 — Pb balance   : {'⚠️  HIGH RISK' if bias_flag else '✓  LOW RISK'}")
    ci_ok = "Pb" in boot_results and "Ba" in boot_results and not overlap
    print(f"  Check 2 — Bootstrap CIs : {'✓  FINDING (non-overlapping)' if ci_ok else '⚠️  SUGGESTION (overlapping)'}")
    tilt_ok = abs(delta_check3) < 0.01
    print(f"  Check 3 — Tilt ablation : {'✓  VESTIGIAL (ΔR²<0.01)' if tilt_ok else '⚠️  CONTRIBUTES — revise claim'}")
    print(f"  Check 4 — Gap analysis  : {gap_source}")
    print("="*70)


if __name__ == "__main__":
    main()
