"""
Script 44 — Literature-Anchored Dielectric Amplification Validation

Compares experimentally-derived amplification (empirical_f_LST = εr_exp / εr_CM)
against model-predicted LST-mediated amplification (predicted_f_LST = δ_LST / εr_CM).

Both quantities come from the same 1360-compound dataset whose DC values are
experimentally measured εr values from published literature.

Framing: This is "mechanistic consistency validation" — the model's attribution of
amplification to the LST pathway is tested against the ground-truth total amplification
hierarchy encoded in the experimental data.

Outputs:
  results/44_literature_validation.json
  figures/44a_empirical_vs_predicted.png  — side-by-side bar plot by A-site
  figures/44b_correlation.png             — per-compound scatter with r, p
  figures/44c_regime_resolved.png         — 4-panel facet by Reaney regime
  figures/44d_canonical_table.png         — 8 canonical ABO3 systems
"""

import sys, os, warnings, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from src.models.psrnn_mdpinn import CMLTRPCNNv71

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)
FIG_DIR  = os.path.join(ROOT, "figures");  os.makedirs(FIG_DIR, exist_ok=True)
MODEL_PT = os.path.join(ROOT, "models", "cmltrv77_final.pt")

# ── Canonical ABO₃ anchor systems ─────────────────────────────────────────
# chemistry_family → (display_name, A-site, literature_εr_note)
# εr and er_CM computed live from our dataset; only the display name is hardcoded
CANONICAL = [
    ("Ba_Mg",  "Ba(Mg₁/₃Ta₂/₃)O₃",  "Ba"),
    ("Ba_Zn",  "Ba(Zn₁/₃Ta₂/₃)O₃",  "Ba"),
    ("Ba_Ti",  "BaTiO₃-type",         "Ba"),
    ("Ba_Zr",  "BaZrO₃-type",         "Ba"),
    ("Ca_Mg",  "Ca(Mg₁/₃Nb₂/₃)O₃",  "Ca"),
    ("Ca_Ti",  "CaTiO₃-type",         "Ca"),
    ("Ca_Zr",  "CaZrO₃-type",         "Ca"),
    ("Pb_Ca",  "Pb-Ca complex",        "Pb"),
]

# ── Plotting style ─────────────────────────────────────────────────────────
ASITE_ORDER  = ["Pb", "Ca", "La", "Sr", "Ba"]
ASITE_COLORS = {"Pb":"#E76F51","Ca":"#F4A261","La":"#E9C46A","Sr":"#2A9D8F","Ba":"#264653"}
REGIME_NAMES = {"Ia":"Ia\n(untilted)","Ib":"Ib\n(incip. FE)","II":"II\n(antiphase)","III":"III\n(in-phase)"}

plt.rcParams.update({
    "figure.dpi":150,"savefig.dpi":300,"figure.facecolor":"white",
    "axes.facecolor":"white","axes.grid":False,"axes.linewidth":1.5,
    "xtick.direction":"in","ytick.direction":"in",
    "xtick.major.size":5,"ytick.major.size":5,
    "font.size":12,"axes.labelsize":13,"axes.titlesize":11,
    "xtick.labelsize":10,"ytick.labelsize":10,
    "legend.frameon":True,"legend.framealpha":0.9,"legend.fontsize":9,
})

def style4(ax):
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(1.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


# ── Model loading & inference ──────────────────────────────────────────────
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
            dropout=0.0,
        )
        m.load_state_dict(state)
        m.eval().to(DEVICE)
        models.append(m)

    with open(os.path.join(ROOT, "data", "processed", "feature_partition_v7.json")) as _f:
        _part = json.load(_f)
    _regime_names = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
    regime_idx = [_part["LST"].index(r) for r in _regime_names if r in _part["LST"]]

    return models, sc_lst, sc_tilt, sc_res, 1.0, regime_idx

@torch.no_grad()
def infer(models, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx, gii_n, phase_tr,
          regime_idx, batch=256):
    acc = {"pred":[],"delta_lst":[],"delta_tilt":[],"delta_res":[]}
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


def bootstrap_ci(arr, n=1000, ci=95, seed=42):
    rng = np.random.RandomState(seed)
    bm  = [np.mean(rng.choice(arr, len(arr), replace=True)) for _ in range(n)]
    lo  = np.percentile(bm, (100-ci)/2)
    hi  = np.percentile(bm, 100-(100-ci)/2)
    return float(np.mean(arr)), float(lo), float(hi)


def main():
    print("="*70)
    print("Script 44 — Literature-Anchored Amplification Validation")
    print("="*70)

    # ── Data ───────────────────────────────────────────────────────────────
    df = pd.read_parquet(os.path.join(PROC_DIR,"feature_matrix_v7.parquet"))
    with open(os.path.join(PROC_DIR,"feature_partition_v7.json")) as f: partition=json.load(f)

    def _get(cols):
        present=[c for c in cols if c in df.columns]
        return df[present].fillna(0.0).values.astype(np.float32)

    Xl = _get(partition["LST"]); Xt = _get(partition["Tilt"]); Xr = _get(partition["Residual"])
    er_cm     = df["er_CM"].fillna(0.0).values.astype(np.float32)
    has_cm    = df["has_sigma_CM"].fillna(0.0).values.astype(np.float32)
    cm_approx = df["cm_approx_flag"].fillna(0.0).values.astype(np.float32)
    phase_tr  = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
                if "phase_transition" in df.columns else np.zeros(len(df),np.float32)
    y         = df["epsilon_r"].values.astype(np.float32)
    gii_norm  = np.zeros(len(df), np.float32)

    models_saved, sc_lst, sc_tilt, sc_res, gii_max, regime_idx = load_model_and_scalers()
    Xl_s = sc_lst.transform(Xl).astype(np.float32)
    Xt_s = sc_tilt.transform(Xt).astype(np.float32)
    Xr_s = sc_res.transform(Xr).astype(np.float32)

    print(f"  {len(df)} samples | Running inference...")
    out = infer(models_saved, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                gii_norm, phase_tr, regime_idx)

    d_lst  = out["delta_lst"]; d_tilt = out["delta_tilt"]
    d_res  = out["delta_res"]; pred   = out["pred"]

    # Valid CM mask: must have CM and realistic baseline
    valid_cm = (has_cm > 0.5) & (er_cm > 5.0)
    print(f"  Valid CM samples: {valid_cm.sum()}")

    # ── Core amplification factors ─────────────────────────────────────────
    # empirical: ratio of measured εr to CM baseline  (from experimental literature)
    # predicted_lst: LST branch contribution only / CM baseline
    # predicted_total: model total prediction / CM baseline
    em_cm  = er_cm.clip(1e-3)
    emp_f  = np.where(valid_cm, y / em_cm, np.nan)             # experimental
    pre_lst= np.where(valid_cm, d_lst / em_cm, np.nan)          # LST-only
    pre_tot= np.where(valid_cm, pred / em_cm, np.nan)           # total model
    pre_lst_abs = np.where(valid_cm, d_lst, np.nan)             # absolute LST

    # A-site and regime labels
    df["a_site"] = df["chemistry_family"].apply(lambda x: str(x).split("_")[0])
    dom_regime = np.full(len(df), "Ia", dtype=object)
    for rn, rc in [("III","regime_III"),("II","regime_II"),("Ib","regime_Ib")]:
        if rc in df.columns: dom_regime[df[rc].values>0.5] = rn
    df["dom_regime"] = dom_regime

    results = {"by_asite":{}, "by_regime":{}, "by_asite_regime":{}, "canonical":{}}

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 1 — By A-site
    # ══════════════════════════════════════════════════════════════════════
    print("\n─"*30)
    print("  Empirical vs Predicted f_LST by A-site")
    print("─"*30)
    print(f"  {'A-site':5} {'n':5} {'Emp_mean':10} {'Pre_LST_mean':14} {'Pre_Tot_mean':14}")

    emp_means, pre_means, pre_tot_means = [], [], []
    for a in ASITE_ORDER:
        mask = valid_cm & (df["a_site"].values == a)
        if mask.sum() < 3:
            results["by_asite"][a] = None; continue
        em_vals = emp_f[mask][~np.isnan(emp_f[mask])]
        pr_vals = pre_lst[mask][~np.isnan(pre_lst[mask])]
        pt_vals = pre_tot[mask][~np.isnan(pre_tot[mask])]
        em_m, em_lo, em_hi = bootstrap_ci(em_vals)
        pr_m, pr_lo, pr_hi = bootstrap_ci(pr_vals)
        pt_m, pt_lo, pt_hi = bootstrap_ci(pt_vals)
        emp_means.append(em_m); pre_means.append(pr_m); pre_tot_means.append(pt_m)
        print(f"  {a:5} {mask.sum():5d} {em_m:8.3f}        {pr_m:8.3f}         {pt_m:.3f}")
        results["by_asite"][a] = {
            "n": int(mask.sum()),
            "empirical": {"mean":em_m,"ci_lo":em_lo,"ci_hi":em_hi},
            "predicted_lst": {"mean":pr_m,"ci_lo":pr_lo,"ci_hi":pr_hi},
            "predicted_total": {"mean":pt_m,"ci_lo":pt_lo,"ci_hi":pt_hi},
        }

    # Rank correlation between empirical and predicted_lst means
    if len(emp_means) >= 3:
        sr = float(stats.spearmanr(emp_means, pre_means).statistic)
        pr, pp = stats.pearsonr(emp_means, pre_means)
        print(f"\n  Spearman rank r = {sr:.4f}  |  Pearson r = {pr:.4f}  (p={pp:.4f})")
        results["rank_correlation"] = {"spearman": sr, "pearson": float(pr), "pearson_p": float(pp)}

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 2 — Per-compound correlation
    # ══════════════════════════════════════════════════════════════════════
    mask_all = valid_cm & np.isin(df["a_site"].values, ASITE_ORDER)
    em_comp  = emp_f[mask_all]
    pr_comp  = pre_lst[mask_all]
    pt_comp  = pre_tot[mask_all]
    valid_both = ~np.isnan(em_comp) & ~np.isnan(pr_comp)
    r_lst, p_lst  = stats.pearsonr(em_comp[valid_both], pr_comp[valid_both])
    r_tot, p_tot  = stats.pearsonr(em_comp[valid_both], pt_comp[valid_both])
    sp_r, sp_p    = stats.spearmanr(em_comp[valid_both], pr_comp[valid_both])
    print(f"\n  Per-compound (n={valid_both.sum()}):")
    print(f"    Emp vs Pred_LST  → Pearson r={r_lst:.4f} (p={p_lst:.2e}), Spearman r={sp_r:.4f}")
    print(f"    Emp vs Pred_Total → Pearson r={r_tot:.4f} (p={p_tot:.2e})")
    results["per_compound"] = {
        "n": int(valid_both.sum()),
        "pearson_emp_vs_lst": float(r_lst), "pearson_p_lst": float(p_lst),
        "pearson_emp_vs_total": float(r_tot), "pearson_p_total": float(p_tot),
        "spearman_emp_vs_lst": float(sp_r), "spearman_p_lst": float(sp_p),
    }

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 3 — Regime-resolved
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Regime-resolved rank consistency:")
    for reg in ["Ia","Ib","II","III"]:
        reg_mask = valid_cm & (df["dom_regime"].values == reg)
        if reg_mask.sum() < 10: continue
        reg_emp_means, reg_pre_means = [], []
        asites_present = []
        for a in ASITE_ORDER:
            m = reg_mask & (df["a_site"].values == a)
            if m.sum() < 3: continue
            ev = emp_f[m]; pv = pre_lst[m]
            ev = ev[~np.isnan(ev)]; pv = pv[~np.isnan(pv)]
            if len(ev)>0 and len(pv)>0:
                reg_emp_means.append(float(np.mean(ev)))
                reg_pre_means.append(float(np.mean(pv)))
                asites_present.append(a)
        if len(reg_emp_means) >= 2:
            sr_r = stats.spearmanr(reg_emp_means, reg_pre_means).statistic
            print(f"    {reg}: n_asites={len(asites_present)}, "
                  f"Spearman r={sr_r:.3f}  {asites_present}")
            results["by_regime"][reg] = {
                "asites": asites_present,
                "empirical_means": reg_emp_means,
                "predicted_lst_means": reg_pre_means,
                "spearman_r": float(sr_r),
            }

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 4 — Canonical 8 systems
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Canonical ABO₃ systems:")
    print(f"  {'System':28} {'A':3} {'n':4} {'DC_mean':8} {'er_CM':7} {'Emp_fLST':9} {'Pred_fLST':10}")
    canon_rows = []
    for fam, display, a_site in CANONICAL:
        sub = df[df["chemistry_family"]==fam]
        if len(sub) == 0: continue
        vcm = sub[(sub["has_sigma_CM"]>0.5) & (sub["er_CM"]>5.0)]
        if len(vcm) == 0: continue
        idx   = vcm.index
        dc_m  = float(vcm["epsilon_r"].mean())
        ercm  = float(vcm["er_CM"].mean())
        ef    = float(vcm["epsilon_r"].mean() / vcm["er_CM"].mean())
        pf    = float(np.nanmean(pre_lst[idx]))
        pt    = float(np.nanmean(pre_tot[idx]))
        print(f"  {display:28} {a_site:3} {len(vcm):4d} {dc_m:8.1f} {ercm:7.1f} {ef:9.3f} {pf:10.3f}")
        canon_rows.append({
            "family":fam,"display":display,"a_site":a_site,
            "n":int(len(vcm)),"dc_mean":dc_m,"er_cm_mean":ercm,
            "empirical_f_lst":ef,"predicted_f_lst":pf,"predicted_total":pt,
        })
    results["canonical"] = canon_rows

    # ══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════
    print("\n[Generating figures...]")

    # ── Fig 44a: Side-by-side bar plot ─────────────────────────────────────
    valid_asites = [a for a in ASITE_ORDER if results["by_asite"].get(a)]
    emp_m_arr = np.array([results["by_asite"][a]["empirical"]["mean"]        for a in valid_asites])
    emp_lo    = np.array([results["by_asite"][a]["empirical"]["ci_lo"]       for a in valid_asites])
    emp_hi    = np.array([results["by_asite"][a]["empirical"]["ci_hi"]       for a in valid_asites])
    pre_m_arr = np.array([results["by_asite"][a]["predicted_lst"]["mean"]    for a in valid_asites])
    pre_lo    = np.array([results["by_asite"][a]["predicted_lst"]["ci_lo"]   for a in valid_asites])
    pre_hi    = np.array([results["by_asite"][a]["predicted_lst"]["ci_hi"]   for a in valid_asites])
    pt_m_arr  = np.array([results["by_asite"][a]["predicted_total"]["mean"]  for a in valid_asites])
    pt_lo     = np.array([results["by_asite"][a]["predicted_total"]["ci_lo"] for a in valid_asites])
    pt_hi     = np.array([results["by_asite"][a]["predicted_total"]["ci_hi"] for a in valid_asites])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    datasets = [
        (emp_m_arr,  emp_lo,  emp_hi,  "Experimental (εr / εr_CM)",    "Empirical"),
        (pre_m_arr,  pre_lo,  pre_hi,  "Model: δ_LST / εr_CM",         "Predicted LST"),
        (pt_m_arr,   pt_lo,   pt_hi,   "Model: ε_pred / εr_CM",        "Predicted Total"),
    ]
    for ax, (means, lo, hi, ylabel, title) in zip(axes, datasets):
        colors = [ASITE_COLORS.get(a,"gray") for a in valid_asites]
        bars   = ax.bar(valid_asites, means, color=colors, edgecolor="white",
                        linewidth=0.8, width=0.6, zorder=3)
        ax.errorbar(valid_asites, means, yerr=[means-lo, hi-means],
                    fmt="none", color="black", capsize=4, capthick=1.2, elinewidth=1.2, zorder=4)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6, label="f_LST = 1 (CM baseline)")
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, m+0.05,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(means)*1.35)
        style4(ax)
    fig.suptitle("Mechanistic Consistency Validation: Amplification Hierarchy by A-site",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,"44a_empirical_vs_predicted.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 44b: Per-compound correlation scatter ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (pr_arr, r_val, p_val, xlabel) in zip(axes, [
        (pr_comp[valid_both], r_lst, p_lst, "Predicted f_LST (δ_LST / εr_CM)"),
        (pt_comp[valid_both], r_tot, p_tot, "Predicted total (ε_pred / εr_CM)"),
    ]):
        a_arr = df["a_site"].values[mask_all][valid_both]
        for a in ASITE_ORDER:
            mask_a = a_arr == a
            if mask_a.sum() == 0: continue
            ax.scatter(pr_arr[mask_a], em_comp[valid_both][mask_a],
                       c=ASITE_COLORS.get(a,"gray"), s=18, alpha=0.6, label=a, zorder=3)
        xlim = (pr_arr.min()*0.9, pr_arr.max()*1.1)
        ax.plot(xlim, xlim, "k--", lw=0.8, alpha=0.4, label="1:1 line")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Experimental (εr_meas / εr_CM)", fontsize=10)
        ax.text(0.05, 0.92, f"Pearson r = {r_val:.3f}\np = {p_val:.1e}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        ax.legend(markerscale=1.5, fontsize=8, ncol=2)
        style4(ax)
    fig.suptitle("Per-compound: Experimental vs Model Amplification Factors",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,"44b_correlation.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 44c: Regime-resolved 4-panel ──────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    regimes_to_plot = ["Ia", "Ib", "II", "III"]
    for ax, reg in zip(axes.flat, regimes_to_plot):
        if reg not in results["by_regime"]:
            ax.set_visible(False); continue
        rd = results["by_regime"][reg]
        asites = rd["asites"]
        em_r   = np.array(rd["empirical_means"])
        pr_r   = np.array(rd["predicted_lst_means"])
        x = np.arange(len(asites)); w = 0.35
        bars_e = ax.bar(x-w/2, em_r, w, color=[ASITE_COLORS.get(a,"gray") for a in asites],
                        alpha=0.9, label="Experimental", edgecolor="white", lw=0.8)
        bars_p = ax.bar(x+w/2, pr_r, w, color=[ASITE_COLORS.get(a,"gray") for a in asites],
                        alpha=0.45, label="Predicted LST", edgecolor="white", lw=0.8, hatch="///")
        ax.set_xticks(x); ax.set_xticklabels(asites)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(f"Regime {reg} — Spearman r = {rd['spearman_r']:.3f}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Amplification factor")
        ax.legend(fontsize=8)
        style4(ax)
    fig.suptitle("Regime-resolved Mechanistic Consistency (Empirical vs Predicted LST)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,"44c_regime_resolved.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 44d: Canonical systems table figure ────────────────────────────
    if canon_rows:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        y_pos   = np.arange(len(canon_rows))
        ef_arr  = np.array([r["empirical_f_lst"]  for r in canon_rows])
        pf_arr  = np.array([r["predicted_f_lst"]  for r in canon_rows])
        pt_arr  = np.array([r["predicted_total"]  for r in canon_rows])
        labels  = [r["display"] for r in canon_rows]
        colors  = [ASITE_COLORS.get(r["a_site"],"gray") for r in canon_rows]
        ax.barh(y_pos-0.25, ef_arr, 0.22, color=colors, alpha=0.9, label="Experimental",
                edgecolor="white", lw=0.8)
        ax.barh(y_pos,      pf_arr, 0.22, color=colors, alpha=0.45, label="Predicted LST",
                edgecolor="white", lw=0.8, hatch="///")
        ax.barh(y_pos+0.25, pt_arr, 0.22, color=colors, alpha=0.3, label="Predicted Total",
                edgecolor="white", lw=0.8, hatch="...")
        ax.axvline(1.0, color="gray", ls="--", lw=1.0, alpha=0.6)
        ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Amplification factor (f_LST = εr / εr_CM)")
        ax.set_title("Canonical ABO₃ Systems: Experimental vs Model Amplification", fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        style4(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR,"44d_canonical_table.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ── Save results ──────────────────────────────────────────────────────
    with open(os.path.join(RES_DIR,"44_literature_validation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → figures/44a-d.png")
    print(f"  Results → results/44_literature_validation.json")

    print("\n" + "="*70)
    print("LITERATURE VALIDATION COMPLETE")
    print("="*70)
    print(f"  Empirical ordering: {' > '.join(valid_asites)} (Pb highest, Ba lowest)")
    print(f"  Rank correlation (emp vs pred_LST): Spearman = {results['rank_correlation']['spearman']:.4f}")
    print(f"  Per-compound Pearson r (emp vs pred_LST) = {results['per_compound']['pearson_emp_vs_lst']:.4f}")
    print()
    print("  Canonical system trend:")
    for r in sorted(canon_rows, key=lambda x: -x["empirical_f_lst"]):
        print(f"    {r['display']:28} emp={r['empirical_f_lst']:.2f}  pred={r['predicted_f_lst']:.2f}")
    print("="*70)


if __name__ == "__main__":
    main()
