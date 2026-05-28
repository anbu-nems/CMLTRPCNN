#!/usr/bin/env python3
"""
Generate 6 remaining supplementary figures using extracted per-sample data
+ Optuna database.

Outputs (all in all_figures/):
  figS3_optuna_convergence.pdf/png
  figS11_shap_importance.pdf/png
  figS12_lofo_parity.pdf/png
  figS13_counterfactual_heatmap.pdf/png
  figS14_permutation_null.pdf/png
  figS15_sigma_conf_vs_error.pdf/png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import sqlite3
import json
import os

# --- self-contained release root (auto-injected) ---
RELEASE_ROOT = os.path.abspath(os.path.dirname(__file__))
while RELEASE_ROOT != os.path.dirname(RELEASE_ROOT) and not os.path.isdir(os.path.join(RELEASE_ROOT, 'model_weights')):
    RELEASE_ROOT = os.path.dirname(RELEASE_ROOT)
# ----------------------------------------------------




OUT_DIR     = os.path.join(RELEASE_ROOT, "all_figures")
EXTRACT_DIR = f"{OUT_DIR}/extracted_data"
PIML        = RELEASE_ROOT

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": True,
    "legend.framealpha": 0.92, "legend.edgecolor": "#CCCCCC",
    "figure.constrained_layout.use": True,
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3,    "ytick.major.size": 3,
    "axes.grid": False,
})

def style4(ax, lw=1.0):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw)
    ax.tick_params(direction="in", top=False, right=False,
                   bottom=True, left=True, width=0.8, length=3)
    ax.set_axisbelow(False)

REGIME_COLORS = {"Ia":"#264653", "Ib":"#2A9D8F", "II":"#E9C46A", "III":"#E76F51"}
REGIME_MARK   = {"Ia":"o", "Ib":"s", "II":"^", "III":"D"}
ASITE_COLORS  = {"Pb":"#E76F51", "Ca":"#F4A261", "La":"#2A9D8F",
                 "Sr":"#E9C46A", "Ba":"#264653"}
OURS = "#E76F51"; GRAY = "#B0BEC5"; TEAL = "#2A9D8F"; GOLD = "#E9C46A"

def save(fig, name):
    for ext in ("pdf", "png"):
        p = f"{OUT_DIR}/{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 1) Optuna convergence — S3
# ════════════════════════════════════════════════════════════════════════════
def fig_optuna_convergence():
    con = sqlite3.connect(f"{PIML}/results/hp_tuning_optuna.db")
    df = pd.read_sql_query(
        "SELECT t.trial_id, t.state, v.value "
        "FROM trials t JOIN trial_values v ON t.trial_id = v.trial_id "
        "WHERE t.state='COMPLETE' AND v.objective=0 "
        "ORDER BY t.trial_id", con)
    con.close()
    df["best_so_far"] = df["value"].cummax()
    best_idx = df["value"].idxmax()
    best_trial = int(df.loc[best_idx, "trial_id"])
    best_val   = float(df.loc[best_idx, "value"])

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    style4(ax)
    ax.scatter(df["trial_id"], df["value"], s=12, color=GRAY,
               alpha=0.6, edgecolors="white", linewidths=0.3,
               label="Trial value", zorder=3)
    ax.plot(df["trial_id"], df["best_so_far"], color=OURS, lw=1.5,
            label="Best so far", zorder=4)
    ax.axhline(best_val, color=OURS, ls=":", lw=0.8, zorder=2)
    ax.scatter([best_trial], [best_val], s=80, color=OURS,
               edgecolor="white", linewidth=1.0, zorder=5,
               marker="*", label=f"Best trial #{best_trial}\n($R^2$={best_val:.4f})")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Formula-split CV $R^2$")
    ax.set_title(f"Optuna HP tuning convergence  ($n$={len(df)} completed trials)",
                 loc="left")
    ax.set_ylim(min(0.4, df["value"].min()-0.02), 1.0)
    ax.legend(loc="lower right")
    save(fig, "figS3_optuna_convergence")


# ════════════════════════════════════════════════════════════════════════════
# 2) SHAP importance — S11
# ════════════════════════════════════════════════════════════════════════════
def fig_shap_importance():
    imp = pd.read_csv(f"{EXTRACT_DIR}/shap_feature_importance.csv")
    # Try to map f_X → actual feature names from FM
    fm = pd.read_parquet(f"{PIML}/data/processed/feature_matrix_v7.parquet")
    feat_cols = [c for c in fm.columns
                 if c not in ("epsilon_r", "formula", "chemistry_family", "DC")]
    if len(feat_cols) >= len(imp):
        def remap(name):
            if name.startswith("f_"):
                idx = int(name[2:])
                return feat_cols[idx] if idx < len(feat_cols) else name
            return name
        imp["feature"] = imp["feature"].apply(remap)

    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    style4(ax)
    colors = [OURS if v > 0 else TEAL for v in top["mean_signed_shap"]]
    bars = ax.barh(np.arange(len(top)), top["mean_abs_shap"],
                   height=0.6, color=colors,
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=6.5)
    ax.set_xlabel(r"Mean |SHAP| value (impact on $\varepsilon_r$)")
    ax.set_title("Top-20 features by SHAP importance", loc="left")
    for b, v in zip(bars, top["mean_abs_shap"]):
        ax.text(b.get_width() + 0.05, b.get_y() + b.get_height()/2,
                f"{v:.2f}", va="center", fontsize=6, color="#333")
    ax.set_xlim(0, top["mean_abs_shap"].max() * 1.18)
    legend_items = [
        mpatches.Patch(color=OURS, label=r"$+$ effect (increases $\varepsilon_r$)"),
        mpatches.Patch(color=TEAL, label=r"$-$ effect (decreases $\varepsilon_r$)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=6.5)
    save(fig, "figS11_shap_importance")


# ════════════════════════════════════════════════════════════════════════════
# 3) LOFO parity (A-site × regime) — S12
# ════════════════════════════════════════════════════════════════════════════
def fig_lofo_parity():
    lofo = pd.read_csv(f"{EXTRACT_DIR}/lofo_predictions_v77.csv")
    # Labels are LOFO-Pb/Ca/Ba/Sr/La (A-site) and LOFO-Ia/Ib/II/III (regime)
    lofo["target"] = lofo["label"].str.replace("LOFO-", "")
    asite_targets  = {"Pb", "Ca", "Ba", "Sr", "La"}
    regime_targets = {"Ia", "Ib", "II", "III"}
    asite_rows  = lofo[lofo["target"].isin(asite_targets)].copy()
    asite_rows["a_site"] = asite_rows["target"]
    regime_rows = lofo[lofo["target"].isin(regime_targets)].copy()
    regime_rows["regime"] = regime_rows["target"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    style4(ax1); style4(ax2)

    lim = 150
    for ax in (ax1, ax2):
        ax.plot([0, lim], [0, lim], color="#333", lw=0.8, ls="--", zorder=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    # A-site LOFO
    if len(asite_rows) > 0:
        order = ["Pb", "Ca", "Ba", "Sr", "La"]
        for a in order:
            sub = asite_rows[asite_rows["a_site"] == a]
            if len(sub) == 0: continue
            r2 = sub["lofo_r2"].iloc[0]
            ax1.scatter(sub["er_measured"], sub["er_predicted"],
                        s=20, color=ASITE_COLORS.get(a, GRAY),
                        alpha=0.75, edgecolors="white", linewidths=0.3,
                        label=f"{a} ($R^2$={r2:+.3f})", zorder=3)
    ax1.set_xlabel(r"Measured $\varepsilon_r$")
    ax1.set_ylabel(r"Predicted $\varepsilon_r$")
    ax1.set_title("a  LOFO by A-site", loc="left")
    ax1.legend(loc="upper left", fontsize=6.5, handlelength=0.8,
               title="Held-out A-site", title_fontsize=6.5)

    # Regime LOFO
    if len(regime_rows) > 0:
        order = ["Ia", "Ib", "II", "III"]
        for r in order:
            sub = regime_rows[regime_rows["regime"] == r]
            if len(sub) == 0: continue
            r2 = sub["lofo_r2"].iloc[0]
            ax2.scatter(sub["er_measured"], sub["er_predicted"],
                        s=20, color=REGIME_COLORS[r],
                        alpha=0.75, edgecolors="white", linewidths=0.3,
                        marker=REGIME_MARK[r],
                        label=f"Regime {r} ($R^2$={r2:+.2f})", zorder=3)
    ax2.set_xlabel(r"Measured $\varepsilon_r$")
    ax2.set_ylabel(r"Predicted $\varepsilon_r$")
    ax2.set_title("b  LOFO by Reaney regime", loc="left")
    ax2.legend(loc="upper left", fontsize=6.5, handlelength=0.8,
               title="Held-out regime", title_fontsize=6.5)

    fig.suptitle("Leave-One-Family-Out generalisation",
                 fontsize=9, fontweight="bold", y=1.03)
    save(fig, "figS12_lofo_parity")


# ════════════════════════════════════════════════════════════════════════════
# 4) Counterfactual A-site swap heatmap — S13
# ════════════════════════════════════════════════════════════════════════════
def fig_counterfactual_heatmap():
    cf = pd.read_csv(f"{EXTRACT_DIR}/counterfactual_asite_swap.csv")
    # Mean Δεr by (original_a_site, swap_to_a_site)
    pivot = cf.groupby(["original_a_site", "swap_to_a_site"])["delta_er"].mean().unstack()
    keep = ["Ba", "Ca", "Sr", "La", "Pb"]
    pivot = pivot.reindex(index=[k for k in keep if k in pivot.index],
                          columns=[k for k in keep if k in pivot.columns])

    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    vmax = float(np.nanmax(np.abs(pivot.values)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    style4(ax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Swap A-site to →")
    ax.set_ylabel("Original A-site")
    ax.set_title("Counterfactual A-site swap: mean $\\Delta\\varepsilon_r$", loc="left")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(v) > vmax*0.5 else "#222",
                        fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"$\Delta\varepsilon_r$ (swap − original)", fontsize=7)
    save(fig, "figS13_counterfactual_heatmap")


# ════════════════════════════════════════════════════════════════════════════
# 5) Permutation null distributions — S14
# ════════════════════════════════════════════════════════════════════════════
def fig_permutation_null():
    asp = pd.read_csv(f"{EXTRACT_DIR}/permutation_asite.csv")
    rgp = pd.read_csv(f"{EXTRACT_DIR}/permutation_regime.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax1, ax2 = axes
    style4(ax1); style4(ax2)

    # A-site permutation: distribution of perm_delta across all samples
    ax1.hist(asp["perm_delta"], bins=40, color=TEAL, alpha=0.85,
             edgecolor="white", linewidth=0.4, zorder=3)
    ax1.axvline(0, color="#333", lw=0.8, ls="--")
    ax1.axvline(asp["perm_delta"].mean(), color=OURS, lw=1.2,
                label=f"Mean = {asp['perm_delta'].mean():+.2f}")
    ax1.set_xlabel(r"$\Delta\varepsilon_r$ (A-site permuted − original)")
    ax1.set_ylabel("Count")
    ax1.set_title("a  A-site permutation null  (50 shuffles)", loc="left")
    ax1.legend(loc="upper right")

    # Regime permutation
    ax2.hist(rgp["perm_delta"], bins=40, color=GOLD, alpha=0.85,
             edgecolor="white", linewidth=0.4, zorder=3)
    ax2.axvline(0, color="#333", lw=0.8, ls="--")
    ax2.axvline(rgp["perm_delta"].mean(), color=OURS, lw=1.2,
                label=f"Mean = {rgp['perm_delta'].mean():+.2f}")
    ax2.set_xlabel(r"$\Delta\varepsilon_r$ (regime permuted − original)")
    ax2.set_ylabel("Count")
    ax2.set_title("b  Regime permutation null  (50 shuffles)", loc="left")
    ax2.legend(loc="upper right")

    save(fig, "figS14_permutation_null")


# ════════════════════════════════════════════════════════════════════════════
# 6) σ_conf vs prediction error — S15
# ════════════════════════════════════════════════════════════════════════════
def fig_sigma_conf_vs_error():
    d = pd.read_csv(f"{EXTRACT_DIR}/decomposition_per_sample.csv")
    d["abs_error"] = (d["er_measured"] - d["er_predicted"]).abs()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    style4(ax1); style4(ax2)

    # σ_conf vs |error|
    for reg in ["Ia", "Ib", "II", "III"]:
        sub = d[d["regime"] == reg]
        ax1.scatter(sub["sigma_conf"], sub["abs_error"],
                    s=10, alpha=0.55, color=REGIME_COLORS[reg],
                    marker=REGIME_MARK[reg],
                    edgecolors="white", linewidths=0.25,
                    label=f"Regime {reg}", zorder=3)
    # Reference horizontal line at conformal q90 = 14.38
    ax1.axhline(14.38, color="#555", lw=0.8, ls="--",
                label="Conformal $\\hat{q}_{90}=14.38$")
    # Reference vertical line at gate threshold 0.35
    ax1.axvline(0.35, color="#888", lw=0.6, ls=":",
                label="Layer-1 gate = 0.35")
    ax1.set_xlabel(r"$\sigma_{\rm conf}$ (confidence gate)")
    ax1.set_ylabel(r"|Predicted − Measured| $\varepsilon_r$")
    ax1.set_title("a  Applicability domain validation", loc="left")
    ax1.legend(loc="upper right", fontsize=6.0, handlelength=0.8)

    # Binned mean abs error by σ_conf bins
    bins = np.linspace(0, 1.0, 11)
    centers = 0.5 * (bins[1:] + bins[:-1])
    means, stds = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d["sigma_conf"] >= lo) & (d["sigma_conf"] < hi)
        if m.sum() < 5:
            means.append(np.nan); stds.append(np.nan); continue
        means.append(d.loc[m, "abs_error"].mean())
        stds.append(d.loc[m, "abs_error"].std())
    means = np.array(means); stds = np.array(stds)
    ax2.errorbar(centers, means, yerr=stds, marker="o",
                 markersize=5, color=OURS, ecolor="#555",
                 elinewidth=0.8, capsize=2, zorder=3,
                 label="Bin mean ± std")
    ax2.axhline(14.38, color="#555", lw=0.8, ls="--")
    ax2.set_xlabel(r"$\sigma_{\rm conf}$ bin centre")
    ax2.set_ylabel(r"Mean $|$error$|$ in bin")
    ax2.set_title("b  Error grows with σ_conf — physics gate works", loc="left")
    ax2.set_xlim(0, 1)
    ax2.legend(loc="upper left", fontsize=6.5)

    save(fig, "figS15_sigma_conf_vs_error")


if __name__ == "__main__":
    fig_optuna_convergence()
    fig_shap_importance()
    fig_lofo_parity()
    fig_counterfactual_heatmap()
    fig_permutation_null()
    fig_sigma_conf_vs_error()
    print("\nAll 6 supplementary figures generated.")
