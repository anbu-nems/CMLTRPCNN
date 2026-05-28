#!/usr/bin/env python3
"""
Additional PCNN figures generated from results JSONs.

Outputs to /Users/anbu/Desktop/NC figures/all_figures/:
  fig_decomposition.pdf       — εr decomposition (CM 49.3 / LST 51.0 / Tilt -1.5 / Res 0.6)
  fig_asite_hierarchy.pdf     — A-site f_LST hierarchy + predicted-vs-literature comparison
  fig_law_ablation.pdf        — 2^3 factorial ablation of physics laws I/II/III
  fig_branch_ablation.pdf     — R² drop when each branch is removed
  fig_flst_by_regime.pdf      — f_LST by Reaney regime (Ia/Ib/II/III)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

# --- self-contained release root (auto-injected) ---
RELEASE_ROOT = os.path.abspath(os.path.dirname(__file__))
while RELEASE_ROOT != os.path.dirname(RELEASE_ROOT) and not os.path.isdir(os.path.join(RELEASE_ROOT, 'model_weights')):
    RELEASE_ROOT = os.path.dirname(RELEASE_ROOT)
# ----------------------------------------------------




RESULTS_DIR = os.path.join(RELEASE_ROOT, "results")
OUT_DIR     = os.path.join(RELEASE_ROOT, "all_figures")
os.makedirs(OUT_DIR, exist_ok=True)

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
        sp.set_visible(True)
        sp.set_linewidth(lw)
    ax.tick_params(direction="in", top=False, right=False,
                   bottom=True, left=True, width=0.8, length=3)
    ax.set_axisbelow(False)

C = {
    "cm":   "#264653",
    "lst":  "#2A9D8F",
    "tilt": "#E9C46A",
    "res":  "#E76F51",
    "ours": "#E76F51",
    "gray": "#78909C",
}

ASITE_COLORS = {
    "Pb": "#E76F51", "Ca": "#F4A261", "La": "#2A9D8F",
    "Sr": "#E9C46A", "Ba": "#264653",
}

REGIME_COLORS = {
    "Ia":  "#264653", "Ib":  "#2A9D8F",
    "II":  "#E9C46A", "III": "#E76F51",
}


def save(fig, name):
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ── 1) Decomposition figure ───────────────────────────────────────────────────
def fig_decomposition():
    # Numbers from CLAUDE.md Finding 3 / Script 39
    labels = ["εr_CM\n(Clausius-Mossotti)",
              "δ_LST\n(soft-mode)",
              "δ_tilt\n(octahedral tilt)",
              "δ_res\n(residual)"]
    values = [49.3, 51.0, -1.5, 0.6]
    colors = [C["cm"], C["lst"], C["tilt"], C["res"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    style4(ax1); style4(ax2)

    # (a) signed bar
    x = np.arange(len(labels))
    bars = ax1.bar(x, values, width=0.55, color=colors,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax1.axhline(0, color="#888", lw=0.8, ls="--", zorder=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("Contribution to predicted εr (%)")
    ax1.set_title("a  Physics-branch contributions", loc="left")
    ax1.set_ylim(-10, 70)
    for b, v in zip(bars, values):
        pos = v + 1.5 if v >= 0 else v - 3.0
        ax1.text(b.get_x() + b.get_width() / 2, pos,
                 f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=7.5, fontweight="bold", color="#222")

    # (b) cumulative waterfall
    sums = [0, values[0],
            values[0] + values[1],
            values[0] + values[1] + values[2],
            values[0] + values[1] + values[2] + values[3]]
    for i, (lab, v, col) in enumerate(zip(labels, values, colors)):
        bottom = sums[i] if v >= 0 else sums[i] + v
        height = abs(v)
        ax2.bar(i, height, bottom=bottom, width=0.55, color=col,
                edgecolor="white", linewidth=0.5, zorder=3)
        ax2.text(i, bottom + height + 1.5, f"{v:+.1f}%",
                 ha="center", va="bottom", fontsize=7.5,
                 fontweight="bold", color="#222")
        if i > 0:
            ax2.plot([i - 1 + 0.275, i - 0.275],
                     [sums[i], sums[i]], color="#888", lw=0.6, ls=":")
    ax2.axhline(100, color="#444", lw=0.8, ls="--", zorder=2)
    ax2.text(3.4, 101, "100% (full εr)", fontsize=6.5, color="#444",
             ha="right", va="bottom")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Cumulative εr contribution (%)")
    ax2.set_title("b  Cumulative buildup", loc="left")
    ax2.set_ylim(-10, 115)

    fig.suptitle("PCNN physics decomposition  ($n$=1304 compositions)",
                 fontsize=9, fontweight="bold", y=1.04)
    save(fig, "fig_decomposition")


# ── 2) A-site hierarchy + literature validation ───────────────────────────────
def fig_asite_hierarchy():
    with open(f"{RESULTS_DIR}/44_literature_validation.json") as f:
        d = json.load(f)

    order = ["Pb", "Ca", "La", "Sr", "Ba"]
    pred_total = [d["by_asite"][a]["predicted_total"]["mean"] for a in order]
    pred_lst   = [d["by_asite"][a]["predicted_lst"]["mean"]   for a in order]
    emp        = [d["by_asite"][a]["empirical"]["mean"]       for a in order]

    n_dict = {a: d["by_asite"][a]["n"] for a in order}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    style4(ax1); style4(ax2)

    # (a) A-site hierarchy: predicted f_LST
    x = np.arange(len(order))
    cols = [ASITE_COLORS[a] for a in order]
    bars = ax1.bar(x, pred_lst, width=0.55, color=cols,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{a}\n($n$={n_dict[a]})" for a in order], fontsize=7)
    ax1.set_ylabel("Predicted f_LST multiplier (×)")
    ax1.set_title("a  A-site soft-mode hierarchy", loc="left")
    for b, v in zip(bars, pred_lst):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.05,
                 f"{v:.2f}×", ha="center", va="bottom",
                 fontsize=7.5, fontweight="bold", color="#222")
    ax1.set_ylim(0, max(pred_lst) * 1.25)
    ax1.axhline(1.0, color="#888", lw=0.6, ls=":")

    # (b) predicted vs literature-aggregated f_LST
    ax2.plot([0, 4], [0, 4], color="#888", lw=0.8, ls="--", zorder=1,
             label="y = x")
    for a, p, e in zip(order, pred_total, emp):
        ax2.scatter(e, p, s=80, color=ASITE_COLORS[a],
                    edgecolor="white", linewidths=0.8, zorder=4,
                    label=f"{a} (n={n_dict[a]})")
        ax2.annotate(a, xy=(e, p), xytext=(6, 4),
                     textcoords="offset points",
                     fontsize=7.5, fontweight="bold",
                     color=ASITE_COLORS[a])
    ax2.set_xlabel("Empirical f_LST (literature)")
    ax2.set_ylabel("PCNN predicted f_LST")
    ax2.set_title("b  PCNN predicted vs literature-aggregated f_LST", loc="left")
    ax2.set_xlim(1.4, 3.6); ax2.set_ylim(1.4, 3.6)
    ax2.legend(loc="lower right", fontsize=6.5, handlelength=0.8)

    fig.suptitle("A-site controls soft-mode magnitude — quantified per family",
                 fontsize=9, fontweight="bold", y=1.04)
    save(fig, "fig_asite_hierarchy")


# ── 3) 2^3 factorial law ablation ─────────────────────────────────────────────
def fig_law_ablation():
    with open(f"{RESULTS_DIR}/59_physics_law_ablation.json") as f:
        d = json.load(f)

    order = ["000", "I", "II", "III", "I+II", "I+III", "II+III", "I+II+III"]
    nice  = ["None",
             "CM only", "LST only", "Tilt only",
             "CM+LST", "CM+Tilt", "LST+Tilt",
             "CM+LST+Tilt"]
    r2  = [d["conditions"][k]["r2"]  for k in order]
    mae = [d["conditions"][k]["mae"] for k in order]

    cols = ["#B0BEC5"] * 8
    cols[order.index("II")]       = C["lst"]
    cols[order.index("I+II")]     = "#F4A261"
    cols[order.index("I+II+III")] = C["ours"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    style4(ax1); style4(ax2)

    y = np.arange(len(order))
    bars = ax1.barh(y, r2, height=0.6, color=cols,
                    edgecolor="white", linewidth=0.5, zorder=3)
    ax1.axvline(0, color="#888", lw=0.8, ls="--")
    ax1.set_yticks(y); ax1.set_yticklabels(nice)
    ax1.set_xlabel("Holdout $R^2$  ($n$=116)")
    ax1.set_title("a  Physics-law factorial ablation", loc="left")
    ax1.invert_yaxis()
    ax1.set_xlim(-1.0, 1.05)
    for b, v in zip(bars, r2):
        xp = v + 0.03 if v >= 0 else v - 0.04
        ha = "left" if v >= 0 else "right"
        ax1.text(xp, b.get_y() + b.get_height()/2,
                 f"{v:.3f}", va="center", ha=ha,
                 fontsize=6.8, color="#222")

    bars2 = ax2.barh(y, mae, height=0.6, color=cols,
                     edgecolor="white", linewidth=0.5, zorder=3)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xlabel("Holdout MAE (εr units)")
    ax2.set_title("b  MAE under same conditions", loc="left")
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(mae) * 1.15)
    for b, v in zip(bars2, mae):
        ax2.text(b.get_width() + 0.4, b.get_y() + b.get_height()/2,
                 f"{v:.2f}", va="center", fontsize=6.8, color="#222")

    legend_items = [
        mpatches.Patch(color=C["lst"], label="LST alone (Law II)"),
        mpatches.Patch(color="#F4A261", label="CM+LST"),
        mpatches.Patch(color=C["ours"], label="Full (CM+LST+Tilt)"),
        mpatches.Patch(color="#B0BEC5", label="Other"),
    ]
    fig.legend(handles=legend_items, loc="upper center",
               bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=6.5,
               framealpha=0.93, edgecolor="#CCCCCC")
    save(fig, "fig_law_ablation")


# ── 4) Branch ablation (drop in R²) ───────────────────────────────────────────
def fig_branch_ablation():
    with open(f"{RESULTS_DIR}/53_ablation_v77.json") as f:
        d = json.load(f)

    full = d["full_model_r2"]
    drops = d["r2_drops"]
    order   = ["no_cm", "no_lst", "no_tilt", "no_res"]
    labels  = ["Remove CM\n(Law I)",
               "Remove LST\n(Law II)",
               "Remove Tilt\n(Law III)",
               "Remove Residual"]
    vals    = [drops[k] for k in order]
    cols    = [C["cm"], C["lst"], C["tilt"], C["res"]]

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    style4(ax)

    x = np.arange(len(order))
    bars = ax.bar(x, vals, width=0.55, color=cols,
                  edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("ΔR² when branch removed")
    ax.set_title(f"Branch importance  (full model $R^2$ = {full:.3f})",
                 loc="left")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2,
                v + 0.02 if v >= 0 else v - 0.02,
                f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8, fontweight="bold", color="#222")

    ax.set_ylim(min(vals) * 1.2 - 0.05, max(vals) * 1.2)
    ax.text(0.02, 0.96,
            "Larger drop  →  more essential branch",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, style="italic", color="#555")
    save(fig, "fig_branch_ablation")


# ── 5) f_LST by Reaney regime ─────────────────────────────────────────────────
def fig_flst_by_regime():
    with open(f"{RESULTS_DIR}/39_physics_decomposition.json") as f:
        d = json.load(f)

    order = ["Ia", "Ib", "II", "III"]
    means = [d["f_lst_by_regime"][r]["mean"]   for r in order]
    stds  = [d["f_lst_by_regime"][r]["std"]    for r in order]
    ns    = [d["f_lst_by_regime"][r]["n"]      for r in order]
    cols  = [REGIME_COLORS[r] for r in order]

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    style4(ax)

    x = np.arange(len(order))
    bars = ax.bar(x, means, yerr=stds, width=0.55, color=cols,
                  edgecolor="white", linewidth=0.5, zorder=3,
                  error_kw=dict(elinewidth=1.0, capsize=3, ecolor="#333"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"Regime {r}\n($n$={n})" for r, n in zip(order, ns)],
                       fontsize=7)
    ax.set_ylabel("f_LST  (mean ± std)")
    ax.set_title("Soft-mode enhancement by Reaney regime", loc="left")
    ax.axhline(1.0, color="#888", lw=0.6, ls=":")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, m + 0.05,
                f"{m:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#222")
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.2)
    ax.text(0.02, 0.96,
            "f_LST > 1 → soft-mode enhances εr above CM",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, style="italic", color="#555")
    save(fig, "fig_flst_by_regime")


# ── 6) Dual-layer Applicability Domain ────────────────────────────────────────
def fig_ad_dual():
    with open(f"{RESULTS_DIR}/49_ad_v77.json") as f:
        d = json.load(f)

    L1 = d["layer1_physics_gate"]
    L2 = d["layer2_kde"]
    COMB = d["combined_ad"]
    VC = d["validation_calibration"]
    asite = d["asite_ad"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    ax1, ax2, ax3 = axes
    for ax in axes:
        style4(ax)

    # ── Panel A: in-domain coverage by layer ─────────────────────────────────
    labels = ["Layer 1\nphysics gate\n(σ_conf)",
              "Layer 2\nKDE density\n(PCA+KDE)",
              "Combined\n(both layers)"]
    pct_total = [L1["pct_physics_reliable"],
                 L2["pct_train_in_domain"],
                 COMB["pct_total_in_ad"]]
    cols = [C["lst"], C["tilt"], C["ours"]]
    x = np.arange(len(labels))
    bars = ax1.bar(x, pct_total, width=0.55, color=cols,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.5)
    ax1.set_ylabel("In-domain coverage (%)")
    ax1.set_title("a  Dual-layer AD coverage", loc="left")
    ax1.set_ylim(0, 110)
    ax1.axhline(100, color="#888", lw=0.6, ls=":")
    for b, v in zip(bars, pct_total):
        ax1.text(b.get_x() + b.get_width()/2, v + 2,
                 f"{v:.1f}%", ha="center", va="bottom",
                 fontsize=7.5, fontweight="bold", color="#222")

    # ── Panel B: R² in vs out for each layer ─────────────────────────────────
    layer_lbls = ["Layer 1", "Layer 2", "Combined"]
    r2_in = [VC["r2_layer1_in"], VC["r2_layer2_in"], VC["r2_combined_in"]]
    r2_out = [VC["r2_layer1_out"], VC["r2_layer2_out"], VC["r2_combined_out"]]

    x2 = np.arange(len(layer_lbls))
    w = 0.35
    b1 = ax2.bar(x2 - w/2, r2_in,  w, color=C["lst"], edgecolor="white",
                 linewidth=0.5, zorder=3, label="In-domain")
    b2 = ax2.bar(x2 + w/2, r2_out, w, color="#B0BEC5", edgecolor="white",
                 linewidth=0.5, zorder=3, label="Out-of-domain")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(layer_lbls, fontsize=7)
    ax2.set_ylabel("Holdout $R^2$")
    ax2.set_title("b  Reliability — in vs out", loc="left")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower left", fontsize=6.5)
    for bars, vals in [(b1, r2_in), (b2, r2_out)]:
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width()/2, v + 0.015,
                     f"{v:.3f}", ha="center", va="bottom",
                     fontsize=6.5, color="#222")

    # ── Panel C: per A-site in-domain by both layers ─────────────────────────
    a_order = ["Pb", "La", "Ba", "Ca", "Sr"]
    pct_L1   = [asite[a]["pct_layer1_physics"] for a in a_order]
    pct_L2   = [asite[a]["pct_layer2_kde"]     for a in a_order]
    pct_comb = [asite[a]["pct_combined_in_ad"] for a in a_order]
    ns       = [asite[a]["n"] for a in a_order]

    y = np.arange(len(a_order))
    h = 0.27
    ax3.barh(y - h, pct_L1,   h, color=C["lst"],  edgecolor="white",
             linewidth=0.4, zorder=3, label="Layer 1 (σ_conf)")
    ax3.barh(y,     pct_L2,   h, color=C["tilt"], edgecolor="white",
             linewidth=0.4, zorder=3, label="Layer 2 (KDE)")
    ax3.barh(y + h, pct_comb, h, color=C["ours"], edgecolor="white",
             linewidth=0.4, zorder=3, label="Combined")
    ax3.set_yticks(y)
    ax3.set_yticklabels([f"{a}\n($n$={n})" for a, n in zip(a_order, ns)],
                        fontsize=7)
    ax3.set_xlabel("In-domain (%)")
    ax3.set_title("c  By A-site cation", loc="left")
    ax3.invert_yaxis()
    ax3.set_xlim(0, 110)
    ax3.legend(loc="lower right", fontsize=6.0, handlelength=0.8)

    fig.suptitle("Dual-layer applicability domain  "
                 "(Layer 1: physics gate · Layer 2: KDE)",
                 fontsize=9, fontweight="bold", y=1.04)
    save(fig, "fig_applicability_domain")


# ── 7) Re-run feature partition figure with corrected 103 count ──────────────
def fig_feature_partition_103():
    """Bar chart showing the 30/30/43 partition of 103 physics features."""
    groups  = ["LST\nbranch", "Tilt\nbranch", "Residual\nbranch"]
    counts  = [30, 30, 43]
    cols    = [C["lst"], C["tilt"], C["res"]]
    descrip = ["d⁰_per_field, chi_A, ionic radii,\nelectronegativity, polarizability …",
               "Goldschmidt t, tilt severity,\nsymmetry score, regime flags …",
               "Composition encoding, atomic\nfraction, mass features …"]

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    style4(ax)
    x = np.arange(len(groups))
    bars = ax.bar(x, counts, width=0.55, color=cols,
                  edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("Number of features")
    ax.set_title("Strict 3-way feature partition  (total = 103)",
                 loc="left")
    ax.set_ylim(0, max(counts) * 1.35)
    for b, n, txt in zip(bars, counts, descrip):
        ax.text(b.get_x() + b.get_width()/2, n + 1.2,
                f"{n}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#222")
        ax.text(b.get_x() + b.get_width()/2, 2.0, txt,
                ha="center", va="bottom", fontsize=6.0,
                color="white", style="italic")
    save(fig, "fig_feature_partition")


if __name__ == "__main__":
    fig_decomposition()
    fig_asite_hierarchy()
    fig_law_ablation()
    fig_branch_ablation()
    fig_flst_by_regime()
    fig_ad_dual()
    fig_feature_partition_103()
    print("\nAll additional figures saved to:", OUT_DIR)
