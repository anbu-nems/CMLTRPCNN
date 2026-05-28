#!/usr/bin/env python3
"""
12-panel structure-property correlation figure (VIBANN Fig 5 style).

Each panel shows one physical descriptor vs PCNN-predicted εr, with points
colored by Reaney regime. Reveals that many independent crystal-chemistry
descriptors track εr in physically expected ways.

Inputs (already extracted):
  ./extracted_data/decomposition_per_sample.csv
  ./data/processed/feature_matrix_v7.parquet

Outputs:
  all_figures/supp/figS21_descriptor_panels.{pdf,png}
  all_figures/extracted_data/descriptors_augmented.csv   ← full data dump for reuse
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os

OUT_FIG = "./figures_output/all_figures/supp"
OUT_CSV = "./extracted_data"
PIML    = "/Users/anbu/Desktop/PIML/piml_ceramic"
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_CSV, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 7, "legend.frameon": True,
    "legend.framealpha": 0.92, "legend.edgecolor": "#CCCCCC",
    "figure.constrained_layout.use": True,
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,  "ytick.major.size": 2.5,
    "axes.grid": False,
})

def style4(ax, lw=0.9):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw)
    ax.tick_params(direction="in", top=False, right=False,
                   bottom=True, left=True, width=0.7, length=2.5)
    ax.set_axisbelow(False)

REGIME_COLORS = {"Ia": "#264653", "Ib": "#2A9D8F", "II": "#E9C46A", "III": "#E76F51"}

# ── Load + augment data ─────────────────────────────────────────────────────
d  = pd.read_csv(f"{OUT_CSV}/decomposition_per_sample.csv")
fm = pd.read_parquet(f"{PIML}/data/processed/feature_matrix_v7.parquet")

# Add descriptors not already in decomposition CSV
extra_cols = ["chi_A", "chi_B", "d0_per_field", "d0_B_fraction",
              "alpha_total", "a_off_centering_frac",
              "phase_transition", "ordering_proxy",
              "soft_mode_tolerance_weighted", "RE_x_chi_A"]
for c in extra_cols:
    if c in fm.columns:
        d[c] = fm[c].values

# Derived: f_LST = δ_LST / εr_CM
d["f_lst"] = np.where(d["er_cm"] > 0, d["delta_lst"] / d["er_cm"], np.nan)

# Save augmented CSV for reproducibility
aug_path = f"{OUT_CSV}/descriptors_augmented.csv"
d.to_csv(aug_path, index=False)
print(f"Saved augmented descriptors CSV: {aug_path}  ({len(d)} rows, {len(d.columns)} cols)")

# Restrict to known regimes for plotting
d_plot = d[d["regime"].isin(["Ia", "Ib", "II", "III"])].copy()

# ── 12 panels ───────────────────────────────────────────────────────────────
panels = [
    # (column, x-label, panel letter, axis log/lin)
    ("tolerance_factor",            r"Goldschmidt $t$",                       "a",  None),
    ("er_cm",                       r"$\varepsilon_r^{CM}$ (CM baseline)",    "b",  None),
    ("delta_lst",                   r"$\delta_{LST}$ (soft-mode)",            "c",  None),
    ("delta_tilt",                  r"$\delta_{tilt}$ (octahedral tilt)",     "d",  None),
    ("delta_res",                   r"$\delta_{res}$ (residual)",             "e",  None),
    ("sigma_conf",                  r"$\sigma_{conf}$ (confidence gate)",     "f",  None),
    ("f_lst",                       r"$f_{LST}=\delta_{LST}/\varepsilon_r^{CM}$", "g", None),
    ("d0_per_field",                r"d$^0$_per_field (B-site)",              "h",  None),
    ("alpha_total",                 r"$\alpha_{total}$ (polarizability, Å³)", "i",  None),
    ("chi_A",                       r"$\chi_A$ (A-site electronegativity)",   "j",  None),
    ("gii",                         "GII (global instability index)",          "k",  None),
    ("ensemble_std",                r"Ensemble $\sigma_{ML}$ (epistemic)",     "l",  None),
]

fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.8))
axes = axes.flatten()

for ax, (col, xlab, letter, _) in zip(axes, panels):
    style4(ax)
    if col not in d_plot.columns:
        ax.text(0.5, 0.5, f"missing\n{col}", transform=ax.transAxes,
                ha="center", va="center", color="#888", fontsize=8)
        continue
    for reg in ["Ia", "Ib", "II", "III"]:
        sub = d_plot[d_plot["regime"] == reg]
        if len(sub) == 0: continue
        ax.scatter(sub[col], sub["er_predicted"],
                   s=4, color=REGIME_COLORS[reg], alpha=0.4,
                   edgecolors="none", zorder=3)
    # Spearman rank correlation as a sanity number
    valid = d_plot[[col, "er_predicted"]].dropna()
    if len(valid) > 10:
        rho = valid[col].corr(valid["er_predicted"], method="spearman")
        ax.text(0.97, 0.96, fr"$\rho$={rho:+.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.0, color="#333",
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="#CCCCCC", lw=0.4, alpha=0.85))
    ax.set_xlabel(xlab, fontsize=7)
    ax.set_ylabel(r"$\varepsilon_r$ (predicted)", fontsize=7)
    # Panel label
    ax.text(-0.18, 1.05, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, fontweight="bold")

# Global legend
legend_items = [
    Line2D([0],[0], marker="o", linestyle="",
           markerfacecolor=REGIME_COLORS[r], markeredgecolor="white",
           markeredgewidth=0.3, markersize=6, label=f"Regime {r}")
    for r in ["Ia", "Ib", "II", "III"]
]
fig.legend(handles=legend_items, loc="upper center",
           bbox_to_anchor=(0.5, 1.04), ncol=4, fontsize=7,
           framealpha=0.92, edgecolor="#CCCCCC")

fig.suptitle("Structure-property correlations — 12 physical descriptors vs PCNN-predicted ε$_r$",
             fontsize=9.5, fontweight="bold", y=1.07)

# Save
for ext in ("pdf", "png"):
    path = f"{OUT_FIG}/figS21_descriptor_panels.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
