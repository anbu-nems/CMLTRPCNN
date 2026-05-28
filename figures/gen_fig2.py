"""
Figure 2 — Parity plot + CV performance summary
Nature Communications: Physics-constrained neural networks for ceramic dielectric constant prediction
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import os

# ── NC journal style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": True,
    "legend.framealpha": 0.9, "legend.edgecolor": "#CCCCCC",
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": False,
    "lines.linewidth": 1.5, "lines.markersize": 5,
})

def style4(ax, lw=1.0):
    """Full 4-sided box with inward ticks on bottom and left only."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
    ax.tick_params(direction="in", top=False, right=False,
                   bottom=True, left=True, width=0.8, length=3)
    ax.set_axisbelow(False)

REGIME_COLORS = {"Ia": "#264653", "Ib": "#2A9D8F", "II": "#E9C46A", "III": "#E76F51"}
OUR_COLOR = "#2874A6"
ORANGE    = "#E67E22"
GRAY      = "#B0BEC5"
FIG_FULL  = (7.2, 3.2)

DATA_CSV = "/Users/anbu/Desktop/PIML/piml_ceramic/data/processed/test_holdout_predictions.csv"
OUT_DIR  = "/Users/anbu/Desktop/NC figures/main"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_CSV)
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=FIG_FULL, layout="constrained")
gs  = fig.add_gridspec(1, 2, width_ratios=[4.0, 3.0])
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
style4(ax_a)
style4(ax_b)

# ── Panel A: Parity scatter ───────────────────────────────────────────────────
for regime, grp in df.groupby("reaney_regime"):
    color = REGIME_COLORS.get(regime, GRAY)
    ax_a.scatter(grp["er_measured"], grp["er_predicted"],
                 s=18, alpha=0.7, color=color,
                 edgecolors="none", zorder=3,
                 label=f"Regime {regime}")

lims = (0, 145)
ax_a.plot(lims, lims, "k--", lw=1.0, zorder=2)
ax_a.set_xlim(lims)
ax_a.set_ylim(lims)
ax_a.set_xlabel("Measured $\\varepsilon_r$")
ax_a.set_ylabel("Predicted $\\varepsilon_r$")
ax_a.set_title("a  Held-out calibration set  ($n$=123,  $R^2$=0.921)", loc="left")

handles = [mpatches.Patch(color=REGIME_COLORS[r], label=f"Regime {r}")
           for r in ["Ia", "Ib", "II", "III"]]
ax_a.legend(handles=handles, loc="upper left", handlelength=1.2, handleheight=1.0)

ax_a.text(0.97, 0.04,
          "Formula-split CV:  $R^2$ = 0.934 ± 0.031\n"
          "Strat-GSS CV:         $R^2$ = 0.893 ± 0.053",
          transform=ax_a.transAxes, fontsize=7,
          ha="right", va="bottom",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                    edgecolor="#CCCCCC", alpha=0.92))

# ── Panel B: Per-fold R² ──────────────────────────────────────────────────────
folds        = [1, 2, 3, 4, 5]
formula_r2   = [0.926, 0.948, 0.968, 0.878, 0.950]
gss_r2       = [0.904, 0.943, 0.811, 0.950, 0.856]
formula_mean = 0.934
gss_mean     = 0.893

ax_b.plot(folds, formula_r2, color=OUR_COLOR, marker="o", markersize=5, label="Formula-split")
ax_b.plot(folds, gss_r2,     color=ORANGE,    marker="s", markersize=5,
          linestyle="--", label="Strat-GSS (OOD)")
ax_b.axhline(formula_mean, color=OUR_COLOR, linestyle=":", lw=1.0)
ax_b.axhline(gss_mean,     color=ORANGE,    linestyle=":", lw=1.0)
ax_b.fill_between(folds, formula_r2, formula_mean, color=OUR_COLOR, alpha=0.08)
ax_b.fill_between(folds, gss_r2,     gss_mean,     color=ORANGE,    alpha=0.08)

ax_b.set_ylim(0.75, 1.0)
ax_b.set_xlabel("Fold")
ax_b.set_ylabel("$R^2$")
ax_b.set_title("b  Per-fold cross-validation $R^2$", loc="left")
ax_b.set_xticks(folds)

legend_elements = [
    Line2D([0], [0], color=OUR_COLOR, marker="o", markersize=5, label="Formula-split"),
    Line2D([0], [0], color=ORANGE,    marker="s", markersize=5,
           linestyle="--", label="Strat-GSS (OOD)"),
    Line2D([0], [0], color=OUR_COLOR, linestyle=":", lw=1.0,
           label=f"Mean = {formula_mean}"),
    Line2D([0], [0], color=ORANGE,    linestyle=":", lw=1.0,
           label=f"Mean = {gss_mean}"),
]
ax_b.legend(handles=legend_elements, handlelength=1.8)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    path = os.path.join(OUT_DIR, f"fig2_parity_cv.{ext}")
    fig.savefig(path, dpi=300)
    print(f"Saved: {path}")
plt.close(fig)
print("Figure 2 done.")
