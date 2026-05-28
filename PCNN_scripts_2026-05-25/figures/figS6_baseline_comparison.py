#!/usr/bin/env python3
"""
Supplementary Fig S6 — Baseline comparison: all models, CV + holdout R²
Nature Communications style (sans-serif, 7.2 in wide)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import os

OUT_DIR = "/Users/anbu/Desktop/NC figures/all_figures/supp"
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

# ── Data (from 57_baseline_comparison.json + CLAUDE.md) ──────────────────────
models = [
    "CM-only",
    "Ridge",
    "Random Forest",
    "XGBoost",
    "CatBoost",
    "BPNN/MLP",
    "PIRNN",
    "PCNN (ours)",
]

cv_r2_mean = [-0.862, 0.721, 0.901, 0.928, 0.930, 0.914, 0.863, 0.893]
cv_r2_std  = [ 0.355, 0.190, 0.057, 0.053, 0.041, 0.093, 0.181, 0.043]
holdout_r2 = [-0.492, 0.867, 0.894, 0.879, 0.910, 0.957, 0.937, 0.941]
holdout_mae= [26.59,  9.19,  7.87,  8.08,  7.04,  5.40,  6.00,  5.77]

# Colors: gray for baselines, coral for ours
OURS   = "#E76F51"
GRAY   = "#78909C"
CM_COL = "#B0BEC5"
colors = [CM_COL, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, OURS]

n = len(models)
y = np.arange(n)

fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2))
ax_cv, ax_ho, ax_mae = axes
for ax in axes:
    style4(ax)

# ── Panel A: Strat-GSS CV R² ──────────────────────────────────────────────────
ax = ax_cv
bars = ax.barh(y, cv_r2_mean, height=0.55, color=colors,
               edgecolor="white")
ax.errorbar(cv_r2_mean, y,
            xerr=cv_r2_std, fmt="none",
            ecolor="#333", elinewidth=1.0, capsize=3, zorder=4)
ax.axvline(0, color="#888", lw=0.8, ls="--", zorder=2)
# Highlight PCNN bar outline
bars[-1].set_edgecolor(OURS)
bars[-1].set_linewidth(1.5)
ax.set_yticks(y)
ax.set_yticklabels(models)
ax.set_xlabel("Strat-GSS CV $R^2$ (mean ± std)")
ax.set_title("a", loc="left", fontweight="bold")
ax.set_xlim(-1.30, 1.05)
# Value labels
for bar, val, err in zip(bars, cv_r2_mean, cv_r2_std):
    xpos = max(val, 0) + 0.04
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=6.5, color="#333")
ax.invert_yaxis()

# ── Panel B: Holdout R² ──────────────────────────────────────────────────────
ax = ax_ho
bars2 = ax.barh(y, holdout_r2, height=0.55, color=colors,
                edgecolor="white")
ax.axvline(0, color="#888", lw=0.8, ls="--", zorder=2)
bars2[-1].set_edgecolor(OURS)
bars2[-1].set_linewidth(1.5)
ax.set_yticks(y)
ax.set_yticklabels([])
ax.set_xlabel("Holdout $R^2$  ($n$=116)")
ax.set_title("b", loc="left", fontweight="bold")
ax.set_xlim(-0.65, 1.06)
for bar, val in zip(bars2, holdout_r2):
    xpos = max(val, 0) + 0.02
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=6.5, color="#333")
ax.invert_yaxis()

# ── Panel C: Holdout MAE ─────────────────────────────────────────────────────
ax = ax_mae
bars3 = ax.barh(y, holdout_mae, height=0.55, color=colors,
                edgecolor="white")
bars3[-1].set_edgecolor(OURS)
bars3[-1].set_linewidth(1.5)
ax.set_yticks(y)
ax.set_yticklabels([])
ax.set_xlabel("Holdout MAE ($\\varepsilon_r$ units)")
ax.set_title("c", loc="left", fontweight="bold")
ax.set_xlim(0, 30)   # bars anchored at zero (touch y-axis)
for bar, val in zip(bars3, holdout_mae):
    ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=6.5, color="#333")
ax.invert_yaxis()

# (interpretive "physics constraints reduce OOD variance…" note moved to the caption)

# ── Legend (bottom-centre, clear of the panel titles and the 26.59 MAE bar) ────
legend_items = [
    mpatches.Patch(color=OURS,   label="PCNN (ours)"),
    mpatches.Patch(color=GRAY,   label="Baselines"),
    mpatches.Patch(color=CM_COL, label="CM-only (physics anchor)"),
]
fig.legend(handles=legend_items, loc="lower center",
           bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=6.5,
           framealpha=0.93, edgecolor="#CCCCCC")

for ext in ["pdf", "png"]:
    path = os.path.join(OUT_DIR, f"figS6_baseline_comparison.{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
print("Fig S6 — Baseline comparison done.")
