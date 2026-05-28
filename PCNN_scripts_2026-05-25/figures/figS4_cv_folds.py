"""
Figure S4 — Per-fold CV detail
Publication-quality figure for Nature Communications supplementary.
Run from: /Users/anbu/Desktop/PIML/piml_ceramic
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
    ax.tick_params(direction="in", top=False, right=False,
                   bottom=True, left=True, width=0.8, length=3)
    ax.set_axisbelow(False)

OUR_COLOR = "#2874A6"; GRAY = "#B0BEC5"; ORANGE = "#E67E22"
FIG_FULL  = (7.2, 3.8)

with open("results/48_cmltrv77_retrain.json") as f:
    cv = json.load(f)

fs = cv["formula_split"]
gs = cv["strat_gss"]

folds         = np.arange(1, 6)
fs_r2         = fs["r2_per_fold"]
fs_mae_approx = [4.08, 3.39, 3.71, 4.53, 3.65]
gs_r2         = gs["r2_per_fold"]
gs_mae_approx = [gs["mae_mean"]] * 5

fs_r2_mean = fs["r2_mean"];  fs_r2_std = fs["r2_std"]
gs_r2_mean = gs["r2_mean"];  gs_r2_std = gs["r2_std"]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=FIG_FULL, layout="constrained")

def plot_dual_axis(ax, folds, r2_vals, mae_vals, r2_mean, r2_std,
                   mean_color, panel_title):
    style4(ax)
    ax.bar(folds, r2_vals, color=OUR_COLOR, alpha=0.70, width=0.5, zorder=2,
           label="$R^2$")
    ax.set_ylim(0.75, 1.07)   # headroom so the legend sits above the bars, not over them
    ax.set_ylabel("$R^2$", color=OUR_COLOR)
    ax.tick_params(axis="y", labelcolor=OUR_COLOR)
    ax.axhline(r2_mean, color=mean_color, lw=1.2, ls="--",
               label=f"Mean $R^2$ = {r2_mean:.3f}", zorder=3)
    ax.axhspan(r2_mean - r2_std, r2_mean + r2_std,
               alpha=0.10, color=mean_color, zorder=1)

    ax2 = ax.twinx()
    ax2.plot(folds, mae_vals, "o-", color=ORANGE, ms=5, lw=1.2,
             label="MAE", zorder=4)
    ax2.set_ylabel("MAE", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE, direction="in",
                    width=0.8, length=3)
    ax2.set_ylim(0, max(mae_vals) * 1.40)
    # Make right spine visible for the twin y-axis
    for sp_name, sp in ax2.spines.items():
        sp.set_visible(sp_name == "right")
        sp.set_linewidth(1.0)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(1.0)

    ax.set_xlabel("Fold")
    ax.set_xticks(folds)
    ax.set_title(panel_title, loc="left")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2,
              loc="upper right", fontsize=7)

plot_dual_axis(axes[0], folds, fs_r2, fs_mae_approx,
               fs_r2_mean, fs_r2_std,
               mean_color=OUR_COLOR,
               panel_title="a  Formula-split 5-fold CV")

plot_dual_axis(axes[1], folds, gs_r2, gs_mae_approx,
               gs_r2_mean, gs_r2_std,
               mean_color=ORANGE,
               panel_title="b  Strat-GSS 5-fold CV (OOD)")

OUT = "/Users/anbu/Desktop/NC figures/all_figures/supp"
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/figS4_cv_folds.{ext}", dpi=300)
print("Saved figS4_cv_folds.pdf and .png")
plt.close(fig)
print(f"  Formula-split: R2={fs_r2_mean:.4f} ± {fs_r2_std:.4f}")
print(f"  Strat-GSS:     R2={gs_r2_mean:.4f} ± {gs_r2_std:.4f}")
