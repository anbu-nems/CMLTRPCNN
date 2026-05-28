"""
Figure S5 — Residual diagnostics
Publication-quality figure for Nature Communications supplementary.
Run from: /Users/anbu/Desktop/PIML/piml_ceramic
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew as scipy_skew

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

REGIME_COLORS = {"Ia": "#264653", "Ib": "#2A9D8F", "II": "#E9C46A", "III": "#E76F51"}
OUR_COLOR = "#2874A6"; GRAY = "#B0BEC5"; RED = "#E74C3C"
FIG_FULL  = (7.2, 3.2)

PROC_DIR = "./data/processed"

df        = pd.read_csv(f"{PROC_DIR}/test_holdout_predictions.csv")
calib_idx = json.load(open(f"{PROC_DIR}/calibration_split_idx.json"))["calib_idx"]
fm        = pd.read_parquet(f"{PROC_DIR}/feature_matrix_v7.parquet")
hold_fm   = fm.iloc[calib_idx].reset_index(drop=True)

def _reg(row):
    for r in ["Ia", "Ib", "II", "III"]:
        if row.get(f"regime_{r}", 0) == 1: return r
    return "unknown"

assert len(df) == len(hold_fm) == 116, f"length mismatch: df={len(df)} hold={len(hold_fm)}"
df["reaney_regime"] = hold_fm.apply(_reg, axis=1).values
df["residual"]      = df["er_measured"] - df["er_predicted"]

y_true    = df["er_measured"].values
y_pred    = df["er_predicted"].values
residuals = df["residual"].values

REGIME_PLOT_COLORS = {**REGIME_COLORS, "unknown": GRAY}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIG_FULL, layout="constrained")
for ax in axes:
    style4(ax)

# Panel A — residuals vs predicted
ax = axes[0]
for regime, grp in df.groupby("reaney_regime"):
    color = REGIME_PLOT_COLORS.get(regime, GRAY)
    ax.scatter(grp["er_predicted"], grp["residual"],
               s=15, alpha=0.7, color=color, label=regime,
               edgecolors="none", zorder=3, rasterized=True)
ax.axhline(0, color="black", lw=1.0, ls="--", zorder=2)
ax.set_xlabel("Predicted $\\varepsilon_r$")
ax.set_ylabel("Residual (Measured $-$ Predicted)")
ax.set_title("a  Residuals vs predicted", loc="left")
ax.text(0.05, 0.93,
        "Spearman $r$(|residual|, pred_std) = 0.278",
        transform=ax.transAxes, fontsize=7)
handles, labels = ax.get_legend_handles_labels()
order   = [l for l in ["Ia", "Ib", "II", "III", "unknown"] if l in labels]
idx     = [labels.index(l) for l in order]
ax.legend([handles[i] for i in idx], [labels[i] for i in idx],
          title="Regime", title_fontsize=7, handlelength=1.2)

# Panel B — residual histogram
ax        = axes[1]
res_mean  = np.mean(residuals)
res_std   = np.std(residuals)
res_skew  = scipy_skew(residuals)
ax.hist(residuals, bins=20, color=OUR_COLOR, alpha=0.70,
        edgecolor="white", density=False, zorder=2)
n_samp    = len(residuals)
bin_width = (residuals.max() - residuals.min()) / 20
x_range   = np.linspace(residuals.min() - 5, residuals.max() + 5, 300)
ax.plot(x_range,
        stats.norm.pdf(x_range, res_mean, res_std) * n_samp * bin_width,
        color=RED, lw=1.2, ls="-", label="Normal fit")
ax.set_xlabel("Residual")
ax.set_ylabel("Count")
ax.set_title("b  Residual distribution", loc="left")
ax.text(0.05, 0.87,
        f"Mean = {res_mean:.2f}\nStd = {res_std:.2f}\nSkew = {res_skew:.2f}",
        transform=ax.transAxes, fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9))
ax.legend()

# Panel C — Q-Q plot
ax = axes[2]
(osm, osr), (slope, intercept, _r) = stats.probplot(residuals)
ax.scatter(osm, osr, s=12, alpha=0.65, color=OUR_COLOR,
           edgecolors="none", zorder=3, rasterized=True)
ax.plot([osm.min(), osm.max()],
        slope * np.array([osm.min(), osm.max()]) + intercept,
        color=RED, lw=1.2, ls="--", label=f"Fit ($r$={_r:.3f})")
ax.set_xlabel("Theoretical quantiles")
ax.set_ylabel("Sample quantiles")
ax.set_title("c  Normal Q-Q plot", loc="left")
ax.legend()

OUT = "./figures_output/all_figures/supp"
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/figS5_residuals.{ext}", dpi=300)
print("Saved figS5_residuals.pdf and .png")
plt.close(fig)
print(f"  Residual stats: mean={res_mean:.3f}, std={res_std:.3f}, skew={res_skew:.3f}")
