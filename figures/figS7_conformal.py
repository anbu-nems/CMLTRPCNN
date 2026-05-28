"""
Figure S1 — Conformal prediction
Publication-quality figure for Nature Communications supplementary.
Run from: .
"""
import numpy as np
import pandas as pd
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

OUR_COLOR = "#2874A6"
GRAY      = "#B0BEC5"
RED       = "#E74C3C"
ORANGE    = "#E67E22"
FIG_FULL  = (7.2, 3.2)

# ── Load data ─────────────────────────────────────────────────────────────────
df     = pd.read_csv("data/processed/test_holdout_predictions.csv")
y_true = df["er_measured"].values
y_pred = df["er_predicted"].values
n      = len(y_true)

abs_scores = np.abs(y_true - y_pred)
rel_scores = np.abs(y_true - y_pred) / (np.abs(y_pred) + 1e-9)

alphas   = [0.20, 0.15, 0.10, 0.05]
nominals = [1 - a for a in alphas]

emp_abs, emp_rel = [], []
for alpha in alphas:
    level = min((1 - alpha) * (1 + 1 / n), 1.0)
    q_abs = np.quantile(abs_scores, level)
    q_rel = np.quantile(rel_scores, level)
    emp_abs.append(np.mean(abs_scores <= q_abs))
    emp_rel.append(np.mean(rel_scores <= q_rel))

alpha90  = 0.10
level90  = min((1 - alpha90) * (1 + 1 / n), 1.0)
q_hat_90 = np.quantile(abs_scores, level90)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIG_FULL, layout="constrained")
for ax in axes:
    style4(ax)
    ax.set_box_aspect(1)   # square panel boxes → all three uniform, aligned, and square

# Panel A — coverage calibration
ax = axes[0]
diag = np.linspace(0.77, 0.98, 100)
ax.plot(diag, diag, "k--", lw=1.0, label="Ideal", zorder=1)
ax.plot(nominals, emp_abs, "o-", color=OUR_COLOR, ms=6, label="Absolute", zorder=3)
ax.plot(nominals, emp_rel, "s-", color=ORANGE,    ms=6, label="Relative", zorder=3)
ax.set_xlim(0.77, 0.98)
ax.set_ylim(0.77, 1.00)
# rectangular (no forced equal aspect) so a/b/c are uniform height & aligned
ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Empirical coverage")
ax.set_title("a  Coverage calibration", loc="left")
ax.legend()

# Panel B — residual vs prediction
ax = axes[1]
ax.scatter(y_pred, abs_scores, s=12, alpha=0.5, color=OUR_COLOR,
           edgecolors="none", zorder=2, rasterized=True)
ax.axhline(q_hat_90, color=RED, lw=1.2, ls="--",
           label=f"90% threshold: ±{q_hat_90:.1f}")
x_line = np.linspace(y_pred.min(), y_pred.max(), 200)
ax.plot(x_line, 0.27 * x_line, color=ORANGE, lw=1.2, label="27% relative")
ax.set_xlabel("Predicted $\\varepsilon_r$")
ax.set_ylabel("|Residual|")
ax.set_title("b  Residual vs prediction", loc="left")
ax.legend()

# Panel C — parity with 90% conformal band
ax = axes[2]
lim_arr = np.linspace(0, 145, 200)
ax.fill_between(lim_arr, lim_arr - q_hat_90, lim_arr + q_hat_90,
                alpha=0.12, color=OUR_COLOR,
                label=f"90% band (±{q_hat_90:.1f})")
ax.plot(lim_arr, lim_arr, "k--", lw=1.0, label="1:1")
ax.scatter(y_true, y_pred, s=12, alpha=0.6, color=OUR_COLOR,
           edgecolors="none", zorder=3, rasterized=True)
emp_90 = np.mean(abs_scores <= q_hat_90) * 100
ax.text(0.97, 0.05,
        f"Coverage = {emp_90:.1f}%\n(target: 90%)",
        transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9))
ax.set_xlim(0, 145)
ax.set_ylim(0, 145)
# rectangular (no forced equal aspect) so a/b/c are uniform height & aligned
ax.set_xlabel("Measured $\\varepsilon_r$")
ax.set_ylabel("Predicted $\\varepsilon_r$")
ax.set_title(f"c  Parity plot, 90% band (±{q_hat_90:.1f})", loc="left")
ax.legend(loc="upper left")

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = "./figures_output/supp"
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/figS7_conformal.{ext}", dpi=300)
print(f"Saved figS7_conformal.pdf and .png  (q_hat_90 = {q_hat_90:.2f})")
plt.close(fig)
