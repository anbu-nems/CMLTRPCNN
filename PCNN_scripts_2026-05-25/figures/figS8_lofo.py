"""
Figure S3 — LOFO generalisability
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

OUR_COLOR = "#2874A6"; GRAY = "#B0BEC5"; RED = "#E74C3C"
GREEN     = "#2A9D8F"; ORANGE = "#E67E22"
FIG_FULL  = (7.2, 3.8)

with open("results/42_lofo_generality.json") as f:
    lofo = json.load(f)

asite_data   = lofo["lofo_asite"]
asite_order  = sorted(asite_data, key=lambda x: asite_data[x]["r2"])
asite_r2     = [asite_data[k]["r2"] for k in asite_order]
asite_n      = [asite_data[k]["n_test"] for k in asite_order]

regime_data  = lofo["lofo_regime"]
regime_order = sorted(regime_data, key=lambda x: regime_data[x]["r2"])
regime_r2    = [regime_data[k]["r2"] for k in regime_order]
regime_n     = [regime_data[k]["n_test"] for k in regime_order]

def bar_color(r2_val):
    if r2_val < 0:     return RED
    elif r2_val < 0.5: return ORANGE
    else:              return GREEN

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=FIG_FULL, layout="constrained")
for ax in axes:
    style4(ax)

ax      = axes[0]
y_pos   = np.arange(len(asite_order))
ax.barh(y_pos, asite_r2, color=[bar_color(v) for v in asite_r2],
        height=0.5, edgecolor="white")
ax.axvline(0, color="black", lw=0.8, ls="--", zorder=5)
# All n-labels placed at right edge with consistent black text — avoids axis-tick collisions
x_nlabel_a = max(max(asite_r2), 0) + 0.06
for i, (key, n_t) in enumerate(zip(asite_order, asite_n)):
    ax.text(x_nlabel_a, i, f"$n$={n_t}", va="center", ha="left", fontsize=7)
ax.set_xlim(right=x_nlabel_a + 0.20)
ax.set_yticks(y_pos); ax.set_yticklabels(asite_order)
ax.set_xlabel("$R^2$ when A-site family held out entirely")
ax.set_title("a  Leave-one-family-out: A-site", loc="left")
# (the Pb "extreme f_LST / data sparsity" interpretation lives in the caption)

ax       = axes[1]
y_pos_b  = np.arange(len(regime_order))
XB_LO    = -1.3   # clip the extreme Regime-Ia bar (R^2 = -7.38) so II/III/Ib stay readable
regime_r2_clip = [max(v, XB_LO) for v in regime_r2]
ax.barh(y_pos_b, regime_r2_clip, color=[bar_color(v) for v in regime_r2],
        height=0.5, edgecolor="white")
ax.axvline(0, color="black", lw=0.8, ls="--", zorder=5)
# n-labels at a fixed column with clear right-hand headroom
x_nlabel_b = 0.85
for i, (key, n_t) in enumerate(zip(regime_order, regime_n)):
    ax.text(x_nlabel_b, i, f"$n$={n_t}", va="center", ha="left", fontsize=7)
# clipped bars carry their true value inline (white, inside the bar)
for i, v in enumerate(regime_r2):
    if v < XB_LO:
        ax.text(XB_LO + 0.05, i, f"$R^2$={v:.2f}", va="center", ha="left",
                fontsize=6.5, color="white", fontweight="bold", zorder=6)
ax.set_xlim(XB_LO, 1.55)
ax.set_yticks(y_pos_b); ax.set_yticklabels(regime_order)
ax.set_xlabel("$R^2$ when regime held out entirely")
ax.set_title("b  Leave-one-family-out: Reaney regime", loc="left")

OUT = "/Users/anbu/Desktop/NC figures/all_figures/supp"
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/figS8_lofo.{ext}", dpi=300)
print("Saved figS8_lofo.pdf and .png")
plt.close(fig)
