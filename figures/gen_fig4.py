"""
Figure 4 — Branch ablation
Nature Communications: Physics-constrained neural networks for ceramic dielectric constant prediction
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
FIG_HALF  = (3.5, 4.0)
OUT_DIR   = "/Users/anbu/Desktop/NC figures/main"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data (top → bottom display order, reversed for barh) ──────────────────────
labels   = ["Full model", "No LST", "No CM anchor", "No Tilt", "No Residual", "CM only"]
r2_vals  = [ 0.882,        0.252,    0.751,           0.884,     0.882,         -0.859]
delta_r2 = [ 0.000,       -0.630,   -0.131,          +0.002,    +0.000,         -1.741]
errors   = [ 0.055,        0.028,    0.082,            0.053,     0.059,          0.030]
bar_colors = [OUR_COLOR, RED, ORANGE, GRAY, GRAY, RED]

# Reverse so "Full model" is at top
labels_r  = labels[::-1]
r2_r      = r2_vals[::-1]
errors_r  = errors[::-1]
delta_r   = delta_r2[::-1]
colors_r  = bar_colors[::-1]

fig, ax = plt.subplots(figsize=FIG_HALF, layout="constrained")
style4(ax)

y_pos = np.arange(len(labels_r))
ax.barh(y_pos, r2_r, height=0.5, color=colors_r,
        edgecolor="white", linewidth=0.5,
        xerr=errors_r,
        error_kw=dict(ecolor="#555555", capsize=2.5, lw=0.8))

ax.axvline(0,     color=GRAY,      linestyle="--", lw=0.8)
ax.axvline(0.882, color=OUR_COLOR, linestyle=":",  lw=1.0,
           label="Full model ($R^2$=0.882)")

# Compact value labels — single line, small font, placed right of bar end
X_RIGHT = 1.22   # right xlim to leave room for labels
for i, (r2, dr2, err) in enumerate(zip(r2_r, delta_r, errors_r)):
    x_label = r2 + err + 0.03
    if x_label < 0.04:
        x_label = 0.04
    sign = "+" if dr2 >= 0 else ""
    label_str = f"$R^2$={r2:.3f}  ({sign}{dr2:.3f})"
    ax.text(x_label, i, label_str, va="center", ha="left",
            fontsize=6.5, color="#222222")

ax.set_xlim(-1.15, X_RIGHT)
ax.set_xlabel("Strat-GSS $R^2$  (3-fold, 3 seeds)")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_r)
ax.set_title("a  Branch ablation — CMLTRPCNNv77", loc="left")
ax.legend(loc="upper left")

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    path = os.path.join(OUT_DIR, f"fig4_ablation.{ext}")
    fig.savefig(path, dpi=300)
    print(f"Saved: {path}")
plt.close(fig)
print("Figure 4 done.")
