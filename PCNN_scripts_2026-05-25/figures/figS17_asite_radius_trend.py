#!/usr/bin/env python3
"""
A-site f_LST hierarchy with ionic-radius x-axis — the "Pb breaks the trend" figure.

X = Shannon 12-coord A-site ionic radius (Å)
Y = predicted f_LST multiplier (soft-mode enhancement factor)
Shows the monotonic radius–f_LST trend EXCEPT for Pb, which sits above the line
due to 6s² lone-pair stereoactivity.

Output: all_figures/supp/figS17_asite_radius_trend.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import json
import os

OUT_DIR = "./figures_output/all_figures/supp"
os.makedirs(OUT_DIR, exist_ok=True)
PIML    = "/Users/anbu/Desktop/PIML/piml_ceramic"

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

# ── Shannon ionic radii, 12-coord (Å) ───────────────────────────────────────
SHANNON_R12 = {
    "Ca": 1.34, "La": 1.36, "Sr": 1.44, "Pb": 1.49, "Ba": 1.61,
}

# Ceramist colours (consistent with Reaney plot)
ASITE_COLORS = {
    "Pb": "#7A6F5B", "Ba": "#5A8C3E", "Sr": "#D4A82E",
    "Ca": "#D67238", "La": "#4A6FA5",
}

# ── Load f_LST data ─────────────────────────────────────────────────────────
d = json.load(open(f"{PIML}/results/44_literature_validation.json"))["by_asite"]

order = ["Ca", "La", "Sr", "Pb", "Ba"]
pred  = np.array([d[a]["predicted_lst"]["mean"]   for a in order])
plo   = np.array([d[a]["predicted_lst"]["ci_lo"] for a in order])
phi   = np.array([d[a]["predicted_lst"]["ci_hi"] for a in order])
emp   = np.array([d[a]["empirical"]["mean"]       for a in order])
elo   = np.array([d[a]["empirical"]["ci_lo"]      for a in order])
ehi   = np.array([d[a]["empirical"]["ci_hi"]      for a in order])
ns    = np.array([d[a]["n"]                       for a in order])
radii = np.array([SHANNON_R12[a]                  for a in order])

# Linear fit through Ca / La / Sr / Ba (exclude Pb)
mask_nonpb = np.array([a != "Pb" for a in order])
slope, intercept = np.polyfit(radii[mask_nonpb], pred[mask_nonpb], 1)
xfit = np.linspace(1.30, 1.65, 100)
yfit = slope * xfit + intercept

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.6, 4.2))
style4(ax)

# Trend line (without Pb)
ax.plot(xfit, yfit, color="#888", lw=1.3, ls="--", zorder=2,
        label=f"Linear trend (excluding Pb)\nslope = {slope:+.2f}/Å")

# Error bars (predicted)
ax.errorbar(radii, pred, yerr=[pred - plo, phi - pred],
            fmt="none", ecolor="#444", elinewidth=1.0, capsize=4, zorder=3)

# Scatter points — element-coloured, large markers
for i, a in enumerate(order):
    ax.scatter(radii[i], pred[i], s=180,
               color=ASITE_COLORS[a],
               edgecolors="white", linewidths=1.2, zorder=4,
               marker="o" if a != "Pb" else "X")
    # Element label above each point
    dy = 0.18 if a != "Pb" else 0.22
    ax.annotate(f"{a}\n$n$={ns[i]}\n$f_{{LST}}$={pred[i]:.2f}",
                (radii[i], pred[i]),
                xytext=(0, 18),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=6.8, fontweight="bold",
                color=ASITE_COLORS[a])

# Pb outlier callout: arrow + annotation
pb_idx = order.index("Pb")
ax.annotate(
    "Pb breaks the trend\n6s² lone-pair stereoactivity\n→ extra A-site off-centering",
    xy=(radii[pb_idx], pred[pb_idx]),
    xytext=(1.32, 2.05),
    fontsize=7, color="#3A2F1F", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.35", fc="#FBF3D6", ec="#A88E4A", lw=0.8),
    arrowprops=dict(arrowstyle="->", color="#A88E4A", lw=1.0,
                    connectionstyle="arc3,rad=-0.2"),
)

# Correlation of the cage-tightening trend (excluding Pb), from the same family means
pearson_r = np.corrcoef(radii[mask_nonpb], pred[mask_nonpb])[0, 1]   # printed/used in caption; not annotated on the panel
# (the "Excl. Pb: Pearson r = ..." interpretation lives in the caption, not on the figure)

ax.set_xlim(1.30, 1.65)
ax.set_ylim(0.4, 2.7)
ax.set_xlabel(r"A-site ionic radius  $r_A$  (Shannon, 12-coord)  [Å]")
ax.set_ylabel(r"Predicted f$_{LST}$ multiplier  $\langle\delta_{LST}/\varepsilon_r^{CM}\rangle$")
ax.set_title("A-site soft-mode hierarchy — monotonic in ionic radius, except Pb",
             loc="left")

ax.legend(loc="upper right", fontsize=6.8, handlelength=1.5)

# Save
for ext in ("pdf", "png"):
    path = f"{OUT_DIR}/figS17_asite_radius_trend.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
