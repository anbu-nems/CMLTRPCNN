#!/usr/bin/env python3
"""
Latent-space PCA projection — VIBANN Fig 2c analog.

The model's 192-dim trunk hidden state (the internal representation BEFORE
splitting into LST/Tilt/Residual branches) is projected to 2D via PCA.
Each composition becomes one point in this learned representation space.

Three panels show three different colorings to reveal what structure the
trunk has internalized:
  a) Coloured by predicted εᵣ        — does the trunk organize by property?
  b) Coloured by Reaney regime       — does it cluster by tilt regime?
  c) Coloured by A-site cation       — does it cluster by chemistry?

Output: all_figures/supp/figS23_latent_space.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import json
import os

OUT_DIR = "./figures_output/all_figures/supp"
os.makedirs(OUT_DIR, exist_ok=True)
EXTRACT = "./extracted_data"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "legend.frameon": True,
    "legend.framealpha": 0.92, "legend.edgecolor": "#CCCCCC",
    "figure.constrained_layout.use": True,
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.grid": False,
})

def style4(ax, lw=0.9):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw)
    ax.tick_params(direction="in", top=False, right=False,
                   bottom=True, left=True, width=0.7, length=2.5)
    ax.set_axisbelow(False)

REGIME_COLORS = {"Ia": "#264653", "Ib": "#2A9D8F", "II": "#E9C46A", "III": "#E76F51"}
ASITE_COLORS  = {"Pb": "#7A6F5B", "Ba": "#5A8C3E", "Sr": "#D4A82E",
                 "Ca": "#D67238", "La": "#4A6FA5"}

d = pd.read_csv(f"{EXTRACT}/latent_space_trunk.csv")
with open(f"{EXTRACT}/latent_space_pca_meta.json") as f:
    meta = json.load(f)
pc_var = meta["explained_variance_ratio"]
print(f"Plotting {len(d)} points; PC1={pc_var[0]:.1%}, PC2={pc_var[1]:.1%}")

# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
ax1, ax2, ax3 = axes
for ax in axes: style4(ax)

# Shared axis limits
pc1_lo, pc1_hi = d["PC1"].quantile(0.001), d["PC1"].quantile(0.999)
pc2_lo, pc2_hi = d["PC2"].quantile(0.001), d["PC2"].quantile(0.999)
mar = 0.05 * (pc1_hi - pc1_lo)
pc1_lo, pc1_hi = pc1_lo - mar, pc1_hi + mar
mar = 0.05 * (pc2_hi - pc2_lo)
pc2_lo, pc2_hi = pc2_lo - mar, pc2_hi + mar

# ── Panel A: coloured by εr ─────────────────────────────────────────────────
sc1 = ax1.scatter(d["PC1"], d["PC2"], c=d["er_measured"],
                  cmap="viridis", s=8, alpha=0.85,
                  edgecolors="white", linewidths=0.2, zorder=3,
                  vmin=0, vmax=150)
ax1.set_xlim(pc1_lo, pc1_hi); ax1.set_ylim(pc2_lo, pc2_hi)
ax1.set_aspect("equal", adjustable="box")
ax1.set_xlabel(f"PC1 ({pc_var[0]:.0%})")
ax1.set_ylabel(f"PC2 ({pc_var[1]:.0%})")
ax1.set_title(r"a   By measured $\varepsilon_r$", loc="left")
cbar = plt.colorbar(sc1, ax=ax1, shrink=0.85, pad=0.02)
cbar.set_label(r"$\varepsilon_r$", fontsize=7)
cbar.ax.tick_params(labelsize=6)

# ── Panel B: coloured by regime ─────────────────────────────────────────────
for r in ["Ia", "Ib", "II", "III"]:
    sub = d[d["regime"] == r]
    if len(sub) == 0: continue
    ax2.scatter(sub["PC1"], sub["PC2"],
                c=REGIME_COLORS[r], s=8, alpha=0.7,
                edgecolors="white", linewidths=0.2, zorder=3,
                label=f"{r} (n={len(sub)})")
ax2.set_xlim(pc1_lo, pc1_hi); ax2.set_ylim(pc2_lo, pc2_hi)
ax2.set_aspect("equal", adjustable="box")
ax2.set_xlabel(f"PC1 ({pc_var[0]:.0%})")
ax2.set_ylabel(f"PC2 ({pc_var[1]:.0%})")
ax2.set_title("b   By Reaney regime", loc="left")
ax2.legend(loc="best", title="Regime", title_fontsize=7,
           handlelength=0.6, markerscale=1.5)

# ── Panel C: coloured by A-site ─────────────────────────────────────────────
for a in ["Pb", "Ba", "Sr", "Ca", "La"]:
    sub = d[d["a_site"] == a]
    if len(sub) == 0: continue
    ax3.scatter(sub["PC1"], sub["PC2"],
                c=ASITE_COLORS[a], s=8, alpha=0.7,
                edgecolors="white", linewidths=0.2, zorder=3,
                label=f"{a} (n={len(sub)})")
# Other compositions in gray
sub_other = d[~d["a_site"].isin(["Pb", "Ba", "Sr", "Ca", "La"])]
if len(sub_other) > 0:
    ax3.scatter(sub_other["PC1"], sub_other["PC2"],
                c="#CCCCCC", s=8, alpha=0.5,
                edgecolors="white", linewidths=0.2, zorder=2,
                label=f"Other (n={len(sub_other)})")
ax3.set_xlim(pc1_lo, pc1_hi); ax3.set_ylim(pc2_lo, pc2_hi)
ax3.set_aspect("equal", adjustable="box")
ax3.set_xlabel(f"PC1 ({pc_var[0]:.0%})")
ax3.set_ylabel(f"PC2 ({pc_var[1]:.0%})")
ax3.set_title("c   By A-site cation", loc="left")
ax3.legend(loc="best", title="A-site", title_fontsize=7,
           handlelength=0.6, markerscale=1.5)

fig.suptitle("Trunk latent representation (192-dim → PCA-2D) — "
             f"{pc_var[0]+pc_var[1]:.0%} of variance shown",
             fontsize=9.5, fontweight="bold", y=1.03)

# (the "n compositions · 192-d trunk hidden state · PCA fit on full set" methodology note lives in the caption)

for ext in ("pdf", "png"):
    path = f"{OUT_DIR}/figS23_latent_space.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
