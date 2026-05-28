#!/usr/bin/env python3
"""
Tolerance factor histogram with Reaney regime boundaries — ceramist-style.

X = Goldschmidt tolerance factor t
Y = count
Stacked colours = A-site cation (Pb/Ba/Sr/Ca/La)
Vertical lines at t=0.985 and t=0.965 mark Reaney regime boundaries
Regime bands shaded; regime labels at top.

Output: all_figures/supp/figS19_tolerance_histogram.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os

OUT_DIR = "/Users/anbu/Desktop/NC figures/all_figures/supp"
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

# A-site colours (consistent with Reaney plot)
ASITE_COLORS = {
    "Pb": "#7A6F5B", "Ba": "#5A8C3E", "Sr": "#D4A82E",
    "Ca": "#D67238", "La": "#4A6FA5",
}
REGIME_BAND = {
    "Ia/Ib": "#E7F3F2",
    "II":    "#FBF3D6",
    "III":   "#FBE1D1",
}

# ── Load data ───────────────────────────────────────────────────────────────
fm = pd.read_parquet(f"{PIML}/data/processed/feature_matrix_v7.parquet")
fm["a_site"] = fm["chemistry_family"].apply(lambda x: str(x).split("_")[0])
fm["t"]      = fm["tolerance_factor"].values

# Restrict to the 5 main A-sites
target_a = ["Pb", "Ba", "Sr", "Ca", "La"]
fm5 = fm[fm["a_site"].isin(target_a)].copy()
print(f"Plotting {len(fm5)} compositions over 5 A-sites")

# Bin edges chosen to cleanly straddle regime boundaries 0.985 and 0.965
bin_edges = np.arange(0.85, 1.07, 0.01)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Compute stacked counts per A-site
counts = {}
for a in target_a:
    sub = fm5[fm5["a_site"] == a]
    c, _ = np.histogram(sub["t"], bins=bin_edges)
    counts[a] = c

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 4.0))
style4(ax)

t_min, t_max = bin_edges.min(), bin_edges.max()
total_max = sum(counts[a] for a in target_a).max()
y_top = int(total_max * 1.20)

# Regime bands
ax.axvspan(0.985, t_max, color=REGIME_BAND["Ia/Ib"], zorder=0, alpha=0.65)
ax.axvspan(0.965, 0.985, color=REGIME_BAND["II"],    zorder=0, alpha=0.65)
ax.axvspan(t_min, 0.965, color=REGIME_BAND["III"],   zorder=0, alpha=0.65)

# Boundary lines
ax.axvline(0.985, color="#666", lw=0.8, ls="--", zorder=1)
ax.axvline(0.965, color="#666", lw=0.8, ls="--", zorder=1)

# Stacked bars
bottom = np.zeros(len(bin_centers))
width  = (bin_edges[1] - bin_edges[0]) * 0.94
# Order Pb on top (so its small count is visible)
stack_order = ["Ba", "Ca", "Sr", "La", "Pb"]
for a in stack_order:
    ax.bar(bin_centers, counts[a], width=width, bottom=bottom,
           color=ASITE_COLORS[a], edgecolor="white", label=f"{a} ($n$={counts[a].sum()})")
    bottom += counts[a]

# Regime labels
ax.text(0.997, y_top * 0.95, "Ia / Ib", ha="center", va="top",
        fontsize=8.5, color="#264653", fontweight="bold")
ax.text(0.975, y_top * 0.95, "II", ha="center", va="top",
        fontsize=8.5, color="#8A6E16", fontweight="bold")
ax.text(0.90, y_top * 0.95, "III", ha="center", va="top",
        fontsize=8.5, color="#A6432B", fontweight="bold")

# Annotation: boundary t values
ax.annotate("$t$ = 0.985", xy=(0.985, y_top*0.75), xytext=(1.005, y_top*0.78),
            fontsize=6.5, color="#444", ha="left",
            arrowprops=dict(arrowstyle="->", color="#666", lw=0.7))
ax.annotate("$t$ = 0.965", xy=(0.965, y_top*0.62), xytext=(0.94, y_top*0.65),
            fontsize=6.5, color="#444", ha="right",
            arrowprops=dict(arrowstyle="->", color="#666", lw=0.7))

ax.set_xlim(t_min, t_max)
ax.set_ylim(0, y_top)
ax.set_xlabel(r"Goldschmidt tolerance factor  $t = (r_A + r_O)/[\sqrt{2}(r_B + r_O)]$")
ax.set_ylabel("Number of compositions")
ax.set_title(r"Tolerance-factor distribution of the dataset, stacked by A-site",
             loc="left")

# A-site legend (large markers)
ax.legend(loc="upper left", fontsize=7, title="A-site", title_fontsize=7,
          handlelength=1.0, ncol=1)

# (the Reaney 1994 attribution lives as a proper numbered citation in the caption)

# Save
for ext in ("pdf", "png"):
    path = f"{OUT_DIR}/figS19_tolerance_histogram.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
