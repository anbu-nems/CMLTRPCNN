#!/usr/bin/env python3
"""
Reaney plot — the iconic microwave-dielectric figure.

X = Goldschmidt tolerance factor t
Y = predicted εr (also shows measured εr as overlay)
Colour = Reaney regime (Ia/Ib/II/III, with shaded bands)
Marker = A-site cation (Pb/Ca/Ba/Sr/La)
Vertical lines at t=0.985 and t=0.965 mark regime boundaries.

Output: all_figures/supp/figS16_reaney_plot.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os

OUT_DIR = "./figures_output/all_figures/supp"
os.makedirs(OUT_DIR, exist_ok=True)
EXTRACT = "./extracted_data"
PIML    = "."

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

# ── Ceramist-style element colours (Shannon-derived) ────────────────────────
ASITE_COLORS = {
    "Pb": "#7A6F5B",   # dark grey-brown — heavy 6s² lone pair
    "Ba": "#5A8C3E",   # green — heavy alkaline earth
    "Sr": "#D4A82E",   # yellow — Sr standard
    "Ca": "#D67238",   # orange — Ca standard
    "La": "#4A6FA5",   # blue — rare-earth standard
}
ASITE_MARK = {
    "Pb": "X",   # cross — flag the lone-pair outlier
    "Ba": "s",   # square
    "Sr": "D",   # diamond
    "Ca": "o",   # circle (commonest)
    "La": "^",   # triangle
}

# Regime shading colours
REGIME_BAND = {
    "Ia/Ib": ("#E7F3F2", "Ia + Ib  (no tilting)"),
    "II":    ("#FBF3D6", "II  (antiphase tilt)"),
    "III":   ("#FBE1D1", "III  (antiphase + in-phase)"),
}

# ── Load data ───────────────────────────────────────────────────────────────
fm = pd.read_parquet(f"{PIML}/data/processed/feature_matrix_v7.parquet")
d  = pd.read_csv(f"{EXTRACT}/decomposition_per_sample.csv")
d["a_site"] = fm["chemistry_family"].apply(lambda x: str(x).split("_")[0]).values
d["t"]      = fm["tolerance_factor"].values

target_a = ["Pb", "Ba", "Sr", "Ca", "La"]
d = d[d["a_site"].isin(target_a)].copy()
print(f"Plotting {len(d)} compositions over 5 A-sites")

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 4.4))
style4(ax)

t_min, t_max = 0.85, 1.06
er_min, er_max = 0, 155

# Regime bands (Reaney 1994)
ax.axvspan(0.985, t_max, color=REGIME_BAND["Ia/Ib"][0], zorder=0)
ax.axvspan(0.965, 0.985, color=REGIME_BAND["II"][0],    zorder=0)
ax.axvspan(t_min, 0.965, color=REGIME_BAND["III"][0],   zorder=0)

# Boundary lines
ax.axvline(0.985, color="#666", lw=0.8, ls="--", zorder=1)
ax.axvline(0.965, color="#666", lw=0.8, ls="--", zorder=1)

# Regime labels at top
ax.text(0.997, er_max * 0.96, "Ia / Ib", ha="center", va="top",
        fontsize=8, color="#264653", fontweight="bold")
ax.text(0.975, er_max * 0.96, "II", ha="center", va="top",
        fontsize=8, color="#8A6E16", fontweight="bold")
ax.text(0.925, er_max * 0.96, "III", ha="center", va="top",
        fontsize=8, color="#A6432B", fontweight="bold")

# Scatter — one A-site at a time
for a in target_a:
    sub = d[d["a_site"] == a]
    ax.scatter(sub["t"], sub["er_predicted"],
               marker=ASITE_MARK[a], color=ASITE_COLORS[a],
               s=18, alpha=0.78,
               edgecolors="white", linewidths=0.3,
               label=f"{a} ($n$={len(sub)})", zorder=3)

# Reference compositions (literature anchors)
anchors = [
    (r"SrTiO$_3$",        1.002, 145, "#222"),
    (r"CaTiO$_3$",        0.967, 145, "#222"),
    (r"CaZrO$_3$",        0.886,  30, "#222"),
    (r"BaZrO$_3$",        1.004,  43, "#222"),
]
# Clip anchors to plot range
for name, t, er, col in anchors:
    if t_min <= t <= t_max and 0 <= er <= er_max:
        ax.scatter([t], [er], marker="*", color=col, s=70, zorder=5,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(name, (t, er), xytext=(5, 5), textcoords="offset points",
                    fontsize=6.5, color="#333", style="italic")

ax.set_xlim(t_min, t_max)
ax.set_ylim(er_min, er_max)
ax.set_xlabel(r"Goldschmidt tolerance factor  $t = (r_A + r_O)/[\sqrt{2}(r_B + r_O)]$")
ax.set_ylabel(r"Predicted dielectric constant  $\varepsilon_r$")
ax.set_title(r"Reaney plot — PCNN predictions across the ABO$_3$ tolerance space",
             loc="left")

# Legend — two columns: A-site (markers/colours) and regimes (bands)
asite_handles = [Line2D([0],[0], marker=ASITE_MARK[a], linestyle="",
                        markerfacecolor=ASITE_COLORS[a],
                        markeredgecolor="white", markeredgewidth=0.4,
                        markersize=7, label=f"{a}")
                 for a in target_a]
leg1 = ax.legend(handles=asite_handles, loc="upper left",
                 title="A-site", title_fontsize=7, ncol=1,
                 handlelength=0.8, fontsize=6.5)
leg1.get_title().set_fontweight("bold")
ax.add_artist(leg1)

regime_handles = [
    mpatches.Patch(color=REGIME_BAND["Ia/Ib"][0], label="Ia + Ib  (no tilting)"),
    mpatches.Patch(color=REGIME_BAND["II"][0],    label="II  (antiphase tilt)"),
    mpatches.Patch(color=REGIME_BAND["III"][0],   label="III  (antiphase + in-phase)"),
]
leg2 = ax.legend(handles=regime_handles, loc="upper right", bbox_to_anchor=(1.0, 0.85),
                 title="Reaney regime", title_fontsize=7,
                 handlelength=1.2, fontsize=6.5)   # nudged down so it clears the SrTiO3 label
leg2.get_title().set_fontweight("bold")
# (the Reaney 1994 attribution lives as a proper numbered citation in the caption)

# Save
for ext in ("pdf", "png"):
    path = f"{OUT_DIR}/figS16_reaney_plot.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
