#!/usr/bin/env python3
"""
Supplementary Fig S1 — Dataset statistics
  a) εr distribution (histogram + KDE, coloured by regime)
  b) Regime breakdown (horizontal bar)
  c) A-site composition (horizontal bar)
Nature Communications style
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os, sys

# (figS1 reads the bundled CSV directly; no feature_engineering import needed)

OUT_DIR  = "./figures_output/supp"
FM_PATH  = "./data/processed/feature_matrix_v7.parquet"
CSV_PATH = "./data/raw/mixed_dataset_clean.csv"
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

REGIME_COLORS = {
    "Ia":  "#264653",
    "Ib":  "#2A9D8F",
    "II":  "#E9C46A",
    "III": "#E76F51",
}
ASITE_COLORS = {
    "Ba":    "#264653",
    "Ca":    "#E76F51",
    "Sr":    "#E9C46A",
    "La":    "#4A6FA5",
    "Pb":    "#F4A261",
    "Mixed": "#999999",
}

# ── Load data ─────────────────────────────────────────────────────────────────
df  = pd.read_csv(CSV_PATH)
fm  = pd.read_parquet(FM_PATH)

# Assign regime per sample
def get_regime(row):
    if row.get("regime_Ia", 0) == 1:  return "Ia"
    if row.get("regime_Ib", 0) == 1:  return "Ib"
    if row.get("regime_II", 0) == 1:  return "II"
    if row.get("regime_III", 0) == 1: return "III"
    return "Other"

fm["regime"] = fm.apply(get_regime, axis=1)
df["regime"] = fm["regime"].values

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
ax_hist, ax_regime, ax_asite = axes
for ax in axes:
    style4(ax)

# ── Panel A: εr distribution coloured by regime ───────────────────────────────
bins = np.linspace(0, 150, 26)
regime_order = ["Ia", "Ib", "II", "III"]
bottoms = np.zeros(len(bins) - 1)
for reg in regime_order:
    sub = df[df["regime"] == reg]["epsilon_r"].values
    counts, _ = np.histogram(sub, bins=bins)
    ax_hist.bar(bins[:-1], counts, width=np.diff(bins),
                bottom=bottoms, color=REGIME_COLORS[reg],
                label=f"Regime {reg}  ($n$={len(sub)})",
                align="edge", edgecolor="white", linewidth=0.3)
    bottoms += counts

ax_hist.set_xlabel(r"$\varepsilon_r$")
ax_hist.set_ylabel("Count")
ax_hist.set_title("a  Dataset distribution  ($n$=1304)", loc="left", fontweight="bold")
ax_hist.set_xlim(0, 150)
ax_hist.legend(loc="best", handlelength=1.0)

# Annotation: median line
med = df["epsilon_r"].median()
ax_hist.axvline(med, color="#555", lw=1.0, ls="--")
ax_hist.text(med + 2, ax_hist.get_ylim()[1] * 0.88,
             f"Median = {med:.0f}", fontsize=6.5, color="#555")

# ── Panel B: Regime breakdown ─────────────────────────────────────────────────
reg_counts = {r: int((fm["regime"] == r).sum()) for r in regime_order}
reg_total  = sum(reg_counts.values())
reg_labels = [f"Regime {r}" for r in regime_order]
reg_vals   = [reg_counts[r] for r in regime_order]
reg_pct    = [v / reg_total * 100 for v in reg_vals]
reg_cols   = [REGIME_COLORS[r] for r in regime_order]

y = np.arange(len(regime_order))
bars = ax_regime.barh(y, reg_vals, height=0.55, color=reg_cols,
                      edgecolor="white")   # match figS2 style — bars sit flush against the y-axis spine
ax_regime.set_yticks(y)
ax_regime.set_yticklabels(reg_labels)
ax_regime.set_xlabel("Number of compositions")
ax_regime.set_title("b  Reaney regime distribution", loc="left", fontweight="bold")
ax_regime.invert_yaxis()
for bar, val, pct in zip(bars, reg_vals, reg_pct):
    ax_regime.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
                   f"{val}  ({pct:.0f}%)", va="center", fontsize=6.5, color="#333")
ax_regime.set_xlim(0, max(reg_vals) * 1.50)   # bars anchored at zero (touch y-axis); right breathing room for value labels

# ── Panel C: A-site composition ───────────────────────────────────────────────
asite_counts = {"Ba": 436, "Ca": 421, "Sr": 180, "La": 125, "Pb": 86, "Mixed": 56}
asite_total  = sum(asite_counts.values())
asite_keys   = list(asite_counts.keys())                                                    # for colour lookup
asite_labels = [k if k != "Mixed" else "Mixed\n(La/Na/Bi/K)" for k in asite_keys]           # fold the explanation into the tick label itself
asite_vals   = list(asite_counts.values())
asite_pct    = [v / asite_total * 100 for v in asite_vals]
asite_cols   = [ASITE_COLORS[k] for k in asite_keys]

y2 = np.arange(len(asite_labels))
bars2 = ax_asite.barh(y2, asite_vals, height=0.55, color=asite_cols,
                      edgecolor="white")   # match figS2 style — bars sit flush against the y-axis spine
ax_asite.set_yticks(y2)
ax_asite.set_yticklabels(asite_labels)
ax_asite.set_xlabel("Number of compositions")
ax_asite.set_title("c  A-site occupancy", loc="left", fontweight="bold")
ax_asite.invert_yaxis()
for bar, val, pct in zip(bars2, asite_vals, asite_pct):
    ax_asite.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                  f"{val}  ({pct:.0f}%)", va="center", fontsize=6.5, color="#333")
ax_asite.set_xlim(0, max(asite_vals) * 1.50)   # bars anchored at zero (touch y-axis); right breathing room for value labels
# (the Mixed= explanation now lives in the y-tick label above; floating note removed)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    path = os.path.join(OUT_DIR, f"figS1_dataset_statistics.{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
print("Fig S1 — Dataset statistics done.")
