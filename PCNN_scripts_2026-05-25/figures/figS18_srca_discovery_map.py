#!/usr/bin/env python3
"""
Sr-Ca-Ti pseudo-binary discovery map — ceramist-style.

X = x in (Sr_{1-x} Ca_x) TiO₃  (composition along the SrTiO₃ ↔ CaTiO₃ axis)
Y = predicted εr  (PCNN ensemble)
Marker = top virtual-screening candidates from PIDR-UGS framework
Color  = Reaney tolerance-factor regime
Shaded band = 90 % conformal prediction interval

Adds:
  - literature anchors for pure SrTiO₃ and CaTiO₃
  - training data overlay (Sr-Ca-Ti family from feature matrix)
  - regime boundary annotation
  - top-3 highlighted with formulas

Output: all_figures/supp/figS18_srca_discovery_map.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import json
import re
import os

OUT_DIR = "./figures_output/all_figures/supp"
os.makedirs(OUT_DIR, exist_ok=True)
PIML    = "/Users/anbu/Desktop/PIML/piml_ceramic"
EXTRACT = "./extracted_data"

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

REGIME_COLORS = {"Ia": "#264653", "Ib": "#2A9D8F", "II": "#E9C46A", "III": "#E76F51"}
REGIME_BAND   = {"Ia/Ib": "#E7F3F2", "II": "#FBF3D6", "III": "#FBE1D1"}

# ── Parse formula → x_Ca for Sr-Ca-Ti family ────────────────────────────────
def parse_xca(formula):
    """Extract x_Ca from formulas like 'Sr0.9Ca0.1TiO3'. Returns NaN if not Sr-Ca-Ti."""
    m_sr = re.search(r"Sr([\d.]*)", formula)
    m_ca = re.search(r"Ca([\d.]*)", formula)
    m_ti = re.search(r"Ti([\d.]*)", formula)
    if not (m_sr or m_ca) or not m_ti:
        return np.nan
    def amount(m):
        if m is None: return 0.0
        v = m.group(1)
        return float(v) if v else 1.0
    sr = amount(m_sr) if m_sr else 0.0
    ca = amount(m_ca) if m_ca else 0.0
    total = sr + ca
    if total <= 0: return np.nan
    return ca / total

def regime_short(r_str):
    """Map 'Regime Ib — incipient ferroelectric' → 'Ib'."""
    for r in ["Ia", "Ib", "II", "III"]:
        if f"Regime {r}" in r_str:
            return r
    return "II"

# ── Load virtual screening top candidates ───────────────────────────────────
vs = json.load(open(f"{PIML}/results/61_virtual_screening.json"))
cand = pd.DataFrame(vs["top_candidates"])
cand["x_ca"]   = cand["formula"].apply(parse_xca)
cand["family_short"] = cand["family"].astype(str)
cand["regime_short"] = cand["regime"].apply(regime_short)
srca = cand[(cand["family_short"] == "Sr-Ca-Ti") & cand["x_ca"].notna()].copy()
srca = srca.sort_values("x_ca").reset_index(drop=True)
other_fams = cand[cand["family_short"] != "Sr-Ca-Ti"]
print(f"Sr-Ca-Ti candidates: {len(srca)}, other families: {len(other_fams)}")

# ── Load training data Sr-Ca-Ti for reference ──────────────────────────────
fm = pd.read_parquet(f"{PIML}/data/processed/feature_matrix_v7.parquet")
fm["x_ca"]  = fm["formula"].apply(parse_xca)
fm["fam"]   = fm["chemistry_family"].apply(lambda x: str(x).split("_")[0])
# Sr-Ca-Ti training points: have BOTH Sr and Ca, no other A-site
train_srca = fm[(fm["fam"].isin(["Sr", "Ca"])) & fm["x_ca"].notna() &
                (fm["x_ca"] > 0.0) & (fm["x_ca"] < 1.0)].copy()
# Also include pure Sr-Ti and pure Ca-Ti
def is_pure(f, pure_str):
    return str(f).strip().replace(" ", "").lower() == pure_str.lower()
pure_sr = fm[fm["formula"].apply(lambda f: is_pure(f, "SrTiO3"))]
pure_ca = fm[fm["formula"].apply(lambda f: is_pure(f, "CaTiO3"))]
print(f"Training Sr-Ca-Ti mixed: {len(train_srca)}  |  pure SrTiO3: {len(pure_sr)}  pure CaTiO3: {len(pure_ca)}")

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 4.4))
style4(ax)

x_min, x_max = -0.05, 1.05
y_min, y_max = 0, 165

# Regime bands along x_Ca — derived from average tolerance factor
# In Sr-Ca-Ti, t decreases roughly linearly from ~1.00 (Sr) to ~0.96 (Ca)
# t = 0.985 is around x_Ca ~ 0.32, t = 0.965 around x_Ca ~ 0.85
ax.axvspan(0.0,  0.32, color=REGIME_BAND["Ia/Ib"], zorder=0, alpha=0.55)
ax.axvspan(0.32, 0.85, color=REGIME_BAND["II"],    zorder=0, alpha=0.55)
ax.axvspan(0.85, 1.0,  color=REGIME_BAND["III"],   zorder=0, alpha=0.55)

# Regime labels
ax.text(0.16, y_max * 0.96, "Regime Ia/Ib", ha="center", fontsize=8,
        color="#264653", fontweight="bold")
ax.text(0.585, y_max * 0.96, "Regime II", ha="center", fontsize=8,
        color="#8A6E16", fontweight="bold")
ax.text(0.925, y_max * 0.96, "Regime III", ha="center", fontsize=8,
        color="#A6432B", fontweight="bold")

# Training Sr-Ca-Ti background (gray, faint)
if len(train_srca) > 0:
    ax.scatter(train_srca["x_ca"], train_srca["epsilon_r"],
               s=18, color="#B0BEC5", alpha=0.55, edgecolors="white",
               linewidths=0.3, zorder=2, label=f"Training data (n={len(train_srca)})")

# Pure-end-member training points
for df_sub, lab in [(pure_sr, "SrTiO₃"), (pure_ca, "CaTiO₃")]:
    if len(df_sub) > 0:
        xv = 0.0 if "Sr" in lab else 1.0
        ymed = df_sub["epsilon_r"].median()
        ax.scatter([xv], [ymed], s=100, marker="*", color="#222",
                   edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(lab, (xv, ymed), xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=7, fontstyle="italic", color="#222")

# Conformal CI band — connect top candidates ordered by x_ca
xs = srca["x_ca"].values
ys = srca["pred"].values
ylo = srca["lower_90"].values
yhi = srca["upper_90"].values
ax.fill_between(xs, ylo, yhi, color="#F4A261", alpha=0.18, zorder=3,
                label="90 % conformal interval")

# Top candidates — diamond markers, coloured by regime
for r in ["Ib", "II", "III"]:
    sub = srca[srca["regime_short"] == r]
    if len(sub) == 0: continue
    ax.scatter(sub["x_ca"], sub["pred"],
               marker="D", s=72, color=REGIME_COLORS[r],
               edgecolors="white", linewidths=0.6, zorder=6,
               label=f"PCNN candidate (Regime {r})")
    ax.errorbar(sub["x_ca"], sub["pred"],
                yerr=[sub["pred"] - sub["lower_90"], sub["upper_90"] - sub["pred"]],
                fmt="none", ecolor=REGIME_COLORS[r], elinewidth=0.7,
                capsize=2.5, alpha=0.7, zorder=5)

# Highlight top-3 — staggered: top1 right+up, top2 right+down, top3 left+up
top3 = srca.sort_values("lower_90", ascending=False).head(3).reset_index(drop=True)
manual_offsets = [(12, 14, "left"), (12, -22, "left"), (-12, 18, "right")]
for i, (_, row) in enumerate(top3.iterrows()):
    dx, dy, ha = manual_offsets[i]
    ax.annotate(row["formula"], (row["x_ca"], row["pred"]),
                xytext=(dx, dy), textcoords="offset points",
                ha=ha, fontsize=6.8, color="#222", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.5))

# Highlight #1 with a halo
best = srca.sort_values("lower_90", ascending=False).iloc[0]
ax.scatter([best["x_ca"]], [best["pred"]],
           s=200, marker="o", facecolor="none",
           edgecolor="#E76F51", linewidth=1.6, zorder=7)
ax.text(0.98, 0.04,
        f"Top pick:  {best['formula']}  →  "
        f"εr = {best['pred']:.1f}  (LCB$_{{90}}$ = {best['lower_90']:.1f})",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7, color="#A6432B", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#E76F51", lw=0.8, alpha=0.94))

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel(r"Composition  $x_{\rm Ca}$  in  (Sr$_{1-x}$Ca$_x$)TiO$_3$")
ax.set_ylabel(r"Dielectric constant  $\varepsilon_r$")
ax.set_title("PIDR-UGS discovery map — Sr–Ca–Ti pseudo-binary",
             loc="left")

ax.legend(loc="center right", fontsize=6.8, handlelength=1.2, ncol=1,
          bbox_to_anchor=(1.0, 0.55))

# (the PIDR-UGS framework description — 72 candidates, 4-gate cascade, 31 retained —
#  lives in the caption, not as a floating footer under the figure)

# Save
for ext in ("pdf", "png"):
    path = f"{OUT_DIR}/figS18_srca_discovery_map.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
