#!/usr/bin/env python3
"""
Parity plot — three protocols overlaid on one axis.
  Colors    : Strat-GSS CV / Formula-split CV / Ensemble holdout
  Markers   : one per Reaney regime (Ia / Ib / II / III)

Output: all_figures/fig_parity_3protocols.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os
from sklearn.metrics import r2_score, mean_absolute_error

PROC_DIR  = "/Users/anbu/Desktop/PIML/piml_ceramic/data/processed"
OUT_DIR   = "/Users/anbu/Desktop/NC figures/all_figures/supp"
CV_CSV    = "/Users/anbu/Desktop/NC figures/all_figures/extracted_data/cv_predictions_v77.csv"
HOLD_CSV  = f"{PROC_DIR}/test_holdout_predictions.csv"
HOLD_RAW  = f"{PROC_DIR}/test_holdout_set.csv"
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
        sp.set_visible(True); sp.set_linewidth(lw)
    ax.tick_params(direction="in", top=False, right=False,
                   bottom=True, left=True, width=0.8, length=3)
    ax.set_axisbelow(False)

# ── Colors per protocol & markers per regime ────────────────────────────────
PROTO_COLOR = {
    "Strat-GSS CV":     "#2A9D8F",   # teal
    "Formula-split CV": "#5E81AC",   # slate blue
    "Ensemble holdout": "#E76F51",   # coral
}
REGIME_MARK = {
    "Ia":  "o",   # circle
    "Ib":  "s",   # square
    "II":  "^",   # triangle up
    "III": "D",   # diamond
}

# ── Load CV predictions ─────────────────────────────────────────────────────
cv = pd.read_csv(CV_CSV)

# ── Load holdout predictions and recover regime ─────────────────────────────
hold = pd.read_csv(HOLD_CSV)
import json
fm = pd.read_parquet(f"{PROC_DIR}/feature_matrix_v7.parquet")
calib_idx = json.load(open(f"{PROC_DIR}/calibration_split_idx.json"))["calib_idx"]
hold_fm = fm.iloc[calib_idx].reset_index(drop=True)
def _reg(row):
    if row.get("regime_Ia", 0)  == 1: return "Ia"
    if row.get("regime_Ib", 0)  == 1: return "Ib"
    if row.get("regime_II", 0)  == 1: return "II"
    if row.get("regime_III", 0) == 1: return "III"
    return "II"
assert len(hold) == len(hold_fm) == 116, f"length mismatch: {len(hold)}/{len(hold_fm)}"
hold["regime"] = hold_fm.apply(_reg, axis=1).values

# ── Build per-protocol records ──────────────────────────────────────────────
gss = cv.dropna(subset=["pred_gss"])
frm = cv.dropna(subset=["pred_formula"])

# Compute R²/MAE per protocol
r2_gss = r2_score(gss["er_measured"], gss["pred_gss"])
r2_frm = r2_score(frm["er_measured"], frm["pred_formula"])
r2_hld = r2_score(hold["er_measured"], hold["er_predicted"])
mae_gss = mean_absolute_error(gss["er_measured"], gss["pred_gss"])
mae_frm = mean_absolute_error(frm["er_measured"], frm["pred_formula"])
mae_hld = mean_absolute_error(hold["er_measured"], hold["er_predicted"])

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.0, 4.6))
style4(ax)

# 1:1 line
lim = 150
ax.plot([0, lim], [0, lim], color="#333", lw=0.9, ls="--", zorder=1)
# ±10% guide bands (faint)
ax.fill_between([0, lim], [0*0.9, lim*0.9], [0*1.1, lim*1.1],
                color="#888", alpha=0.06, zorder=0)

# Draw three protocols, smallest first so coral sits on top
def draw_scatter(df, x_col, y_col, color, label_proto, alpha, size):
    for reg in ["Ia", "Ib", "II", "III"]:
        sub = df[df["regime"] == reg]
        if len(sub) == 0: continue
        ax.scatter(sub[x_col], sub[y_col],
                   marker=REGIME_MARK[reg], color=color,
                   s=size, alpha=alpha,
                   edgecolors="white", linewidths=0.4, zorder=3)

draw_scatter(gss,  "er_measured", "pred_gss",      PROTO_COLOR["Strat-GSS CV"],     "Strat-GSS CV",     0.55, 22)
draw_scatter(frm,  "er_measured", "pred_formula",  PROTO_COLOR["Formula-split CV"], "Formula-split CV", 0.55, 22)
draw_scatter(hold, "er_measured", "er_predicted",  PROTO_COLOR["Ensemble holdout"], "Ensemble holdout", 0.95, 32)

ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"Measured $\varepsilon_r$")
ax.set_ylabel(r"Predicted $\varepsilon_r$")
ax.set_title("Parity plot — three evaluation protocols", loc="left")

# ── Two legends: protocol (color) and regime (marker) ───────────────────────
proto_handles = [
    Line2D([0],[0], marker="o", linestyle="",
           markerfacecolor=PROTO_COLOR["Strat-GSS CV"],     markeredgecolor="white",
           markeredgewidth=0.4, markersize=7,
           label=f"Strat-GSS CV     R²={r2_gss:.3f}  MAE={mae_gss:.2f}  ($n$={len(gss)})"),
    Line2D([0],[0], marker="o", linestyle="",
           markerfacecolor=PROTO_COLOR["Formula-split CV"], markeredgecolor="white",
           markeredgewidth=0.4, markersize=7,
           label=f"Formula-split CV  R²={r2_frm:.3f}  MAE={mae_frm:.2f}  ($n$={len(frm)})"),
    Line2D([0],[0], marker="o", linestyle="",
           markerfacecolor=PROTO_COLOR["Ensemble holdout"], markeredgecolor="white",
           markeredgewidth=0.4, markersize=7,
           label=f"Ensemble holdout   R²={r2_hld:.3f}  MAE={mae_hld:.2f}  ($n$={len(hold)})"),
]
leg1 = ax.legend(handles=proto_handles, loc="upper left",
                 title="Protocol (colour)",
                 title_fontsize=7, handlelength=0.8)
leg1.get_title().set_fontweight("bold")
ax.add_artist(leg1)

reg_handles = [
    Line2D([0],[0], marker=REGIME_MARK[r], linestyle="",
           markerfacecolor="#666", markeredgecolor="white",
           markeredgewidth=0.4, markersize=7,
           label=f"Regime {r}")
    for r in ["Ia", "Ib", "II", "III"]
]
leg2 = ax.legend(handles=reg_handles, loc="lower right",
                 title="Reaney regime (marker)",
                 title_fontsize=7, handlelength=0.8, ncol=1)
leg2.get_title().set_fontweight("bold")

# Save
for ext in ("pdf", "png"):
    path = os.path.join(OUT_DIR, f"fig_parity_3protocols.{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
print(f"\nGSS:    R²={r2_gss:.4f}  MAE={mae_gss:.3f}  n={len(gss)}")
print(f"FRM:    R²={r2_frm:.4f}  MAE={mae_frm:.3f}  n={len(frm)}")
print(f"Hold:   R²={r2_hld:.4f}  MAE={mae_hld:.3f}  n={len(hold)}")
