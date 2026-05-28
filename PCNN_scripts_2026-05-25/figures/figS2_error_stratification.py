"""
Figure S2 — Error stratification
Publication-quality figure for Nature Communications supplementary.
Run from: .
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

REGIME_COLORS = {"Ia": "#264653", "Ib": "#2A9D8F", "II": "#E9C46A", "III": "#E76F51"}
OUR_COLOR = "#2874A6"; GRAY = "#B0BEC5"; RED = "#E74C3C"
FIG_FULL  = (7.2, 3.8)

# ── Load data ─────────────────────────────────────────────────────────────────
PROC_DIR    = "./data/processed"
LATENT_CSV  = "./extracted_data/latent_space_trunk.csv"

df        = pd.read_csv(f"{PROC_DIR}/test_holdout_predictions.csv")
calib_idx = json.load(open(f"{PROC_DIR}/calibration_split_idx.json"))["calib_idx"]
fm        = pd.read_parquet(f"{PROC_DIR}/feature_matrix_v7.parquet")
hold_fm   = fm.iloc[calib_idx].reset_index(drop=True)

def _reg(row):
    for r in ["Ia", "Ib", "II", "III"]:
        if row.get(f"regime_{r}", 0) == 1: return r
    return "Unknown"

assert len(df) == len(hold_fm) == 116, f"length mismatch: df={len(df)} hold={len(hold_fm)}"
df["reaney_regime"] = hold_fm.apply(_reg, axis=1).values
# A-site cation strings live in the latent-space export (idx == row position)
df["a_site"] = pd.read_csv(LATENT_CSV).iloc[calib_idx]["a_site"].values

y_true = df["er_measured"].values
y_pred = df["er_predicted"].values

def smape(yt, yp):
    return np.mean(2 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp) + 1e-9)) * 100

def r2(yt, yp):
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-9)

regime_df    = df[df["reaney_regime"].isin(["Ia", "Ib", "II", "III"])]
regime_order = ["III", "II", "Ib", "Ia"]
regime_stats = {}
for reg in regime_order:
    sub = regime_df[regime_df["reaney_regime"] == reg]
    regime_stats[reg] = {
        "smape": smape(sub["er_measured"].values, sub["er_predicted"].values),
        "r2":    r2(sub["er_measured"].values, sub["er_predicted"].values),
        "n":     len(sub),
    }

asite_stats = {}
for asite, sub in df.groupby("a_site"):
    if len(sub) >= 5:
        asite_stats[asite] = {
            "smape": smape(sub["er_measured"].values, sub["er_predicted"].values),
            "n":     len(sub),
        }
asite_order = sorted(asite_stats, key=lambda x: asite_stats[x]["smape"], reverse=True)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=FIG_FULL, layout="constrained")
for ax in axes:
    style4(ax)

# Panel A
ax         = axes[0]
y_pos      = np.arange(len(regime_order))
smape_vals = [regime_stats[r]["smape"] for r in regime_order]
colors_a   = [REGIME_COLORS[r] for r in regime_order]
ax.barh(y_pos, smape_vals, color=colors_a, height=0.5, edgecolor="white")
for i, reg in enumerate(regime_order):
    s = regime_stats[reg]
    ax.text(smape_vals[i] + 0.3, i,
            f"$n$={s['n']}  $R^2$={s['r2']:.2f}",
            va="center", fontsize=7)
ax.set_yticks(y_pos)
ax.set_yticklabels(regime_order)
ax.set_xlabel("SMAPE (%)")
ax.set_ylabel("Reaney regime")
ax.set_title("a  Prediction error by structural regime", loc="left")
ax.set_xlim(0, max(smape_vals) * 1.70)   # bars anchored at zero (touch y-axis)
# (the "*Regime Ia (n=0 in holdout): evaluated via leave-one-family-out" note lives in the caption)

# Panel B
ax        = axes[1]
smape_b   = [asite_stats[a]["smape"] for a in asite_order]
norm      = mcolors.Normalize(vmin=min(smape_b), vmax=max(smape_b))
cmap      = mcolors.LinearSegmentedColormap.from_list("lo_hi", [OUR_COLOR, RED])
colors_b  = [cmap(norm(v)) for v in smape_b]
y_pos_b   = np.arange(len(asite_order))
ax.barh(y_pos_b, smape_b, color=colors_b, height=0.5, edgecolor="white")
for i, asite in enumerate(asite_order):
    ax.text(smape_b[i] + 0.3, i,
            f"$n$={asite_stats[asite]['n']}",
            va="center", fontsize=7)
ax.set_yticks(y_pos_b)
ax.set_yticklabels(asite_order)
ax.set_xlabel("SMAPE (%)")
ax.set_ylabel("A-site cation")
ax.set_title("b  Prediction error by A-site cation", loc="left")
ax.set_xlim(0, max(smape_b) * 1.55)   # bars anchored at zero (touch y-axis)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = "./figures_output/all_figures/supp"
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/figS2_error_stratification.{ext}", dpi=300)
print("Saved figS2_error_stratification.pdf and .png")
plt.close(fig)
for reg in regime_order:
    s = regime_stats[reg]
    print(f"  {reg}: SMAPE={s['smape']:.1f}%  R²={s['r2']:.3f}  n={s['n']}")
