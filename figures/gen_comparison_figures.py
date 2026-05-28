"""
Generate publication-quality comparison figures for NC paper.
Saves to: NC figures/comparison/

Figures:
  A. Train vs Test R² grouped bar (overfitting gap)
  B. Test R² horizontal ranking bar
  C. Strat-GSS CV R² with error bars
  D. CV vs Holdout scatter (calibration)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
})

# ── Data ──────────────────────────────────────────────────────────────────────
# Train / Test performance
models_tt = [
    ("CM-only",           -0.517,  22.252, -0.950, 37.730),
    ("Ridge",              0.848,   6.814,  0.875,  9.038),
    ("Random Forest",      0.986,   1.604,  0.926,  6.471),
    ("XGBoost",            0.999,   0.327,  0.921,  6.338),
    ("CatBoost",           0.998,   0.817,  0.919,  6.587),
    ("BPNN / MLP",         0.980,   2.668,  0.891,  7.642),
    ("PIRNN",              None,    None,   0.878,  7.568),
    ("PCNN\n(ours)",       0.926,   4.549,  0.921,  6.645),
]

# Strat-GSS CV data
models_cv = [
    ("CM-only",           -0.806, 0.074),
    ("Ridge",              0.664, 0.174),
    ("BPNN / MLP",         0.921, 0.072),
    ("PIRNN",              0.908, 0.109),
    ("Random Forest",      0.926, 0.046),
    ("CatBoost",           0.953, 0.043),
    ("XGBoost",            0.959, 0.026),
    ("PCNN\n(ours)",       0.882, 0.055),
]

PCNN_COLOR   = "#E76F51"   # coral — our model
TRAIN_COLOR  = "#2A9D8F"   # teal
TEST_COLOR   = "#264653"   # dark teal
BASE_COLOR   = "#B0BEC5"   # gray for baselines
BAD_COLOR    = "#F4A261"   # orange for high gap
CV_COLOR     = "#5E81AC"   # blue for CV bars

# ── Figure A: Train vs Test R² grouped bar ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 3.6))

labels  = [m[0] for m in models_tt]
train_r = [m[1] for m in models_tt]
test_r  = [m[3] for m in models_tt]
n = len(labels)
x = np.arange(n)
w = 0.35

# Color train bars: PCNN highlighted, others gray
train_colors = [PCNN_COLOR if "PCNN" in l else TRAIN_COLOR for l in labels]
test_colors  = [PCNN_COLOR if "PCNN" in l else TEST_COLOR  for l in labels]

# Train bars (skip PIRNN — no train R²)
for i, (tr, col) in enumerate(zip(train_r, train_colors)):
    if tr is not None:
        bar = ax.bar(x[i] - w/2, tr, w, color=col, alpha=0.75,
                     edgecolor="white", linewidth=0.5, zorder=3)

# Test bars
for i, (te, col) in enumerate(zip(test_r, test_colors)):
    ax.bar(x[i] + w/2, te, w, color=col, alpha=0.95,
           edgecolor="white", linewidth=0.5, zorder=3)

# Gap annotation for XGBoost and CatBoost (biggest gap)
for i, m in enumerate(models_tt):
    if m[0] in ("XGBoost", "CatBoost") and m[1] is not None:
        gap = m[1] - m[3]
        ax.annotate(f"gap\n{gap:+.3f}",
                    xy=(x[i], (m[1]+m[3])/2),
                    fontsize=6.5, ha="center", va="center",
                    color="#E11D48", fontweight="bold")

# PCNN gap annotation
pcnn_i = [i for i,m in enumerate(models_tt) if "PCNN" in m[0]][0]
pcnn_gap = models_tt[pcnn_i][1] - models_tt[pcnn_i][3]
ax.annotate(f"gap\n{pcnn_gap:+.3f}",
            xy=(x[pcnn_i], 0.60),
            fontsize=6.5, ha="center", va="center",
            color=PCNN_COLOR, fontweight="bold")

ax.axhline(0, color="black", linewidth=0.6, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("R²")
ax.set_ylim(-1.25, 1.08)
ax.set_title("(a)  Train vs Test R² — Overfitting Comparison")

train_patch = mpatches.Patch(color=TRAIN_COLOR, alpha=0.75, label="Train R²")
test_patch  = mpatches.Patch(color=TEST_COLOR,  alpha=0.95, label="Test R² (held-out)")
pcnn_patch  = mpatches.Patch(color=PCNN_COLOR,  alpha=0.90, label="PCNN (ours)")
ax.legend(handles=[train_patch, test_patch, pcnn_patch],
          loc="lower right", ncol=3, fontsize=7.5)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figA_train_test_gap.pdf"))
fig.savefig(os.path.join(OUT_DIR, "figA_train_test_gap.png"), dpi=300)
plt.close()
print("Saved figA_train_test_gap")

# ── Figure B: Test R² horizontal ranking ─────────────────────────────────────
# Sort by test R², exclude CM-only for visual clarity
data_b = [(m[0], m[3], m[4]) for m in models_tt]
data_b_sorted = sorted(data_b, key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(5.5, 3.2))

names  = [d[0] for d in data_b_sorted]
test_r = [d[1] for d in data_b_sorted]
maes   = [d[2] for d in data_b_sorted]

colors = [PCNN_COLOR if "PCNN" in n else BASE_COLOR for n in names]
colors = ["#EF4444" if v < 0 else c for v, c in zip(test_r, colors)]

bars = ax.barh(names, test_r, color=colors, height=0.6,
               edgecolor="white", linewidth=0.5, zorder=3)

# Value labels
for bar, val, mae in zip(bars, test_r, maes):
    xpos = max(val, 0) + 0.01
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f"R²={val:.3f}  MAE={mae:.1f}",
            va="center", ha="left", fontsize=7.5,
            color="#111" if "PCNN" not in names[bars.patches.index(bar)] else PCNN_COLOR,
            fontweight="bold" if "PCNN" in names[bars.patches.index(bar)] else "normal")

ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlabel("Test R² (held-out, n=123)")
ax.set_xlim(-1.3, 1.25)
ax.set_title("(b)  Test Set Performance Ranking")
ax.grid(axis="x", alpha=0.2, linestyle="--")
ax.set_axisbelow(True)

pcnn_patch = mpatches.Patch(color=PCNN_COLOR, label="PCNN (ours)")
base_patch = mpatches.Patch(color=BASE_COLOR, label="Baselines")
ax.legend(handles=[pcnn_patch, base_patch], loc="lower right", fontsize=7.5)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figB_test_ranking.pdf"))
fig.savefig(os.path.join(OUT_DIR, "figB_test_ranking.png"), dpi=300)
plt.close()
print("Saved figB_test_ranking")

# ── Figure C: Strat-GSS CV R² with error bars ────────────────────────────────
data_c = sorted(models_cv, key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(5.5, 3.2))

names_c = [d[0] for d in data_c]
means_c = [d[1] for d in data_c]
stds_c  = [d[2] for d in data_c]

colors_c = [PCNN_COLOR if "PCNN" in n else
            "#EF4444" if v < 0 else CV_COLOR
            for n, v in zip(names_c, means_c)]

bars = ax.barh(names_c, means_c, color=colors_c, height=0.6,
               edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
ax.errorbar(means_c, names_c, xerr=stds_c, fmt="none",
            ecolor="#333", elinewidth=1.2, capsize=3.5, zorder=4)

for bar, val, std in zip(bars, means_c, stds_c):
    if val > -0.5:
        ax.text(max(val, 0) + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}±{std:.3f}",
                va="center", ha="left", fontsize=7.5)

ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlabel("Strat-GSS CV R² (mean ± std, 5-fold)")
ax.set_xlim(-1.3, 1.25)
ax.set_title("(c)  Cross-Validation Stability (Strat-GSS)")
ax.grid(axis="x", alpha=0.2, linestyle="--")
ax.set_axisbelow(True)

pcnn_patch = mpatches.Patch(color=PCNN_COLOR, alpha=0.85, label="PCNN (ours)")
cv_patch   = mpatches.Patch(color=CV_COLOR,   alpha=0.85, label="Baselines")
ax.legend(handles=[pcnn_patch, cv_patch], loc="lower right", fontsize=7.5)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figC_cv_stability.pdf"))
fig.savefig(os.path.join(OUT_DIR, "figC_cv_stability.png"), dpi=300)
plt.close()
print("Saved figC_cv_stability")

# ── Figure D: CV vs Holdout scatter — calibration plot ───────────────────────
cv_ho_data = [
    ("CM-only",       -0.806, -0.950),
    ("Ridge",          0.664,  0.875),
    ("Random Forest",  0.926,  0.926),
    ("XGBoost",        0.959,  0.921),
    ("CatBoost",       0.953,  0.919),
    ("BPNN / MLP",     0.921,  0.914),
    ("PIRNN",          0.908,  0.878),
    ("PCNN (ours)",    0.882,  0.921),
]

fig, ax = plt.subplots(figsize=(4.0, 3.8))

for name, cv, ho in cv_ho_data:
    color  = PCNN_COLOR if "PCNN" in name else BASE_COLOR
    marker = "*" if "PCNN" in name else "o"
    size   = 120 if "PCNN" in name else 55
    ax.scatter(cv, ho, color=color, s=size, marker=marker,
               zorder=4, edgecolors="white", linewidth=0.5)
    offset_x = 0.008
    offset_y = 0.008
    if name == "XGBoost":   offset_y = -0.018
    if name == "CatBoost":  offset_x = -0.055; offset_y = -0.018
    if name == "PIRNN":     offset_y = -0.018
    if name == "CM-only":   offset_x = 0.015
    ax.text(cv + offset_x, ho + offset_y, name,
            fontsize=7, ha="left", va="bottom",
            color=PCNN_COLOR if "PCNN" in name else "#444",
            fontweight="bold" if "PCNN" in name else "normal")

# y=x diagonal (perfect calibration)
lims = [-1.1, 1.05]
ax.plot(lims, lims, "k--", linewidth=0.9, alpha=0.4, label="y = x (perfect)")

# Shade overfitting zone
ax.fill_between(lims, lims, [1.1, 1.1], alpha=0.05, color="red",
                label="Overfit zone (CV > holdout)")

ax.set_xlabel("Strat-GSS CV R²")
ax.set_ylabel("Held-out Test R²")
ax.set_xlim(-1.15, 1.05)
ax.set_ylim(-1.15, 1.05)
ax.set_title("(d)  CV vs Holdout Calibration")
ax.legend(fontsize=7, loc="upper left")
ax.set_aspect("equal")
ax.grid(alpha=0.2, linestyle="--")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figD_cv_vs_holdout.pdf"))
fig.savefig(os.path.join(OUT_DIR, "figD_cv_vs_holdout.png"), dpi=300)
plt.close()
print("Saved figD_cv_vs_holdout")

# ── Figure E: Combined 2×2 panel ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle("Model Comparison — CMLTRPCNNv7.7 vs Baselines", fontsize=11, fontweight="bold", y=1.01)

# Panel A — Train vs Test bar
ax = axes[0, 0]
x = np.arange(len(models_tt))
for i, m in enumerate(models_tt):
    if m[1] is not None:
        c = PCNN_COLOR if "PCNN" in m[0] else TRAIN_COLOR
        ax.bar(x[i]-w/2, m[1], w, color=c, alpha=0.75, edgecolor="white", linewidth=0.4, zorder=3)
    c = PCNN_COLOR if "PCNN" in m[0] else TEST_COLOR
    ax.bar(x[i]+w/2, m[3], w, color=c, alpha=0.95, edgecolor="white", linewidth=0.4, zorder=3)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([m[0] for m in models_tt], fontsize=7.5, rotation=15, ha="right")
ax.set_ylabel("R²")
ax.set_ylim(-1.2, 1.08)
ax.set_title("(a)  Train vs Test R²")
ax.legend(handles=[mpatches.Patch(color=TRAIN_COLOR, alpha=0.75, label="Train"),
                   mpatches.Patch(color=TEST_COLOR,  alpha=0.95, label="Test"),
                   mpatches.Patch(color=PCNN_COLOR,              label="PCNN")],
          fontsize=7, ncol=3, loc="lower right")

# Panel B — Test ranking
ax = axes[0, 1]
bars2 = ax.barh(names, test_r, color=colors, height=0.55,
                edgecolor="white", linewidth=0.4, zorder=3)
for bar, val in zip(bars2, test_r):
    ax.text(max(val,0)+0.01, bar.get_y()+bar.get_height()/2,
            f"{val:.3f}", va="center", ha="left", fontsize=7.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Test R²")
ax.set_xlim(-1.3, 1.2)
ax.set_title("(b)  Test R² Ranking")

# Panel C — CV stability
ax = axes[1, 0]
bars3 = ax.barh(names_c, means_c, color=colors_c, height=0.55,
                edgecolor="white", linewidth=0.4, alpha=0.85, zorder=3)
ax.errorbar(means_c, names_c, xerr=stds_c, fmt="none",
            ecolor="#333", elinewidth=1.0, capsize=3, zorder=4)
for val, name in zip(means_c, names_c):
    if val > -0.3:
        ax.text(max(val,0)+0.01, names_c.index(name),
                f"{val:.3f}", va="center", ha="left", fontsize=7)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Strat-GSS CV R² (mean ± std)")
ax.set_xlim(-1.3, 1.25)
ax.set_title("(c)  CV Stability")

# Panel D — CV vs Holdout
ax = axes[1, 1]
for name, cv, ho in cv_ho_data:
    color  = PCNN_COLOR if "PCNN" in name else BASE_COLOR
    marker = "*" if "PCNN" in name else "o"
    size   = 100 if "PCNN" in name else 45
    ax.scatter(cv, ho, color=color, s=size, marker=marker,
               zorder=4, edgecolors="white", linewidth=0.4)
    ax.text(cv+0.01, ho+0.01, name, fontsize=6.5,
            color=PCNN_COLOR if "PCNN" in name else "#555",
            fontweight="bold" if "PCNN" in name else "normal")
ax.plot([-1.1, 1.05], [-1.1, 1.05], "k--", linewidth=0.8, alpha=0.4)
ax.set_xlabel("CV R²")
ax.set_ylabel("Holdout R²")
ax.set_xlim(-1.15, 1.05)
ax.set_ylim(-1.15, 1.05)
ax.set_title("(d)  CV vs Holdout Calibration")
ax.set_aspect("equal")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_comparison_panel.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig_comparison_panel.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig_comparison_panel (2×2)")

# ── Save all CSVs ─────────────────────────────────────────────────────────────
# 1. Train-test table
rows_tt = []
for m in models_tt:
    rows_tt.append({
        "model": m[0].replace("\n",""),
        "train_r2": m[1], "train_mae": m[2],
        "test_r2": m[3],  "test_mae": m[4],
        "gap_r2": round(m[1]-m[3], 4) if m[1] is not None else None
    })
pd.DataFrame(rows_tt).to_csv(os.path.join(OUT_DIR, "01_train_test_performance.csv"), index=False)

# 2. CV stability table
rows_cv = [{"model": d[0].replace("\n",""),
            "cv_r2_mean": d[1], "cv_r2_std": d[2]} for d in models_cv]
pd.DataFrame(rows_cv).to_csv(os.path.join(OUT_DIR, "02_cv_stability.csv"), index=False)

# 3. CV vs holdout table
rows_cal = [{"model": d[0], "cv_r2": d[1], "holdout_r2": d[2],
             "cv_holdout_gap": round(d[1]-d[2], 4)} for d in cv_ho_data]
pd.DataFrame(rows_cal).to_csv(os.path.join(OUT_DIR, "03_cv_vs_holdout.csv"), index=False)

print("\nSaved CSVs:")
print("  comparison/01_train_test_performance.csv")
print("  comparison/02_cv_stability.csv")
print("  comparison/03_cv_vs_holdout.csv")
print(f"\nAll figures saved to: {OUT_DIR}")
