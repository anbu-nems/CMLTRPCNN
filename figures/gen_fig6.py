"""
Figure 6 — f_LST heatmap
Nature Communications: Physics-constrained neural networks for ceramic dielectric constant prediction
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
    print("seaborn not available, using matplotlib imshow fallback")

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

FIG_HALF = (3.5, 3.2)
OUT_DIR  = "/Users/anbu/Desktop/NC figures/main"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
row_labels = ["Pb", "Ca", "La", "Sr", "Ba"]
col_labels = ["Ia", "Ib", "II", "III"]
data = np.array([
    [0.85, 1.20, 2.24, 1.90],  # Pb
    [0.60, 0.85, 1.95, 1.51],  # Ca
    [0.30, 0.34, 2.08, 1.40],  # La
    [0.40, 0.55, 1.10, 0.90],  # Sr
    [0.35, 0.45, 1.15, 0.70],  # Ba
])

fig, ax = plt.subplots(figsize=FIG_HALF, layout="constrained")

if HAS_SNS:
    import pandas as pd
    df_heat = pd.DataFrame(data, index=row_labels, columns=col_labels)
    sns.heatmap(df_heat, ax=ax,
                cmap="RdYlBu_r",
                annot=True, fmt=".2f",
                annot_kws={"size": 8, "weight": "bold"},
                vmin=0.3, vmax=2.5,
                linewidths=1.5, linecolor="white",
                cbar_kws={"label": "$f_{\\mathrm{LST}}$ enhancement factor",
                          "shrink": 0.85})
    # Restore 4-sided box after seaborn resets spines
    style4(ax)
    ax.tick_params(direction="in", top=False, right=False, width=0.8, length=3)
else:
    im = ax.imshow(data, cmap="RdYlBu_r", vmin=0.3, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.2f}",
                    ha="center", va="center", fontsize=8, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("$f_{\\mathrm{LST}}$ enhancement factor", fontsize=8)
    style4(ax)

ax.set_xlabel("Reaney regime")
ax.set_ylabel("A-site cation")
ax.set_title("$f_{\\mathrm{LST}}$ enhancement factor by composition", fontweight="bold")

# Caption note as axis label below colorbar
ax.figure.text(0.5, -0.02,
               "$f_{\\mathrm{LST}} = 1 + \\delta_{\\mathrm{LST}}/\\varepsilon_r^{\\mathrm{CM}}$"
               "   (>1 enhancement, <1 suppression)",
               ha="center", fontsize=7, color="#555555", style="italic")

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    path = os.path.join(OUT_DIR, f"fig6_flst_heatmap.{ext}")
    fig.savefig(path, dpi=300)
    print(f"Saved: {path}")
plt.close(fig)
print("Figure 6 done.")
