"""
Figure 3 — Physics decomposition
Nature Communications: Physics-constrained neural networks for ceramic dielectric constant prediction
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

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

OUR_COLOR = "#2874A6"
GRAY      = "#B0BEC5"
RED       = "#E74C3C"
ORANGE    = "#E67E22"
FIG_FULL  = (7.2, 3.5)
OUT_DIR   = "./figures_output/main"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
regimes  = ["Ia", "Ib", "II", "III"]
cm_pct   = [60.0, 60.0, 44.0, 39.0]
lst_pct  = [31.0, 39.0, 55.0, 58.0]
tilt_pct = [ 2.0,  0.5,  0.5,  1.5]
res_pct  = [ 7.0,  0.5,  0.5,  1.5]

colors_stack = {
    "CM":       "#264653",
    "LST":      "#2A9D8F",
    "Tilt":     "#E9C46A",
    "Residual": "#E76F51",
}

asites    = ["Pb", "Ca", "La", "Sr", "Ba"]
flst_vals = [2.27, 1.73, 1.38, 0.91, 0.70]
bar_colors = [RED, ORANGE, "#E9C46A", GRAY, "#B0C4DE"]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIG_FULL, layout="constrained")
style4(ax_a)
style4(ax_b)

# ── Panel A: Stacked bar ──────────────────────────────────────────────────────
x     = np.arange(len(regimes))
width = 0.5

ax_a.bar(x, cm_pct,   width, color=colors_stack["CM"],       label="CM anchor")
ax_a.bar(x, lst_pct,  width, bottom=cm_pct,                  color=colors_stack["LST"],      label="LST")
ax_a.bar(x, tilt_pct, width,
         bottom=[c + l for c, l in zip(cm_pct, lst_pct)],    color=colors_stack["Tilt"],     label="Tilt")
bot_res = [c + l + t for c, l, t in zip(cm_pct, lst_pct, tilt_pct)]
ax_a.bar(x, res_pct,  width, bottom=bot_res,                  color=colors_stack["Residual"], label="Residual")

ax_a.set_xticks(x)
ax_a.set_xticklabels(regimes)
ax_a.set_xlabel("Reaney regime")
ax_a.set_ylabel("Contribution to $\\varepsilon_r$ (%)")
ax_a.set_title("a  Branch contributions by structural regime", loc="left")
ax_a.set_ylim(0, 112)
ax_a.legend(loc="upper right")

# Label the dominant LST bar (Regime II, index 2)
ax_a.text(x[2], cm_pct[2] + lst_pct[2] / 2,
          "53%", ha="center", va="center",
          color="white", fontsize=7.5, fontweight="bold")

# ── Panel B: f_LST horizontal bars ───────────────────────────────────────────
y_pos = np.arange(len(asites))
bars  = ax_b.barh(y_pos, flst_vals, color=bar_colors, height=0.5, edgecolor="none")
ax_b.axvline(1.0, color=GRAY, linestyle="--", lw=1.0, label="No enhancement")
ax_b.set_xlim(0, 3.0)
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(asites)
ax_b.set_xlabel("LST enhancement factor  $f_{\\mathrm{LST}}$")
ax_b.set_ylabel("A-site cation")
ax_b.set_title("b  Soft-mode enhancement factor by A-site", loc="left")
ax_b.invert_yaxis()

for bar, val in zip(bars, flst_vals):
    ax_b.text(val + 0.06, bar.get_y() + bar.get_height() / 2,
              f"{val:.2f}×", va="center", ha="left", fontsize=7)

ax_b.legend()

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    path = os.path.join(OUT_DIR, f"fig3_decomposition.{ext}")
    fig.savefig(path, dpi=300)
    print(f"Saved: {path}")
plt.close(fig)
print("Figure 3 done.")
