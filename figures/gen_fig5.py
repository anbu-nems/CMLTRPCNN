"""
Figure 5 — Applicability Domain reliability
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
FIG_FULL  = (7.2, 3.2)
OUT_DIR   = "./figures_output/main"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIG_FULL, layout="constrained")
style4(ax_a)
style4(ax_b)

# ── Panel A: AD reliability grouped bars ─────────────────────────────────────
groups    = ["Layer 1\n(physics gate)", "Layer 2\n(KDE density)", "Combined AD"]
in_ad_r2  = [0.938, 0.916, 0.916]
out_ad_r2 = [0.756, 0.889, 0.889]
pct_in    = ["71.7%", "51.2%", "66.3%"]

x     = np.arange(len(groups))
width = 0.30

bars_in  = ax_a.bar(x - width / 2, in_ad_r2,  width, color=OUR_COLOR,
                    label="In-AD",  edgecolor="white")
bars_out = ax_a.bar(x + width / 2, out_ad_r2, width, color=GRAY,
                    label="Out-AD", edgecolor="white")

ax_a.set_ylim(0.6, 1.05)
ax_a.set_ylabel("$R^2$  (calibration split, $n$=123)")
ax_a.set_xticks(x)
ax_a.set_xticklabels(groups, fontsize=7)
ax_a.set_title("a  AD reliability: in-domain vs out-of-domain", loc="left")
ax_a.legend()

# Fraction in-AD annotations — above the taller bar, inside ylim
for i, pct in enumerate(pct_in):
    top = max(in_ad_r2[i], out_ad_r2[i]) + 0.012
    ax_a.text(i, top, f"{pct} in-AD", ha="center", va="bottom",
              fontsize=6.5, color="#555555")

# ── Panel B: sigma_conf by A-site ─────────────────────────────────────────────
asites_b   = ["Pb", "La", "Ca", "Sr", "Ba"]
sigma_vals = [0.158, 0.205, 0.326, 0.350, 0.387]

def sigma_color(v):
    if v < 0.20:   return OUR_COLOR
    elif v <= 0.35: return "#E9C46A"
    else:           return ORANGE

bar_colors_b = [sigma_color(v) for v in sigma_vals]

y_pos_b = np.arange(len(asites_b))
ax_b.barh(y_pos_b, sigma_vals, color=bar_colors_b, height=0.5, edgecolor="none")
ax_b.axvline(0.35, color=GRAY, linestyle="--", lw=1.0, label="Threshold = 0.35")

ax_b.set_xlim(0, 0.52)
ax_b.set_yticks(y_pos_b)
ax_b.set_yticklabels(asites_b)
ax_b.set_xlabel("Mean $\\sigma_{\\mathrm{conf}}$ (physics uncertainty gate)")
ax_b.set_ylabel("A-site cation")
ax_b.set_title("b  Physics gate $\\sigma_{\\mathrm{conf}}$ by A-site\n"
               "(low = reliable physics,  high = uncertain)", loc="left")
ax_b.invert_yaxis()
ax_b.legend()


# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    path = os.path.join(OUT_DIR, f"fig5_ad_reliability.{ext}")
    fig.savefig(path, dpi=300)
    print(f"Saved: {path}")
plt.close(fig)
print("Figure 5 done.")
