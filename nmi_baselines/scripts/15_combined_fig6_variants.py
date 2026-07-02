"""
15_combined_fig6_variants.py — combine new standalone Fig 4 (variants
accuracy bars) + Fig 5 (variants UQ forest plot) into a single 2-panel
"Fig 6" for the NMI 6-figure layout (Option A).

Each sub-panel is a clean reuse of the existing single-panel design from
script 13. Side-by-side 2-panel layout at NMI double-column width. No
twin-y axis (which was what broke earlier composites).

Output: fig_NMI_6_architectural_variants.{pdf,png}
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

ROOT     = "../.."  # repo root (run from nmi_baselines/scripts/)
RES_DIR  = os.path.join(ROOT, "nmi_baselines", "results")
FIG_DIR  = os.path.join(ROOT, "nmi_baselines", "figures")

# NMI Nature-family preset (matches script 13)
plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset":    "dejavusans",
    "font.size":           10,
    "axes.titlesize":      11,
    "axes.titleweight":    "bold",
    "axes.labelsize":      10,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     8.5,
    "legend.frameon":      True,
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "#AAAAAA",
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "axes.linewidth":      1.0,
    "xtick.direction":     "in", "ytick.direction": "in",
    "xtick.top":           False, "ytick.right": False,
    "xtick.bottom":        True, "ytick.left": True,
    "xtick.major.width":   0.9, "ytick.major.width": 0.9,
    "xtick.major.size":    3,   "ytick.major.size": 3,
    "axes.grid":           False,
    "lines.linewidth":     1.6,
    "lines.markersize":    5,
    "axes.unicode_minus":  True,
})

NC_WIDTH_DOUBLE = 7.205

def style4(ax, lw=1.0, square=True):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw)
    ax.tick_params(direction="in", top=False, right=False,
                   bottom=True, left=True, width=0.9, length=3)
    ax.set_axisbelow(False)
    if square: ax.set_box_aspect(1)

def panel_letter(ax, letter, fontsize=12, x=-0.18, y=1.10):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize, fontweight="bold",
            family=plt.rcParams["font.family"])

OUR_COLOR       = "#E76F51"
COLOR_VARIANT_A = "#009E73"
COLOR_VARIANT_B = "#CC79A7"
COLOR_VARIANT_C = "#0072B2"

with open(os.path.join(RES_DIR, "04_ml_literature_baselines.json")) as f:   r04 = json.load(f)
with open(os.path.join(RES_DIR, "08_monotonic_lst_head.json")) as f:        r08 = json.load(f)
with open(os.path.join(RES_DIR, "09_quantile_conformal_heads.json")) as f:  r09 = json.load(f)
with open(os.path.join(RES_DIR, "10_laplace_last_layer.json")) as f:        r10 = json.load(f)

print("[fig6] PCNN architectural variants — accuracy + UQ ...")
fig, axes = plt.subplots(1, 2, figsize=(NC_WIDTH_DOUBLE, 3.6),
                         gridspec_kw={"wspace": 0.45})

# ─── Panel (a): variants accuracy ───────────────────────────────────────
ax = axes[0]; style4(ax)
variants = ["PCNN", "Mono. (A)", "Quant. (B)", "Laplace (C)"]
v_r2  = [r04["PCNN_reference"]["holdout_r2"], r08["holdout_r2"],
         r09["holdout_r2"], r10["holdout_r2_post_laplace"]]
v_mae = [r04["PCNN_reference"]["holdout_mae"], r08["holdout_mae"],
         r09["holdout_mae"], r10["holdout_mae_post_laplace"]]
v_colors = [OUR_COLOR, COLOR_VARIANT_A, COLOR_VARIANT_B, COLOR_VARIANT_C]

bars = ax.bar(variants, v_r2, color=v_colors,
              edgecolor="white", linewidth=0.7, width=0.66)
for bar, r, m in zip(bars, v_r2, v_mae):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0015,
            f"{r:.3f}", ha="center", va="bottom", fontsize=8, color="#222")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0080,
            f"MAE {m:.2f}", ha="center", va="bottom", fontsize=7, color="#666")
ax.axhline(y=r04["PCNN_reference"]["holdout_r2"], color=OUR_COLOR,
           linestyle=":", linewidth=0.7, alpha=0.5, zorder=1)
ax.set_ylabel(r"Held-out $R^2$")
ax.set_ylim(0.935, max(v_r2) * 1.014)
ax.tick_params(axis="x", rotation=20)
for t in ax.get_xticklabels(): t.set_horizontalalignment("right")
panel_letter(ax, "a")


# ─── Panel (b): variants UQ forest plot ─────────────────────────────────
ax = axes[1]; style4(ax, square=False)

def wilson_ci(p, n, z=1.96):
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return center - half, center + half

n_ho = 116
forest = [
    ("PCNN\n(post-hoc conformal)",      0.905, 28.76, OUR_COLOR),
    ("+ Mono. (A)\n(post-hoc conformal)", 0.900, 28.76, COLOR_VARIANT_A),
    ("+ Quant. (B)",                     r09["holdout_90pct_coverage"],
                                         r09["holdout_pi_width_mean"], COLOR_VARIANT_B),
    ("+ Laplace (C)",                    r10["holdout_bayesian_90pct_coverage"],
                                         r10["holdout_bayesian_pi_width"], COLOR_VARIANT_C),
]
y_positions = np.arange(len(forest))[::-1]

ax.axvline(x=90, color="black", linestyle="--", linewidth=0.9, alpha=0.6,
           zorder=1, label="Target 90%")

for ypos, (lab, cov, wid, c) in zip(y_positions, forest):
    lo, hi = wilson_ci(cov, n_ho)
    ax.plot([100*lo, 100*hi], [ypos, ypos], color=c, linewidth=2.0, zorder=3)
    ax.plot([100*lo, 100*lo], [ypos-0.12, ypos+0.12], color=c, linewidth=1.5, zorder=3)
    ax.plot([100*hi, 100*hi], [ypos-0.12, ypos+0.12], color=c, linewidth=1.5, zorder=3)
    ax.scatter([100*cov], [ypos], s=80, color=c, edgecolors="black",
               linewidths=0.9, zorder=4)
    ax.text(100*cov, ypos + 0.28, f"{100*cov:.1f}% (w={wid:.1f})",
            va="bottom", ha="center", fontsize=7.5, color="#222")

ax.set_yticks(y_positions)
ax.set_yticklabels([lab for lab, *_ in forest], fontsize=8)
ax.set_xlabel("90% PI coverage (%), Wilson 95% CI")
ax.set_xlim(35, 100)
ax.set_ylim(-0.6, len(forest) - 0.1)
ax.legend(loc="lower left", framealpha=0.92, fontsize=7.5)
panel_letter(ax, "b")

# Save
pdf = os.path.join(FIG_DIR, "fig_NMI_6_architectural_variants.pdf")
png = os.path.join(FIG_DIR, "fig_NMI_6_architectural_variants.png")
fig.savefig(pdf); fig.savefig(png, dpi=300); plt.close(fig)
print(f"   → fig_NMI_6_architectural_variants.{{pdf,png}}")
print(f"   (a) variants accuracy: PCNN={v_r2[0]:.3f} → Laplace={v_r2[3]:.3f}")
print(f"   (b) variants UQ forest plot")
