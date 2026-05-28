"""
Figure 1 — Failure of classical theory and motivation for physics-constrained ML
Nature Communications

FAITHFUL TO CAPTIONS (rebuilt 2026-05-24). All numbers computed live from the
per-sample decomposition CSV (valid-CM, n=1,124).

Panels (captions.md Fig. 1):
  a  CM parity: predicted (ε_CM) vs measured ε_r; R^2 = -0.584, MAPE 45.9%
  b  per-regime CM residuals (ε_measured - ε_CM); medians II 25.0, III 19.1
  c  enhancement-ratio histogram ε_measured / ε_CM (median marked)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np, pandas as pd, os

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.titleweight": "bold", "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "#CCCCCC",
    "figure.dpi": 150, "savefig.dpi": 600, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": False, "lines.linewidth": 1.5, "lines.markersize": 5,
})
def style4(ax, lw=0.8):
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(lw)
    ax.tick_params(direction="in", top=False, right=False, bottom=True, left=True, width=0.8, length=3)
    ax.set_axisbelow(True)

REG = {"Ia": "#264653", "Ib": "#2A9D8F", "II": "#E9C46A", "III": "#E76F51"}
GRAY = "#B0BEC5"
CSV = "./extracted_data/decomposition_per_sample.csv"
OUT_DIR = "./figures_output/main"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV)
d = df[df["has_cm"] == True].copy()
yt, yc = d["er_measured"].values, d["er_cm"].values
r2 = 1 - np.sum((yt - yc) ** 2) / np.sum((yt - yt.mean()) ** 2)
mape = 100 * np.mean(np.abs((yt - yc) / yt))
d["resid"] = d["er_measured"] - d["er_cm"]
d["ratio"] = d["er_measured"] / d["er_cm"]
regs = ["Ia", "Ib", "II", "III"]

fig = plt.figure(figsize=(7.2, 2.6), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0])
ax_a, ax_b, ax_c = (fig.add_subplot(gs[i]) for i in range(3))
for ax in (ax_a, ax_b, ax_c): style4(ax)

# Panel a — CM parity
for rname in ["III", "II", "Ib", "Ia"]:  # densest last so sparse regimes stay visible
    g = d[d["regime"] == rname]
    ax_a.scatter(g["er_measured"], g["er_cm"], s=8, alpha=0.45,
                 color=REG.get(rname, GRAY), edgecolors="none", zorder=3, rasterized=True)
lims = [0, 150]
ax_a.plot(lims, lims, "k--", lw=0.9, zorder=4, label="1:1")
ax_a.fill_between(lims, [0.9*x for x in lims], [1.1*x for x in lims],
                  color="#999", alpha=0.12, zorder=1, label="±10%")
ax_a.set_xlim(0, 150); ax_a.set_ylim(0, 80)  # CM predictions never exceed ~45 → tighter y-scale
ax_a.set_xlabel(r"Measured $\varepsilon_r$"); ax_a.set_ylabel(r"CM-predicted $\varepsilon_{r}^{\mathrm{CM}}$")
ax_a.set_title("a  Clausius–Mossotti fails", loc="left")
ax_a.text(0.96, 0.05, f"$R^2$ = {r2:.3f}\nMAPE = {mape:.1f}%",
          transform=ax_a.transAxes, ha="right", va="bottom", fontsize=7,
          bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", alpha=0.92))
hh = [mpatches.Patch(color=REG[r], label=f"Regime {r}") for r in regs]
ax_a.legend(handles=hh, loc="upper left", fontsize=6.0, handlelength=1.0)

# Panel b — per-regime CM residuals (box plot: IQR box + whiskers + outlier dots + mean diamond)
data = [d.loc[d["regime"] == r, "resid"].values for r in regs]
positions = np.arange(len(regs))
bp = ax_b.boxplot(data, positions=positions, widths=0.55, patch_artist=True, whis=1.5,
                  medianprops=dict(color="#222", lw=1.2),
                  whiskerprops=dict(color="#444", lw=0.8),
                  capprops=dict(color="#444", lw=0.8),
                  flierprops=dict(marker="o", markersize=2.0,
                                  markerfacecolor="#888", markeredgecolor="none", alpha=0.55))
for box, r in zip(bp["boxes"], regs):
    box.set_facecolor(REG[r]); box.set_alpha(0.75); box.set_edgecolor("#222"); box.set_linewidth(0.6)
for i, r in enumerate(regs):
    v = data[i]
    med_r = np.median(v)
    ax_b.scatter(i, v.mean(), marker="D", s=30, color="white", edgecolor="#222", lw=0.8, zorder=6)
    ax_b.text(i + 0.32, med_r, f"{med_r:.0f}", ha="left", va="center", fontsize=6.2, fontweight="bold", color="#222")
ax_b.axhline(0, color="#888", lw=0.7, ls=":")
ax_b.set_xticks(positions); ax_b.set_xticklabels(regs)
ax_b.set_xlabel("Reaney regime")
ax_b.set_ylabel(r"CM residual  $\varepsilon_{\mathrm{meas}}-\varepsilon_{r}^{\mathrm{CM}}$")
ax_b.set_title("b  Systematic underprediction", loc="left")
ax_b.set_ylim(-20, 120)

# Panel c — enhancement-ratio histogram
ratio = d["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
med = ratio.median()
ax_c.hist(ratio[ratio < 6], bins=44, color="#2A9D8F", edgecolor="white", linewidth=0.3, zorder=3)
ytop = ax_c.get_ylim()[1]
ax_c.axvline(1.0, color="#888", lw=0.9, ls=":", zorder=4)
ax_c.axvline(med, color="#C0392B", lw=1.2, ls="--", zorder=5)
ax_c.text(med + 0.08, ytop*0.92, f"median = {med:.2f}", color="#C0392B",
          fontsize=6.4, ha="left", va="top", fontweight="bold")
ax_c.text(1.05, ytop*0.60, "no\nenhancement", color="#888",
          fontsize=5.6, ha="left", va="top")   # nudged right of the x=1 line so it clears the y-axis tick labels
ax_c.set_xlabel(r"enhancement ratio  $\varepsilon_{\mathrm{meas}}/\varepsilon_{r}^{\mathrm{CM}}$")
ax_c.set_ylabel("count")
ax_c.set_title("c  Soft-mode enhancement", loc="left")
ax_c.set_xlim(0, 6)

for ext in ["pdf", "png", "svg"]:
    fig.savefig(os.path.join(OUT_DIR, f"fig1_motivation.{ext}"), dpi=600)
    print(f"Saved fig1_motivation.{ext}")
plt.close(fig)
print(f"Figure 1 rebuilt. R2={r2:.3f} MAPE={mape:.1f}% ratio_median={med:.2f}")
print("  residual medians:", {r: round(float(np.median(d.loc[d.regime==r,'resid'])),1) for r in regs})
