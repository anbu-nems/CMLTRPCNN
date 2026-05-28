"""
Supplementary Fig. 10 — Feature partition + crystal-system feature ablation.
a: strict 30 / 30 / 43 LST/Tilt/Residual partition (103 physics features) feeding the
   zero-parameter Clausius–Mossotti anchor.
b: crystal-system feature ablation — holdout R² with the 8 crystal-system features zeroed
   (loaded live from results/55_cs_ablation.json; no hard-coded numbers).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import json, os

PIML = "."
OUT  = "./figures_output/supp"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3, "axes.grid": False,
})
C_ANCHOR, C_LST, C_TILT, C_RES, C_GREY = "#264653", "#2A9D8F", "#E9C46A", "#E76F51", "#B0BEC5"

# ── live data ────────────────────────────────────────────────────────────────
part = json.load(open(f"{PIML}/data/processed/feature_partition_v7.json"))
n_lst, n_tilt, n_res = len(part["LST"]), len(part["Tilt"]), len(part["Residual"])
total = n_lst + n_tilt + n_res
ab = json.load(open(f"{PIML}/results/55_cs_ablation.json"))
r2_full, r2_zero = ab["original"]["r2"], ab["cs_zeroed"]["r2"]
dr2, dmae, n_cal = ab["delta_r2"], ab["delta_mae"], ab["n_calib_samples"]
n_csf = ab["n_cs_features"]

fig = plt.figure(figsize=(7.2, 3.4), layout="constrained")
gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.06)
ax_a = fig.add_subplot(gs[0]); ax_b = fig.add_subplot(gs[1])

# ── Panel a — partition map ──────────────────────────────────────────────────
ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1); ax_a.axis("off")
def band(y, h, color, title, items, tcol="white", icol="white"):
    ax_a.add_patch(FancyBboxPatch((0.02, y), 0.80, h, boxstyle="round,pad=0.008",
                   linewidth=0, facecolor=color, zorder=2))
    ax_a.text(0.05, y + h - 0.03, title, va="top", ha="left",
              fontsize=8.3, fontweight="bold", color=tcol, zorder=3)
    if items:
        ax_a.text(0.055, y + h - 0.075, items, va="top", ha="left",
                  fontsize=6.4, color=icol, zorder=3, linespacing=1.45)

BANNER_H, GAP = 0.105, 0.014
avail = 1.0 - BANNER_H - GAP*4
bands = [("Residual", n_res, C_RES, "white", "#fde6dd",
          "sintering $T$ / time, crystal system, space group,\n"
          "configurational entropy, A-off-centring fraction,\nVCA flag, GII…"),
         ("Tilt", n_tilt, C_TILT, "#264653", "#5c4a00",
          "Goldschmidt tolerance $t$, octahedral factor,\n"
          "tilt-angle proxy, bond-angle variance,\nA-site size mismatch, A-site electronegativity…"),
         ("LST", n_lst, C_LST, "white", "#e8f5f4",
          "ionic polarizabilities ($\\alpha_A,\\alpha_B$), soft-mode proxy,\n"
          "Reaney regime $I_a/I_b/II/III$, field strengths,\n"
          "CM baseline ($\\varepsilon_r^{CM}$, $\\sigma_{CM}$), SOJT weighting…")]
y = 0.0; tops = []
for name, n, col, tc, ic, items in bands:
    h = n/total * avail
    band(y, h, col, f"{name} branch  |  {n} features", items, tc, ic)
    tops.append((y, y+h)); y += h + GAP
# CM anchor banner on top
band(y, BANNER_H, C_ANCHOR,
     "$\\varepsilon_r^{\\mathrm{CM}}$  (0 parameters) — Clausius–Mossotti anchor", "", "white")

# right-side bracket: "103 physics features"
y0, y1 = tops[0][0], tops[-1][1]
ax_a.plot([0.85, 0.85], [y0, y1], color="#999", lw=1.0, zorder=4)
for yy in (y0, y1): ax_a.plot([0.83, 0.85], [yy, yy], color="#999", lw=1.0, zorder=4)
ax_a.text(0.875, (y0+y1)/2, f"{total}\nphysics\nfeatures", va="center", ha="left",
          fontsize=7.5, fontweight="bold", color="#333", zorder=4)
ax_a.set_title("a   Strict feature partition", loc="left")

# ── Panel b — crystal-system feature ablation ────────────────────────────────
xb = [0, 1]
bars = ax_b.bar(xb, [r2_full, r2_zero], width=0.6,
                color=[C_LST, C_GREY], edgecolor="white")
for x, v in zip(xb, [r2_full, r2_zero]):
    ax_b.text(x, v + 0.0012, f"{v:.3f}", ha="center", va="bottom",
              fontsize=8, fontweight="bold", color="#222")
ax_b.set_xticks(xb)
ax_b.set_xticklabels(["Full model", f"crystal-system\nfeatures zeroed (n={n_csf})"], fontsize=7.2)
ax_b.set_ylabel(f"Holdout $R^2$  ($n$={n_cal})")
lo = min(r2_full, r2_zero); hi = max(r2_full, r2_zero)
ax_b.set_ylim(lo - 0.010, hi + 0.014)
ax_b.set_title("b   Crystal-system feature ablation", loc="left")
# delta annotation
ax_b.annotate("", xy=(1, r2_zero), xytext=(1, r2_full),
              arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=1.1))
ax_b.text(1.18, (r2_full + r2_zero)/2,
          f"$\\Delta R^2$ = {dr2:+.3f}\n$\\Delta$MAE = {dmae:+.2f}",
          va="center", ha="left", fontsize=6.8, color="#C0392B")
# (interpretive note lives in the caption, not the figure — keeps the panel clean for NC)
ax_b.set_xlim(-0.6, 1.9)
# keep all 4 spines visible so panel b matches the box layout of every other supp panel

for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/figS10_feature_partition.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved figS10_feature_partition.{ext}")
print(f"  partition {n_lst}/{n_tilt}/{n_res}={total} | ablation R² {r2_full:.4f}->{r2_zero:.4f} (Δ{dr2:+.4f})")
plt.close(fig)
