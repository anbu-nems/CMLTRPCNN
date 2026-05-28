"""
Figure 6 — Extrapolation boundaries, uncertainty-aware screening, and inverse design
Nature Communications

FAITHFUL TO CAPTIONS (rebuilt 2026-05-24). GA inverse design folded in as panel d
(per author decision). Regime-LOFO + dual-AD detail live in Supplementary.
All numbers loaded live from canonical results JSONs — no stale hardcoding.

Panels:
  a  LOFO A-site transfer limits (Pb -0.097 .. La 0.849)           [results/42]
  b  Confident-but-wrong: per-A-site in-AD% vs LOFO R^2            [results/49,42]
  c  Uncertainty-aware virtual screening (70/72 pass)             [results/61]
  d  Genetic-algorithm inverse design — convergence + Pb-free opt [results/67,68]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json, os

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.2, "legend.frameon": True,
    "legend.framealpha": 0.9, "legend.edgecolor": "#CCCCCC",
    "figure.dpi": 150, "savefig.dpi": 600, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": False, "lines.linewidth": 1.5, "lines.markersize": 5,
})

def style4(ax, lw=0.8):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw)
    ax.tick_params(direction="in", top=False, right=False,
                   bottom=True, left=True, width=0.8, length=3)
    ax.set_axisbelow(True)

OUR, GRAY, RED, ORANGE, GREEN = "#2874A6", "#B0BEC5", "#C0392B", "#E67E22", "#2A9D8F"
ASITE = {"Pb": "#C0392B", "Ca": "#E67E22", "La": "#D4A017", "Sr": "#7F8C8D", "Ba": "#5B8FB0"}
RES = "./results"
OUT_DIR = "./figures_output/all_figures/main"
os.makedirs(OUT_DIR, exist_ok=True)
L = lambda p: json.load(open(os.path.join(RES, p)))

# ── Data ──────────────────────────────────────────────────────────────────────
lofo = L("42_lofo_generality.json")["lofo_asite"]
ad   = L("49_ad_v77.json")["asite_ad"]
scr  = L("61_virtual_screening.json")
gate = scr["screening_summary"]["gate_cascade"]
cand = sorted(scr["top_candidates"], key=lambda c: c["lower_90"], reverse=True)[:14]
ga   = {r: L(f"67_ga_{r}.json") for r in ["pb_allowed", "pb_free", "d0_strict"]}
rech = L("68_ga_candidate_ad_recheck_v77.json")

fig = plt.figure(figsize=(7.2, 5.6), layout="constrained")
gs = fig.add_gridspec(2, 2, hspace=0.04, wspace=0.04)
ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0]); ax_d = fig.add_subplot(gs[1, 1])
for ax in (ax_a, ax_b, ax_c, ax_d): style4(ax)

# ── Panel a — LOFO A-site (sorted ascending R^2 for barh) ─────────────────────
order_a = sorted(lofo.keys(), key=lambda a: lofo[a]["r2"])
r2 = [lofo[a]["r2"] for a in order_a]
y = np.arange(len(order_a))
ax_a.barh(y, r2, height=0.6, color=[ASITE[a] for a in order_a], edgecolor="white", lw=0.5, zorder=3)
ax_a.axvline(0, color="#666", lw=0.8, ls="--", zorder=4)
for i, a in enumerate(order_a):
    xt = lofo[a]["r2"] + (0.04 if lofo[a]["r2"] >= 0 else -0.04)
    ax_a.text(xt, i, f"{lofo[a]['r2']:.2f}", va="center",
              ha="left" if lofo[a]["r2"] >= 0 else "right",
              fontsize=6.2, fontweight="bold")
    ax_a.text(1.02, i, f"n={lofo[a]['n_test']}", va="center", ha="left",
              fontsize=5.8, color="#777", transform=ax_a.get_yaxis_transform())
ax_a.set_yticks(y); ax_a.set_yticklabels(order_a)
ax_a.set_xlim(-1.2, 1.05)
ax_a.set_xlabel(r"$R^2$ (A-site family held out)")
ax_a.set_title("a  Leave-one-family-out transfer", loc="left")
ax_a.text(-1.15, order_a.index("Pb"), "Pb: lone-pair extrapolation",
          fontsize=5.4, color=RED, ha="left", va="center", style="italic")

# ── Panel b — confident-but-wrong: in-AD% (bars) vs LOFO R^2 (markers) ─────────
order_b = sorted(ad.keys(), key=lambda a: ad[a]["mean_sigma_conf"])  # most→least confident
xb = np.arange(len(order_b))
inad = [ad[a]["pct_combined_in_ad"] for a in order_b]
ax_b.bar(xb, inad, width=0.6, color=[ASITE[a] for a in order_b], edgecolor="white",
         lw=0.5, zorder=3, label="in-AD (%)")
for xi, a in zip(xb, order_b):
    ax_b.text(xi, ad[a]["pct_combined_in_ad"] + 1.5, f"{ad[a]['pct_combined_in_ad']:.0f}%",
              ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=ASITE[a])
ax_b.set_xticks(xb)
ax_b.set_xticklabels([f"{a}\nσ={ad[a]['mean_sigma_conf']:.2f}" for a in order_b], fontsize=6.4)
ax_b.set_ylabel("compositions in-AD (%)"); ax_b.set_ylim(0, 100)
ax_b.set_title("b  Confident-but-wrong failure mode", loc="left")
axb2 = ax_b.twinx()
axb2.plot(xb, [lofo[a]["r2"] for a in order_b], "o-", color="#222", lw=1.0,
          ms=4.5, zorder=5, label=r"LOFO $R^2$")
axb2.axhline(0, color="#999", lw=0.6, ls=":")
axb2.set_ylabel(r"LOFO $R^2$"); axb2.set_ylim(-0.3, 1.0)
axb2.tick_params(direction="in", labelsize=7)
# (the "Pb confident-but-wrong" interpretation lives in the caption, not the panel)
h1, l1 = ax_b.get_legend_handles_labels(); h2, l2 = axb2.get_legend_handles_labels()
ax_b.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=5.6)

# ── Panel c — virtual screening (top candidates, conformal LB) ─────────────────
n_pass = gate["after_g2"]; n_tot = gate["n_total"]
yc = np.arange(len(cand))[::-1]
preds = [c["pred"] for c in cand]
lo = [c["lower_90"] for c in cand]; hi = [c["upper_90"] for c in cand]
ax_c.errorbar(preds, yc, xerr=[np.array(preds) - np.array(lo), np.array(hi) - np.array(preds)],
              fmt="o", color=OUR, ecolor="#9CC3E0", elinewidth=1.4, capsize=2,
              ms=4.5, zorder=3)
ax_c.axvline(80, color=GREEN, lw=1.0, ls="--", zorder=2)
ax_c.axvline(65, color=ORANGE, lw=1.0, ls=":", zorder=2)
ax_c.text(80, len(cand) - 0.3, " ε$_r$≥80", color=GREEN, fontsize=5.8, ha="left", va="top")
ax_c.text(65, len(cand) - 0.3, "LB≥65 ", color=ORANGE, fontsize=5.8, ha="right", va="top")
ax_c.set_yticks(yc)
ax_c.set_yticklabels([c["formula"] for c in cand], fontsize=5.2)
ax_c.set_xlabel(r"predicted $\varepsilon_r$ (point + 90% conformal interval)")
ax_c.set_xlim(60, 165)
ax_c.set_title("c  Uncertainty-aware virtual screening", loc="left")
ax_c.text(0.03, 0.05, f"{n_pass}/{n_tot} pass (97%)", transform=ax_c.transAxes,
          ha="left", va="bottom", fontsize=6.2, fontweight="bold", color=OUR,
          bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CCCCCC", lw=0.5, alpha=0.95))

# ── Panel d — GA inverse design: convergence + Pb-free optimum ─────────────────
runs = [("pb_allowed", "Pb-allowed", RED), ("d0_strict", r"$d^{0}$-strict", ORANGE),
        ("pb_free", "Pb-free", OUR)]
for key, lab, col in runs:
    be = ga[key]["convergence"]["best_er"]
    ax_d.plot(np.arange(1, len(be) + 1), be, "-", color=col, lw=1.6, label=lab, zorder=3)
ax_d.axhline(140.1, color=GRAY, lw=0.9, ls="--", zorder=2, label="curated optimum (140)")
pf = ga["pb_free"]["top10"][0]
ax_d.scatter([50], [pf["predicted_er"]], s=42, color=OUR, edgecolors="white",
             lw=0.8, zorder=5)
ax_d.annotate(
    "Pb-free optimum (extrapolation)\n"
    r"Ba$_{0.05}$Sr$_{0.95}$Ti$_{0.84}$Zr$_{0.15}$O$_3$" "\n"
    f"ε$_r$≈{pf['predicted_er']:.0f} (LCB {pf['lower_90']:.0f}), $f$$_{{LST}}$=2.85",
    xy=(50, pf["predicted_er"]), xytext=(1.5, 92),
    fontsize=5.3, color="#222", ha="left", va="bottom",
    arrowprops=dict(arrowstyle="->", color=OUR, lw=0.8),
    bbox=dict(boxstyle="round,pad=0.25", fc="#F2F8FD", ec=OUR, lw=0.7, alpha=0.95))
ax_d.set_xlabel("generation"); ax_d.set_ylabel(r"best predicted $\varepsilon_r$")
ax_d.set_xlim(0, 52); ax_d.set_ylim(90, 178)
ax_d.set_title("d  Genetic-algorithm inverse design", loc="left")
ax_d.legend(loc="lower right", fontsize=5.8)

for ext in ["pdf", "png", "svg"]:
    fig.savefig(os.path.join(OUT_DIR, f"fig6_limits.{ext}"), dpi=600)
    print(f"Saved fig6_limits.{ext}")
plt.close(fig)
print("Figure 6 rebuilt (LOFO / AD / screening / GA).")
print("  LOFO:", {a: round(lofo[a]["r2"], 3) for a in lofo})
print("  in-AD%:", {a: ad[a]["pct_combined_in_ad"] for a in ad})
print("  screening pass:", f'{gate["after_g2"]}/{gate["n_total"]}')
print("  GA pb_free best:", round(ga["pb_free"]["convergence"]["best_er"][-1], 1))
