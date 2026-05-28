"""
Figure 5 — Mechanistic decomposition and causal physics audit
Nature Communications

FAITHFUL TO CAPTIONS (rebuilt 2026-05-24). Counterfactual + tilt-probe + REGIME_SCALES
loaded live from results (43 counterfactual, 40 tilt_proof, F03F routing CSV).

Panels (captions.md Fig. 5):
  a  branch ablation: 3-fold Strat-GSS CV R^2 with each branch removed (LST removal collapses R^2)
  b  counterfactual: % of Δε_r routed via δ_LST per B-site family; mean 96.0%
  c  tilt causal probe: tilt features informative (R^2=0.38) yet branch ablation ΔR^2≈0
     -> tilt acts through electronic coupling (χ_A), not direct geometric suppression
  d  REGIME_SCALES (architectural prior) vs observed f_LST: A-site chemistry overrides prior
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, json, os

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.titleweight": "bold", "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.2,
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

CM, LST, TILT, OUR = "#264653", "#2A9D8F", "#E9C46A", "#2874A6"
RED, ORANGE, GRAY = "#C0392B", "#E67E22", "#B0BEC5"
RES = "/Users/anbu/Desktop/PIML/piml_ceramic/results"
F03F = "/Users/anbu/Desktop/NC figures/Manuscript figures 1/findings_csvs/F03F_counterfactual_LST_routing.csv"
OUT_DIR = "/Users/anbu/Desktop/NC figures/all_figures/main"
os.makedirs(OUT_DIR, exist_ok=True)

tp = json.load(open(f"{RES}/40_tilt_proof.json"))
cf = json.load(open(f"{RES}/43_counterfactual_permutation.json"))["counterfactual"]
mean_lst = cf["mean_lst_fraction"] * 100
routing = pd.read_csv(F03F).sort_values("lst_fraction")

fig = plt.figure(figsize=(7.2, 5.4), layout="constrained")
gs = fig.add_gridspec(2, 2)
ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
for ax in (ax_a, ax_b, ax_c, ax_d): style4(ax)

# a — branch ablation (values trace to results/53_ablation_v77 + captions)
labels = ["Full", "No CM", "No LST", "No tilt", "No res", "CM only"]
vals = [0.894, 0.725, 0.206, 0.895, 0.893, -0.875]
cols = [OUR, ORANGE, RED, GRAY, GRAY, RED]
xa = np.arange(len(labels))
ax_a.bar(xa, vals, width=0.66, color=cols, edgecolor="white")   # bars flush against axis spine
ax_a.axhline(0, color="#666", lw=0.7)
ax_a.axhline(0.894, color=OUR, lw=0.8, ls=":", zorder=2)
for xi, v in zip(xa, vals):
    ax_a.text(xi, v + (0.03 if v >= 0 else -0.05), f"{v:.3f}", ha="center",
              va="bottom" if v >= 0 else "top", fontsize=5.8, fontweight="bold")
ax_a.set_xticks(xa); ax_a.set_xticklabels(labels, fontsize=6.4, rotation=20, ha="right")
ax_a.set_ylabel(r"Strat-GSS CV $R^2$"); ax_a.set_ylim(-1.05, 1.05)
ax_a.set_title("a  Branch ablation", loc="left")
# (the "LST removal collapses the model" interpretation lives in the caption)

# b — counterfactual LST routing by B-site family
yb = np.arange(len(routing))
ax_b.barh(yb, routing["lst_fraction"].values * 100, height=0.7, color=LST,
          edgecolor="white")
ax_b.axvline(mean_lst, color=RED, lw=1.1, ls="--", zorder=4)
ax_b.text(mean_lst, len(routing) - 0.3, f" mean {mean_lst:.0f}%", color=RED,
          fontsize=6.0, ha="left", va="top", fontweight="bold")
for i, (frac, np_) in enumerate(zip(routing["lst_fraction"].values, routing["n_pairs"].values)):
    ax_b.text(frac * 100 - 0.4, i, f"{frac*100:.0f}", va="center", ha="right",
              fontsize=5.0, color="white", fontweight="bold")
ax_b.set_yticks(yb); ax_b.set_yticklabels(routing["B_site"].values, fontsize=6.0)
ax_b.set_xlabel(r"% of $\Delta\varepsilon_r$ routed via $\delta_{\mathrm{LST}}$")
ax_b.set_xlim(90, 101)
ax_b.set_title("b  B-site-independent LST routing", loc="left")

# c — tilt causal probe
pa = tp["proof_A_tilt_feature_informativeness"]
bars = [("tilt feats\nalone", pa["r2_tilt_linear"], TILT),
        ("LST feats\nalone", pa["r2_lst_linear"], LST),
        ("all feats", pa["r2_all_linear"], OUR)]
xc = np.arange(len(bars))
ax_c.bar(xc, [b[1] for b in bars], width=0.6, color=[b[2] for b in bars],
         edgecolor="white")
for xi, b in zip(xc, bars):
    ax_c.text(xi, b[1] + 0.02, f"{b[1]:.2f}", ha="center", va="bottom",
              fontsize=6.2, fontweight="bold")
ax_c.set_xticks(xc); ax_c.set_xticklabels([b[0] for b in bars], fontsize=6.4)
ax_c.set_ylabel(r"linear $R^2$ for $(\varepsilon_r-\varepsilon_r^{\mathrm{CM}})$")
ax_c.set_ylim(0, 1.0)
ax_c.set_title("c  Tilt acts indirectly", loc="left")
# (the "tilt routed via electronic coupling" interpretation lives in the caption)

# d — REGIME_SCALES (prior) vs observed f_LST
pb = tp["proof_B_reaney_reconciliation"]
regs = ["Ia", "Ib", "II", "III"]
arch = [pb[r]["arch_scale"] for r in regs]
obs = [pb[r]["mean_f_lst"] for r in regs]
xd = np.arange(len(regs)); w = 0.38
ax_d.bar(xd - w/2, arch, w, color=GRAY, edgecolor="white", label="architectural prior")
ax_d.bar(xd + w/2, obs, w, color=LST, edgecolor="white", label="observed $f_{\\mathrm{LST}}$")
for xi, a, o in zip(xd, arch, obs):
    ax_d.text(xi - w/2, a + 0.03, f"{a:.1f}", ha="center", va="bottom", fontsize=5.4, color="#555")
    ax_d.text(xi + w/2, o + 0.03, f"{o:.2f}", ha="center", va="bottom", fontsize=5.4,
              color=LST, fontweight="bold")
ax_d.set_xticks(xd); ax_d.set_xticklabels(regs)
ax_d.set_xlabel("Reaney regime"); ax_d.set_ylabel("scale / $f_{\\mathrm{LST}}$")
ax_d.set_ylim(0, 1.85)
ax_d.set_title("d  A-site chemistry overrides tilt prior", loc="left")
ax_d.legend(loc="upper left", fontsize=5.8, handlelength=1.1)
# (the "Regime III: damped prior yet highest observed" interpretation lives in the caption)

for ext in ["pdf", "png", "svg"]:
    fig.savefig(os.path.join(OUT_DIR, f"fig5_causality.{ext}"), dpi=600)
    print(f"Saved fig5_causality.{ext}")
plt.close(fig)
print(f"Figure 5 rebuilt. counterfactual mean={mean_lst:.1f}%")
print("  REGIME observed f_LST:", {r: round(pb[r]['mean_f_lst'],3) for r in regs})
