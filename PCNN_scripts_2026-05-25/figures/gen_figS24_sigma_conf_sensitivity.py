"""
Generate Supplementary Fig. S15c — σ_conf gate β sensitivity panel.

Inputs: results/sweep_full.json (from sweep_full.py)
Output: figures/figS15c_sigma_conf_sensitivity.png (+ .pdf .svg)

Panels:
  (i)  per-A-site f_LST under each β perturbation
       (HEADLINE: shows hierarchy is invariant to ±50% β perturbation)
  (ii) per-A-site mean σ_conf under each β perturbation
       (shows that σ_conf shifts as expected, validating the gate's responsiveness)
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

IN  = "/Users/anbu/Desktop/sigma_conf_sweep/results/sweep_full.json"
OUT = "/Users/anbu/Desktop/sigma_conf_sweep/figures"
os.makedirs(OUT, exist_ok=True)

with open(IN) as f:
    data = json.load(f)

FAMILIES = ["Pb", "Ca", "La", "Sr", "Ba"]
ASITE_CLR = {"Pb": "#C0392B", "Ca": "#E67E22", "La": "#D4A017",
             "Sr": "#7F8C8D", "Ba": "#5B8FB0"}
PERT_ORDER = ["b0_low", "b0_high", "b1_low", "b1_high",
              "b2_low", "b2_high", "b3_low", "b3_high", "baseline"]
PERT_LABEL = {
    "baseline":  "Baseline\n(−2, 3, 2, 2)",
    "b0_low":    "β₀ ×0.5\n(−1, 3, 2, 2)",
    "b0_high":   "β₀ ×1.5\n(−3, 3, 2, 2)",
    "b1_low":    "β₁ ×0.5\n(−2, 1.5, 2, 2)",
    "b1_high":   "β₁ ×1.5\n(−2, 4.5, 2, 2)",
    "b2_low":    "β₂ ×0.5\n(−2, 3, 1, 2)",
    "b2_high":   "β₂ ×1.5\n(−2, 3, 3, 2)",
    "b3_low":    "β₃ ×0.5\n(−2, 3, 2, 1)",
    "b3_high":   "β₃ ×1.5\n(−2, 3, 2, 3)",
}

# Extract per-perturbation per-A-site f_LST + σ_conf
def get_mat(metric):
    """Return (n_perturbations, n_families) array of metric."""
    mat = np.zeros((len(PERT_ORDER), len(FAMILIES)))
    for i, p in enumerate(PERT_ORDER):
        for j, fam in enumerate(FAMILIES):
            d = data["runs"][p]["per_asite"][fam]
            if d.get("n", 0) > 0:
                mat[i, j] = d[metric]
            else:
                mat[i, j] = np.nan
    return mat

flst = get_mat("mean_f_lst")
sigc = get_mat("mean_sigma_conf")
r2_per = {p: data["runs"][p]["holdout_r2"] for p in PERT_ORDER}

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 6.8, "ytick.labelsize": 7,
    "legend.fontsize": 6.4, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")

# Panel 1: per-A-site f_LST under each perturbation
ax = axes[0]
x = np.arange(len(PERT_ORDER))
width = 0.16
for j, fam in enumerate(FAMILIES):
    ax.bar(x + (j - 2) * width, flst[:, j], width=width,
           color=ASITE_CLR[fam], edgecolor="white", lw=0.4,
           label=f"{fam}", zorder=3)
# Highlight baseline
baseline_idx = PERT_ORDER.index("baseline")
ax.axvspan(baseline_idx - 0.5, baseline_idx + 0.5, color="0.95", zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([PERT_LABEL[p] for p in PERT_ORDER], rotation=0)
ax.set_ylabel(r"per-A-site mean $f_{\rm LST} = \delta_{\rm LST}/\varepsilon_{\rm CM}$")
ax.set_title("a  Per-A-site amplification hierarchy is invariant to β perturbation", loc="left")
ax.legend(loc="upper right", ncol=5, frameon=True, framealpha=0.9)
ax.grid(axis="y", linestyle=":", linewidth=0.5, color="0.85", zorder=0)
ax.set_axisbelow(True)
# Add holdout R² annotation under each perturbation
for i, p in enumerate(PERT_ORDER):
    ax.text(i, -0.18, f"$R^2$ = {r2_per[p]:.2f}", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=5.5, color="0.35")

# Panel 2: per-A-site σ_conf under each perturbation (shows gate IS responsive)
ax = axes[1]
for j, fam in enumerate(FAMILIES):
    ax.bar(x + (j - 2) * width, sigc[:, j], width=width,
           color=ASITE_CLR[fam], edgecolor="white", lw=0.4,
           label=f"{fam}", zorder=3)
ax.axvspan(baseline_idx - 0.5, baseline_idx + 0.5, color="0.95", zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([PERT_LABEL[p] for p in PERT_ORDER], rotation=0)
ax.set_ylabel(r"per-A-site mean $\sigma_{\rm conf}$")
ax.set_title("b  σ_conf shifts as expected (gate is responsive)", loc="left")
ax.legend(loc="upper right", ncol=5, frameon=True, framealpha=0.9)
ax.grid(axis="y", linestyle=":", linewidth=0.5, color="0.85", zorder=0)
ax.set_axisbelow(True)
ax.set_ylim(0, max(0.7, sigc.max() * 1.15))

for ext in ["png", "pdf", "svg"]:
    fig.savefig(os.path.join(OUT, f"figS15c_sigma_conf_sensitivity.{ext}"), dpi=600)
plt.close()

# Also write a small text summary
print(f"=== σ_conf β sensitivity sweep summary ===")
print(f"  Baseline β = {data['baseline_beta']}")
print(f"  Perturbations: {len(PERT_ORDER)}, seeds per perturbation: {data['n_seeds']}, "
      f"epochs: {data['n_epochs']}")
print()
print(f"=== Per-A-site f_LST hierarchy across all perturbations ===")
print(f"  {'Perturbation':<14}  {'Pb':>6}  {'Ca':>6}  {'La':>6}  {'Sr':>6}  {'Ba':>6}   "
      f"{'Ordering preserved?':>20}")
canonical_order = ["Pb", "Ca", "La", "Sr", "Ba"]
for i, p in enumerate(PERT_ORDER):
    fvals = {fam: data["runs"][p]["per_asite"][fam].get("mean_f_lst", float("nan"))
             for fam in FAMILIES}
    # Check whether order is preserved
    pred_order = sorted(fvals, key=fvals.get, reverse=True)
    ok = "✓" if pred_order == canonical_order else f"✗ ({' > '.join(pred_order)})"
    line = f"  {p:<14}  " + "  ".join([f"{fvals[fam]:>6.2f}" for fam in canonical_order])
    line += f"   {ok:>20}"
    print(line)

print()
print(f"=== Holdout R² per perturbation ===")
for p in PERT_ORDER:
    print(f"  {p:<14}  R² = {r2_per[p]:.3f}")
