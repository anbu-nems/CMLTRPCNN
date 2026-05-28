"""
Figure 3 — Predictive performance, uncertainty calibration and OOD stability
Nature Communications

FAITHFUL TO CAPTIONS (rebuilt 2026-05-24). Parity + calibration computed from the
holdout predictions; baseline panels loaded live from results/57 (CV) and 58 (gap).

Panels (captions.md Fig. 3):
  a  parity on 116-sample holdout (R^2=0.941), coloured by regime
  b  conformal calibration (empirical vs nominal coverage)
  c  CV stability across seven baselines (Strat-GSS R^2 mean ± SD)
  d  train->test R^2 gap (overfitting) per baseline; PCNN 0.047, XGBoost 0.17, RF 0.12
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd, numpy as np, json, os

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
OUR, ORANGE, GRAY = "#2874A6", "#F4A261", "#B0BEC5"
PROC = "./data/processed"
RES = "./results"
OUT_DIR = "./figures_output/main"
os.makedirs(OUT_DIR, exist_ok=True)

# Holdout predictions + regime
df = pd.read_csv(f"{PROC}/test_holdout_predictions.csv")
calib_idx = json.load(open(f"{PROC}/calibration_split_idx.json"))["calib_idx"]
fm = pd.read_parquet(f"{PROC}/feature_matrix_v7.parquet").iloc[calib_idx].reset_index(drop=True)
def _reg(r):
    for k in ["Ia", "Ib", "II", "III"]:
        if r.get(f"regime_{k}", 0) == 1: return k
    return "II"
df["regime"] = fm.apply(_reg, axis=1).values
yt, yp = df["er_measured"].values, df["er_predicted"].values
n = len(yt); abss = np.abs(yt - yp)
alphas = [0.20, 0.15, 0.10, 0.05]; nominals = [1 - a for a in alphas]
emp = [float(np.mean(abss <= np.quantile(abss, min((1-a)*(1+1/n), 1.0)))) for a in alphas]

# Baselines
cv = json.load(open(f"{RES}/57_baseline_comparison.json"))["models"]
tt = json.load(open(f"{RES}/58_train_test_comparison.json"))
def short(k):
    return ("PCNN" if ("ours" in k or "PCNN" in k) else "Ridge" if "Ridge" in k
            else "RF" if "Random" in k else "XGB" if "XGB" in k else "CatB" if "Cat" in k
            else "BPNN" if "BPNN" in k else "PIRNN" if "PIRNN" in k else "CM")
order = ["Ridge Regression", "Random Forest", "XGBoost", "CatBoost", "BPNN / MLP",
         "PIRNN (phys. features only)", "PCNN / CMLTRPCNNv7.7 (ours)"]
ttkey = {"Ridge Regression": "Ridge Regression", "Random Forest": "Random Forest",
         "XGBoost": "XGBoost", "CatBoost": "CatBoost", "BPNN / MLP": "BPNN / MLP",
         "PIRNN (phys. features only)": "PIRNN", "PCNN / CMLTRPCNNv7.7 (ours)": "PCNN (CMLTRPCNNv7.7)"}
labels = [short(k) for k in order]
cvm = [cv[k]["strat_gss_r2_mean"] for k in order]
cvs = [cv[k]["strat_gss_r2_std"] for k in order]
gap = [tt[ttkey[k]]["train_r2"] - tt[ttkey[k]]["test_r2"] for k in order]
is_ours = [k.startswith("PCNN") for k in order]

fig = plt.figure(figsize=(7.2, 5.4), layout="constrained")
gs = fig.add_gridspec(2, 2)
ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
for ax in (ax_a, ax_b, ax_c, ax_d): style4(ax)

# a — parity
for rname, g in df.groupby("regime"):
    ax_a.scatter(g["er_measured"], g["er_predicted"], s=18, alpha=0.78,
                 color=REG.get(rname, GRAY), edgecolors="none", zorder=3)
ax_a.plot([0, 145], [0, 145], "k--", lw=0.9, zorder=2)
ax_a.fill_between([0, 145], [0, 130.5], [0, 159.5], color="#999", alpha=0.10, zorder=1)
ax_a.set_xlim(0, 145); ax_a.set_ylim(0, 145); ax_a.set_aspect("equal", adjustable="box")
ax_a.set_xlabel(r"Measured $\varepsilon_r$"); ax_a.set_ylabel(r"Predicted $\varepsilon_r$")
ax_a.set_title(r"a  Held-out set ($n$=116, $R^2$=0.941)", loc="left")
hh = [mpatches.Patch(color=REG[r], label=f"Regime {r}") for r in ["Ia", "Ib", "II", "III"]]
ax_a.legend(handles=hh, loc="upper left", fontsize=6.0, handlelength=1.0)
ax_a.text(0.97, 0.04, "Formula-split: $R^2$=0.946±0.008\nStrat-GSS:     $R^2$=0.893±0.044",
          transform=ax_a.transAxes, fontsize=6.0, ha="right", va="bottom",
          bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CCCCCC", alpha=0.92))

# b — conformal calibration
diag = np.linspace(0.77, 0.99, 50)
ax_b.plot(diag, diag, "k--", lw=0.9, label="ideal", zorder=1)
ax_b.plot(nominals, emp, "o-", color=OUR, ms=6, zorder=3, label="empirical")
for nm, e in zip(nominals, emp):
    ax_b.annotate(f"{e*100:.1f}%", (nm, e), textcoords="offset points", xytext=(4, -8),
                  fontsize=5.6, color=OUR)
ax_b.set_xlim(0.77, 0.99); ax_b.set_ylim(0.77, 1.0); ax_b.set_aspect("equal", adjustable="box")
ax_b.set_xlabel("nominal coverage"); ax_b.set_ylabel("empirical coverage")
ax_b.set_title("b  Conformal calibration", loc="left")
ax_b.legend(fontsize=6.2, loc="upper left", handlelength=1.4)

# c — baseline CV stability
xb = np.arange(len(order))
cols = [OUR if o else GRAY for o in is_ours]
ax_c.bar(xb, cvm, yerr=cvs, width=0.62, color=cols, edgecolor="white",
         capsize=2.5, error_kw=dict(elinewidth=0.8, ecolor="#444"))
for xi, m, s, o in zip(xb, cvm, cvs, is_ours):
    ax_c.text(xi, m + s + 0.012, f"±{s:.3f}", ha="center", va="bottom",
              fontsize=5.2, color=OUR if o else "#666",
              fontweight="bold" if o else "normal")
ax_c.axhline(1.0, color="#bbb", lw=0.7, ls=":", zorder=1)   # R^2 = 1 ceiling
ax_c.set_xticks(xb); ax_c.set_xticklabels(labels, fontsize=6.2, rotation=25, ha="right")
ax_c.set_ylabel(r"Strat-GSS CV $R^2$ (mean ± SD)"); ax_c.set_ylim(0.55, 1.08)
ax_c.set_title("c  Cross-validation stability", loc="left")

# d — train->test gap
ax_d.bar(xb, gap, width=0.62, color=cols, edgecolor="white")
for xi, gv, o in zip(xb, gap, is_ours):
    ax_d.text(xi, gv + 0.004, f"{gv:.3f}", ha="center", va="bottom",
              fontsize=5.4, color=OUR if o else "#444", fontweight="bold" if o else "normal")
ax_d.set_xticks(xb); ax_d.set_xticklabels(labels, fontsize=6.2, rotation=25, ha="right")
ax_d.set_ylabel(r"train$\rightarrow$test $R^2$ gap"); ax_d.set_ylim(0, 0.2)
ax_d.set_title("d  Overfitting gap (lower = better)", loc="left")

for ext in ["pdf", "png", "svg"]:
    fig.savefig(os.path.join(OUT_DIR, f"fig3_performance.{ext}"), dpi=600)
    print(f"Saved fig3_performance.{ext}")
plt.close(fig)
print("Figure 3 rebuilt. coverage:", [round(e*100,1) for e in emp])
print("  gaps:", {short(k): round(tt[ttkey[k]]['train_r2']-tt[ttkey[k]]['test_r2'],3) for k in order})
