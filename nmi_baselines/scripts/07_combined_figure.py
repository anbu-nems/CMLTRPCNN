"""
07_combined_figure.py — Combined 2×2 ML-methodology-audit figure for NMI main MS.

Consolidates the strongest single panel from each of the four standalone
figures (script 06) into one publication-quality 2×2 grid. Anchors the new
Results §2.5 narrative end-to-end and saves a main-figure slot for NMI.

Panel layout:
  (a) Per-A-site f_LST hierarchy (Pb > Ca > Sr > La > Ba)  ← discovery
  (b) ML-literature baseline comparison: holdout R² (MAE labelled above bars)
  (c) Reaney-LOFO sign-guarantee (twinx: R² + violation count)
  (d) Loss-penalty tuning burden: holdout violation rate vs λ_sign

Per-regime f_LST, holdout-MAE bars, and CV-variance-vs-λ are saved separately
as supplementary figures.

Styling: NMI / Nature-family preset, 4-sided spines via style4, twinx via
style_twinx, panel letters via panel_letter, Okabe-Ito + PCNN coral highlight,
Unicode minus, constrained layout.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT     = "../.."  # repo root (run from nmi_baselines/scripts/)
RES_DIR  = os.path.join(ROOT, "nmi_baselines", "results")
FIG_DIR  = os.path.join(ROOT, "nmi_baselines", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# Style — Nature family preset + helpers (academic-plotting skill)
# ════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.size":           9,
    "axes.titlesize":      10,
    "axes.titleweight":    "bold",
    "axes.labelsize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     7.5,
    "legend.frameon":      True,
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "#AAAAAA",
    "figure.constrained_layout.use": True,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "axes.linewidth":      1.0,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           False,
    "ytick.right":         False,
    "xtick.bottom":        True,
    "ytick.left":          True,
    "xtick.major.width":   0.9,
    "ytick.major.width":   0.9,
    "xtick.major.size":    3,
    "ytick.major.size":    3,
    "axes.grid":           False,
    "lines.linewidth":     1.4,
    "lines.markersize":    5,
    "axes.unicode_minus":  True,
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset":    "dejavusans",
})

NC_WIDTH_DOUBLE = 7.205   # 183 mm


def style4(ax, lw=1.0):
    """Full 4-sided box spine — journal standard."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
    ax.tick_params(direction="in", top=False, right=False,
                   bottom=True, left=True, width=0.9, length=3)
    ax.set_axisbelow(False)


def style_twinx(ax2, lw=1.0):
    """Right-y twin axis: right spine only.
    Never call style4 on a twinx — would overdraw primary spines."""
    for sp in ["top", "left", "bottom"]:
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(lw)
    ax2.tick_params(direction="in", right=True, left=False,
                    top=False, bottom=False, width=0.9, length=3)


def panel_letter(ax, letter, fontsize=13, x=-0.16, y=1.07):
    """Standard bold top-left panel letter for multi-panel figures."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize, fontweight="bold",
            family=plt.rcParams["font.family"])


# Okabe-Ito + PCNN highlight
OUR_COLOR      = "#E76F51"   # coral — PCNN
BASELINE_COLOR = "#B0BEC5"   # cool gray — baseline models
COLOR_LST_MECH = "#0072B2"   # blue — LST/soft-mode
COLOR_TILT     = "#E69F00"   # amber — tilt

MINUS = "−"  # U+2212


# ════════════════════════════════════════════════════════════════════════════
# Load results
# ════════════════════════════════════════════════════════════════════════════
with open(os.path.join(RES_DIR, "01_loss_penalty_pinn.json")) as f:           r01 = json.load(f)
with open(os.path.join(RES_DIR, "02_loss_penalty_lambda_sweep.json")) as f:    r02 = json.load(f)
with open(os.path.join(RES_DIR, "04_ml_literature_baselines.json")) as f:      r04 = json.load(f)
with open(os.path.join(RES_DIR, "05_subclass_generalizability.json")) as f:    r05 = json.load(f)
print("[loaded] 4 result JSONs\n")


# ════════════════════════════════════════════════════════════════════════════
# Build the combined 2x2 figure
# ════════════════════════════════════════════════════════════════════════════
print("[combine] building 2x2 ML-methodology-audit figure…")

fig, axes = plt.subplots(2, 2, figsize=(NC_WIDTH_DOUBLE, 6.0),
                         layout="constrained")


# ─── PANEL (a): per-A-site f_LST hierarchy ──────────────────────────────────
ax = axes[0, 0]
style4(ax)

by_asite    = r05["experiment_B_subclass_attribution"]["by_asite"]
asite_order = ["Pb", "Ca", "Sr", "La", "Ba"]
asite_data  = {a["label"].replace("asite_", ""): a for a in by_asite}
asites      = [a for a in asite_order if a in asite_data]
flst        = [asite_data[a]["f_LST"] for a in asites]
ns          = [asite_data[a]["n"]    for a in asites]

bars = ax.bar(asites, flst, color=COLOR_LST_MECH,
              edgecolor="white", linewidth=0.6, width=0.66)
for bar, v, n in zip(bars, flst, ns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color="#222")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.045,
            f"n={n}", ha="center", va="bottom", fontsize=6.5, color="#888")
ax.set_xlabel("Dominant A-site cation")
ax.set_ylabel(r"$f_{\mathrm{LST}} = \langle\delta_{\mathrm{LST}}\rangle\,/\,\langle\epsilon_{\mathrm{CM}}\rangle$")
ax.set_ylim(0, max(flst) * 1.25)
panel_letter(ax, "a")


# ─── PANEL (b): baseline holdout R² with MAE labels above bars ──────────────
ax = axes[0, 1]
style4(ax)

models = ["GAM", "EBM", "PMN", "Soft-pen.\nPCNN", "PCNN"]
ho_r2  = [r04["GAM"]["holdout_r2"],
          r04["EBM"]["holdout_r2"],
          r04["PMN"]["holdout_r2"],
          0.946,
          r04["PCNN_reference"]["holdout_r2"]]
ho_mae = [r04["GAM"]["holdout_mae"],
          r04["EBM"]["holdout_mae"],
          r04["PMN"]["holdout_mae"],
          5.38,
          r04["PCNN_reference"]["holdout_mae"]]
colors = [BASELINE_COLOR]*4 + [OUR_COLOR]

bars = ax.bar(models, ho_r2, color=colors,
              edgecolor="white", linewidth=0.6, width=0.66)
# Two-line label per bar: R² value (large) + MAE (small)
for bar, r, m in zip(bars, ho_r2, ho_mae):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"R²={r:.3f}", ha="center", va="bottom", fontsize=7, color="#222")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.022,
            f"MAE={m:.2f}", ha="center", va="bottom", fontsize=6.5, color="#666")
ax.axhline(y=r04["PCNN_reference"]["holdout_r2"], color=OUR_COLOR,
           linestyle=":", linewidth=0.7, alpha=0.6, zorder=1)
ax.set_ylabel(r"Held-out $R^2$")
ax.set_ylim(0.83, max(ho_r2) * 1.045)
panel_letter(ax, "b")


# ─── PANEL (c): Reaney-LOFO sign-guarantee (twinx) ──────────────────────────
ax = axes[1, 0]
style4(ax)

lofo         = r05["experiment_A_reaney_lofo"]["results"]
regimes_lofo = [x["held_out_regime"] for x in lofo]
r2s          = [x["r2"]               for x in lofo]
lst_v        = [x["lst_violations"]   for x in lofo]
tilt_v       = [x["tilt_violations"]  for x in lofo]
total_v      = [a + b for a, b in zip(lst_v, tilt_v)]
n_te         = [x["n_test"]           for x in lofo]

x  = np.arange(len(regimes_lofo))
bw = 0.36

b1 = ax.bar(x - bw/2, r2s, bw,
            color=BASELINE_COLOR, edgecolor="white", linewidth=0.6,
            label=r"Held-out $R^2$")
ax.set_xlabel("Reaney regime held out from training")
ax.set_ylabel(r"Held-out $R^2$", color="#555555")
ax.set_xticks(x)
ax.set_xticklabels([f"{r}\n(n={n})" for r, n in zip(regimes_lofo, n_te)])
ax.set_ylim(0, 1.0)
ax.axhline(y=0, color="black", linewidth=0.5)
for bar, v in zip(b1, r2s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="#444")
ax.tick_params(axis="y", labelcolor="#555555")

ax2 = ax.twinx()
style_twinx(ax2)
b2 = ax2.bar(x + bw/2, total_v, bw,
             color=OUR_COLOR, edgecolor="white", linewidth=0.6,
             label="Sign violations")
ax2.set_ylabel("Sign violations (count)", color=OUR_COLOR)
ax2.set_ylim(0, max(max(total_v) + 1, 5))
for bar, v in zip(b2, total_v):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f"{v}", ha="center", va="bottom",
             fontsize=8.5, color=OUR_COLOR, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=OUR_COLOR)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="best", framealpha=0.92, fontsize=7.5)
panel_letter(ax, "c")


# ─── PANEL (d): λ_sign sweep, violation rate (LST + tilt) ───────────────────
ax = axes[1, 1]
style4(ax)

sweep        = r02["results"]
lams         = [r["lambda_sign"]                  for r in sweep]
ho_lst_pct   = [100 * r["ho_lst_violation_rate"]  for r in sweep]
ho_tilt_pct  = [100 * r["ho_tilt_violation_rate"] for r in sweep]

ax.plot(lams, ho_lst_pct,  "o-", color=COLOR_LST_MECH, linewidth=1.6, markersize=6,
        markerfacecolor="white", markeredgewidth=1.4,
        label=r"$\delta_{\mathrm{LST}} < 0$")
ax.plot(lams, ho_tilt_pct, "s-", color=COLOR_TILT, linewidth=1.6, markersize=6,
        markerfacecolor="white", markeredgewidth=1.4,
        label=r"$\delta_{\mathrm{tilt}} > 0$")
ax.axhline(y=0, color=OUR_COLOR, linestyle=":", linewidth=1.2,
           label="PCNN (architecture)")
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda_{\mathrm{sign}}$ (sign-penalty weight)")
ax.set_ylabel("Holdout violation rate (%)")
ax.set_ylim(-1, max(ho_lst_pct) * 1.18)
ax.legend(loc="best", framealpha=0.92, fontsize=7.5)
panel_letter(ax, "d")


# ─── Save ───────────────────────────────────────────────────────────────────
pdf_path = os.path.join(FIG_DIR, "fig_NMI_combined_methodology.pdf")
png_path = os.path.join(FIG_DIR, "fig_NMI_combined_methodology.png")
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=300)
plt.close(fig)

print(f"   → fig_NMI_combined_methodology.pdf / .png")
print(f"     Panel (a): f_LST hierarchy Pb={flst[0]:.3f} > ... > Ba={flst[-1]:.3f}")
print(f"     Panel (b): PCNN holdout R²={ho_r2[-1]:.3f}, MAE={ho_mae[-1]:.2f}; vs GAM/EBM/PMN/Soft-pen.")
print(f"     Panel (c): R-LOFO R²={[round(v,2) for v in r2s]}, total violations={total_v}")
print(f"     Panel (d): LST viol% across λ: {[round(v,1) for v in ho_lst_pct]}")
print(f"\n[done] combined figure saved → {FIG_DIR}/")
