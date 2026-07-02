"""
11_combined_figure_v2.py — Updated combined 2×3 NMI methodology + architectural-
variants figure. Replaces script 07's 2×2 with a richer 6-panel layout that
incorporates the A/B/C architectural-variant findings (scripts 08-10).

Layout:
  (a) Per-A-site f_LST hierarchy             [kept from v1 — discovery]
  (b) ML-literature baseline comparison      [kept from v1 — empirical advantage]
  (c) Reaney-LOFO sign-guarantee twinx       [kept from v1 — guarantee under failure]
  (d) PCNN architectural variants — holdout accuracy (R² + MAE labels)   [NEW]
  (e) PCNN architectural variants — uncertainty quality (PI width vs coverage) [NEW]
  (f) λ_sign sensitivity sweep               [kept from v1 — tuning burden]

Variants compared in (d) and (e):
  PCNN canonical            — sign-bounded architecture, post-hoc conformal
  PCNN + monotonic LST (A)  — Softplus + positive-weight monotonicity over soft-mode features
  PCNN + quantile heads (B) — intrinsic q5/q50/q95 trained via pinball loss
  PCNN + Laplace last (C)   — last-layer Bayesian posterior via Daxberger 2021

Styling: NMI / Nature-family preset (Arial sans-serif, NC double width, style4 +
style_twinx, panel_letter, Okabe-Ito + PCNN coral, Unicode minus, constrained layout).
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
# Style (academic-plotting skill: NMI preset)
# ════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset":    "dejavusans",
    "font.size":           8.5,
    "axes.titlesize":      9.5,
    "axes.titleweight":    "bold",
    "axes.labelsize":      8.5,
    "xtick.labelsize":     7.5,
    "ytick.labelsize":     7.5,
    "legend.fontsize":     7,
    "legend.frameon":      True,
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "#AAAAAA",
    "figure.constrained_layout.use": True,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "axes.linewidth":      0.9,
    "xtick.direction":     "in", "ytick.direction": "in",
    "xtick.top":           False, "ytick.right": False,
    "xtick.bottom":        True,  "ytick.left":   True,
    "xtick.major.width":   0.8, "ytick.major.width": 0.8,
    "xtick.major.size":    3,   "ytick.major.size":  3,
    "axes.grid":           False,
    "lines.linewidth":     1.4,
    "lines.markersize":    5,
    "axes.unicode_minus":  True,
})

NC_WIDTH_DOUBLE = 7.205


def style4(ax, lw=0.9, square=True):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw)
    ax.tick_params(direction="in", top=False, right=False,
                   bottom=True, left=True, width=0.8, length=3)
    ax.set_axisbelow(False)
    # Force the plot box (area inside the 4 spines) to be square — independent
    # of how much vertical/horizontal space the axis labels consume.
    if square:
        ax.set_box_aspect(1)


def style_twinx(ax2, lw=0.9):
    for sp in ["top", "left", "bottom"]:
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(lw)
    ax2.tick_params(direction="in", right=True, left=False,
                    top=False, bottom=False, width=0.8, length=3)


def panel_letter(ax, letter, fontsize=11, x=-0.20, y=1.10):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize, fontweight="bold",
            family=plt.rcParams["font.family"])


OUR_COLOR      = "#E76F51"
BASELINE_COLOR = "#B0BEC5"
COLOR_LST_MECH = "#0072B2"
COLOR_TILT     = "#E69F00"
COLOR_VARIANT_A = "#009E73"   # green — monotonic
COLOR_VARIANT_B = "#CC79A7"   # magenta — quantile
COLOR_VARIANT_C = "#0072B2"   # blue — Laplace
MINUS = "−"


# ════════════════════════════════════════════════════════════════════════════
# Load results
# ════════════════════════════════════════════════════════════════════════════
with open(os.path.join(RES_DIR, "01_loss_penalty_pinn.json")) as f:        r01 = json.load(f)
with open(os.path.join(RES_DIR, "02_loss_penalty_lambda_sweep.json")) as f: r02 = json.load(f)
with open(os.path.join(RES_DIR, "04_ml_literature_baselines.json")) as f:   r04 = json.load(f)
with open(os.path.join(RES_DIR, "05_subclass_generalizability.json")) as f: r05 = json.load(f)
with open(os.path.join(RES_DIR, "08_monotonic_lst_head.json")) as f:        r08 = json.load(f)
with open(os.path.join(RES_DIR, "09_quantile_conformal_heads.json")) as f:  r09 = json.load(f)
with open(os.path.join(RES_DIR, "10_laplace_last_layer.json")) as f:        r10 = json.load(f)
with open(os.path.join(RES_DIR, "12_calibration_curves.json")) as f:        r12 = json.load(f)
print("[loaded] 8 result JSONs\n")


# ════════════════════════════════════════════════════════════════════════════
# Build 2×3 combined figure
# ════════════════════════════════════════════════════════════════════════════
print("[combine v2] building 2×3 NMI methodology + variants figure…")

# MANUAL GridSpec (no constrained_layout) — required because Panel (c)'s twin-y
# axis confuses constrained_layout's uniform-size logic and ends up stretching
# that panel. Manual gridspec gives equal cell sizes; per-axis set_box_aspect(1)
# then enforces square plot boxes inside each cell. This is the "GridSpec
# Pattern" exception in the academic-plotting skill.
fig = plt.figure(figsize=(NC_WIDTH_DOUBLE, 5.6), constrained_layout=False)
gs  = fig.add_gridspec(2, 3, wspace=0.50, hspace=0.55,
                       left=0.07, right=0.98, top=0.94, bottom=0.10)
axes = np.empty((2, 3), dtype=object)
for i in range(2):
    for j in range(3):
        axes[i, j] = fig.add_subplot(gs[i, j])


# ─── PANEL (a): per-A-site f_LST hierarchy ──────────────────────────────────
ax = axes[0, 0]; style4(ax)
by_asite    = r05["experiment_B_subclass_attribution"]["by_asite"]
asite_order = ["Pb", "Ca", "Sr", "La", "Ba"]
asite_data  = {a["label"].replace("asite_", ""): a for a in by_asite}
asites      = [a for a in asite_order if a in asite_data]
flst        = [asite_data[a]["f_LST"] for a in asites]
ns          = [asite_data[a]["n"]    for a in asites]
bars = ax.bar(asites, flst, color=COLOR_LST_MECH,
              edgecolor="white", linewidth=0.5, width=0.65)
for bar, v, n in zip(bars, flst, ns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
            f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color="#222")
ax.set_xlabel("Dominant A-site")
ax.set_ylabel(r"$f_{\mathrm{LST}}$")
ax.set_ylim(0, max(flst) * 1.20)
panel_letter(ax, "a")


# ─── PANEL (b): ML-literature baseline comparison ──────────────────────────
ax = axes[0, 1]; style4(ax)
# Single-line labels — narrow panels can't fit multi-line. "Soft-pen." identifies
# the soft-sign-penalty PCNN variant (described in the caption).
models = ["GAM", "EBM", "PMN", "Soft-pen.", "PCNN"]
ho_r2  = [r04["GAM"]["holdout_r2"], r04["EBM"]["holdout_r2"],
          r04["PMN"]["holdout_r2"], 0.946,
          r04["PCNN_reference"]["holdout_r2"]]
ho_mae = [r04["GAM"]["holdout_mae"], r04["EBM"]["holdout_mae"],
          r04["PMN"]["holdout_mae"], 5.38, r04["PCNN_reference"]["holdout_mae"]]
colors = [BASELINE_COLOR]*4 + [OUR_COLOR]
bars = ax.bar(models, ho_r2, color=colors, edgecolor="white", linewidth=0.5, width=0.66)
for bar, r, m in zip(bars, ho_r2, ho_mae):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
            f"{r:.3f}", ha="center", va="bottom", fontsize=6.5, color="#222")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.020,
            f"MAE {m:.2f}", ha="center", va="bottom", fontsize=5.5, color="#666")
ax.axhline(y=r04["PCNN_reference"]["holdout_r2"], color=OUR_COLOR,
           linestyle=":", linewidth=0.6, alpha=0.6, zorder=1)
ax.set_ylabel(r"Held-out $R^2$")
ax.set_ylim(0.83, max(ho_r2) * 1.043)
ax.tick_params(axis="x", rotation=25)
for tick in ax.get_xticklabels():
    tick.set_horizontalalignment("right"); tick.set_fontsize(7)
panel_letter(ax, "b")


# ─── PANEL (c): Reaney-LOFO sign-guarantee (twinx) ──────────────────────────
ax = axes[0, 2]; style4(ax)
lofo         = r05["experiment_A_reaney_lofo"]["results"]
regimes_lofo = [x["held_out_regime"] for x in lofo]
r2s          = [x["r2"]               for x in lofo]
lst_v        = [x["lst_violations"]   for x in lofo]
tilt_v       = [x["tilt_violations"]  for x in lofo]
total_v      = [a + b for a, b in zip(lst_v, tilt_v)]
n_te         = [x["n_test"]           for x in lofo]
x  = np.arange(len(regimes_lofo)); bw = 0.36

b1 = ax.bar(x - bw/2, r2s, bw, color=BASELINE_COLOR,
            edgecolor="white", linewidth=0.5, label=r"$R^2$")
ax.set_xlabel("R-LOFO held-out regime")
ax.set_ylabel(r"Held-out $R^2$", color="#555555")
ax.set_xticks(x); ax.set_xticklabels(regimes_lofo)
ax.set_ylim(0, 1.0)
for bar, v in zip(b1, r2s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color="#444")
ax.tick_params(axis="y", labelcolor="#555555")

ax2 = ax.twinx(); style_twinx(ax2)
b2 = ax2.bar(x + bw/2, total_v, bw, color=OUR_COLOR,
             edgecolor="white", linewidth=0.5, label="Sign viol.")
ax2.set_ylabel("Sign violations", color=OUR_COLOR)
ax2.set_ylim(0, max(max(total_v)+1, 5))
for bar, v in zip(b2, total_v):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f"{v}", ha="center", va="bottom",
             fontsize=8, color=OUR_COLOR, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=OUR_COLOR)
panel_letter(ax, "c")


# ─── PANEL (d): PCNN architectural variants — accuracy ─────────────────────
ax = axes[1, 0]; style4(ax)
# Short single-line labels with the variant letter as parenthetical suffix.
# Caption defines: A=monotonic LST head, B=intrinsic quantile heads, C=Laplace last-layer.
variants = ["PCNN", "Mono. (A)", "Quant. (B)", "Laplace (C)"]
v_r2  = [r04["PCNN_reference"]["holdout_r2"], r08["holdout_r2"],
         r09["holdout_r2"], r10["holdout_r2_post_laplace"]]
v_mae = [r04["PCNN_reference"]["holdout_mae"], r08["holdout_mae"],
         r09["holdout_mae"], r10["holdout_mae_post_laplace"]]
v_colors = [OUR_COLOR, COLOR_VARIANT_A, COLOR_VARIANT_B, COLOR_VARIANT_C]
bars = ax.bar(variants, v_r2, color=v_colors,
              edgecolor="white", linewidth=0.5, width=0.66)
for bar, r, m in zip(bars, v_r2, v_mae):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0015,
            f"{r:.3f}", ha="center", va="bottom", fontsize=6.5, color="#222")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"MAE {m:.2f}", ha="center", va="bottom", fontsize=5.5, color="#666")
ax.axhline(y=r04["PCNN_reference"]["holdout_r2"], color=OUR_COLOR,
           linestyle=":", linewidth=0.6, alpha=0.5, zorder=1)
ax.set_ylabel(r"Held-out $R^2$")
ax.set_ylim(0.93, max(v_r2) * 1.018)
ax.tick_params(axis="x", rotation=25)
for tick in ax.get_xticklabels():
    tick.set_horizontalalignment("right"); tick.set_fontsize(7)
panel_letter(ax, "d")


# ─── PANEL (e): Forest plot — 90% PI coverage with Wilson 95% CI ────────────
# Each variant gets one horizontal row: a dot at the empirical 90% PI coverage
# value, with a horizontal line spanning the Wilson 95% binomial confidence
# interval for that coverage estimate (n=116 holdout). Vertical reference line
# at the 90% target. PI width annotated to the right of each row.
ax = axes[1, 1]; style4(ax)


def wilson_ci(p, n, z=1.96):
    """Wilson 95% binomial CI for proportion p observed in n trials."""
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return center - half, center + half


n_ho = 116
forest = [
    # (label,                         coverage%, width, color)
    ("PCNN\n(post-hoc conformal)",     0.905, 28.76,  OUR_COLOR),
    ("+ Mono. (A)\n(post-hoc conformal)", 0.900, 28.76,  COLOR_VARIANT_A),
    ("+ Quant. (B)",                    r09["holdout_90pct_coverage"],
                                        r09["holdout_pi_width_mean"],
                                        COLOR_VARIANT_B),
    ("+ Laplace (C)",                   r10["holdout_bayesian_90pct_coverage"],
                                        r10["holdout_bayesian_pi_width"],
                                        COLOR_VARIANT_C),
]

y_positions = np.arange(len(forest))[::-1]   # top-down ordering

# 90% reference line
ax.axvline(x=90, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
           zorder=1, label="Target 90%")

# Draw each row as a Wilson-CI line + central dot
for ypos, (lab, cov, wid, c) in zip(y_positions, forest):
    lo, hi = wilson_ci(cov, n_ho)
    # CI horizontal line
    ax.plot([100*lo, 100*hi], [ypos, ypos], color=c, linewidth=1.8, zorder=3)
    ax.plot([100*lo, 100*lo], [ypos-0.10, ypos+0.10], color=c, linewidth=1.4, zorder=3)
    ax.plot([100*hi, 100*hi], [ypos-0.10, ypos+0.10], color=c, linewidth=1.4, zorder=3)
    # Center dot
    ax.scatter([100*cov], [ypos], s=70, color=c, edgecolors="black",
               linewidths=0.8, zorder=4)
    # Annotation ABOVE the dot (avoids overlap with dashed target line at x=90
    # and with the right edge of the plot area).
    ax.text(100*cov, ypos + 0.28, f"{100*cov:.1f}% (w={wid:.1f})",
            va="bottom", ha="center", fontsize=6.5, color="#222")

ax.set_yticks(y_positions)
ax.set_yticklabels([lab for lab, *_ in forest], fontsize=6.5)
ax.set_xlabel("90% PI coverage (%), Wilson 95% CI")
# xlim symmetric around 50-100% range; data span ≈ 48-95%
ax.set_xlim(35, 100)
# Extra ylim headroom so the top-row annotation doesn't clip the panel
ax.set_ylim(-0.6, len(forest) - 0.1)
ax.legend(loc="lower left", framealpha=0.92, fontsize=6)
panel_letter(ax, "e")


# ─── PANEL (f): λ_sign sensitivity sweep ───────────────────────────────────
ax = axes[1, 2]; style4(ax)
sweep        = r02["results"]
lams         = [r["lambda_sign"]                  for r in sweep]
ho_lst_pct   = [100 * r["ho_lst_violation_rate"]  for r in sweep]
ho_tilt_pct  = [100 * r["ho_tilt_violation_rate"] for r in sweep]
ax.plot(lams, ho_lst_pct,  "o-", color=COLOR_LST_MECH, linewidth=1.4, markersize=5,
        markerfacecolor="white", markeredgewidth=1.2,
        label=r"$\delta_{\mathrm{LST}} < 0$")
ax.plot(lams, ho_tilt_pct, "s-", color=COLOR_TILT, linewidth=1.4, markersize=5,
        markerfacecolor="white", markeredgewidth=1.2,
        label=r"$\delta_{\mathrm{tilt}} > 0$")
ax.axhline(y=0, color=OUR_COLOR, linestyle=":", linewidth=1.0,
           label="PCNN (arch.)")
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda_{\mathrm{sign}}$")
ax.set_ylabel("Holdout violation rate (%)")
ax.set_ylim(-1, max(ho_lst_pct) * 1.18)
ax.legend(loc="best", framealpha=0.9, fontsize=6.5)
panel_letter(ax, "f")


# ─── Save (bbox_inches="tight" required since no constrained_layout) ─────
pdf = os.path.join(FIG_DIR, "fig_NMI_combined_v2.pdf")
png = os.path.join(FIG_DIR, "fig_NMI_combined_v2.png")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"   → fig_NMI_combined_v2.pdf / .png")
print(f"     (a) f_LST hierarchy: Pb={flst[0]:.3f} → Ba={flst[-1]:.3f}")
print(f"     (b) ML-literature baselines: PCNN R²={ho_r2[-1]:.3f}, MAE={ho_mae[-1]:.2f}")
print(f"     (c) R-LOFO: R²={[round(v,2) for v in r2s]}, total violations={total_v}")
print(f"     (d) Variants R²: PCNN={v_r2[0]:.3f}, Mono={v_r2[1]:.3f}, Quant={v_r2[2]:.3f}, Laplace={v_r2[3]:.3f}")
print(f"     (e) UQ coverage/width: see scatter")
print(f"     (f) λ sweep: violation% {[round(v,1) for v in ho_lst_pct]}")
print(f"\n[done] saved → {FIG_DIR}/")
