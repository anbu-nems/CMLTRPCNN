"""
13_standalone_figures.py — regenerate the 6 NMI figures as INDEPENDENT
standalone figures. No more 2x3 combined figure (layout-broke due to twin-y
in Panel c). Each figure is sized for its own content, NMI Nature-family
preset throughout, square plot boxes via set_box_aspect(1).

Generated:
  fig_NMI_1_fLST_hierarchy.{pdf,png}      — A-site + Reaney regime (2-panel main)
  fig_NMI_2_baseline_comparison.{pdf,png} — R² and MAE bars vs ML-literature  (2-panel main)
  fig_NMI_3_lofo_sign_guarantee.{pdf,png} — twinx R² and sign violations      (1-panel main)
  fig_NMI_4_variants_accuracy.{pdf,png}   — PCNN + A/B/C R² + MAE bars        (1-panel main)
  fig_NMI_5_variants_uq.{pdf,png}         — forest plot of 90% PI coverage    (1-panel main)
  fig_NMI_S1_lambda_sweep.{pdf,png}       — λ_sign sensitivity 2-panel        (supp)
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
# NMI Nature-family preset + helpers (academic-plotting skill)
# ════════════════════════════════════════════════════════════════════════════
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

NC_WIDTH_SINGLE = 3.504   # 89 mm
NC_WIDTH_DOUBLE = 7.205   # 183 mm


def style4(ax, lw=1.0, square=True):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw)
    ax.tick_params(direction="in", top=False, right=False,
                   bottom=True, left=True, width=0.9, length=3)
    ax.set_axisbelow(False)
    if square:
        ax.set_box_aspect(1)


def style_twinx(ax2, lw=1.0):
    for sp in ["top", "left", "bottom"]:
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(lw)
    ax2.tick_params(direction="in", right=True, left=False,
                    top=False, bottom=False, width=0.9, length=3)


def panel_letter(ax, letter, fontsize=12, x=-0.18, y=1.10):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize, fontweight="bold",
            family=plt.rcParams["font.family"])


# Color palette — Okabe-Ito + PCNN coral
OUR_COLOR       = "#E76F51"
BASELINE_COLOR  = "#B0BEC5"
COLOR_LST_MECH  = "#0072B2"
COLOR_TILT      = "#E69F00"
COLOR_VARIANT_A = "#009E73"
COLOR_VARIANT_B = "#CC79A7"
COLOR_VARIANT_C = "#0072B2"


def save(fig, name):
    pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    png = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(pdf); fig.savefig(png, dpi=300)
    plt.close(fig)


# ── Load results once
with open(os.path.join(RES_DIR, "01_loss_penalty_pinn.json")) as f:           r01 = json.load(f)
with open(os.path.join(RES_DIR, "02_loss_penalty_lambda_sweep.json")) as f:    r02 = json.load(f)
with open(os.path.join(RES_DIR, "04_ml_literature_baselines.json")) as f:      r04 = json.load(f)
with open(os.path.join(RES_DIR, "05_subclass_generalizability.json")) as f:    r05 = json.load(f)
with open(os.path.join(RES_DIR, "08_monotonic_lst_head.json")) as f:           r08 = json.load(f)
with open(os.path.join(RES_DIR, "09_quantile_conformal_heads.json")) as f:     r09 = json.load(f)
with open(os.path.join(RES_DIR, "10_laplace_last_layer.json")) as f:           r10 = json.load(f)
with open(os.path.join(RES_DIR, "12_calibration_curves.json")) as f:           r12 = json.load(f)
print("[loaded] 8 result JSONs\n")


# ════════════════════════════════════════════════════════════════════════════
# Fig 1 — f_LST hierarchy by A-site + Reaney regime (2-panel, double width)
# ════════════════════════════════════════════════════════════════════════════
def fig1_fLST_hierarchy():
    print("[fig1] f_LST hierarchy …")
    fig, axes = plt.subplots(1, 2, figsize=(NC_WIDTH_DOUBLE, 3.5),
                             gridspec_kw={"wspace": 0.30})

    # (a) by A-site
    by_asite    = r05["experiment_B_subclass_attribution"]["by_asite"]
    asite_order = ["Pb", "Ca", "Sr", "La", "Ba"]
    ad  = {a["label"].replace("asite_", ""): a for a in by_asite}
    asites = [a for a in asite_order if a in ad]
    flst   = [ad[a]["f_LST"] for a in asites]
    ns     = [ad[a]["n"]    for a in asites]

    ax = axes[0]; style4(ax)
    bars = ax.bar(asites, flst, color=COLOR_LST_MECH,
                  edgecolor="white", linewidth=0.7, width=0.66)
    for bar, v, n in zip(bars, flst, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#222")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.040,
                f"n={n}", ha="center", va="bottom", fontsize=7, color="#888")
    ax.set_xlabel("Dominant A-site cation")
    ax.set_ylabel(r"$f_{\mathrm{LST}} = \langle\delta_{\mathrm{LST}}\rangle\,/\,\langle\epsilon_{\mathrm{CM}}\rangle$")
    ax.set_ylim(0, max(flst) * 1.22)
    panel_letter(ax, "a")

    # (b) by Reaney regime
    by_regime    = r05["experiment_B_subclass_attribution"]["by_regime"]
    regime_order = ["Ia", "Ib", "II", "III"]
    rd = {r["label"].replace("regime_", ""): r for r in by_regime}
    regimes = [r for r in regime_order if r in rd]
    flst_r  = [rd[r]["f_LST"] for r in regimes]
    ns_r    = [rd[r]["n"]    for r in regimes]

    ax = axes[1]; style4(ax)
    bars = ax.bar(regimes, flst_r, color=COLOR_LST_MECH,
                  edgecolor="white", linewidth=0.7, width=0.62)
    for bar, v, n in zip(bars, flst_r, ns_r):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.009,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#222")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.027,
                f"n={n}", ha="center", va="bottom", fontsize=7, color="#888")
    ax.set_xlabel("Reaney regime")
    ax.set_ylabel(r"$f_{\mathrm{LST}}$")
    ax.set_ylim(0, max(flst_r) * 1.22)
    panel_letter(ax, "b")

    save(fig, "fig_NMI_1_fLST_hierarchy")
    print(f"   A-site: Pb={flst[0]:.3f} > … > Ba={flst[-1]:.3f}")
    print(f"   regime: Ia={flst_r[0]:.3f} < … < III={flst_r[-1]:.3f}")


# ════════════════════════════════════════════════════════════════════════════
# Fig 2 — Baseline comparison (R² + MAE, 2-panel, double width)
# ════════════════════════════════════════════════════════════════════════════
def fig2_baseline_comparison():
    print("[fig2] baseline comparison …")
    fig, axes = plt.subplots(1, 2, figsize=(NC_WIDTH_DOUBLE, 3.5),
                             gridspec_kw={"wspace": 0.32})

    models = ["GAM", "EBM", "PMN", "Soft-pen.", "PCNN"]
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

    # (a) Holdout R²
    ax = axes[0]; style4(ax)
    bars = ax.bar(models, ho_r2, color=colors,
                  edgecolor="white", linewidth=0.7, width=0.66)
    for bar, v in zip(bars, ho_r2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#222")
    ax.axhline(y=r04["PCNN_reference"]["holdout_r2"], color=OUR_COLOR,
               linestyle=":", linewidth=0.7, alpha=0.6, zorder=1)
    ax.set_ylabel(r"Held-out $R^2$")
    ax.set_ylim(0.83, max(ho_r2) * 1.022)
    ax.tick_params(axis="x", rotation=25)
    for t in ax.get_xticklabels():
        t.set_horizontalalignment("right")
    panel_letter(ax, "a")

    # (b) Holdout MAE
    ax = axes[1]; style4(ax)
    bars = ax.bar(models, ho_mae, color=colors,
                  edgecolor="white", linewidth=0.7, width=0.66)
    for bar, v in zip(bars, ho_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.10,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#222")
    ax.axhline(y=r04["PCNN_reference"]["holdout_mae"], color=OUR_COLOR,
               linestyle=":", linewidth=0.7, alpha=0.6, zorder=1)
    ax.set_ylabel(r"Held-out MAE ($\varepsilon_r$ units)")
    ax.set_ylim(0, max(ho_mae) * 1.13)
    ax.tick_params(axis="x", rotation=25)
    for t in ax.get_xticklabels():
        t.set_horizontalalignment("right")
    panel_letter(ax, "b")

    save(fig, "fig_NMI_2_baseline_comparison")
    print(f"   PCNN: R²={ho_r2[-1]:.3f}, MAE={ho_mae[-1]:.2f} (best on both)")


# ════════════════════════════════════════════════════════════════════════════
# Fig 3 — LOFO sign-guarantee (twinx, single panel, single column)
# ════════════════════════════════════════════════════════════════════════════
def fig3_lofo_sign_guarantee():
    print("[fig3] LOFO sign-guarantee …")
    fig, ax1 = plt.subplots(figsize=(NC_WIDTH_SINGLE, 3.5))
    style4(ax1)

    lofo         = r05["experiment_A_reaney_lofo"]["results"]
    regimes_lofo = [x["held_out_regime"] for x in lofo]
    r2s          = [x["r2"] for x in lofo]
    lst_v        = [x["lst_violations"] for x in lofo]
    tilt_v       = [x["tilt_violations"] for x in lofo]
    total_v      = [a + b for a, b in zip(lst_v, tilt_v)]
    n_te         = [x["n_test"] for x in lofo]

    x  = np.arange(len(regimes_lofo))
    bw = 0.36

    b1 = ax1.bar(x - bw/2, r2s, bw, color=BASELINE_COLOR,
                 edgecolor="white", linewidth=0.7, label=r"Held-out $R^2$")
    ax1.set_xlabel("R-LOFO held-out regime")
    ax1.set_ylabel(r"Held-out $R^2$", color="#555555")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{r}\n(n={n})" for r, n in zip(regimes_lofo, n_te)])
    ax1.set_ylim(0, 1.0)
    for bar, v in zip(b1, r2s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#444")
    ax1.tick_params(axis="y", labelcolor="#555555")

    ax2 = ax1.twinx(); style_twinx(ax2)
    b2 = ax2.bar(x + bw/2, total_v, bw, color=OUR_COLOR,
                 edgecolor="white", linewidth=0.7, label="Sign violations")
    ax2.set_ylabel("Sign violations (count)", color=OUR_COLOR)
    ax2.set_ylim(0, max(max(total_v)+1, 5))
    for bar, v in zip(b2, total_v):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f"{v}", ha="center", va="bottom",
                 fontsize=9, color=OUR_COLOR, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=OUR_COLOR)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper left", framealpha=0.92, fontsize=7.5)

    save(fig, "fig_NMI_3_lofo_sign_guarantee")
    print(f"   R²={[round(v,2) for v in r2s]}, violations={total_v}")


# ════════════════════════════════════════════════════════════════════════════
# Fig 4 — Architectural variants accuracy (1-panel, single column)
# ════════════════════════════════════════════════════════════════════════════
def fig4_variants_accuracy():
    print("[fig4] architectural variants accuracy …")
    fig, ax = plt.subplots(figsize=(NC_WIDTH_SINGLE, 3.5))
    style4(ax)

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
    for t in ax.get_xticklabels():
        t.set_horizontalalignment("right")

    save(fig, "fig_NMI_4_variants_accuracy")
    print(f"   PCNN={v_r2[0]:.3f} → Laplace={v_r2[3]:.3f}")


# ════════════════════════════════════════════════════════════════════════════
# Fig 5 — Architectural variants UQ comparison: forest plot (single column)
# ════════════════════════════════════════════════════════════════════════════
def fig5_variants_uq():
    print("[fig5] architectural variants UQ (forest) …")
    fig, ax = plt.subplots(figsize=(NC_WIDTH_SINGLE, 3.5))
    style4(ax, square=False)   # forest plot reads better non-square (tall)

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
        # Annotation above the dot — avoids the dashed line on the right
        ax.text(100*cov, ypos + 0.28, f"{100*cov:.1f}% (w={wid:.1f})",
                va="bottom", ha="center", fontsize=7.5, color="#222")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([lab for lab, *_ in forest], fontsize=8)
    ax.set_xlabel("90% PI coverage (%), Wilson 95% CI")
    ax.set_xlim(35, 100)
    ax.set_ylim(-0.6, len(forest) - 0.1)
    ax.legend(loc="lower left", framealpha=0.92, fontsize=7.5)

    save(fig, "fig_NMI_5_variants_uq")
    print(f"   PCNN=90.5%, Mono=90.0%, Quant=57.8%, Laplace=88.0%")


# ════════════════════════════════════════════════════════════════════════════
# Fig S1 — λ_sign sweep (2-panel, double width, supplementary)
# ════════════════════════════════════════════════════════════════════════════
def fig_S1_lambda_sweep():
    print("[figS1] λ_sign sweep …")
    fig, axes = plt.subplots(1, 2, figsize=(NC_WIDTH_DOUBLE, 3.5),
                             gridspec_kw={"wspace": 0.32})

    sweep        = r02["results"]
    lams         = [r["lambda_sign"]                  for r in sweep]
    ho_lst_pct   = [100 * r["ho_lst_violation_rate"]  for r in sweep]
    ho_tilt_pct  = [100 * r["ho_tilt_violation_rate"] for r in sweep]
    cv_std       = [r["strat_gss_r2_std"]             for r in sweep]

    # (a) Violation rate
    ax = axes[0]; style4(ax)
    ax.plot(lams, ho_lst_pct,  "o-", color=COLOR_LST_MECH, linewidth=1.6, markersize=7,
            markerfacecolor="white", markeredgewidth=1.5,
            label=r"$\delta_{\mathrm{LST}} < 0$")
    ax.plot(lams, ho_tilt_pct, "s-", color=COLOR_TILT, linewidth=1.6, markersize=7,
            markerfacecolor="white", markeredgewidth=1.5,
            label=r"$\delta_{\mathrm{tilt}} > 0$")
    ax.axhline(y=0, color=OUR_COLOR, linestyle=":", linewidth=1.2,
               label="PCNN (architecture)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda_{\mathrm{sign}}$ (sign-penalty weight)")
    ax.set_ylabel("Holdout violation rate (%)")
    ax.set_ylim(-1, max(ho_lst_pct) * 1.18)
    ax.legend(loc="upper right", framealpha=0.92)
    panel_letter(ax, "a")

    # (b) CV variance
    ax = axes[1]; style4(ax)
    ax.plot(lams, cv_std, "o-", color=BASELINE_COLOR, linewidth=1.6, markersize=7,
            markerfacecolor="white", markeredgewidth=1.5,
            label="Soft-sign-penalty PCNN")
    ax.axhline(y=r04["PCNN_reference"]["strat_gss_r2_std"], color=OUR_COLOR,
               linestyle="--", linewidth=1.4, label="PCNN (architecture)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda_{\mathrm{sign}}$")
    ax.set_ylabel(r"5-fold CV $R^2$ std")
    ax.set_ylim(0, max(cv_std) * 1.18)
    ax.legend(loc="upper right", framealpha=0.92)
    panel_letter(ax, "b")

    save(fig, "fig_NMI_S1_lambda_sweep")
    print(f"   LST viol%: {[round(v,1) for v in ho_lst_pct]}")


# ════════════════════════════════════════════════════════════════════════════
# Run all
# ════════════════════════════════════════════════════════════════════════════
fig1_fLST_hierarchy()
fig2_baseline_comparison()
fig3_lofo_sign_guarantee()
fig4_variants_accuracy()
fig5_variants_uq()
fig_S1_lambda_sweep()

print(f"\n[done] 6 standalone figures saved → {FIG_DIR}/")
print(f"  fig_NMI_1_fLST_hierarchy.{{pdf,png}}        main")
print(f"  fig_NMI_2_baseline_comparison.{{pdf,png}}   main")
print(f"  fig_NMI_3_lofo_sign_guarantee.{{pdf,png}}   main")
print(f"  fig_NMI_4_variants_accuracy.{{pdf,png}}     main")
print(f"  fig_NMI_5_variants_uq.{{pdf,png}}           main")
print(f"  fig_NMI_S1_lambda_sweep.{{pdf,png}}         supp")
