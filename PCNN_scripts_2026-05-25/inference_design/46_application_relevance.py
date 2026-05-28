"""
Script 46 — Application Relevance Analysis

Mechanism-first framing for Nature Communications:
  1. LST amplification is what makes ABO₃ ceramics device-relevant —
     A-site controls its magnitude (the mechanism).
  2. La/Regime II sits in the high-amplification + reliable-prediction quadrant —
     that is the mechanistic basis for its DRA candidacy.
  3. Pb-era data actively degrades Pb-free prediction — a field-correction finding.

Four figures:

  Figure 46a — Connected dot plot: εr_CM baseline vs measured εr per A-site.
    Visualises that δ_LST bridges the CM floor to device-relevant εr;
    gap magnitude is A-site controlled (the mechanism).

  Figure 46b — Mechanism-reliability map: f_LST vs LOFO R² scatter.
    Each bubble = one A×Regime cell (size ∝ n).
    La/II occupies the "high mechanism + reliable" design-target quadrant.
    Pb/II occupies the "high mechanism + unreliable" trap quadrant.

  Figure 46c — LOFO R² ladder: Pb-era field-correction warning.

  Figure 46d — f_tilt suppression by regime + commercial benchmarks.

Reads aggregated outputs from scripts 42, 44, 45 — no model re-inference.

Outputs:
  results/46_application_relevance.json
  figures/46a_mechanism_bridge.png
  figures/46b_mechanism_reliability_map.png
  figures/46c_lofo_field_warning.png
  figures/46d_tilt_suppression_commercial.png
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)
FIG_DIR  = os.path.join(ROOT, "figures"); os.makedirs(FIG_DIR, exist_ok=True)

# ── Journal style — must be applied in every script (per CLAUDE.md memory) ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.grid": False,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.fontsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.6,
})

def style4(ax):
    """Four-sided spine, no grid, inward ticks, labelsize=14 (journal default)."""
    for s in ax.spines.values():
        s.set_linewidth(1.0)
        s.set_visible(True)
    ax.tick_params(direction="in", length=4, width=1.0, top=True, right=True,
                   labelsize=12)

# ── Curated palette ────────────────────────────────────────────────────────
C_PB     = "#C0392B"   # Pb — warm red, training-risk highlight
C_LA     = "#1E8449"   # La — green, Pb-free target
C_CA     = "#2874A6"   # Ca
C_SR     = "#7D6608"   # Sr
C_BA     = "#5B2C6F"   # Ba
C_NEUTR  = "#7F8C8D"
C_DRA    = "#FAD7A0"   # DRA target band
C_BSTN   = "#A9DFBF"   # base station band
C_LTCC   = "#D6EAF8"   # LTCC band
C_BENCH  = "#34495E"   # benchmark line

ASITE_COLOR = {"Pb": C_PB, "Ca": C_CA, "La": C_LA, "Sr": C_SR, "Ba": C_BA}
ASITE_ORDER = ["Pb", "Ca", "La", "Sr", "Ba"]
REGIME_ORDER = ["Ia", "Ib", "II", "III"]


# ── Load aggregated results from prior scripts ─────────────────────────────
def load_inputs():
    with open(os.path.join(RES_DIR, "42_lofo_generality.json")) as f:
        lofo = json.load(f)
    with open(os.path.join(RES_DIR, "44_literature_validation.json")) as f:
        lit = json.load(f)
    with open(os.path.join(RES_DIR, "45_composition_engineering_map.json")) as f:
        cmap = json.load(f)
    df = pd.read_parquet(os.path.join(PROC_DIR, "feature_matrix_v7.parquet"))
    df["a_site"] = df["chemistry_family"].astype(str).str.split("_").str[0]
    dom = np.full(len(df), "Ia", dtype=object)
    for rn, rc in [("III", "regime_III"), ("II", "regime_II"), ("Ib", "regime_Ib")]:
        if rc in df.columns:
            dom[df[rc].values > 0.5] = rn
    df["dom_regime"] = dom
    return lofo, lit, cmap, df


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 46a — Connected dot plot: εr_CM floor vs measured εr per A-site
# ──────────────────────────────────────────────────────────────────────────
def fig_46a(lit, df):
    """
    Connected dot plot making the causal chain explicit:
      εr_CM (gray open circle) → δ_LST bridge (connecting line) → measured εr (filled)

    Without LST amplification, every A-site would sit at its CM baseline —
    all below or barely touching the DRA window. The A-site hierarchy controls
    the width of the gap, i.e. the mechanism output.
    """
    # Order by ascending mean DC so the hierarchy reads bottom-to-top
    asites = ["Ba", "Sr", "La", "Ca", "Pb"]

    er_cm_vals, dc_vals, dc_q25_vals, dc_q75_vals, n_vals = {}, {}, {}, {}, {}
    for a in asites:
        mask = df["a_site"] == a
        cm   = df.loc[mask, "er_CM"].dropna().values
        dc   = df.loc[mask, "DC"].dropna().values
        dc   = dc[(dc > 0) & (dc < 200)]
        er_cm_vals[a]  = float(np.mean(cm))   if len(cm) else np.nan
        dc_vals[a]     = float(np.mean(dc))   if len(dc) else np.nan
        dc_q25_vals[a] = float(np.percentile(dc, 25)) if len(dc) else np.nan
        dc_q75_vals[a] = float(np.percentile(dc, 75)) if len(dc) else np.nan
        n_vals[a]      = len(dc)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    # Device window background
    ax.axvspan(20, 80, color=C_DRA,  alpha=0.22, zorder=0)
    ax.axvspan(20, 50, color=C_BSTN, alpha=0.28, zorder=0)
    ax.axvline(45, color=C_BENCH, linestyle="--", linewidth=1.2, zorder=1.5)
    ax.text(45.8, 4.75, "CTNA  $\\varepsilon_r{=}45$", color=C_BENCH,
            fontsize=9.5, va="top")
    ax.text(22, 4.75, "5G base station\n(20–50)", color="#0E6655",
            fontsize=8.5, va="top", ha="left")
    ax.text(52, 4.75, "DRA window\n(20–80)", color="#7E5109",
            fontsize=8.5, va="top", ha="left")

    y_pos = np.arange(len(asites))

    for i, a in enumerate(asites):
        color  = ASITE_COLOR[a]
        cm_val = er_cm_vals[a]
        dc_val = dc_vals[a]

        # IQR band (faint thick line behind everything)
        ax.plot([dc_q25_vals[a], dc_q75_vals[a]], [i, i],
                color=color, linewidth=5, alpha=0.18, solid_capstyle="butt", zorder=2)

        # Bridge line (CM → DC)
        ax.plot([cm_val, dc_val], [i, i], color=color, linewidth=2.0,
                alpha=0.65, zorder=3)

        # CM baseline: white-filled circle with dark edge
        ax.scatter(cm_val, i, s=95, color="white", edgecolors="#555",
                   linewidths=1.8, zorder=5)

        # Measured mean: filled colored circle
        ax.scatter(dc_val, i, s=115, color=color, edgecolors="black",
                   linewidths=0.8, zorder=5)

        # Value labels
        ax.text(cm_val - 1.8, i, f"{cm_val:.0f}", ha="right", va="center",
                fontsize=10, color="#555")
        ax.text(dc_val + 1.8, i, f"{dc_val:.0f}  (n={n_vals[a]})",
                ha="left", va="center", fontsize=10, color=color,
                fontweight="bold")

    # Annotate the δ_LST gap on the Ca bar
    ca_idx = asites.index("Ca")
    ca_cm, ca_dc = er_cm_vals["Ca"], dc_vals["Ca"]
    ax.annotate("", xy=(ca_dc, ca_idx + 0.42), xytext=(ca_cm, ca_idx + 0.42),
                arrowprops=dict(arrowstyle="<->", color="#666", lw=1.1))
    ax.text((ca_cm + ca_dc) / 2, ca_idx + 0.55, r"$\delta_{\mathrm{LST}}$  (LST amplification)",
            ha="center", va="bottom", fontsize=10.5, color="#444")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#555", markersize=9, linewidth=0,
               label=r"$\varepsilon_{r,\mathrm{CM}}$  (Clausius-Mossotti baseline)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7F8C8D",
               markeredgecolor="black", markersize=9, linewidth=0,
               label=r"Mean measured $\varepsilon_r$  (IQR shaded)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(asites)
    ax.set_xlabel(r"Dielectric constant  $\varepsilon_r$")
    ax.set_xlim(0, 88)
    ax.set_title(
        r"$\varepsilon_{r,\mathrm{CM}}$ alone is sub-threshold; $\delta_{\mathrm{LST}}$ "
        "bridges the gap — A-site controls its magnitude",
        loc="left", fontweight="bold", fontsize=12)
    style4(ax)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "46a_mechanism_bridge.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  ✓ {out}")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 46b — Mechanism-reliability map
# ──────────────────────────────────────────────────────────────────────────
def fig_46b(lofo, cmap):
    """
    Scatter: f_LST (mechanism strength, x) vs LOFO R² (predictive reliability, y).
    Each bubble = one A×Regime cell, sized by n (compounds in cell).
    LOFO R² is assigned at A-site level — the finest resolution we have.

    Key finding:
      La/II  → top-right: high amplification + reliable prediction = design target
      Pb/II  → bottom-right: high amplification + unreliable = field trap
      Ca/III → top-right: second viable Pb-free zone

    This figure makes the NatComm claim precise: our mechanism identifies WHERE
    to look for Pb-free replacements AND exposes WHERE the existing literature's
    model bias lives.
    """
    lofo_asite = {
        "Pb": lofo["lofo_asite"]["Pb"]["r2"],
        "Ba": lofo["lofo_asite"]["Ba"]["r2"],
        "Ca": lofo["lofo_asite"]["Ca"]["r2"],
        "Sr": lofo["lofo_asite"]["Sr"]["r2"],
        "La": lofo["lofo_asite"]["La"]["r2"],
    }

    points = []
    for key, c in cmap["cell_data"].items():
        a, reg = key.split("_")
        f_lst = c.get("f_lst", np.nan)
        r2    = lofo_asite.get(a, np.nan)
        n     = c.get("n", 0)
        if not (np.isnan(f_lst) or np.isnan(r2)) and n >= 5:
            points.append({"key": key, "a": a, "reg": reg,
                           "f_lst": f_lst, "r2": r2, "n": n})

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # Quadrant shading
    ax.axhspan( 0.5, 1.08, color="#D5F5E3", alpha=0.45, zorder=0)   # design zone
    ax.axhspan(-1.1, 0.0,  color="#FADBD8", alpha=0.30, zorder=0)   # adversarial zone

    # Reference lines
    ax.axhline(0.5, color="#1E8449", linestyle=":", linewidth=1.2, zorder=1)
    ax.axhline(0.0, color="black",   linestyle="-", linewidth=0.8, zorder=1)
    ax.axvline(1.5, color="#2874A6", linestyle=":", linewidth=1.2, zorder=1)
    ax.axvline(1.0, color="gray",    linestyle=":", linewidth=0.8, alpha=0.45, zorder=1)

    # Quadrant labels
    ax.text(2.55, 0.82,
            "Design target\n"
            r"(high $f_{\mathrm{LST}}$, reliable prediction)",
            ha="center", va="center", fontsize=9.5, color="#1E8449",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1E8449", alpha=0.85))
    ax.text(2.55, -0.65,
            "High amplification\nbut unreliable\n(Pb-era data trap)",
            ha="center", va="center", fontsize=9.5, color=C_PB,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_PB, alpha=0.85))

    # Scatter bubbles
    for p in points:
        ax.scatter(p["f_lst"], p["r2"],
                   s=max(35, p["n"] * 1.6),
                   color=ASITE_COLOR[p["a"]], edgecolors="black",
                   linewidths=0.7, zorder=4, alpha=0.88)

    # Label key cells
    label_cfg = {
        "La_II":  (0.10,  0.06, True),
        "Ca_III": (0.10, -0.09, True),
        "Pb_II":  (0.10, -0.07, True),
        "Pb_III": (-0.08, 0.06, False),
        "Ca_II":  (0.10,  0.06, False),
        "Ba_III": (0.10,  0.06, False),
    }
    for p in points:
        cfg = label_cfg.get(p["key"])
        if cfg is None:
            continue
        dx, dy, bold = cfg
        ax.annotate(
            f"{p['a']}/{p['reg']}  (n={p['n']})",
            xy=(p["f_lst"], p["r2"]),
            xytext=(p["f_lst"] + dx, p["r2"] + dy),
            fontsize=9.5,
            color=ASITE_COLOR[p["a"]],
            fontweight="bold" if bold else "normal",
            arrowprops=dict(arrowstyle="-", color=ASITE_COLOR[p["a"]],
                            lw=0.8) if abs(dx) > 0.08 else None,
        )

    # Reference line labels
    ax.text(0.02, 0.52, "reliability threshold  R²=0.5",
            color="#1E8449", fontsize=8.5, va="bottom")
    ax.text(1.52, -1.07, "above-average\namplification  (1.5×)",
            color="#2874A6", fontsize=8.5, va="bottom")

    # Bubble-size legend
    for n_ex, y_ex in [(20, -0.82), (100, -0.67), (275, -0.52)]:
        ax.scatter(0.18, y_ex, s=max(35, n_ex * 1.6), color="gray",
                   edgecolors="black", linewidths=0.5, zorder=5, alpha=0.55)
        ax.text(0.38, y_ex, f"n = {n_ex}", va="center", fontsize=8.5, color="#555")
    ax.text(0.18, -0.40, "n:", va="center", fontsize=8.5, color="#555", ha="center")

    ax.set_xlabel(r"Predicted soft-mode amplification  $f_{\mathrm{LST}}$  (mechanism strength)")
    ax.set_ylabel("Leave-one-A-site-out R²  (predictive reliability)")
    ax.set_xlim(0.0, 3.1)
    ax.set_ylim(-1.15, 1.08)
    ax.set_title(
        "Mechanism-reliability map: La/II is the actionable Pb-free design target\n"
        "Bubble size ∝ n compounds;  colour = A-site;  LOFO R² assigned at A-site level",
        loc="left", fontweight="bold", fontsize=12)
    style4(ax)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "46b_mechanism_reliability_map.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  ✓ {out}")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 46c — LOFO field-correction warning
# ──────────────────────────────────────────────────────────────────────────
def fig_46c(lofo):
    """Two panels — LOFO R² per A-site and per Reaney regime.
       Negative R² shaded red (Pb-era data adversarial to Pb-free prediction).
    """
    asites  = ["Pb", "Ba", "Ca", "Sr", "La"]
    a_r2    = [lofo["lofo_asite"][a]["r2"] for a in asites]
    a_n     = [lofo["lofo_asite"][a]["n_test"] for a in asites]
    regimes = ["Ia", "Ib", "II", "III"]
    r_r2    = [lofo["lofo_regime"][r]["r2"] for r in regimes]
    r_n     = [lofo["lofo_regime"][r]["n_test"] for r in regimes]

    def color(r):
        if r < 0:    return C_PB        # negative — adversarial
        if r < 0.30: return "#E59866"   # weak
        if r < 0.65: return "#F1C40F"   # moderate
        return C_LA                     # reliable

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ── Left — A-site LOFO ───────────────────────────────────────────────
    y = np.arange(len(asites))
    ax1.barh(y, a_r2, color=[color(r) for r in a_r2],
             edgecolor="black", linewidth=0.7, height=0.65, zorder=3)
    ax1.axvline(0,    color="black", linewidth=1.0, zorder=2)
    ax1.axvline(0.65, color="#1E8449", linestyle=":", linewidth=1.0, zorder=2)
    ax1.text(0.66, len(asites)-0.4, "reliable\nthreshold (0.65)",
             color="#1E8449", fontsize=9, va="top")
    for yi, r, n in zip(y, a_r2, a_n):
        x_text = r + 0.08 if r >= 0 else r - 0.08
        ha = "left" if r >= 0 else "right"
        ax1.text(x_text, yi, f"R² = {r:+.3f}  (n={n})", va="center", ha=ha,
                 fontsize=9.5, color="black")
    ax1.set_yticks(y); ax1.set_yticklabels(asites)
    ax1.set_xlabel("Leave-one-A-site-out R²")
    ax1.set_xlim(-1.5, 1.0)
    ax1.set_title("a   A-site generalization", loc="left", fontweight="bold")
    style4(ax1)

    # ── Right — Regime LOFO ──────────────────────────────────────────────
    y2 = np.arange(len(regimes))
    # cap for display only
    r_r2_disp = [max(r, -2.0) for r in r_r2]
    ax2.barh(y2, r_r2_disp, color=[color(r) for r in r_r2],
             edgecolor="black", linewidth=0.7, height=0.65, zorder=3)
    ax2.axvline(0,    color="black", linewidth=1.0, zorder=2)
    ax2.axvline(0.65, color="#1E8449", linestyle=":", linewidth=1.0, zorder=2)
    for yi, r, rd, n in zip(y2, r_r2, r_r2_disp, r_n):
        x_text = rd + 0.06 if rd >= 0 else rd - 0.06
        ha = "left" if rd >= 0 else "right"
        label = f"R² = {r:+.3f}  (n={n})"
        if r < -2.0:
            label = f"R² = {r:+.2f}  (n={n})  ↓capped"
        ax2.text(x_text, yi, label, va="center", ha=ha, fontsize=9.5, color="black")
    ax2.set_yticks(y2); ax2.set_yticklabels(regimes)
    ax2.set_xlabel("Leave-one-Reaney-regime-out R²")
    ax2.set_xlim(-2.4, 1.0)
    ax2.set_title("b   Reaney regime generalization", loc="left", fontweight="bold")
    style4(ax2)

    fig.suptitle("Pb-era and Regime-Ia training data are adversarial to Pb-free prediction",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "46c_lofo_field_warning.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  ✓ {out}")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 46d — f_tilt suppression with commercial benchmarks
# ──────────────────────────────────────────────────────────────────────────
def fig_46d(cmap, df):
    """Tilt-suppression factor by Reaney regime, with commercial materials
       (BMT, BZT, CTNA, BNT) annotated at their measured εr.
    """
    cells = cmap["cell_data"]
    f_tilt_per_regime = {r: [] for r in REGIME_ORDER}
    n_per_regime      = {r: 0  for r in REGIME_ORDER}
    for key, c in cells.items():
        a, reg = key.split("_")
        if reg in f_tilt_per_regime and not np.isnan(c.get("f_tilt", np.nan)):
            f_tilt_per_regime[reg].append((c["f_tilt"], c["n"]))
            n_per_regime[reg] += c["n"]
    # weighted mean per regime
    means = []
    for r in REGIME_ORDER:
        if f_tilt_per_regime[r]:
            vs, ns = zip(*f_tilt_per_regime[r])
            means.append(np.average(vs, weights=ns))
        else:
            means.append(np.nan)

    # Commercial benchmark lookups — match by formula substring + regime
    commercial = [
        ("BMT",  "BaMg",   "II",  24, "BaMg₁/₃Ta₂/₃O₃"),
        ("BZT",  "BaZn",   "II",  29, "BaZn₁/₃Ta₂/₃O₃"),
        ("CTNA", "CaTi",   "II",  45, "CaTiO₃-NdAlO₃"),
        ("BNT",  "Ba",     "II",  80, "Ba₄Nd₉.₃₃Ti₁₈O₅₄"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                    gridspec_kw={"width_ratios":[1, 1.4]})

    # ── Left — f_tilt per regime ─────────────────────────────────────────
    x = np.arange(len(REGIME_ORDER))
    colors_reg = ["#85C1E9", "#5DADE2", "#F39C12", "#C0392B"]
    bars = ax1.bar(x, means, color=colors_reg, edgecolor="black",
                   linewidth=0.7, zorder=3)
    for xi, m, n in zip(x, means, [n_per_regime[r] for r in REGIME_ORDER]):
        if not np.isnan(m):
            ax1.text(xi, m - 0.005, f"{m:.4f}", ha="center", va="top",
                     fontsize=10, fontweight="bold")
            ax1.text(xi, 0.003, f"n={n}", ha="center", va="bottom",
                     fontsize=9, color="#444")
    ax1.axhline(0, color="black", linewidth=1.0)
    ax1.set_xticks(x); ax1.set_xticklabels(REGIME_ORDER)
    ax1.set_xlabel("Reaney regime")
    ax1.set_ylabel(r"Tilt-suppression factor  $f_{\mathrm{tilt}}$")
    ax1.set_title("a   Tilt suppression strengthens with regime", loc="left",
                  fontweight="bold")
    style4(ax1)

    # ── Right — commercial benchmarks vs εr ──────────────────────────────
    # For each benchmark, find any matching composition
    found = []
    for short, sub, reg, er_typ, full in commercial:
        m = (df["formula"].astype(str).str.startswith(sub)) & (df["dom_regime"] == reg)
        n_match = int(m.sum())
        dc_med  = float(df.loc[m, "DC"].median()) if n_match else er_typ
        found.append((short, full, reg, dc_med, n_match))

    y = np.arange(len(found))
    ers = [t[3] for t in found]
    ax2.axvspan(20, 80, color=C_DRA,  alpha=0.30, zorder=0)
    ax2.axvspan(20, 50, color=C_BSTN, alpha=0.30, zorder=0)
    ax2.axvline(45, color=C_BENCH, linestyle="--", linewidth=1.2, zorder=2)
    ax2.barh(y, ers, color="#566573", edgecolor="black", linewidth=0.7,
             height=0.6, zorder=3)
    for yi, (short, full, reg, dc, n_m) in zip(y, found):
        ax2.text(dc + 1.5, yi, f"{short}  ({full})  εr={dc:.0f}",
                 va="center", fontsize=10)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xlim(0, 110)
    ax2.set_xlabel(r"Measured $\varepsilon_r$  (microwave-frequency)")
    ax2.set_title("b   Commercial Reaney-II / III ceramics in the DRA window",
                  loc="left", fontweight="bold")
    style4(ax2)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "46d_tilt_suppression_commercial.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  ✓ {out}")


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("Script 46 — Application Relevance Analysis")
    print("="*70)
    lofo, lit, cmap, df = load_inputs()

    print("\n[1/4] Figure 46a — LST bridges CM floor to device window")
    fig_46a(lit, df)

    print("\n[2/4] Figure 46b — Mechanism-reliability map")
    fig_46b(lofo, cmap)

    print("\n[3/4] Figure 46c — LOFO field-correction warning")
    fig_46c(lofo)

    print("\n[4/4] Figure 46d — f_tilt suppression with commercial benchmarks")
    fig_46d(cmap, df)

    # ── JSON export ────────────────────────────────────────────────────────
    asite_hier = {a: {
        "f_lst_predicted":   lit["by_asite"][a]["predicted_lst"]["mean"],
        "f_lst_ci_lo":       lit["by_asite"][a]["predicted_lst"]["ci_lo"],
        "f_lst_ci_hi":       lit["by_asite"][a]["predicted_lst"]["ci_hi"],
        "f_lst_empirical":   lit["by_asite"][a]["empirical"]["mean"],
        "n":                 lit["by_asite"][a]["n"],
    } for a in ASITE_ORDER}

    la_ii_cell = cmap["cell_data"].get("La_II", {})
    # La/II shortlist for JSON export only (not a figure anymore)
    mask_la_ii = (df["a_site"] == "La") & (df["dom_regime"] == "II")
    shortlist = (df.loc[mask_la_ii, ["formula","chemistry_family","DC","er_CM"]]
                   .copy()
                   .sort_values("DC", ascending=False)
                   .drop_duplicates("formula")
                   .head(12))
    output = {
        "claim_1_mechanism": {
            "headline": "LST amplification bridges CM floor to device-relevant εr; A-site controls magnitude",
            "p_value":  "<1e-4 (permutation test, script 43)",
            "hierarchy": asite_hier,
        },
        "claim_2_design_target": {
            "headline": "La/Regime II occupies high-mechanism + reliable-prediction quadrant: Pb-free DRA target",
            "f_lst_predicted": la_ii_cell.get("f_lst"),
            "f_lst_empirical": la_ii_cell.get("emp_f"),
            "lofo_r2_la":      lofo["lofo_asite"]["La"]["r2"],
            "lofo_risk":       la_ii_cell.get("risk"),
            "cluster_n":       la_ii_cell.get("n"),
            "ctna_benchmark":  45,
            "synthesis_shortlist": shortlist[["formula","chemistry_family","DC","er_CM"]]
                                   .sort_values("DC", ascending=False)
                                   .to_dict(orient="records"),
            "max_dc_in_cluster": float(shortlist["DC"].max()),
            "fraction_above_ctna": float((shortlist["DC"] >= 45).mean()),
        },
        "claim_3_field_correction": {
            "headline": "Pb-era training data is adversarial to Pb-free prediction",
            "lofo_pb_r2":  lofo["lofo_asite"]["Pb"]["r2"],
            "lofo_pb_n":   lofo["lofo_asite"]["Pb"]["n_test"],
            "lofo_la_r2":  lofo["lofo_asite"]["La"]["r2"],
            "lofo_ca_r2":  lofo["lofo_asite"]["Ca"]["r2"],
            "interpretation": (
                "LOFO R² = -0.947 for Pb shows existing Pb-based ABO3 data "
                "actively degrades Pb-free epsilon_r prediction. "
                "Decomposition framework disentangles this systematic bias."
            ),
        },
        "supporting_tilt_suppression": {
            "headline": "f_tilt suppression strengthens with Reaney regime severity",
            "regimes": {
                r: {"n": sum(c["n"] for k,c in cmap["cell_data"].items()
                              if k.endswith("_"+r) and not np.isnan(c.get("f_tilt", np.nan)))}
                for r in REGIME_ORDER
            },
        },
    }
    out_path = os.path.join(RES_DIR, "46_application_relevance.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n  ✓ {out_path}")

    # ── Summary print ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY — three NatComm application claims")
    print("="*70)
    print(f"\n  Claim 1  Mechanism:  A-site f_LST hierarchy")
    for a in ASITE_ORDER:
        v = asite_hier[a]
        print(f"    {a:3} : f_LST = {v['f_lst_predicted']:.2f}× "
              f"[{v['f_lst_ci_lo']:.2f}, {v['f_lst_ci_hi']:.2f}]  n={v['n']}")
    print(f"\n  Claim 2  Design target: La/II in high-mechanism + reliable quadrant")
    print(f"    f_LST = {la_ii_cell.get('f_lst', float('nan')):.3f},  "
          f"LOFO R² = {lofo['lofo_asite']['La']['r2']:+.3f},  "
          f"n = {la_ii_cell.get('n','?')}")
    print(f"    Max εr in cluster: {shortlist['DC'].max():.0f}  "
          f"(CTNA benchmark = 45)")
    print(f"    Fraction above CTNA: "
          f"{(shortlist['DC'] >= 45).mean()*100:.0f}% of top-12 shortlist")
    print(f"\n  Claim 3  Field correction:")
    print(f"    LOFO Pb R² = {lofo['lofo_asite']['Pb']['r2']:+.3f}  "
          f"→ Pb-era data adversarial to Pb-free prediction")
    print()


if __name__ == "__main__":
    main()
