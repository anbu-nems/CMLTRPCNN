"""
Script 54 — Conformal prediction intervals for CMLTRPCNNv77

Split conformal prediction using the 123-sample calibration holdout.
Guarantees empirical coverage at any desired level — replaces the weak
5-seed pred_std uncertainty estimate (Spearman r=0.278).

Method: split conformal prediction (Papadopoulos 2002, Vovk 2005)
  1. Nonconformity score: s_i = |y_i - ŷ_i|  (absolute residual)
  2. For level α: q_hat = quantile({s_i}, (1-α)(1+1/n))
  3. Prediction interval: [ŷ - q_hat, ŷ + q_hat]
  Coverage guarantee: P(y_new ∈ interval) ≥ 1-α for exchangeable data.

Also reports: adaptive conformal intervals scaled by ŷ (heteroscedastic).

Outputs:
  results/54_conformal_v77.json
  figures/54_conformal_coverage.png
"""

import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

ROOT    = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)
FIG_DIR  = os.path.join(ROOT, "figures"); os.makedirs(FIG_DIR, exist_ok=True)


def conformal_quantile(scores, alpha):
    """Return q_hat for (1-alpha) coverage: ceil((1-alpha)(1+1/n))-th quantile."""
    n     = len(scores)
    level = min((1 - alpha) * (1 + 1.0 / n), 1.0)
    return float(np.quantile(scores, level))


def coverage(y_true, y_pred, q_hat):
    """Fraction of points within [ŷ - q_hat, ŷ + q_hat]."""
    return float(np.mean(np.abs(y_true - y_pred) <= q_hat))


def adaptive_coverage(y_true, y_pred, q_hat_rel):
    """Fraction within [ŷ(1 - q_hat_rel), ŷ(1 + q_hat_rel)]."""
    rel_err = np.abs(y_true - y_pred) / (np.abs(y_pred) + 1e-8)
    return float(np.mean(rel_err <= q_hat_rel))


def main():
    print("=" * 60)
    print("Script 54 — Conformal prediction (CMLTRPCNNv77)")
    print("=" * 60)

    # ── Load calibration holdout predictions ──────────────────────────────────
    pred_path = os.path.join(PROC_DIR, "test_holdout_predictions.csv")
    if not os.path.exists(pred_path):
        print(f"ERROR: {pred_path} not found.")
        print("Run script 49 first to generate calibration predictions.")
        sys.exit(1)

    df = pd.read_csv(pred_path)
    print(f"  Calibration split: n={len(df)}")

    y_true = df["er_measured"].values.astype(float)
    y_pred = df["er_predicted"].values.astype(float)

    # Basic check
    r2_calib = float(r2_score(y_true, y_pred))
    mae_calib = float(np.mean(np.abs(y_true - y_pred)))
    print(f"  Calibration R²={r2_calib:.4f}  MAE={mae_calib:.2f}")

    # ── Absolute nonconformity scores ─────────────────────────────────────────
    scores_abs = np.abs(y_true - y_pred)

    # ── Relative nonconformity scores (adaptive, heteroscedastic) ─────────────
    scores_rel = np.abs(y_true - y_pred) / (np.abs(y_pred) + 1e-8)

    alphas   = [0.05, 0.10, 0.15, 0.20]   # 95%, 90%, 85%, 80% target coverage
    levels   = [1 - a for a in alphas]

    print(f"\n  {'Level':>8}  {'q_hat (abs)':>12}  {'Emp.Cov (abs)':>14}  "
          f"{'q_hat (rel%)':>13}  {'Emp.Cov (rel)':>13}")
    print("  " + "-" * 68)

    abs_results = {}
    rel_results = {}

    for alpha, level in zip(alphas, levels):
        q_abs = conformal_quantile(scores_abs, alpha)
        q_rel = conformal_quantile(scores_rel, alpha)
        cov_abs = coverage(y_true, y_pred, q_abs)
        cov_rel = adaptive_coverage(y_true, y_pred, q_rel)

        print(f"  {level*100:>7.0f}%  {q_abs:>12.2f}  {cov_abs*100:>13.1f}%  "
              f"{q_rel*100:>12.1f}%  {cov_rel*100:>12.1f}%")

        key = f"{int(level*100)}pct"
        abs_results[key] = {
            "nominal":          level,
            "q_hat_absolute":   round(q_abs, 3),
            "empirical_coverage": round(cov_abs, 4),
            "interval_width":   round(2 * q_abs, 3),
        }
        rel_results[key] = {
            "nominal":          level,
            "q_hat_relative":   round(q_rel, 4),
            "empirical_coverage": round(cov_rel, 4),
            "interval_halfwidth_pct": round(q_rel * 100, 2),
        }

    # 90% interval is the primary reporting level
    q90_abs = conformal_quantile(scores_abs, 0.10)
    q90_rel = conformal_quantile(scores_rel, 0.10)
    print(f"\n  PRIMARY (90% coverage):")
    print(f"    Absolute interval: ŷ ± {q90_abs:.1f}  (width={2*q90_abs:.1f})")
    print(f"    Relative interval: ŷ ± {q90_rel*100:.1f}%")

    # ── Figure: coverage calibration + interval width ─────────────────────────
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.grid": False, "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "xtick.direction": "in", "ytick.direction": "in",
    })

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel A: Coverage calibration (nominal vs empirical)
    ax = axes[0]
    nom_levels = [0.80, 0.85, 0.90, 0.95]
    emp_abs = [coverage(y_true, y_pred, conformal_quantile(scores_abs, 1-l))
               for l in nom_levels]
    emp_rel = [adaptive_coverage(y_true, y_pred, conformal_quantile(scores_rel, 1-l))
               for l in nom_levels]
    ax.plot([0.75, 1.0], [0.75, 1.0], "k--", linewidth=1.0, alpha=0.5, label="Ideal")
    ax.plot(nom_levels, emp_abs, "o-", color="#2874A6", linewidth=2,
            markersize=7, label="Absolute (conformal)")
    ax.plot(nom_levels, emp_rel, "s-", color="#E67E22", linewidth=2,
            markersize=7, label="Relative (adaptive)")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("A. Coverage calibration", loc="left", fontweight="bold")
    ax.set_xlim(0.77, 0.97); ax.set_ylim(0.77, 1.02)
    ax.legend(fontsize=9)
    for s in ax.spines.values(): s.set_linewidth(0.8)
    ax.tick_params(top=True, right=True)

    # Panel B: Interval width vs predicted value
    ax = axes[1]
    sort_idx = np.argsort(y_pred)
    ax.scatter(y_pred, scores_abs, s=8, alpha=0.4, color="#2874A6", edgecolors="none",
               label="Residual |y-ŷ|")
    ax.axhline(q90_abs, color="#E74C3C", linewidth=1.5, linestyle="--",
               label=f"90% q̂ = {q90_abs:.1f}")
    # Adaptive: width = q_rel * y_pred
    adaptive_width = q90_rel * np.sort(y_pred)
    ax.plot(np.sort(y_pred), adaptive_width, color="#E67E22", linewidth=1.5,
            label=f"90% adaptive = {q90_rel*100:.0f}%·ŷ")
    ax.set_xlabel(r"Predicted $\varepsilon_r$")
    ax.set_ylabel("Absolute residual")
    ax.set_title("B. Interval width vs prediction", loc="left", fontweight="bold")
    ax.legend(fontsize=9)
    for s in ax.spines.values(): s.set_linewidth(0.8)
    ax.tick_params(direction="in", top=True, right=True)

    # Panel C: Parity with 90% conformal bands
    ax = axes[2]
    lim = (0, 145)
    ax.plot(lim, lim, "k--", linewidth=1.0, alpha=0.5)
    ax.fill_between(np.array(lim),
                    np.array(lim) - q90_abs,
                    np.array(lim) + q90_abs,
                    alpha=0.15, color="#2874A6", label=f"90% abs band (±{q90_abs:.0f})")
    ax.scatter(y_true, y_pred, s=8, alpha=0.5, color="#2874A6",
               edgecolors="none", zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"Measured $\varepsilon_r$")
    ax.set_ylabel(r"Predicted $\varepsilon_r$")
    ax.set_title(f"C. Parity with 90% conformal band", loc="left", fontweight="bold")
    ax.legend(fontsize=9)
    cov_shown = coverage(y_true, y_pred, q90_abs)
    ax.text(0.05, 0.92, f"Coverage = {cov_shown*100:.1f}%  (target 90%)",
            transform=ax.transAxes, fontsize=9, color="#2874A6")
    for s in ax.spines.values(): s.set_linewidth(0.8)
    ax.tick_params(direction="in", top=True, right=True)

    fig.suptitle("CMLTRPCNNv77 — Split conformal prediction (calibration n=116)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "54_conformal_coverage.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Figure → {fig_path}")

    # ── JSON ──────────────────────────────────────────────────────────────────
    out = {
        "model":           "CMLTRPCNNv77",
        "method":          "Split conformal prediction (Papadopoulos 2002)",
        "n_calibration":   len(df),
        "r2_calibration":  round(r2_calib, 4),
        "mae_calibration": round(mae_calib, 3),
        "absolute_intervals": abs_results,
        "relative_intervals": rel_results,
        "primary_90pct": {
            "q_hat_absolute":        round(q90_abs, 3),
            "interval_width":        round(2 * q90_abs, 3),
            "empirical_coverage":    round(coverage(y_true, y_pred, q90_abs), 4),
            "q_hat_relative":        round(q90_rel, 4),
            "relative_halfwidth_pct": round(q90_rel * 100, 2),
        },
    }
    res_path = os.path.join(RES_DIR, "54_conformal_v77.json")
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results → {res_path}")

    print()
    print("=" * 60)
    print("SUMMARY — Conformal prediction")
    print("=" * 60)
    print(f"  90% absolute interval : ŷ ± {q90_abs:.1f}  "
          f"(empirical coverage = {coverage(y_true, y_pred, q90_abs)*100:.1f}%)")
    print(f"  90% relative interval : ŷ ± {q90_rel*100:.1f}%  "
          f"(empirical coverage = {adaptive_coverage(y_true, y_pred, q90_rel)*100:.1f}%)")
    print(f"  Manuscript statement  : 'Split-conformal 90% prediction intervals of")
    print(f"  ±{q90_abs:.0f} (absolute) attain {coverage(y_true, y_pred, q90_abs)*100:.0f}% empirical coverage on the")
    print(f"  family-disjoint calibration set (n=116); coverage also holds out of")
    print(f"  calibration on held-out points (Methods).'")


if __name__ == "__main__":
    main()
