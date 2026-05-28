"""
Script 61 — Virtual Screening for High-εr ABO₃ Candidates (Pb-free)
PIDR-UGS Framework: Physics-Informed Design Rules + Uncertainty-Gated Screening

Physics-Informed Design Rules (PIDR):
  R1 — Regime III (t ≤ 0.965): f_LST=1.516 vs Ia=0.380 → enforced via family design
  R2 — d⁰ B-site (Ti/Zr/Nb): f_LST=0.989 vs d¹⁰=0.526 → all 8 families use Ti or Zr
  R3 — A-site: Ca/Sr/La + RE dopants (Nd, Sm, Ce): hierarchy Ca(1.66×) > Sr(0.91×) > La(1.13×)
  R4 — Pb-free: LOFO R²=−0.097, 6s² lone-pair OOD → fundamental physics boundary
  R5 — Perovskite stability: GII < 0.3 enforced via λ_GII=0.030 during training

Uncertainty-Gated Screening (UGS):
  G1 — Performance:  εr_pred ≥ 80
  G2 — Confidence:   LCB = lower_90 ≥ 65   (conformal 90% coverage, q̂₉₀=14.38)
  G3 — Novelty:      min Euclidean dist to training set ≥ d_threshold (80th pct)
  G4 — Uncertainty:  seed_std ≤ 2.5 × seed_std_ref_q80 (ensemble disagreement cap)

Ranking: by lower_90 (conservative conformal LCB), not pred mean.

Saves: results/61_virtual_screening.json, figures/61_virtual_screening.png
"""
import sys, os, json, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import euclidean_distances

_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_HERE)
_APP     = "/Users/anbu/Desktop/NC figures/explorer_app"
_MCODE   = "/Users/anbu/Desktop/NC figures/model_code"
sys.path.insert(0, _APP)
sys.path.insert(0, _MCODE)
sys.path.insert(0, os.path.join(_MCODE, "physics"))

import inference as _inf
from feature_engineering import normalize_formula, parse_formula

# ── Paths ─────────────────────────────────────────────────────────────────────
RES_DIR  = os.path.join(_ROOT, "results")
FIG_DIR  = os.path.join(_ROOT, "figures")
DATA_CSV = os.path.join(_ROOT, "data", "raw", "mixed_dataset_clean.csv")

ER_MIN_TARGET   = 80.0
ER_LOWER_CUTOFF = 65.0

# ── Composition vector for G3 novelty filter ─────────────────────────────────
def _comp_vector(formula, element_list):
    """Convert formula to fixed-length element fraction vector (sums to 1)."""
    fracs = {}
    try:
        norm = normalize_formula(formula)
        a_site, b_site, _, _, _ = parse_formula(norm)
        all_sites = {**a_site, **b_site}
        total = sum(all_sites.values())
        if total > 0:
            for el, n in all_sites.items():
                fracs[el] = n / total
    except Exception:
        pass
    return np.array([fracs.get(el, 0.0) for el in element_list], dtype=np.float32)


def build_training_comp_matrix(formulas):
    """Build composition matrix and element list from training formulas."""
    element_set = set()
    for f in formulas:
        try:
            norm = normalize_formula(str(f))
            a_site, b_site, _, _, _ = parse_formula(norm)
            element_set.update(a_site.keys())
            element_set.update(b_site.keys())
        except Exception:
            pass
    element_list = sorted(element_set)
    X = np.array([_comp_vector(str(f), element_list) for f in formulas],
                 dtype=np.float32)
    return X, element_list


def compute_novelty_threshold(X_train_comp):
    """80th percentile of min pairwise distances in training set."""
    pdist = euclidean_distances(X_train_comp)
    np.fill_diagonal(pdist, np.inf)
    min_dists = pdist.min(axis=1)
    return float(np.percentile(min_dists, 80)), min_dists


# ── Predict single composition via inference.py ───────────────────────────────
def predict_one(formula):
    try:
        r = _inf.predict(formula)   # no ST/CT — predicting composition alone
        if not r.get("parse_ok"):
            return None
        r["formula"] = formula
        return r
    except Exception:
        return None

# ── Build composition library (Pb-free) ──────────────────────────────────────
def build_library():
    candidates = []

    # Family 1: Sr₁₋ₓCeₓTiO₃ — near-optimal Ce fraction, unexplored gaps
    # Known: x=0.25(143), x=0.286(136), x=0.333(123), x=0.4(113)
    # Gap: x=0.20-0.28 and x=0.30 not well covered
    for x in [0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.36, 0.38, 0.42, 0.45]:
        sr = round(1.0 - x, 3)
        if sr <= 0: continue
        f = f"Sr{sr}Ce{x}TiO3"
        candidates.append((f, "Sr-Ce-Ti"))

    # Family 2: Ca₁₋ₓSmₓTiO₃ — Sm-doped CaTiO₃
    # Known: Ca0.5Li0.25Sm0.25TiO3=130, but pure Ca-Sm-Ti less explored
    for x in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
        ca = round(1.0 - x, 3)
        if ca <= 0: continue
        f = f"Ca{ca}Sm{x}TiO3"
        candidates.append((f, "Ca-Sm-Ti"))

    # Family 3: Ca-Li-Sm-Ti charge compensated (analogous to known Ca-Li-Nd-Ti)
    # Li compensates Sm³⁺: Ca(1-x) + Li(x/4) + Sm(3x/4) + Ti → approximate
    for x in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        ca = round(1.0 - x, 3)
        li = round(x / 4.0, 3)
        sm = round(3*x / 4.0, 3)
        if ca <= 0: continue
        f = f"Ca{ca}Li{li}Sm{sm}TiO3"
        candidates.append((f, "Ca-Li-Sm-Ti"))

    # Family 4: Ca-Zr-Ti with Nd additions — Zr+Nd co-substitution
    # CaZr0.2Ti0.8=141 and CaZr0.3Ti0.7=138 are known; add Nd dopant
    for zr in [0.10, 0.15, 0.20, 0.25]:
        for nd in [0.05, 0.10, 0.15]:
            ca = round(1.0 - nd, 3)
            ti = round(1.0 - zr, 3)
            if ca <= 0 or ti <= 0: continue
            f = f"Ca{ca}Nd{nd}Zr{zr}Ti{ti}O3"
            candidates.append((f, "Ca-Nd-Zr-Ti"))

    # Family 5: Ca-Li-Nd-Sm-Ti — partial Nd→Sm replacement in known high-εr system
    # Known: Ca0.6Li0.14La0.04Nd0.18TiO3=142; vary Nd/Sm ratio
    for nd in [0.10, 0.15, 0.18]:
        for sm in [0.03, 0.05, 0.08, 0.10]:
            re_tot = nd + sm
            ca = round(0.6, 2)
            li = round(0.14, 2)
            if ca + li + re_tot > 1.05: continue
            f = f"Ca{ca}Li{li}Nd{nd}Sm{sm}TiO3"
            candidates.append((f, "Ca-Li-Nd-Sm-Ti"))

    # Family 6: Sr₁₋ₓCaₓTiO₃ mixed A-site — unexplored ratios
    # Known: Sr0.8Ca0.2TiO3=133.9; explore other ratios
    for x in [0.05, 0.10, 0.15, 0.25, 0.30, 0.35, 0.40, 0.50]:
        sr = round(1.0 - x, 2)
        if sr <= 0: continue
        f = f"Sr{sr}Ca{x}TiO3"
        candidates.append((f, "Sr-Ca-Ti"))

    # Family 7: La₁₋ₓNaₓTiO₃ — unexplored fractions
    # Known: La0.5Na0.5TiO3=122; explore nearby fractions
    for x in [0.30, 0.35, 0.40, 0.45, 0.55, 0.60]:
        la = round(1.0 - x, 2)
        if la <= 0: continue
        f = f"La{la}Na{x}TiO3"
        candidates.append((f, "La-Na-Ti"))

    # Family 8: Ca-Zr-Ti with Sm additions
    for zr in [0.10, 0.20]:
        for sm in [0.05, 0.10, 0.15]:
            ca = round(1.0 - sm, 3)
            ti = round(1.0 - zr, 3)
            if ca <= 0 or ti <= 0: continue
            f = f"Ca{ca}Sm{sm}Zr{zr}Ti{ti}O3"
            candidates.append((f, "Ca-Sm-Zr-Ti"))

    return candidates

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("Script 61 — Virtual Screening for High-εr ABO₃ Candidates")
    print("="*70)

    # Load existing dataset formulas (to exclude + build G3 reference)
    df_existing = pd.read_csv(DATA_CSV)
    known = set(df_existing["formula"].apply(normalize_formula).tolist())
    print(f"\nKnown compositions to exclude: {len(known)}")

    # G3 — build training composition matrix for novelty threshold
    print("Building training composition matrix for novelty filter (G3)...")
    X_train_comp, element_list = build_training_comp_matrix(df_existing["formula"])
    d_threshold, train_min_dists = compute_novelty_threshold(X_train_comp)
    print(f"  G3 novelty threshold (80th pct of training min-dist): {d_threshold:.4f}")

    # Load model (warm up inference cache)
    print("Loading model...")
    _inf._load()

    # Build library
    library = build_library()
    print(f"Library size: {len(library)} compositions")

    # Filter out known compositions
    library = [(f, fam) for f, fam in library
               if normalize_formula(f) not in known]
    print(f"After excluding known: {len(library)} novel compositions")

    # Run predictions
    print("\nRunning predictions...")
    results = []
    failed = 0
    for i, (formula, family) in enumerate(library):
        r = predict_one(formula)
        if r is None:
            failed += 1
            continue
        r["family"] = family
        results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(library)} done...")

    print(f"  Predicted: {len(results)} | Failed to parse: {failed}")

    # ── Apply PIDR-UGS gates ──────────────────────────────────────────────────
    df = pd.DataFrame(results)

    # G3 — novelty: min distance to training set in composition space
    print("Computing novelty distances (G3)...")
    X_cand_comp = np.array([_comp_vector(r["formula"], element_list) for r in results],
                           dtype=np.float32)
    cand_min_dists = euclidean_distances(X_cand_comp, X_train_comp).min(axis=1)
    df["novelty_dist"] = cand_min_dists
    df["g3_pass"] = cand_min_dists >= d_threshold

    # G4 — uncertainty cap: seed_std ≤ 2.5 × reference q80
    seed_std_ref_q80 = float(np.percentile(df["seed_std"], 80))
    seed_std_cap = 2.5 * seed_std_ref_q80
    df["g4_pass"] = df["seed_std"] <= seed_std_cap
    print(f"  G4 seed_std ref q80: {seed_std_ref_q80:.3f} → cap: {seed_std_cap:.3f}")

    # Gate cascade counts
    n_total   = len(df)
    n_g1      = int((df["pred"] >= ER_MIN_TARGET).sum())
    n_g1g2    = int(((df["pred"] >= ER_MIN_TARGET) & (df["lower_90"] >= ER_LOWER_CUTOFF)).sum())
    n_g1g2g3  = int(((df["pred"] >= ER_MIN_TARGET) & (df["lower_90"] >= ER_LOWER_CUTOFF) & df["g3_pass"]).sum())
    n_final   = int(((df["pred"] >= ER_MIN_TARGET) & (df["lower_90"] >= ER_LOWER_CUTOFF) & df["g3_pass"] & df["g4_pass"]).sum())

    candidates = df[
        (df["pred"]      >= ER_MIN_TARGET) &
        (df["lower_90"]  >= ER_LOWER_CUTOFF) &
        df["g3_pass"] &
        df["g4_pass"]
    ].copy()
    # Rank by LCB (lower_90), then by pred for ties
    candidates = candidates.sort_values(["lower_90", "pred"], ascending=False).reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"  PIDR-UGS SCREENING RESULTS")
    print(f"{'='*70}")
    print(f"  Compositions screened : {n_total}")
    print(f"  After G1 (εr ≥ 80)    : {n_g1}")
    print(f"  After G2 (LCB ≥ 65)   : {n_g1g2}")
    print(f"  After G3 (novelty)     : {n_g1g2g3}")
    print(f"  After G4 (σ cap)       : {n_final}  ← final candidates")
    print(f"\n  Top 20 candidates (ranked by LCB = lower_90):")
    print(f"  {'Formula':<45} {'εr':>6} {'LCB':>6} {'90% CI':>16} {'δ_LST':>7} {'std':>6} {'d_nov':>6} {'Family'}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*16} {'-'*7} {'-'*6} {'-'*6} {'-'*15}")

    top20 = candidates.head(20)
    for _, row in top20.iterrows():
        ci = f"[{row['lower_90']:.0f}, {row['upper_90']:.0f}]"
        print(f"  {row['formula']:<45} {row['pred']:>6.1f} {row['lower_90']:>6.1f} {ci:>16} "
              f"{row['delta_lst']:>7.1f} {row['seed_std']:>6.2f} {row['novelty_dist']:>6.3f}  {row['family']}")

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    FAMILY_COLORS = {
        "Sr-Ce-Ti":        "#2A9D8F",
        "Ca-Sm-Ti":        "#264653",
        "Ca-Li-Sm-Ti":     "#E9C46A",
        "Ca-Nd-Zr-Ti":     "#F4A261",
        "Ca-Li-Nd-Sm-Ti":  "#E76F51",
        "Sr-Ca-Ti":        "#A8DADC",
        "La-Na-Ti":        "#8ecae6",
        "Ca-Sm-Zr-Ti":     "#6d6875",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel A — PIDR-UGS gate funnel
    ax = axes[0]
    gate_labels = [
        f"Library\n(n={n_total})",
        f"G1: εr ≥ 80\n(n={n_g1})",
        f"G2: LCB ≥ 65\n(n={n_g1g2})",
        f"G3: Novelty\n(n={n_g1g2g3})",
        f"G4: σ cap\n(n={n_final})",
    ]
    gate_counts = [n_total, n_g1, n_g1g2, n_g1g2g3, n_final]
    gate_colors = ["#B0BEC5", "#78909C", "#546E7A", "#E76F51", "#264653"]
    y_pos_funnel = np.arange(len(gate_labels))
    bars = ax.barh(y_pos_funnel, gate_counts, color=gate_colors, height=0.55, alpha=0.88)
    for bar, count in zip(bars, gate_counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(count), va="center", ha="left", fontsize=9, fontweight="bold")
    ax.set_yticks(y_pos_funnel)
    ax.set_yticklabels(gate_labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Compositions surviving gate")
    ax.set_title("A  PIDR-UGS screening cascade", fontweight="bold", loc="left")
    ax.set_xlim(0, n_total * 1.15)

    # Panel B — top candidates ranked by LCB, with 90% CI error bars
    ax2 = axes[1]
    top_plot = candidates.head(20).copy().reset_index(drop=True)
    yp = np.arange(len(top_plot))
    colors = [FAMILY_COLORS.get(f, "#888") for f in top_plot["family"]]
    # Plot lower_90 (LCB) as the bar base, showing conservative estimate
    ax2.barh(yp, top_plot["lower_90"], color=colors, height=0.6, alpha=0.85,
             label="LCB (lower_90)")
    ax2.errorbar(top_plot["pred"], yp,
                 xerr=[top_plot["pred"] - top_plot["lower_90"],
                       top_plot["upper_90"] - top_plot["pred"]],
                 fmt="o", color="#333", lw=1.2, capsize=3, markersize=4,
                 label="Pred ± 90% CI")
    ax2.axvline(80, color="#264653", lw=1.2, ls="--", label="εr=80 target")
    ax2.set_yticks(yp)
    ax2.set_yticklabels(top_plot["formula"], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("εr  (bar = LCB, marker = pred, whiskers = 90% CI)")
    ax2.set_title("B  Top candidates ranked by LCB", fontweight="bold", loc="left")

    # Family legend
    seen = set()
    patches = []
    for fam, col in FAMILY_COLORS.items():
        if fam in top_plot["family"].values and fam not in seen:
            patches.append(mpatches.Patch(color=col, label=fam))
            seen.add(fam)
    ax2.legend(handles=patches, fontsize=7, loc="lower right")

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "61_virtual_screening.png")
    fig.savefig(fig_path, dpi=300)
    print(f"\n  Figure → {fig_path}")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "screening_summary": {
            "framework": "PIDR-UGS",
            "library_size":         len(library),
            "n_predicted":          len(results),
            "gate_cascade": {
                "n_total":   n_total,
                "after_g1":  n_g1,
                "after_g2":  n_g1g2,
                "after_g3":  n_g1g2g3,
                "after_g4":  n_final,
            },
            "gate_parameters": {
                "G1_er_min":           ER_MIN_TARGET,
                "G2_lower_90_min":     ER_LOWER_CUTOFF,
                "G3_novelty_threshold": round(d_threshold, 4),
                "G3_percentile":       80,
                "G4_seed_std_ref_q80": round(seed_std_ref_q80, 4),
                "G4_seed_std_cap":     round(seed_std_cap, 4),
            },
            "ranking": "lower_90 (LCB) descending",
        },
        "top_candidates": candidates.head(20).to_dict(orient="records"),
        "all_candidates": candidates.to_dict(orient="records"),
    }
    res_path = os.path.join(RES_DIR, "61_virtual_screening.json")
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results → {res_path}")
    print("="*70)


if __name__ == "__main__":
    main()
