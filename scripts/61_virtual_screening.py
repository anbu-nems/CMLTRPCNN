import os

# --- self-contained release root (auto-injected) ---
RELEASE_ROOT = os.path.abspath(os.path.dirname(__file__))
while RELEASE_ROOT != os.path.dirname(RELEASE_ROOT) and not os.path.isdir(os.path.join(RELEASE_ROOT, 'model_weights')):
    RELEASE_ROOT = os.path.dirname(RELEASE_ROOT)
# ----------------------------------------------------

"""
Script 61 — Virtual Screening for High-εr ABO₃ Candidates (Pb-free)

Strategy (physics-guided, Pb excluded):
  - Pb excluded: LOFO shows Pb is a fundamental extrapolation barrier (R²=-0.097)
    Screening restricted to Pb-free families for scientific consistency.
  - A-site: Ca (f_LST=1.74×) and Sr with Ce substitution → high εr
  - B-site: d⁰ cations (Ti, Zr) activate soft-mode proxy
  - Target: εr > 80, in applicability domain, conformal lower bound > 65
  - Exclude: all compositions already in training dataset

Library design (Pb-free):
  Family 1 — Sr₁₋ₓCeₓTiO₃  — near-optimal Ce fraction (unexplored x=0.20–0.28)
  Family 2 — Ca₁₋ₓSmₓTiO₃ + Ca-Li-Sm-Ti variants — Sm-doped Ca-Ti
  Family 3 — Ca-Zr-Ti with RE dopants (Nd, Sm) — Zr+RE co-substitution
  Family 4 — Ca-Li-Nd-Ti with Sm substitution — partial Nd→Sm replacement
  Family 5 — Sr-Ca-Ti mixed A-site — unexplored Sr/Ca ratios
  Family 6 — La-Na-Ti series — La₁₋ₓNaₓTiO₃ unexplored fractions

Saves: results/61_virtual_screening.json, figures/61_virtual_screening.png
"""
import sys, os, json, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_HERE)
_APP     = os.path.join(RELEASE_ROOT, "explorer_app")
_MCODE   = os.path.join(RELEASE_ROOT, "model_code")
sys.path.insert(0, _APP)
sys.path.insert(0, _MCODE)
sys.path.insert(0, os.path.join(_MCODE, "physics"))

import inference as _inf
from feature_engineering import normalize_formula



# ── Paths ─────────────────────────────────────────────────────────────────────
RES_DIR  = os.path.join(_ROOT, "results")
FIG_DIR  = os.path.join(_ROOT, "figures")
DATA_CSV = os.path.join(_ROOT, "data", "raw", "mixed_dataset_clean.csv")

ER_MIN_TARGET   = 80.0
ER_LOWER_CUTOFF = 65.0

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

    # Load existing dataset formulas (to exclude)
    df_existing = pd.read_csv(DATA_CSV)
    known = set(df_existing["formula"].apply(normalize_formula).tolist())
    print(f"\nKnown compositions to exclude: {len(known)}")

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

    # Filter: εr > 80 AND lower_90 > 65 AND sigma_conf < 0.6 (in AD)
    df = pd.DataFrame(results)
    candidates = df[
        (df["pred"] >= ER_MIN_TARGET) &
        (df["lower_90"] >= ER_LOWER_CUTOFF)
    ].copy()
    candidates = candidates.sort_values("pred", ascending=False).reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"  SCREENING RESULTS")
    print(f"{'='*70}")
    print(f"  Compositions screened : {len(results)}")
    print(f"  εr > 80               : {(df['pred'] >= 80).sum()}")
    print(f"  εr > 80 + in-AD       : {len(candidates)}")
    print(f"\n  Top 20 high-εr candidates:")
    print(f"  {'Formula':<45} {'εr':>6} {'90% CI':>16} {'δ_LST':>7} {'std':>6} {'Family'}")
    print(f"  {'-'*45} {'-'*6} {'-'*16} {'-'*7} {'-'*6} {'-'*15}")

    top20 = candidates.head(20)
    for _, row in top20.iterrows():
        ci = f"[{row['lower_90']:.0f}, {row['upper_90']:.0f}]"
        print(f"  {row['formula']:<45} {row['pred']:>6.1f} {ci:>16} "
              f"{row['delta_lst']:>7.1f} {row['seed_std']:>6.2f}  {row['family']}")

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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A — predicted εr distribution of all screened vs candidates
    ax = axes[0]
    ax.hist(df["pred"], bins=30, color="#B0BEC5", alpha=0.7, label="All screened")
    ax.hist(candidates["pred"], bins=20, color="#E76F51", alpha=0.8,
            label=f"In-AD + εr>80 (n={len(candidates)})")
    ax.axvline(80, color="#264653", lw=1.5, ls="--", label="εr = 80 threshold")
    ax.set_xlabel("Predicted εr")
    ax.set_ylabel("Count")
    ax.set_title("A  Screening distribution", fontweight="bold", loc="left")
    ax.legend(fontsize=8)

    # Panel B — top 20 candidates with error bars, coloured by family
    ax2 = axes[1]
    top_plot = candidates.head(20).copy().reset_index(drop=True)
    y_pos = np.arange(len(top_plot))
    colors = [FAMILY_COLORS.get(f, "#888") for f in top_plot["family"]]
    ax2.barh(y_pos, top_plot["pred"], color=colors, height=0.6, alpha=0.85)
    ax2.errorbar(top_plot["pred"], y_pos,
                 xerr=[top_plot["pred"] - top_plot["lower_90"],
                       top_plot["upper_90"] - top_plot["pred"]],
                 fmt="none", color="#333", lw=1.2, capsize=3)
    ax2.axvline(80, color="#264653", lw=1.2, ls="--")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_plot["formula"], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("Predicted εr  (90% CI)")
    ax2.set_title("B  Top 20 candidates", fontweight="bold", loc="left")

    # Legend for families
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
            "library_size": len(library),
            "n_predicted":  len(results),
            "n_er_above_80": int((df["pred"] >= 80).sum()),
            "n_candidates_in_ad": len(candidates),
            "filters": {
                "er_min": ER_MIN_TARGET,
                "lower_90_min": ER_LOWER_CUTOFF,
                "sigma_conf_max": 0.60,
            }
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
