"""
Script 33 — Feature Matrix v7: Add d10_B_fraction + RE_B_fraction

Targeted fix for hard GSS folds (F2=0.579, F3=0.667).
Diagnostics identified two missing features:

  d10_B_fraction  : fraction of B-site occupied by d¹⁰ cations (Zn²⁺, Cd²⁺, Ga³⁺, Hg²⁺)
                    Ba_Zn (n=59×2 in hard folds) completely missed by current features
  RE_B_fraction   : fraction of B-site occupied by rare-earth cations
                    (La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Yb, Lu, Y)
                    Ba_La (n=59×2), Ca_La (n=53), Sr_La (n=30) partially missed
  has_d10_B       : binary flag — any d¹⁰ on B-site

All 3 go into FEATURE_RES (compositional complexity, not physics mechanism).

Outputs:
  data/processed/feature_matrix_v7.parquet   (104 features = 101 + 3)
  data/processed/feature_partition_v7.json
  results/33_feature_v7_extend.json
"""

import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd

from src.features.feature_engineering import (
    normalize_formula, parse_formula, avg_radius
)
from src.physics.polarizability import calc_alpha_total
from src.physics.clausius_mossotti import (
    calc_molar_mass, calc_pseudocubic_a, calc_density,
    calc_er_CM, propagate_CM_uncertainty
)
from src.physics.ionic_radii import A_SITE_RADII, B_SITE_RADII

ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)

# d¹⁰ B-site cations — closed d shell, no crystal field stabilisation,
# tetrahedral preference → anomalous εr behaviour
D10_ELEMENTS = {"Zn", "Cd", "Hg", "Ga", "In", "Tl", "Cu", "Ag", "Au", "Ge", "Sn", "Pb"}

# Rare-earth elements that can occupy B-site in complex perovskites
RE_ELEMENTS = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
               "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc"}


def compute_b_site_flags(formula_str):
    """Return (d10_B_fraction, RE_B_fraction, has_d10_B) from formula string."""
    try:
        norm    = normalize_formula(str(formula_str))
        _, b_site, _, _, _ = parse_formula(norm)
        if not b_site:
            return 0.0, 0.0, 0.0
        b_tot = sum(b_site.values())
        if b_tot <= 0:
            return 0.0, 0.0, 0.0

        d10_n = sum(n for el, n in b_site.items() if el in D10_ELEMENTS)
        re_n  = sum(n for el, n in b_site.items() if el in RE_ELEMENTS)

        d10_frac = float(d10_n / b_tot)
        re_frac  = float(re_n  / b_tot)
        has_d10  = float(d10_frac > 0)
        return d10_frac, re_frac, has_d10
    except Exception:
        return 0.0, 0.0, 0.0


def _patch_cm_for_row(formula_str):
    """
    Re-derive er_CM and cm_approx_flag for one formula using the updated
    VCA normalisation logic (handles re_b_site with b_tot > 1).

    Returns (er_CM, sigma_CM, cm_approx_flag) or None if formula is fine.
    Used only to patch samples previously assigned er_CM=0 with flag=0.
    """
    try:
        a, b, o, _, ptype = parse_formula(formula_str)
        if not a and not b:
            return None   # parse_ok=False, can't fix
        a_tot = sum(a.values())
        b_tot = sum(b.values())

        if ptype not in ("re_b_site",) or b_tot <= 1.0:
            return None   # not a re_b_site with large b_tot, already handled

        # Try VCA by b_tot
        o_norm_check = o / b_tot
        a_norm_check = a_tot / b_tot
        if (abs(o_norm_check - 3.0) < 0.25) and (abs(a_norm_check - 1.0) < 0.25):
            scale = b_tot
            a_n = {el: n / scale for el, n in a.items()}
            b_n = {el: n / scale for el, n in b.items()}
            o_n = o / scale
            alpha_n = calc_alpha_total(a_n, b_n, o_n)
            r_A = avg_radius(a_n, A_SITE_RADII)
            r_B = avg_radius(b_n, B_SITE_RADII)
            M_n = calc_molar_mass(a_n, b_n, o_n)
            a_pc_n = calc_pseudocubic_a(r_A, r_B)
            rho_n = calc_density(M_n, a_pc_n)
            er_vca = calc_er_CM(alpha_n, M_n, rho_n)
            sigma_vca = propagate_CM_uncertainty(alpha_n, M_n, rho_n, d_alpha=0.10)
            flag = 1.0
        else:
            er_vca = float("nan")
            sigma_vca = float("nan")
            flag = 2.0
        return er_vca, sigma_vca, flag
    except Exception:
        return None


def main():
    print("=" * 65)
    print("Script 33 — Feature Matrix v7 (add d10_B + RE_B features)")
    print("=" * 65)

    # Load existing v6 matrix
    v6_path = os.path.join(PROC_DIR, "feature_matrix_v6.parquet")
    df = pd.read_parquet(v6_path)
    print(f"\n  Loaded v6: {df.shape[0]} samples × {df.shape[1]} columns")

    # Compute 3 new features
    print("\n  Computing B-site flags from formulas...")
    results = df["formula"].apply(compute_b_site_flags)
    df["d10_B_fraction"] = [r[0] for r in results]
    df["RE_B_fraction"]  = [r[1] for r in results]
    df["has_d10_B"]      = [r[2] for r in results]

    # Patch CM for samples with er_CM=0 and cm_approx_flag=0 (re_b_site VCA fix)
    print("\n  Patching CM for re_b_site VCA cases...")
    broken_mask = (df["er_CM"] == 0) & (df["cm_approx_flag"] == 0)
    n_broken = broken_mask.sum()
    n_fixed, n_layered = 0, 0
    for idx in df.index[broken_mask]:
        patch = _patch_cm_for_row(df.at[idx, "formula"])
        if patch is None:
            continue
        er_new, sigma_new, flag_new = patch
        if not (isinstance(er_new, float) and math.isnan(er_new)):
            df.at[idx, "er_CM"] = er_new
            df.at[idx, "sigma_CM"] = sigma_new
            df.at[idx, "cm_approx_flag"] = flag_new
            df.at[idx, "has_sigma_CM"] = 1.0
            n_fixed += 1
        else:
            df.at[idx, "er_CM"] = 0.0          # keep as 0 (layered fallback)
            df.at[idx, "cm_approx_flag"] = 2.0
            df.at[idx, "has_sigma_CM"] = 0.0
            n_layered += 1
    print(f"    {n_broken} previously unresolved: "
          f"{n_fixed} recovered (VCA flag=1), {n_layered} reclassified as layered")

    # Validate on hard families
    print("\n  Validation on hard families:")
    hard = ["Ba_La", "Ba_Zn", "Ca_La", "Sr_La", "Sr_Ga", "Ca_Zr"]
    for fam in hard:
        sub = df[df["chemistry_family"] == fam]
        if len(sub) == 0:
            continue
        print(f"    {fam:25s}  n={len(sub):3d}  "
              f"d10={sub['d10_B_fraction'].mean():.2f}  "
              f"RE={sub['RE_B_fraction'].mean():.2f}  "
              f"has_d10={sub['has_d10_B'].mean():.2f}")

    print(f"\n  Global stats:")
    print(f"    d10_B_fraction > 0 : {(df['d10_B_fraction']>0).sum()} samples")
    print(f"    RE_B_fraction  > 0 : {(df['RE_B_fraction']>0).sum()} samples")
    print(f"    has_d10_B = 1      : {df['has_d10_B'].sum():.0f} samples")

    # Load v6 partition and extend residual branch
    with open(os.path.join(PROC_DIR, "feature_partition_v6.json")) as f:
        partition_v6 = json.load(f)

    new_res_features = ["d10_B_fraction", "RE_B_fraction", "has_d10_B"]
    partition_v7 = {
        "LST":      partition_v6["LST"],
        "Tilt":     partition_v6["Tilt"],
        "Residual": partition_v6["Residual"] + new_res_features,
        "counts": {
            "LST":      len(partition_v6["LST"]),
            "Tilt":     len(partition_v6["Tilt"]),
            "Residual": len(partition_v6["Residual"]) + len(new_res_features),
            "Total":    len(partition_v6["LST"]) + len(partition_v6["Tilt"]) +
                        len(partition_v6["Residual"]) + len(new_res_features),
        }
    }

    print(f"\n  Partition v7:")
    print(f"    LST:      {partition_v7['counts']['LST']} features (unchanged)")
    print(f"    Tilt:     {partition_v7['counts']['Tilt']} features (unchanged)")
    print(f"    Residual: {partition_v7['counts']['Residual']} features (+3)")
    print(f"    Total:    {partition_v7['counts']['Total']} features")

    # Save
    v7_path = os.path.join(PROC_DIR, "feature_matrix_v7.parquet")
    df.to_parquet(v7_path, index=False)
    print(f"\n  Saved → {v7_path}")

    part_path = os.path.join(PROC_DIR, "feature_partition_v7.json")
    with open(part_path, "w") as f:
        json.dump(partition_v7, f, indent=2)
    print(f"  Saved → {part_path}")

    results_out = {
        "new_features": new_res_features,
        "total_features": partition_v7["counts"]["Total"],
        "partition": partition_v7["counts"],
        "d10_samples": int((df["d10_B_fraction"] > 0).sum()),
        "RE_B_samples": int((df["RE_B_fraction"] > 0).sum()),
    }
    with open(os.path.join(RES_DIR, "33_feature_v7_extend.json"), "w") as f:
        json.dump(results_out, f, indent=2)

    print("\n  Done. Run script 34 to retrain with new features.")
    print("=" * 65)


if __name__ == "__main__":
    main()
