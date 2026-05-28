"""
Full feature engineering pipeline for PIML V6.

Feature layers:
  Layer 1 — Electronic (polarizability, CM baseline)
  Layer 2 — Structural (tolerance factor, tilting, regime, phonon)
  Layer 3 — Bonding / covalency
  Layer 4 — Process (CT, C_time, ST, ST_time) + interaction terms
"""

import re
import math
import numpy as np
import pandas as pd

from src.physics.ionic_radii import (
    A_SITE_ELEMENTS, B_SITE_ELEMENTS, A_SITE_RADII, B_SITE_RADII, O_RADIUS
)
from src.physics.polarizability import ION_POLARIZABILITY, calc_alpha_total
from src.physics.clausius_mossotti import (
    calc_molar_mass, calc_pseudocubic_a, calc_density,
    calc_er_CM, propagate_CM_uncertainty, ATOMIC_MASS
)
from src.physics.tolerance_factor import (
    avg_radius, calc_tolerance_factor, calc_octahedral_factor,
    calc_tilt_severity, calc_octahedral_volume, calc_b_size_mismatch
)
from src.physics.reaney_regime import (
    classify_regime, soft_mode_proxy_binary, soft_mode_proxy_continuous,
    d0_B_fraction, polarizable_A_fraction
)

# ── Lookup tables ─────────────────────────────────────────────────────────────
ELECTRONEGATIVITY = {
    "Ba":0.89,"Sr":0.95,"Ca":1.00,"La":1.10,"Ce":1.12,"Pr":1.13,"Nd":1.14,
    "Sm":1.17,"Eu":1.20,"Gd":1.20,"Tb":1.22,"Dy":1.22,"Ho":1.23,"Er":1.24,
    "Tm":1.25,"Yb":1.10,"Lu":1.27,"Y":1.22,"Bi":2.02,"Pb":2.33,"Na":0.93,
    "K":0.82,"Li":0.98,"Ag":1.93,"Ti":1.54,"Zr":1.33,"Nb":1.60,"Ta":1.50,
    "Sn":1.96,"W":2.36,"Mg":1.31,"Zn":1.65,"Al":1.61,"Fe":1.83,"Mn":1.55,
    "In":1.78,"Sc":1.36,"Ga":1.81,"Ge":2.01,"Si":1.90,"Co":1.88,"Ni":1.91,
    "Cu":1.90,"V":1.63,"Cr":1.66,"Mo":2.16,"Te":2.10,"Hf":1.30,"Ru":2.20,
    "Ir":2.20,"Re":1.90,"Os":2.20,"Sb":2.05,"O":3.44,
}

# Ionic charges for field-strength calculation
A_CHARGE = {"Ba":2,"Sr":2,"Ca":2,"Pb":2,"La":3,"Ce":3,"Pr":3,"Nd":3,"Sm":3,
             "Eu":3,"Gd":3,"Tb":3,"Dy":3,"Ho":3,"Er":3,"Yb":3,"Lu":3,"Y":3,
             "Bi":3,"Na":1,"K":1,"Li":1,"Ag":1}
B_CHARGE = {"Ti":4,"Zr":4,"Sn":4,"Hf":4,"Ge":4,"Si":4,"Nb":5,"Ta":5,"V":5,
             "W":6,"Mo":6,"Mg":2,"Zn":2,"Co":2,"Ni":2,"Cu":2,"Fe":3,"Mn":3,
             "Al":3,"Cr":3,"Ga":3,"In":3,"Sc":3,
             # RE³⁺ when reclassified to B-site
             "La":3,"Ce":3,"Pr":3,"Nd":3,"Sm":3,"Eu":3,"Gd":3,"Tb":3,
             "Dy":3,"Ho":3,"Er":3,"Tm":3,"Yb":3,"Lu":3,"Y":3}

# Covalent radii (Å)
COV_RADII = {"Ba":2.15,"Sr":1.95,"Ca":1.76,"La":1.87,"Ce":1.83,"Pr":1.82,
             "Nd":1.81,"Sm":1.80,"Eu":1.80,"Gd":1.79,"Tb":1.76,"Dy":1.75,
             "Ho":1.74,"Er":1.73,"Yb":1.70,"Lu":1.72,"Y":1.90,"Bi":1.60,
             "Pb":1.46,"Na":1.66,"K":2.03,"Ti":1.60,"Zr":1.75,"Nb":1.64,
             "Ta":1.70,"Sn":1.39,"W":1.62,"Mg":1.41,"Zn":1.22,"Al":1.21,
             "Fe":1.52,"Mn":1.61,"In":1.42,"Sc":1.70,"Ga":1.22,"Si":1.11,
             "Co":1.50,"Ni":1.24,"Cu":1.38,"V":1.53,"Cr":1.66,"Mo":1.54,"Hf":1.75}

# SOJT hierarchy weights: Ti(3d)>Zr/Nb(4d)>Ta/W/Mo(5d)
SOJT_WEIGHTS = {"Ti":1.0,"Nb":0.6,"Zr":0.6,"Hf":0.6,"Ta":0.3,"W":0.3,"Mo":0.3,"V":0.6}

# Lone-pair A-site cations (drive off-centering)
LONE_PAIR_A = {"Pb","Bi","Na","Li"}

# Melting point proxies (°C) for ST normalisation
MELT_PROXY = {"Ba":1000,"Sr":900,"Ca":850,"La":1200,"Nd":1200,"Sm":1200,"Gd":1200,
              "Ti":1843,"Zr":1855,"Nb":2477,"Ta":2996,"W":3422,"Mg":650,"Sn":232,
              "Al":660,"Pb":327,"Bi":271}

# Brown (2009) bond-valence R0 parameters for A-site cations in ABO₃ (Å)
BVS_R0_A = {
    "Ba":2.285,"Sr":2.118,"Ca":1.967,"Pb":2.112,"Bi":2.094,
    "La":2.172,"Ce":2.152,"Pr":2.130,"Nd":2.105,"Sm":2.068,
    "Eu":2.054,"Gd":2.054,"Tb":2.038,"Dy":2.021,"Ho":2.007,
    "Er":1.994,"Yb":1.965,"Lu":1.958,"Y":2.014,
    "Na":1.803,"K":2.132,"Li":1.466,
}

# A-site family flags
BA_ELEMENTS = {"Ba"}
CA_ELEMENTS = {"Ca"}
SR_ELEMENTS = {"Sr"}
RE_ELEMENTS = {"La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Yb","Lu","Y"}

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d+\.?\d*)?")

# Unicode subscript digits → ASCII  (₀₁₂₃₄₅₆₇₈₉ → 0-9)
_UNICODE_SUB_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def normalize_formula(formula: str) -> str:
    """
    Normalize raw Materials string before formula parsing:
      1. Unicode subscript digits → ASCII   (Ba₅Nb₃TaO₁₅  → Ba5Nb3TaO15)
      2. Non-breaking spaces stripped        (\xa0Li…\xa0   → Li…)
      3. Leading digit+comma artifact removed (0,Ca4La2Ti5O17 → Ca4La2Ti5O17)
      4. Explicit coefficient-1 collapsed     (Sr1Ti1O3      → SrTiO3)
         Safe rule: element+1 followed by a non-digit (uppercase letter or end).
         Ti10O3 → Ti10O3 (1 followed by 0, a digit → not removed).
      5. Leading/trailing whitespace removed
    """
    formula = formula.translate(_UNICODE_SUB_MAP)
    formula = formula.replace("\xa0", "").strip()
    formula = re.sub(r"^\d+,\s*", "", formula)
    # Remove explicit-1 coefficient: [Element]1 followed by another element or end-of-string
    formula = re.sub(r"([A-Z][a-z]?)1(?!\d)", r"\1", formula)
    return formula.strip()


def parse_formula(formula: str):
    """
    Parse ABO₃ formula → (a_site, b_site, o_count, unknown, ptype).

    ptype values:
      "simple"     — standard ABO₃ (a_tot≈1, single B element)
      "complex_b"  — ABO₃ family, multiple B-site elements (majority of dataset)
      "re_b_site"  — RE reclassified to B-site (BaRE₀.₅Nb₀.₅O₃ type)
      "double"     — true double/complex perovskite A₂BB'O₆ or similar
                     (a_tot>1, but VCA valid: o/a_tot≈3, b_tot/a_tot≈1)
      "layered"    — Ruddlesden-Popper / Aurivillius / intergrowth structures
                     (a_tot>1 AND VCA not valid: stoichiometry deviates from ABO₃)

    Physics note:
      "simple"/"complex_b"/"re_b_site": Clausius-Mossotti, Goldschmidt tolerance
        factor, Reaney regime classification all apply directly.
      "double": VCA normalisation gives effective ABO₃ stoichiometry; CM valid
        with higher uncertainty (cm_approx_flag=1).
      "layered": Standard ABO₃ physics does NOT apply. CM is not computed.
        Model uses pure-ML fallback (has_cm=0).

    RE→B-site reclassification:
    Triggered when a_tot > 1.05 AND RE elements present AND high-valence
    B-site elements (Nb⁵⁺, Ta⁵⁺, W⁶⁺, Mo⁶⁺, Re⁶⁺) present. If moving
    all RE from A→B brings a_tot closer to 1.0 AND charge balance holds, accept.
    """
    formula = normalize_formula(formula)
    tokens = _FORMULA_RE.findall(formula)
    a_site, b_site, o_count, unknown = {}, {}, 0.0, []
    for element, stoich_str in tokens:
        stoich = float(stoich_str) if stoich_str else 1.0
        if element == "O":
            o_count += stoich
        elif element in A_SITE_ELEMENTS:
            a_site[element] = a_site.get(element, 0.0) + stoich
        elif element in B_SITE_ELEMENTS:
            b_site[element] = b_site.get(element, 0.0) + stoich
        else:
            unknown.append(element)

    # RE→B-site reclassification for BaRE₀.₅Nb₀.₅O₃-type complex perovskites
    _HIGH_VALENCE_B = {"Nb", "Ta", "W", "Mo", "Re"}
    a_tot = sum(a_site.values())
    ptype = "simple"

    if a_tot > 1.05 and any(el in _HIGH_VALENCE_B for el in b_site):
        re_on_a = {el: n for el, n in a_site.items() if el in RE_ELEMENTS}
        if re_on_a:
            a_trial = {el: n for el, n in a_site.items() if el not in re_on_a}
            b_trial = {**b_site, **re_on_a}
            a_tot_t = sum(a_trial.values())
            if abs(a_tot_t - 1.0) < abs(a_tot - 1.0):
                z_a = sum(A_CHARGE.get(el, 0) * n for el, n in a_trial.items())
                z_b = sum(
                    (B_CHARGE.get(el) or A_CHARGE.get(el, 0)) * n
                    for el, n in b_trial.items()
                )
                if o_count > 0 and abs(z_a + z_b - 2.0 * o_count) < 1.0:
                    a_site, b_site = a_trial, b_trial
                    ptype = "re_b_site"

    if ptype == "simple":
        a_tot_f = sum(a_site.values())
        b_tot_f = sum(b_site.values())
        if a_tot_f > 1.05 or o_count > 3.5:
            # Determine if VCA is valid (true double/complex perovskite)
            # or if this is a layered/intergrowth structure where CM doesn't apply.
            # VCA valid when normalising by a_tot gives ABO₃-like stoichiometry:
            #   o_count / a_tot ≈ 3.0   (oxygen per A-site ≈ 3)
            #   b_tot   / a_tot ≈ 1.0   (B-site count matches A-site count)
            if a_tot_f > 0:
                o_per_a = o_count / a_tot_f
                b_per_a = b_tot_f / a_tot_f
                vca_valid = (abs(o_per_a - 3.0) < 0.25) and (abs(b_per_a - 1.0) < 0.25)
            else:
                vca_valid = False
            ptype = "double" if vca_valid else "layered"
        elif len(b_site) > 1:
            ptype = "complex_b"

    return a_site, b_site, o_count, unknown, ptype


def _wavg(site_dict: dict, lookup: dict) -> float:
    total = sum(site_dict.values())
    if total == 0:
        return float("nan")
    val = 0.0
    for el, n in site_dict.items():
        if el not in lookup:
            return float("nan")
        val += (n / total) * lookup[el]
    return val


def _config_entropy(site: dict) -> float:
    total = sum(site.values())
    if total == 0:
        return 0.0
    return -sum((n / total) * math.log(n / total) for n in site.values() if n > 0)


def _field_strength(site: dict, radii: dict, charges: dict) -> float:
    total = sum(site.values())
    if total == 0:
        return float("nan")
    fs = 0.0
    for el, n in site.items():
        r = radii.get(el, float("nan"))
        z = charges.get(el, float("nan"))
        if math.isnan(r) or math.isnan(z):
            return float("nan")
        fs += (n / total) * (z / r ** 2)
    return fs


def compute_features(row: pd.Series) -> dict:
    formula = normalize_formula(str(row.get("Materials") or row.get("formula") or ""))
    a_site, b_site, o_count, unknown, ptype = parse_formula(formula)

    feats = {}
    feats["formula"] = formula  # always normalized
    feats["parse_ok"] = int(len(a_site) > 0 and len(b_site) > 0)
    feats["n_A_elements"] = len(a_site)
    feats["n_B_elements"] = len(b_site)
    feats["unknown_elements"] = ",".join(unknown) if unknown else ""

    if not feats["parse_ok"]:
        return feats

    a_tot = sum(a_site.values())
    b_tot = sum(b_site.values())

    # ── Layer 1: Electronic ───────────────────────────────────────────────────
    try:
        alpha_total = calc_alpha_total(a_site, b_site, o_count)
    except KeyError:
        alpha_total = float("nan")

    alpha_A = _wavg(a_site, ION_POLARIZABILITY)
    alpha_B = _wavg(b_site, ION_POLARIZABILITY)
    feats["alpha_A"] = alpha_A
    feats["alpha_B"] = alpha_B
    feats["alpha_total"] = alpha_total

    try:
        r_A_avg = avg_radius(a_site, A_SITE_RADII)
        r_B_avg = avg_radius(b_site, B_SITE_RADII)
        M = calc_molar_mass(a_site, b_site, o_count)
        a_pc = calc_pseudocubic_a(r_A_avg, r_B_avg)
        rho = calc_density(M, a_pc)
        er_CM = calc_er_CM(alpha_total, M, rho)
        sigma_CM = propagate_CM_uncertainty(alpha_total, M, rho)
    except Exception:
        M = a_pc = rho = er_CM = sigma_CM = float("nan")

    feats["molar_mass"] = M
    feats["a_pc"] = a_pc
    feats["density_calc"] = rho

    if ptype in ("double", "re_b_site") and b_tot > 1.0:
        # VCA normalisation: scale to 1 B-site equivalent.
        # Covers both explicit double perovskites (A₂BB'O₆) and re_b_site
        # compounds where multiple RE ions moved to B-site (e.g. Ba₄La(TiNb₃)O₁₅).
        # Valid when normalising by b_tot gives ABO₃-like stoichiometry:
        #   o/b_tot ≈ 3 and a_tot/b_tot ≈ 1.
        scale = b_tot
        o_norm_check = o_count / scale
        a_norm_check = a_tot / scale
        vca_valid = (abs(o_norm_check - 3.0) < 0.25) and (abs(a_norm_check - 1.0) < 0.25)
        if not vca_valid and ptype == "double":
            # Fallback: try A-site normalisation for classic double perovskites
            scale = a_tot
            o_norm_check = o_count / scale
            b_norm_check = b_tot / scale
            vca_valid = (abs(o_norm_check - 3.0) < 0.25) and (abs(b_norm_check - 1.0) < 0.25)
        if vca_valid:
            try:
                a_norm = {el: n / scale for el, n in a_site.items()}
                b_norm = {el: n / scale for el, n in b_site.items()}
                o_norm = o_count / scale
                alpha_n = calc_alpha_total(a_norm, b_norm, o_norm)
                r_A_n = avg_radius(a_norm, A_SITE_RADII)
                r_B_n = avg_radius(b_norm, B_SITE_RADII)
                M_n = calc_molar_mass(a_norm, b_norm, o_norm)
                a_pc_n = calc_pseudocubic_a(r_A_n, r_B_n)
                rho_n = calc_density(M_n, a_pc_n)
                er_CM_vca = calc_er_CM(alpha_n, M_n, rho_n)
                sigma_CM_vca = propagate_CM_uncertainty(alpha_n, M_n, rho_n, d_alpha=0.10)
            except Exception:
                er_CM_vca = sigma_CM_vca = float("nan")
            feats["er_CM"] = er_CM_vca
            feats["sigma_CM"] = sigma_CM_vca
            feats["cm_approx_flag"] = 1.0
        else:
            # Stoichiometry cannot be reduced to ABO₃ — treat as layered
            feats["er_CM"] = float("nan")
            feats["sigma_CM"] = float("nan")
            feats["cm_approx_flag"] = 2.0
    elif ptype == "double" and b_tot <= 1.0:
        # Classic A₂BB'O₆: normalise by a_tot
        try:
            scale = a_tot
            a_norm = {el: n / scale for el, n in a_site.items()}
            b_norm = {el: n / scale for el, n in b_site.items()}
            o_norm = o_count / scale
            alpha_n = calc_alpha_total(a_norm, b_norm, o_norm)
            r_A_n = avg_radius(a_norm, A_SITE_RADII)
            r_B_n = avg_radius(b_norm, B_SITE_RADII)
            M_n = calc_molar_mass(a_norm, b_norm, o_norm)
            a_pc_n = calc_pseudocubic_a(r_A_n, r_B_n)
            rho_n = calc_density(M_n, a_pc_n)
            er_CM_vca = calc_er_CM(alpha_n, M_n, rho_n)
            sigma_CM_vca = propagate_CM_uncertainty(alpha_n, M_n, rho_n, d_alpha=0.10)
        except Exception:
            er_CM_vca = sigma_CM_vca = float("nan")
        feats["er_CM"] = er_CM_vca
        feats["sigma_CM"] = sigma_CM_vca
        feats["cm_approx_flag"] = 1.0
    elif ptype == "layered":
        # Ruddlesden-Popper / Aurivillius / intergrowth: CM not applicable
        feats["er_CM"] = float("nan")
        feats["sigma_CM"] = float("nan")
        feats["cm_approx_flag"] = 2.0
    else:
        feats["er_CM"] = er_CM
        feats["sigma_CM"] = sigma_CM
        feats["cm_approx_flag"] = 0.0

    # Structural stoichiometry features (continuous — help model distinguish types)
    b_tot_f = sum(b_site.values())
    total_cat = a_tot + b_tot_f
    feats["o_per_cation"] = o_count / total_cat if total_cat > 0 else float("nan")
    feats["b_per_a"]      = b_tot_f / a_tot    if a_tot > 0    else float("nan")

    # polarizability derived
    feats["ab_pol_diff"] = abs(alpha_A - alpha_B) if not (math.isnan(alpha_A) or math.isnan(alpha_B)) else float("nan")

    # Phonon softness proxy: (α_B − α_O) / α_O
    # How much more polarizable the B-site is than oxygen → predicts LST enhancement magnitude
    _alpha_O = ION_POLARIZABILITY["O"]
    feats["phonon_softness_proxy"] = (
        (alpha_B - _alpha_O) / _alpha_O if not math.isnan(alpha_B) else float("nan")
    )

    # ── Layer 2: Structural ───────────────────────────────────────────────────
    t   = calc_tolerance_factor(a_site, b_site)
    mu  = calc_octahedral_factor(b_site)
    tilt = calc_tilt_severity(t)
    oct_vol = calc_octahedral_volume(b_site)

    feats["tolerance_factor"] = t
    feats["octahedral_factor"] = mu
    feats["tilt_severity"] = tilt
    feats["tol_sq_dev"] = tilt ** 2 if not math.isnan(tilt) else float("nan")
    feats["goldschmidt_deviation"] = abs(t - 1.0) if not math.isnan(t) else float("nan")
    feats["symmetry_score"] = max(0.0, 1.0 - min(tilt, 1.0)) if not math.isnan(tilt) else float("nan")
    feats["octahedral_volume"] = oct_vol
    feats["b_size_mismatch"] = calc_b_size_mismatch(b_site)

    regime = classify_regime(t, a_site, b_site)
    feats["reaney_regime"] = regime
    feats["regime_Ia"] = int(regime == "Ia")
    feats["regime_Ib"] = int(regime == "Ib")
    feats["regime_II"] = int(regime == "II")
    feats["regime_III"] = int(regime == "III")

    # Soft-mode proxies
    feats["d0_B_polarizable_A"] = int(soft_mode_proxy_binary(a_site, b_site))
    feats["d0_B_fraction"] = d0_B_fraction(b_site)
    feats["polarizable_A_fraction"] = polarizable_A_fraction(a_site)
    feats["soft_mode_proxy"] = soft_mode_proxy_continuous(a_site, b_site)

    # Phonon / SOJT features
    b_fs = _field_strength(b_site, B_SITE_RADII, B_CHARGE)
    d0_fr = feats["d0_B_fraction"]
    feats["b_field_strength"] = b_fs
    feats["a_field_strength"] = _field_strength(a_site, A_SITE_RADII, A_CHARGE)
    feats["d0_per_field"] = d0_fr / b_fs if (not math.isnan(b_fs) and b_fs > 0) else float("nan")

    # weighted SOJT: Ti(3d)>Zr/Nb(4d)>Ta/W/Mo(5d)
    feats["weighted_SOJT"] = (
        sum(n * SOJT_WEIGHTS.get(el, 0) for el, n in b_site.items()) / b_tot
        if b_tot > 0 else 0.0
    )

    # B-site average atomic mass (phonon mass proxy)
    feats["b_avg_atomic_mass"] = _wavg(b_site, ATOMIC_MASS)

    # B–O reduced mass: μ = m_B_avg × m_O / (m_B_avg + m_O)
    # Lower μ → higher ω_TO → weaker LST enhancement; higher μ → higher εr
    M_O = 15.999
    m_B_avg = feats["b_avg_atomic_mass"]
    feats["b_o_reduced_mass"] = (
        (m_B_avg * M_O) / (m_B_avg + M_O)
        if not math.isnan(m_B_avg) else float("nan")
    )

    # Soft-mode activity: SOJT strength × A-site polarizability (LST driving force)
    # Explicit product — quantifies how much εr can exceed CM via soft-mode
    feats["soft_mode_activity"] = (
        feats["weighted_SOJT"] * alpha_A
        if not math.isnan(alpha_A) else float("nan")
    )

    # LST enhancement proxy: soft_mode_activity / reduced_mass
    # Higher = lower ω_TO expected = stronger LST enhancement
    feats["lst_enhancement_proxy"] = (
        feats["soft_mode_activity"] / feats["b_o_reduced_mass"]
        if (not math.isnan(feats.get("soft_mode_activity", float("nan"))) and
            not math.isnan(feats["b_o_reduced_mass"]) and
            feats["b_o_reduced_mass"] > 0)
        else float("nan")
    )

    # Phase boundary proximity (|t − boundary|)
    # Materials near tilt transition have competing order parameters → εr anomalies
    feats["phase_boundary_dist_ib_ii"] = abs(t - 0.985) if not math.isnan(t) else float("nan")
    feats["phase_boundary_dist_ii_iii"] = abs(t - 0.965) if not math.isnan(t) else float("nan")
    feats["phase_boundary_min_dist"] = (
        min(feats["phase_boundary_dist_ib_ii"], feats["phase_boundary_dist_ii_iii"])
        if not math.isnan(t) else float("nan")
    )
    feats["regime_boundary_crossing"] = (
        int(feats["phase_boundary_min_dist"] < 0.01) if not math.isnan(t) else 0
    )

    # Charge imbalance proxy (BVS strain proxy)
    # For ABO₃: A-charge + B-charge = 6 (balances 3×O²⁻). Deviation = structural strain.
    a_charge_avg = _wavg(a_site, A_CHARGE)
    b_charge_avg = _wavg(b_site, B_CHARGE)
    feats["charge_imbalance_proxy"] = (
        abs(a_charge_avg + b_charge_avg - 6.0) / 6.0
        if (not math.isnan(a_charge_avg) and not math.isnan(b_charge_avg))
        else float("nan")
    )

    # BVS strain: A-site bond-valence deviation from formal charge
    # R_AO = a_pc/√2 for pseudocubic coordination; Brown (2009) R0 table
    if not math.isnan(a_pc) and a_pc > 0 and all(el in BVS_R0_A for el in a_site):
        _R_AO = a_pc / math.sqrt(2)
        _bvs_a = sum(
            (n / a_tot) * 12 * math.exp((BVS_R0_A[el] - _R_AO) / 0.37)
            for el, n in a_site.items()
        )
        _z_avg = _wavg(a_site, A_CHARGE)
        feats["bvs_strain_proxy"] = (
            abs(_bvs_a - _z_avg) / _z_avg
            if (not math.isnan(_z_avg) and _z_avg > 0) else float("nan")
        )
    else:
        feats["bvs_strain_proxy"] = float("nan")

    # SOJT × tolerance factor: soft-mode strength modulated by structural geometry
    # High SOJT in well-matched structure (t≈1) → maximum soft-mode εr enhancement
    feats["soft_mode_tolerance_weighted"] = (
        feats["weighted_SOJT"] * (1.0 - abs(t - 1.0))
        if not math.isnan(t) else float("nan")
    )

    # Phase boundary instability: exponential spike near tilt transition boundaries
    # Materials within Δt=0.01 of a boundary have anomalously high εr variance
    feats["phase_boundary_instability"] = (
        math.exp(-feats["phase_boundary_min_dist"] / 0.01)
        if not math.isnan(t) else float("nan")
    )

    # Continuous tilt strain: tilt severity scaled by charge imbalance
    # Combines structural and electronic driving forces for octahedral tilting
    feats["continuous_tilt_strain"] = (
        tilt * feats["charge_imbalance_proxy"]
        if (not math.isnan(tilt) and
            not math.isnan(feats["charge_imbalance_proxy"]))
        else float("nan")
    )
    feats["bond_angle_variance_proxy"] = (
        tilt ** 2 * mu if (not math.isnan(tilt) and not math.isnan(mu)) else float("nan")
    )

    # ── Layer 3: Bonding / Covalency ──────────────────────────────────────────
    chi_A = _wavg(a_site, ELECTRONEGATIVITY)
    chi_B = _wavg(b_site, ELECTRONEGATIVITY)
    chi_O = ELECTRONEGATIVITY["O"]

    feats["chi_A"] = chi_A
    feats["chi_B"] = chi_B
    feats["delta_chi_A"] = abs(chi_A - chi_O) if not math.isnan(chi_A) else float("nan")
    feats["delta_chi_B"] = abs(chi_B - chi_O) if not math.isnan(chi_B) else float("nan")
    feats["covalency_index"] = (
        1.0 - feats["delta_chi_B"] / chi_O
        if not math.isnan(feats["delta_chi_B"]) else float("nan")
    )

    # Phillips ionicity (van Vechten 1970): exact formula from quantum chemistry
    # ionicity = 1 − exp(−0.25 × ΔEN²); covalency = 1 − ionicity
    # More quantitative than covalency_index which uses a linear approximation
    dEN_BO = feats["delta_chi_B"]
    feats["phillips_ionicity_bo"] = (
        1.0 - math.exp(-0.25 * dEN_BO ** 2)
        if not math.isnan(dEN_BO) else float("nan")
    )
    feats["phillips_covalency_bo"] = (
        1.0 - feats["phillips_ionicity_bo"]
        if not math.isnan(feats["phillips_ionicity_bo"]) else float("nan")
    )

    feats["covalency_ratio"] = (
        feats["delta_chi_B"] / feats["delta_chi_A"]
        if (not math.isnan(feats["delta_chi_B"]) and
            not math.isnan(feats["delta_chi_A"]) and
            feats["delta_chi_A"] > 0) else float("nan")
    )

    r_A_avg_f = _wavg(a_site, A_SITE_RADII)
    r_B_avg_f = _wavg(b_site, B_SITE_RADII)
    feats["r_A_avg"] = r_A_avg_f
    feats["r_B_avg"] = r_B_avg_f
    feats["r_A_r_B_ratio"] = (
        r_A_avg_f / r_B_avg_f
        if (not math.isnan(r_A_avg_f) and r_B_avg_f > 0) else float("nan")
    )

    # A-site ionic radius variance (0.0 for single-element A-site)
    if not math.isnan(r_A_avg_f) and a_tot > 0:
        _a_fracs = [n / a_tot for n in a_site.values()]
        _a_radii = [A_SITE_RADII.get(el, float("nan")) for el in a_site]
        feats["a_site_variance"] = (
            sum(f * (r - r_A_avg_f) ** 2 for f, r in zip(_a_fracs, _a_radii))
            if not any(math.isnan(r) for r in _a_radii) else float("nan")
        )
    else:
        feats["a_site_variance"] = float("nan")

    # Soft-mode proxy v2: d0 fraction × polarizable_A × A-site size amplification
    # Larger A-site (Ba > Sr > Ca) → better d0-polarizable coupling → stronger soft mode
    feats["soft_mode_proxy_v2"] = (
        feats["d0_B_fraction"] * feats["polarizable_A_fraction"] * (r_A_avg_f / 1.61)
        if not (math.isnan(feats.get("d0_B_fraction", float("nan"))) or
                math.isnan(feats.get("polarizable_A_fraction", float("nan"))) or
                math.isnan(r_A_avg_f))
        else float("nan")
    )

    feats["b_site_ratio"] = (
        max(list(b_site.values())) / min(list(b_site.values()))
        if len(b_site) == 2 and min(b_site.values()) > 0 else 1.0
    )

    # Covalent radius (for interaction features)
    all_site = {**a_site, **b_site}
    all_tot = sum(all_site.values())
    r_cov = (
        sum((n / all_tot) * COV_RADII[el] for el, n in all_site.items() if el in COV_RADII)
        if all_tot > 0 else float("nan")
    )
    feats["r_cov"] = r_cov

    # Entropy features
    feats["a_config_entropy"] = _config_entropy(a_site)
    feats["b_config_entropy"] = _config_entropy(b_site)
    feats["num_cations"] = len(set(list(a_site.keys()) + list(b_site.keys())))

    # A-site off-centering (lone-pair cations Pb, Bi, Na, Li)
    feats["a_off_centering_frac"] = (
        sum(n for el, n in a_site.items() if el in LONE_PAIR_A) / a_tot
        if a_tot > 0 else 0.0
    )

    # Family flags (dominant A-site element > 40%)
    feats["is_Ba"] = int(a_site.get("Ba", 0) / a_tot > 0.4) if a_tot > 0 else 0
    feats["is_Ca"] = int(a_site.get("Ca", 0) / a_tot > 0.4) if a_tot > 0 else 0
    feats["is_Sr"] = int(a_site.get("Sr", 0) / a_tot > 0.4) if a_tot > 0 else 0
    feats["is_RE"] = int(
        sum(a_site.get(el, 0) for el in RE_ELEMENTS) / a_tot > 0.4
    ) if a_tot > 0 else 0

    # Perovskite structure type flags
    feats["ptype_simple"]    = int(ptype == "simple")
    feats["ptype_complex_b"] = int(ptype == "complex_b")
    feats["ptype_re_b_site"] = int(ptype == "re_b_site")
    feats["ptype_double"]    = int(ptype == "double")
    feats["ptype_layered"]   = int(ptype == "layered")

    return feats


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        feats = compute_features(row)

        # Process variables
        def _get_num(row, *keys):
            for k in keys:
                v = row.get(k)
                if v is not None and str(v).strip() not in ("", "nan", "NaN"):
                    return pd.to_numeric(v, errors="coerce")
            return float("nan")

        CT      = _get_num(row, "CT", "C-T")
        C_time  = _get_num(row, "C_time", "C-Time")
        ST      = _get_num(row, "ST")
        ST_time = _get_num(row, "ST_time", "ST-time")

        feats["CT"]      = CT
        feats["C_time"]  = C_time
        feats["ST"]      = ST
        feats["ST_time"] = ST_time

        # Interaction: sintering T × covalent radius (V7 rank #1)
        r_cov = feats.get("r_cov", float("nan"))
        feats["T_sint_x_r_cov"] = ST * r_cov if (not math.isnan(ST) and not math.isnan(r_cov)) else float("nan")
        feats["T_calc_x_r_cov"] = CT * r_cov if (not math.isnan(CT) and not math.isnan(r_cov)) else float("nan")

        # Sintering T normalised by melting point proxy
        a_site_f, b_site_f, _, _, _ = parse_formula(
            str(row.get("Materials") or row.get("formula") or "")
        )
        all_site_f = {**a_site_f, **b_site_f}
        all_tot_f  = sum(all_site_f.values())
        melt_est = (
            sum((n / all_tot_f) * MELT_PROXY[el]
                for el, n in all_site_f.items() if el in MELT_PROXY)
            if all_tot_f > 0 else float("nan")
        )
        feats["ST_norm_melt"] = ST / melt_est if (not math.isnan(ST) and not math.isnan(melt_est) and melt_est > 0) else float("nan")

        # Thermal budget: log(T × t + 1) — joint thermal exposure proxy
        feats["thermal_budget_sinter"] = (
            math.log1p(ST * ST_time)
            if (not math.isnan(ST) and not math.isnan(ST_time) and ST > 0 and ST_time > 0)
            else float("nan")
        )
        feats["thermal_budget_calc"] = (
            math.log1p(CT * C_time)
            if (not math.isnan(CT) and not math.isnan(C_time) and CT > 0 and C_time > 0)
            else float("nan")
        )

        # Family × physics interactions (safe .get() for failed-parse rows)
        _is_Ba = feats.get("is_Ba", 0)
        _is_Ca = feats.get("is_Ca", 0)
        _is_Sr = feats.get("is_Sr", 0)
        _is_RE = feats.get("is_RE", 0)
        _t_f   = feats.get("tolerance_factor", float("nan"))
        _tsrc  = feats.get("T_sint_x_r_cov", float("nan"))
        _chiA  = feats.get("chi_A", float("nan"))
        _alpB  = feats.get("alpha_B", float("nan"))
        feats["Ba_x_T_sint_r_cov"] = _is_Ba * (_tsrc if not math.isnan(_tsrc) else 0.0)
        feats["Ca_x_tol_factor"]   = _is_Ca * (_t_f  if not math.isnan(_t_f)  else 0.0)
        feats["RE_x_chi_A"]        = _is_RE * (_chiA  if not math.isnan(_chiA) else 0.0)
        feats["Sr_x_alpha_B"]      = _is_Sr * (_alpB  if not math.isnan(_alpB) else 0.0)

        # Crystal system raw string
        cs_raw = row.get("crystal_system") or row.get("crystalsystem") or ""
        feats["crystal_system"] = str(cs_raw).strip().lower()

        # Crystal system one-hot (structural encoding for regime-aware models)
        _cs = feats["crystal_system"]
        feats["cs_cubic"]        = int("cubic"        in _cs)
        feats["cs_tetragonal"]   = int("tetragonal"   in _cs)
        feats["cs_orthorhombic"] = int("orthorhombic" in _cs)
        feats["cs_rhombohedral"] = int("rhombohedral" in _cs or "trigonal" in _cs)
        feats["cs_monoclinic"]   = int("monoclinic"   in _cs)
        feats["cs_hexagonal"]    = int("hexagonal"    in _cs)

        # Additive
        feats["additive_pct"] = pd.to_numeric(
            row.get("additive_pct") or row.get("additive %") or 0, errors="coerce"
        )
        add_type = row.get("additive_type") or row.get("Additive type ")
        feats["has_additive"] = int(
            pd.notna(add_type) and str(add_type).strip() not in ("", "nan", "NaN")
        )

        # Targets
        feats["DC"]     = row.get("DC", float("nan"))
        feats["Qf_GHz"] = row.get("Qf_GHz") or row.get("Qf (GHz)", float("nan"))
        feats["tau_f"]  = row.get("tau_f") or row.get("𝜏f", float("nan"))
        feats["ref"]    = row.get("ref", "")
        records.append(feats)

    return pd.DataFrame(records)
