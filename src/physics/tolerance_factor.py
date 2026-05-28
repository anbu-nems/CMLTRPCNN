"""
Goldschmidt tolerance factor and octahedral factor for ABO₃ perovskites.

t = (r_A + r_O) / [√2 · (r_B + r_O)]   (Goldschmidt 1926)
μ = r_B / r_O                           (octahedral factor)

For mixed A/B sites: weighted averages of ionic radii.
"""

import math
from src.physics.ionic_radii import A_SITE_RADII, B_SITE_RADII, O_RADIUS


def avg_radius(site_dict: dict, radii_table: dict) -> float:
    """Weighted average ionic radius for a mixed-occupancy site."""
    total_stoich = sum(site_dict.values())
    r_avg = sum(n * radii_table[el] for el, n in site_dict.items()) / total_stoich
    return r_avg


def calc_tolerance_factor(a_site: dict, b_site: dict) -> float:
    """
    Goldschmidt tolerance factor t.
    Returns NaN if any element is missing from the radii tables.
    """
    try:
        r_A = avg_radius(a_site, A_SITE_RADII)
        r_B = avg_radius(b_site, B_SITE_RADII)
    except KeyError:
        return float("nan")
    return (r_A + O_RADIUS) / (math.sqrt(2) * (r_B + O_RADIUS))


def calc_octahedral_factor(b_site: dict) -> float:
    """
    Octahedral (tau) factor μ = r_B / r_O.
    Stability range: μ ∈ [0.41, 0.73].
    """
    try:
        r_B = avg_radius(b_site, B_SITE_RADII)
    except KeyError:
        return float("nan")
    return r_B / O_RADIUS


def calc_tilt_severity(t: float) -> float:
    """
    Tilt severity = |t - 1|. Larger values → more tilting → smaller εr.
    Used as a monotonically constrained feature (∂Δεr/∂tilt_severity ≤ 0).
    """
    if math.isnan(t):
        return float("nan")
    return abs(t - 1.0)


def calc_octahedral_volume(b_site: dict) -> float:
    """
    Approximate BO₆ octahedral volume ∝ (r_B + r_O)³ in Å³.
    Proxy for the rattling volume available to the B-cation.
    """
    try:
        r_B = avg_radius(b_site, B_SITE_RADII)
    except KeyError:
        return float("nan")
    return (r_B + O_RADIUS) ** 3


def calc_b_size_mismatch(b_site: dict) -> float:
    """
    Variance of B-site ionic radii — measures cation size mismatch.
    Returns 0.0 for single-B-site compositions.
    """
    if len(b_site) <= 1:
        return 0.0
    total = sum(b_site.values())
    fracs = {el: n / total for el, n in b_site.items()}
    r_avg = sum(f * B_SITE_RADII[el] for el, f in fracs.items() if el in B_SITE_RADII)
    variance = sum(
        f * (B_SITE_RADII[el] - r_avg) ** 2
        for el, f in fracs.items()
        if el in B_SITE_RADII
    )
    return variance
