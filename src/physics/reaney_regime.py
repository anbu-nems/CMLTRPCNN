"""
Reaney regime classifier for ABO₃ perovskites.

Regime Ia : t > 0.985  AND soft_mode_proxy = False  (paraelectric, untilted)
Regime Ib : t > 0.985  AND soft_mode_proxy = True   (incipient ferroelectric)
Regime II : 0.965 < t ≤ 0.985                        (antiphase tilts only)
Regime III: t ≤ 0.965                                 (antiphase + in-phase tilts)

soft_mode_proxy = True  if  d0_B_site AND polarizable_A_site
  d0_B_site:       Ti, Nb, Ta, or Zr on B-site
  polarizable_A:   Ba, Sr, Pb, or Bi on A-site

Source: Reaney et al. (1994) Jpn. J. Appl. Phys. 33, 3984-3990.
"""

D0_B_ELEMENTS = {"Ti", "Nb", "Ta", "Zr"}
POLARIZABLE_A_ELEMENTS = {"Ba", "Sr", "Pb", "Bi"}


def soft_mode_proxy_binary(a_site: dict, b_site: dict) -> bool:
    """
    Binary soft-mode proxy.
    True if composition has d0 B-cation AND polarizable A-cation.
    """
    has_d0_B = any(el in D0_B_ELEMENTS for el in b_site)
    has_pol_A = any(el in POLARIZABLE_A_ELEMENTS for el in a_site)
    return has_d0_B and has_pol_A


def d0_B_fraction(b_site: dict) -> float:
    """Fraction of d0 cations (Ti, Nb, Ta, Zr) on the B-site."""
    total = sum(b_site.values())
    if total == 0:
        return 0.0
    d0 = sum(n for el, n in b_site.items() if el in D0_B_ELEMENTS)
    return d0 / total


def polarizable_A_fraction(a_site: dict) -> float:
    """Fraction of polarizable cations (Ba, Sr, Pb, Bi) on the A-site."""
    total = sum(a_site.values())
    if total == 0:
        return 0.0
    pol = sum(n for el, n in a_site.items() if el in POLARIZABLE_A_ELEMENTS)
    return pol / total


def soft_mode_proxy_continuous(a_site: dict, b_site: dict) -> float:
    """
    Continuous soft-mode proxy = d0_B_fraction × polarizable_A_fraction.
    Range [0, 1]; captures partial-substitution effects.
    """
    return d0_B_fraction(b_site) * polarizable_A_fraction(a_site)


def classify_regime(t: float, a_site: dict, b_site: dict) -> str:
    """
    Classify Reaney regime as 'Ia', 'Ib', 'II', or 'III'.
    Returns 'unknown' if t is NaN.
    """
    import math
    if math.isnan(t):
        return "unknown"
    soft = soft_mode_proxy_binary(a_site, b_site)
    if t > 0.985:
        return "Ib" if soft else "Ia"
    elif t > 0.965:
        return "II"
    else:
        return "III"
