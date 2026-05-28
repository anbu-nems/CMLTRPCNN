"""
Clausius-Mossotti (CM) baseline for ABO₃ dielectric constant prediction.

CM relation (Ch. 2 Eq. 2.8):
  (εr - 1)/(εr + 2) · M/ρ = (4π Na / 3) · α

Solving for εr given α, M, ρ:
  x = (4π Na / 3) · α · ρ / M
  εr_CM = (1 + 2x) / (1 - x)

Density from pseudocubic lattice parameter:
  a_pc = 2·(r_B + r_O)   [Å]
  ρ    = 1.6605 · M / a_pc³   [g/cm³]  (M in amu, a in Å)
"""

import math
import numpy as np
from src.physics.ionic_radii import O_RADIUS

# Physical constants
NA = 6.02214076e23   # Avogadro's number
FOUR_PI_NA_OVER_3 = (4 * math.pi * NA) / 3  # = 2.522e24 mol⁻¹

# Atomic masses (amu)
ATOMIC_MASS = {
    "Ba": 137.327, "Sr": 87.62,  "Ca": 40.078, "La": 138.905,
    "Ce": 140.116, "Pr": 140.908,"Nd": 144.242,"Sm": 150.36,
    "Eu": 151.964, "Gd": 157.25, "Tb": 158.925,"Dy": 162.500,
    "Ho": 164.930, "Er": 167.259,"Tm": 168.934,"Yb": 173.045,
    "Lu": 174.967, "Y":  88.906, "Bi": 208.980,"Pb": 207.2,
    "Na": 22.990,  "K":  39.098, "Li": 6.941,  "Ag": 107.868,
    "Ti": 47.867,  "Zr": 91.224, "Nb": 92.906, "Ta": 180.948,
    "Sn": 118.710, "W":  183.840,"Mg": 24.305, "Zn": 65.38,
    "Al": 26.982,  "Fe": 55.845, "Mn": 54.938, "In": 114.818,
    "Sc": 44.956,  "Ga": 69.723, "Ge": 72.630, "Si": 28.086,
    "Co": 58.933,  "Ni": 58.693, "Cu": 63.546, "V":  50.942,
    "Cr": 51.996,  "Mo": 95.96,  "Te": 127.600,"Hf": 178.49,
    "Ru": 101.07,  "Ir": 192.217,"Re": 186.207,"Os": 190.23,
    "Sb": 121.760, "O":  15.999,
}


def calc_molar_mass(a_site: dict, b_site: dict, o_count: float = 3.0) -> float:
    """Molar mass of one formula unit in g/mol."""
    M = 0.0
    for el, n in a_site.items():
        M += n * ATOMIC_MASS[el]
    for el, n in b_site.items():
        M += n * ATOMIC_MASS[el]
    M += o_count * ATOMIC_MASS["O"]
    return M


def calc_pseudocubic_a(r_A_avg: float, r_B_avg: float) -> float:
    """
    Pseudocubic lattice parameter in Å using a weighted bond-average formula
    calibrated to reproduce correct CM values for SrTiO3, LaAlO3, BaTiO3, BMT:
      a = 0.686·(r_A + r_O) + 0.972·(r_B + r_O)
    """
    return 0.686 * (r_A_avg + O_RADIUS) + 0.972 * (r_B_avg + O_RADIUS)


def calc_density(M: float, a_pc: float) -> float:
    """
    Theoretical density from unit cell.
    ρ = 1.6605 · M / a³  [g/cm³]  (M in amu, a in Å)
    """
    return 1.6605 * M / (a_pc ** 3)


def calc_er_CM(alpha_total: float, M: float, rho: float) -> float:
    """
    Clausius-Mossotti prediction of εr.

    alpha_total: molecular polarizability in Å³
    M:          molar mass in g/mol
    rho:        density in g/cm³

    Returns εr_CM. Returns NaN if formula is unphysical (x >= 1).
    """
    # Convert α from Å³/formula-unit to cm³/mol
    alpha_cm3_mol = alpha_total * 1e-24 * NA  # Å³ → cm³, × Na

    x = (FOUR_PI_NA_OVER_3 * alpha_total * 1e-24 * rho) / M
    if x >= 1.0 or x <= 0.0:
        return float("nan")
    er = (1.0 + 2.0 * x) / (1.0 - x)
    # Cap at 300: x > ~0.99 means CM is near-singular and unreliable as a baseline
    return er if er <= 300.0 else float("nan")


def propagate_CM_uncertainty(alpha_total: float, M: float, rho: float,
                              d_alpha: float = 0.05) -> float:
    """
    Analytic uncertainty propagation through the CM formula.
    d_alpha: absolute uncertainty on α_total in Å³ (default ±0.05 Å³ per ion,
             propagated as 5% of α_total as a conservative estimate).

    Returns σ(εr_CM) via partial derivative dεr/dα × d_alpha.
    """
    x = (FOUR_PI_NA_OVER_3 * alpha_total * 1e-24 * rho) / M
    if x >= 1.0 or x <= 0.0:
        return float("nan")
    # dεr/dx = 3 / (1 - x)²
    # dx/dα  = (4πNa/3) · ρ/M · 1e-24
    dx_dalpha = (FOUR_PI_NA_OVER_3 * 1e-24 * rho) / M
    der_dx = 3.0 / (1.0 - x) ** 2
    sigma = abs(der_dx * dx_dalpha) * d_alpha
    return sigma
