"""
CN-dependent ionic polarizabilities from Qin et al. (2023).

Reference:
  Qin et al., npj Computational Materials 9, 202 (2023)
  DOI: 10.1038/s41524-023-01093-6
  Title: Machine-learning of dielectric properties of crystalline compounds

Three tiers (in decreasing precision):
  ML Extended   — 915 ions, ML-predicted, GHz-calibrated
  MLR Optimized — 141 ions, MLR-fitted to GHz measurements
  Shannon       — 59 ions, original Shannon (1993) basis

Key lookup format: element → αD (Å³) at typical ABO₃ perovskite coordination:
  A-site  CN=12 (cuboctahedral), B-site CN=6 (octahedral), O²⁻ CN=6

CALIBRATION NOTE:
  These αD values are calibrated assuming ACTUAL crystal densities (from DFT
  or experimental lattice parameters). They are NOT compatible with the
  pseudocubic lattice parameter approximation in clausius_mossotti.py without
  independent density recalibration. Do NOT substitute for Shannon-calibrated
  values in get_polarizability() without recalibrating calc_pseudocubic_a().

Usage:
  get_qin_alpha(element, site='B')
    → αD from ML Extended if available, else MLR Optimized, else None
"""

# ---------------------------------------------------------------------------
# Qin et al. ML Extended (915 ions) — perovskite-appropriate CN
# A-site: valence from typical charge state, CN=12
# B-site: valence from typical charge state, CN=6
# O²⁻:    CN=6
# ---------------------------------------------------------------------------
QIN_ML_A = {   # A-site ions, CN=12
    "Ba": 7.362245, "Sr": 4.911142, "Ca": 3.968000, "Pb": 5.851772,
    "La": 4.029052, "Ce": 3.776591, "Pr": 3.247178, "Nd": 2.994257,
    "Sm": 2.702051, "Eu": 2.620390, "Gd": 2.403493, "Tb": 2.356156,
    "Dy": 2.254067, "Ho": 2.156760, "Er": 2.880864, "Tm": 1.990075,
    "Yb": 1.912449, "Lu": 2.736167, "Y":  2.687591,
    "Bi": 4.608389, "Na": 4.094398, "K":  6.841893,
    "Li": 2.261957, "Ag": 5.585507,
}

QIN_ML_B = {   # B-site ions, CN=6
    "Ti": 2.401146, "Zr": 3.756683, "Sn": 3.235466, "Hf": 3.399022,
    "Ge": 1.691362, "Si": 0.808751,
    "Nb": 3.835331, "Ta": 3.416416, "V":  2.646876,
    "W":  3.583176, "Mo": 4.084436,
    "Mg": 1.602571, "Zn": 1.848990, "Co": 1.288907, "Ni": 1.520549,
    "Cu": 1.777702, "Fe": 1.218108, "Mn": 1.399618, "In": 3.366124,
    "Sc": 2.833325, "Al": 1.061962, "Ga": 1.700577, "Cr": 1.640422,
    "Ru": 2.493690, "Ir": 2.590501, "Os": 2.623171, "Re": 3.052744,
    "Sb": 3.113850, "Te": 3.407108, "Pd": 2.433189, "Pt": 2.603661,
}

QIN_ML_O = 1.624745   # O²⁻, CN=6

# ---------------------------------------------------------------------------
# Qin et al. MLR Optimized (141 ions) — same CN conventions
# Only entries differing substantially from ML Extended are listed here;
# fall back to QIN_ML_A / QIN_ML_B for unlisted ions.
# ---------------------------------------------------------------------------
QIN_MLR_A = {
    "Ba": 7.350000, "Sr": 5.010000, "Ca": 3.990000, "La": 3.940000,
}
QIN_MLR_B = {
    "Ti": 2.430000, "Zr": 3.690000, "Nb": 3.790000, "Ta": 3.340000,
    "Al": 1.067000, "Mg": 1.590000,
}
QIN_MLR_O = 1.858400   # O²⁻, CN=6 from MLR Optimized


def get_qin_alpha(element: str, site: str = "B") -> float | None:
    """
    Return Qin et al. polarizability (Å³) at perovskite-appropriate CN.

    site='A'  → CN=12 lookup
    site='B'  → CN=6  lookup
    site='O'  → returns QIN_ML_O (CN=6)

    Priority: ML Extended → MLR Optimized → None

    Returns None if element is not in Qin database.
    """
    if site == "O":
        return QIN_ML_O

    ml_table  = QIN_ML_A  if site == "A" else QIN_ML_B
    mlr_table = QIN_MLR_A if site == "A" else QIN_MLR_B

    if element in ml_table:
        return ml_table[element]
    if element in mlr_table:
        return mlr_table[element]
    return None
