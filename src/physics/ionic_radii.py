"""
Shannon ionic radii (Å) for ABO₃ perovskite elements.
A-site: CN=12 coordination; B-site: CN=6 (octahedral); O²⁻: CN=6.
Source: Shannon (1976) Acta Cryst. A32, 751-767.
"""

# A-site cations (CN=12), in Å
A_SITE_RADII = {
    "Ba": 1.61,
    "Sr": 1.44,
    "Ca": 1.34,
    "La": 1.36,
    "Ce": 1.34,   # Ce³⁺ CN12
    "Pr": 1.30,
    "Nd": 1.27,
    "Sm": 1.24,
    "Eu": 1.23,
    "Gd": 1.22,
    "Tb": 1.18,
    "Dy": 1.17,
    "Ho": 1.15,
    "Er": 1.14,
    "Tm": 1.13,
    "Yb": 1.11,
    "Lu": 1.10,
    "Y":  1.16,   # Y³⁺ CN9=1.075; use interpolated CN12≈1.16
    "Bi": 1.45,   # Bi³⁺ CN12 (estimated from CN8=1.17)
    "Pb": 1.49,   # Pb²⁺ CN12
    "Na": 1.39,   # Na⁺ CN12
    "K":  1.64,   # K⁺ CN12
    "Ag": 1.48,   # Ag⁺ CN12
    "Li": 1.06,   # Li⁺ CN12 (estimated)
}

# B-site cations (CN=6, octahedral), in Å
B_SITE_RADII = {
    "Ti": 0.605,
    "Zr": 0.720,
    "Nb": 0.640,
    "Ta": 0.640,
    "Sn": 0.690,
    "W":  0.600,
    "Mg": 0.720,
    "Zn": 0.740,
    "Al": 0.535,
    "Fe": 0.645,  # Fe³⁺ high-spin
    "Mn": 0.645,  # Mn³⁺
    "In": 0.800,
    "Sc": 0.745,
    "Ga": 0.620,
    "Ge": 0.530,
    "Si": 0.400,
    "Co": 0.745,  # Co²⁺ high-spin
    "Ni": 0.690,
    "Cu": 0.730,
    "V":  0.540,  # V⁵⁺
    "Cr": 0.615,  # Cr³⁺
    "Mo": 0.590,  # Mo⁶⁺
    "Te": 0.560,  # Te⁶⁺
    "Hf": 0.710,
    "Ru": 0.620,  # Ru⁴⁺
    "Ir": 0.625,  # Ir⁴⁺
    "Re": 0.580,  # Re⁶⁺
    "Os": 0.630,  # Os⁴⁺
    "Sb": 0.600,  # Sb⁵⁺
    "Bi": 0.760,  # Bi³⁺ B-site (rare)
    # RE³⁺ CN=6 — used when RE is reclassified to B-site in complex perovskites
    # e.g. Ba(RE₀.₅Nb₀.₅)O₃ type. Source: Shannon (1976) Acta Cryst. A32, 751.
    "La": 1.032, "Ce": 1.010, "Pr": 0.990, "Nd": 0.983,
    "Sm": 0.958, "Eu": 0.947, "Gd": 0.938, "Tb": 0.923,
    "Dy": 0.912, "Ho": 0.901, "Er": 0.890, "Tm": 0.880,
    "Yb": 0.868, "Lu": 0.861, "Y":  0.900,
}

# Oxygen anion (CN=6), in Å
O_RADIUS = 1.40

# Unified lookup — A-site elements (for parsing)
A_SITE_ELEMENTS = set(A_SITE_RADII.keys())
# B-site elements (for parsing)
B_SITE_ELEMENTS = set(B_SITE_RADII.keys())


def get_r_A(element: str) -> float:
    """Return A-site (CN=12) ionic radius for element. Raises KeyError if unknown."""
    if element not in A_SITE_RADII:
        raise KeyError(f"No A-site ionic radius for element '{element}'")
    return A_SITE_RADII[element]


def get_r_B(element: str) -> float:
    """Return B-site (CN=6) ionic radius for element. Raises KeyError if unknown."""
    if element not in B_SITE_RADII:
        raise KeyError(f"No B-site ionic radius for element '{element}'")
    return B_SITE_RADII[element]
