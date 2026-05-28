"""
Script 67 — Physics-Constrained Genetic Algorithm (GA) for Inverse Design of
ABO₃ Perovskite Ceramics with High Dielectric Constant (εr).

Why GA instead of BO?
---------------------
Script 63 (BO with Pb)       → converged to Pb-rich rare-earth corners (εr≈102, confident-but-wrong).
Script 64 (BO Pb-free)       → trapped in OOD (σ_AD=0.35 cutoff too tight). Best in-AD εr≈68.
Script 61 (curated screening)→ Sr₀.₈₅Ca₀.₁₅TiO₃, εr=140 — but only on a hand-picked 72-entry library.

The GA performs a genuine *inverse design* search: it draws compositions from a
physics-constrained simplex×simplex space, filters them with Goldschmidt tolerance
factor / d⁰ B-site / charge balance constraints *before* the PCNN sees them, and
uses a soft confidence penalty (fitness = εr − 0.5·σ_conf) rather than hard rejection.

Three GA instances are run in parallel:
  GA-Pb-allowed : full A-site {Ba,Sr,Ca,La,Pb,Nd,Sm,Y,Bi,Ce}
  GA-Pb-free    : Pb fraction == 0
  GA-d⁰-strict  : B-site restricted to {Ti,Nb,Ta,Zr} only

Hyperparameters: pop=100, gen=50, tournament=3, crossover=0.7, mutation=0.3,
elite=5 ⇒ 5,000 PCNN forward passes per GA instance.
"""
from __future__ import annotations

import os, sys, json, time, math, warnings, csv, copy
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(_HERE)
_APP   = "./figures_output/explorer_app"
_MCODE = "./figures_output/model_code"
sys.path.insert(0, _APP)
sys.path.insert(0, _MCODE)

import inference as _inf          # noqa: E402
from feature_engineering import compute_features  # noqa: E402

RES_DIR = os.path.join(_ROOT, "results")
os.makedirs(RES_DIR, exist_ok=True)

# ── Search space ──────────────────────────────────────────────────────────────
A_ELEMENTS_FULL = ["Ba", "Sr", "Ca", "La", "Pb", "Nd", "Sm", "Y", "Bi", "Ce"]
B_ELEMENTS_FULL = ["Ti", "Zr", "Nb", "Ta", "Sn"]

# Cation charges (per spec: A-site Ce as Ce⁴⁺, B-site Sn as Sn⁴⁺)
A_CHARGE = {"Ba": 2, "Sr": 2, "Ca": 2, "Pb": 2,
            "La": 3, "Nd": 3, "Sm": 3, "Y": 3, "Bi": 3,
            "Ce": 4}
B_CHARGE = {"Ti": 4, "Zr": 4, "Sn": 4, "Nb": 5, "Ta": 5}

# d⁰ B-site cations (required for soft-mode activation)
D0_BSITE = {"Ti", "Zr", "Nb", "Ta"}    # Sn is NOT d⁰ (5s² lone pair)

# Ionic radii (Shannon CN=12 for A, CN=6 for B)
A_RADII = {"Ba": 1.61, "Sr": 1.44, "Ca": 1.34, "Pb": 1.49,
           "La": 1.36, "Nd": 1.27, "Sm": 1.24, "Y": 1.16,
           "Bi": 1.45, "Ce": 1.34}
B_RADII = {"Ti": 0.605, "Zr": 0.720, "Nb": 0.640, "Ta": 0.640, "Sn": 0.690}
O_RADIUS = 1.40

# Physics constants
GII_MAX = 1.166
Q90 = 14.38
SIGMA_AD_CUTOFF = 0.35
TOL_LO, TOL_HI = 0.85, 1.05
D0_MIN_FRACTION = 0.50
CHARGE_TOL = 0.10

# GA hyperparameters (spec values)
POP_SIZE = 100
N_GEN = 50
TOURN_K = 3
CROSS_RATE = 0.70
MUT_RATE = 0.30
ELITE = 5
MUT_SIGMA = 0.05  # ±5% absolute shift per cation
SIGMA_PENALTY = 0.5  # fitness = εr − 0.5·σ_conf

# Diversity controls: occasionally inject fresh random individuals to avoid premature
# convergence (5% per generation, replacing the bottom 5 individuals after selection)
IMMIGRATION_RATE = 0.05

# A-site family classification
def asite_family(formula_frac: dict) -> str:
    """Pick dominant A-site element family for grouping."""
    if not formula_frac:
        return "?"
    items = [(el, f) for el, f in formula_frac.items() if f > 1e-3]
    items.sort(key=lambda t: -t[1])
    return items[0][0] if items else "?"


# ── Composition vector ⇄ formula ─────────────────────────────────────────────
def vec_to_formula(a_frac: np.ndarray, b_frac: np.ndarray, A_els, B_els,
                   tol: float = 1e-3) -> str:
    a = a_frac / max(a_frac.sum(), 1e-12)
    b = b_frac / max(b_frac.sum(), 1e-12)
    parts = []
    for el, f in zip(A_els, a):
        if f > tol:
            parts.append(f"{el}{round(float(f), 4)}")
    for el, f in zip(B_els, b):
        if f > tol:
            parts.append(f"{el}{round(float(f), 4)}")
    parts.append("O3")
    return "".join(parts)


# ── Physics constraints ──────────────────────────────────────────────────────
def tolerance_factor(a_frac: np.ndarray, b_frac: np.ndarray, A_els, B_els) -> float:
    a = a_frac / max(a_frac.sum(), 1e-12)
    b = b_frac / max(b_frac.sum(), 1e-12)
    r_A = sum(a[i] * A_RADII[el] for i, el in enumerate(A_els))
    r_B = sum(b[i] * B_RADII[el] for i, el in enumerate(B_els))
    return (r_A + O_RADIUS) / (math.sqrt(2) * (r_B + O_RADIUS))


def d0_fraction(b_frac: np.ndarray, B_els) -> float:
    b = b_frac / max(b_frac.sum(), 1e-12)
    return sum(b[i] for i, el in enumerate(B_els) if el in D0_BSITE)


def charge_balance_residual(a_frac: np.ndarray, b_frac: np.ndarray, A_els, B_els) -> float:
    """Return |z_A + z_B − 6| (ABO3 needs total cation charge = 6)."""
    a = a_frac / max(a_frac.sum(), 1e-12)
    b = b_frac / max(b_frac.sum(), 1e-12)
    z_A = sum(a[i] * A_CHARGE[el] for i, el in enumerate(A_els))
    z_B = sum(b[i] * B_CHARGE[el] for i, el in enumerate(B_els))
    return abs((z_A + z_B) - 6.0)


def passes_physics(a_frac: np.ndarray, b_frac: np.ndarray, A_els, B_els,
                   d0_strict: bool = False) -> bool:
    t = tolerance_factor(a_frac, b_frac, A_els, B_els)
    if not (TOL_LO < t < TOL_HI):
        return False
    f_d0 = d0_fraction(b_frac, B_els)
    if d0_strict:
        if f_d0 < 0.999:
            return False
    else:
        if f_d0 < D0_MIN_FRACTION:
            return False
    if charge_balance_residual(a_frac, b_frac, A_els, B_els) > CHARGE_TOL:
        return False
    return True


def project_charge_balance(a_frac: np.ndarray, b_frac: np.ndarray,
                           A_els, B_els, n_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively adjust A-site to satisfy z_A + z_B = 6, keeping B-site fixed.

    Strategy: compute target z_A = 6 - z_B. Solve for A-site fractions
    that hit this target while staying close to current distribution.
    Uses a projection onto the simplex hyperplane constraint.
    """
    a = a_frac / max(a_frac.sum(), 1e-12)
    b = b_frac / max(b_frac.sum(), 1e-12)
    z_B = sum(b[i] * B_CHARGE[el] for i, el in enumerate(B_els))
    target_zA = 6.0 - z_B
    charges_A = np.array([A_CHARGE[el] for el in A_els], dtype=float)
    # Bound check: target must lie within [min,max] of A-site charges
    if not (charges_A.min() - CHARGE_TOL <= target_zA <= charges_A.max() + CHARGE_TOL):
        return a, b  # infeasible — leave caller to reject
    # Iterative reweighting: scale fractions by exp(λ·z) and renormalize
    lo, hi = -10.0, 10.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        w = a * np.exp(mid * charges_A)
        w = w / max(w.sum(), 1e-12)
        z_now = (w * charges_A).sum()
        if z_now < target_zA:
            lo = mid
        else:
            hi = mid
        if abs(z_now - target_zA) < 1e-4:
            break
    return w, b


def sample_valid(n: int, A_els, B_els, rng, d0_strict: bool = False,
                 max_tries: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Sample n physics-valid (A,B) simplex points using charge projection + rejection."""
    A_out = np.empty((n, len(A_els)))
    B_out = np.empty((n, len(B_els)))
    n_filled = 0
    tries = 0
    alpha_A = 0.6
    alpha_B = 0.8
    while n_filled < n:
        batch = max(200, (n - n_filled) * 4)
        Acand = rng.dirichlet(np.full(len(A_els), alpha_A), size=batch)
        Bcand = rng.dirichlet(np.full(len(B_els), alpha_B), size=batch)
        for i in range(batch):
            if n_filled >= n:
                break
            # Project A-site to satisfy charge balance given B-site
            a_proj, b_proj = project_charge_balance(Acand[i], Bcand[i], A_els, B_els)
            if passes_physics(a_proj, b_proj, A_els, B_els, d0_strict):
                A_out[n_filled] = a_proj
                B_out[n_filled] = b_proj
                n_filled += 1
        tries += 1
        if tries > max_tries:
            alpha_A = min(2.0, alpha_A * 1.5)
            alpha_B = min(2.0, alpha_B * 1.5)
        if tries > 2 * max_tries:
            raise RuntimeError(
                f"Could not sample {n} physics-valid candidates after {tries} batches"
            )
    return A_out, B_out


# ── Batched PCNN inference (lifted from script 64) ───────────────────────────
def _build_feature_row_dict(formula: str) -> dict | None:
    row = pd.Series({"Materials": formula, "ST": None, "CT": None})
    feats = compute_features(row)
    if not feats.get("parse_ok", 0):
        return None
    _nan = float("nan")
    feats["CT"]      = _nan; feats["C_time"]  = _nan
    feats["ST"]      = _nan; feats["ST_time"] = _nan
    er_cm_val = feats.get("er_CM", _nan)
    feats["has_sigma_CM"] = 1.0 if (not math.isnan(er_cm_val) and er_cm_val > 0) else 0.0
    feats["T_sint_x_r_cov"] = _nan
    feats["T_calc_x_r_cov"] = _nan
    feats["ST_norm_melt"]   = _nan
    feats["thermal_budget_sinter"] = _nan
    feats["thermal_budget_calc"]   = _nan
    feats["Ba_x_T_sint_r_cov"] = 0.0
    feats["Ca_x_tol_factor"]   = (feats.get("is_Ca", 0) *
                                  (feats.get("tolerance_factor", 0.0)
                                   if not math.isnan(feats.get("tolerance_factor", _nan)) else 0.0))
    feats["RE_x_chi_A"]        = (feats.get("is_RE", 0) *
                                  (feats.get("chi_A", 0.0)
                                   if not math.isnan(feats.get("chi_A", _nan)) else 0.0))
    feats["Sr_x_alpha_B"]      = (feats.get("is_Sr", 0) *
                                  (feats.get("alpha_B", 0.0)
                                   if not math.isnan(feats.get("alpha_B", _nan)) else 0.0))
    feats["additive_pct"]    = 0.0
    feats["has_additive"]    = 0
    feats["ST_time_missing"] = 1
    feats.setdefault("phase_transition", 0.0)
    feats["ST_time_x_phase"] = 0.0
    feats["ordering_proxy"] = 0.0
    for cs in ["cs_cubic","cs_tetragonal","cs_orthorhombic","cs_rhombohedral",
               "cs_monoclinic","cs_hexagonal","cs_trigonal","cs_perovskite"]:
        feats.setdefault(cs, 0)
    d10, re_b, hd10 = _inf._b_site_flags(formula)
    feats["d10_B_fraction"] = d10
    feats["RE_B_fraction"]  = re_b
    feats["has_d10_B"]      = hd10
    a_pc = feats.get("a_pc", float("nan"))
    gii, bvvs, tbv, di = _inf.calc_bvs_features(formula, a_pc)
    feats["GII"]  = gii  if not math.isnan(gii)  else 0.0
    feats["BVVS"] = bvvs if not math.isnan(bvvs) else 0.0
    feats["tBV"]  = tbv  if not math.isnan(tbv)  else 0.0
    feats["Di"]   = di   if not math.isnan(di)   else 0.0
    return feats


def predict_batch(formulas):
    state = _inf._load()
    lst_cols  = state["lst_cols"]
    tilt_cols = state["tilt_cols"]
    res_cols  = state["res_cols"]
    regime_idx = state["regime_idx"]
    models     = state["models"]

    N = len(formulas)
    n_lst, n_tilt, n_res = len(lst_cols), len(tilt_cols), len(res_cols)
    Xl_raw = np.full((N, n_lst),  np.nan, dtype=np.float32)
    Xt_raw = np.full((N, n_tilt), np.nan, dtype=np.float32)
    Xr_raw = np.full((N, n_res),  np.nan, dtype=np.float32)
    er_cm  = np.full(N, np.nan, dtype=np.float32)
    has_cm = np.zeros(N, dtype=np.float32)
    cm_approx = np.zeros(N, dtype=np.float32)
    gii_n   = np.zeros(N, dtype=np.float32)
    phase_tr = np.zeros(N, dtype=np.float32)
    valid    = np.zeros(N, dtype=bool)

    for i, f in enumerate(formulas):
        feats = _build_feature_row_dict(f)
        if feats is None:
            continue
        valid[i] = True
        for j, c in enumerate(lst_cols):
            v = feats.get(c, np.nan); Xl_raw[i, j] = v if v is not None else np.nan
        for j, c in enumerate(tilt_cols):
            v = feats.get(c, np.nan); Xt_raw[i, j] = v if v is not None else np.nan
        for j, c in enumerate(res_cols):
            v = feats.get(c, np.nan); Xr_raw[i, j] = v if v is not None else np.nan
        er_cm_raw = feats.get("er_CM", np.nan)
        if math.isnan(er_cm_raw) or er_cm_raw <= 0:
            er_cm[i] = 10.0; has_cm[i] = 0.0
        else:
            er_cm[i] = er_cm_raw; has_cm[i] = float(feats.get("has_sigma_CM", 1.0))
        cm_approx[i] = float(feats.get("cm_approx_flag", 0.0))
        gii_raw = float(feats.get("GII", 0.0))
        gii_n[i] = float(np.clip(gii_raw / GII_MAX, 0.0, 1.0))
        phase_tr[i] = float(feats.get("phase_transition", 0.0))

    Xl_s = np.nan_to_num(state["sc_lst"].transform(Xl_raw),  nan=0.0).astype(np.float32)
    Xt_s = np.nan_to_num(state["sc_tilt"].transform(Xt_raw), nan=0.0).astype(np.float32)
    Xr_s = np.nan_to_num(state["sc_res"].transform(Xr_raw),  nan=0.0).astype(np.float32)
    tXl = torch.from_numpy(Xl_s); tXt = torch.from_numpy(Xt_s); tXr = torch.from_numpy(Xr_s)
    t_er_cm = torch.from_numpy(er_cm); t_has_cm = torch.from_numpy(has_cm)
    t_cm_approx = torch.from_numpy(cm_approx); t_gii_n = torch.from_numpy(gii_n)
    t_phase_tr = torch.from_numpy(phase_tr)

    preds = np.zeros((len(models), N), dtype=np.float32)
    with torch.no_grad():
        for k, m in enumerate(models):
            out = m(tXl, tXt, tXr, t_er_cm, t_has_cm, t_cm_approx,
                    t_gii_n, t_phase_tr, regime_idx)
            preds[k] = out["pred"].cpu().numpy()
    er_mean = preds.mean(axis=0); er_std = preds.std(axis=0)
    er_mean[~valid] = np.nan; er_std[~valid] = np.nan
    return er_mean, er_std, valid


# ── GA operators ─────────────────────────────────────────────────────────────
def tournament_select(fitness: np.ndarray, k: int, rng) -> int:
    idx = rng.integers(0, len(fitness), size=k)
    return int(idx[np.argmax(fitness[idx])])


def crossover(parent1, parent2, A_els, B_els, d0_strict, rng,
              max_tries: int = 30):
    """Return one child via either (a) site-swap or (b) convex combination,
    followed by charge-balance projection + physics rejection."""
    a1, b1 = parent1
    a2, b2 = parent2
    for _ in range(max_tries):
        if rng.random() < 0.5:
            child_a = a1.copy(); child_b = b2.copy()
        else:
            lam = rng.uniform(0.2, 0.8)
            child_a = lam * a1 + (1 - lam) * a2
            child_b = lam * b1 + (1 - lam) * b2
        child_a = child_a / max(child_a.sum(), 1e-12)
        child_b = child_b / max(child_b.sum(), 1e-12)
        child_a, child_b = project_charge_balance(child_a, child_b, A_els, B_els)
        if passes_physics(child_a, child_b, A_els, B_els, d0_strict):
            return child_a, child_b
    return a1.copy(), b1.copy()


def mutate(individual, A_els, B_els, d0_strict, rng, max_tries: int = 30):
    a, b = individual
    for _ in range(max_tries):
        a_m = np.clip(a + rng.normal(0, MUT_SIGMA, size=len(a)), 0, None)
        b_m = np.clip(b + rng.normal(0, MUT_SIGMA, size=len(b)), 0, None)
        a_m = a_m / max(a_m.sum(), 1e-12)
        b_m = b_m / max(b_m.sum(), 1e-12)
        a_m, b_m = project_charge_balance(a_m, b_m, A_els, B_els)
        if passes_physics(a_m, b_m, A_els, B_els, d0_strict):
            return a_m, b_m
    return a.copy(), b.copy()


# ── Main GA loop ─────────────────────────────────────────────────────────────
def run_ga(name: str, A_els, B_els, d0_strict: bool,
           pop_size: int = POP_SIZE, n_gen: int = N_GEN,
           seed: int = 7, time_budget_sec: float = 1500.0,
           log_writer=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    print(f"\n=== GA [{name}] | A={A_els} | B={B_els} | d0_strict={d0_strict} ===")
    print(f"  pop={pop_size}  gen={n_gen}  elite={ELITE}  "
          f"x={CROSS_RATE}  mut={MUT_RATE}  time_budget={time_budget_sec/60:.0f} min")

    # Initial population (physics-valid Dirichlet samples)
    t0 = time.time()
    A_init, B_init = sample_valid(pop_size, A_els, B_els, rng, d0_strict=d0_strict)
    pop = [(A_init[i], B_init[i]) for i in range(pop_size)]
    print(f"  Initial population sampled in {time.time()-t0:.1f}s")

    convergence = {"best_fit": [], "mean_fit": [], "best_er": [], "in_ad_count": [], "gen_time_s": []}
    history = []  # list of (gen, idx, formula, er, sigma, fitness)
    all_seen = []  # cumulative for final population

    for gen in range(n_gen):
        t_gen = time.time()
        formulas = [vec_to_formula(a, b, A_els, B_els) for (a, b) in pop]
        ers, sigs, valid = predict_batch(formulas)
        # Replace invalids with -inf fitness
        ers_safe = np.where(valid, ers, -1e9)
        sigs_safe = np.where(valid, sigs, 1e6)
        fitness = ers_safe - SIGMA_PENALTY * sigs_safe
        in_ad = (sigs_safe <= SIGMA_AD_CUTOFF) & valid

        # Log every individual
        for j in range(pop_size):
            row = {
                "ga": name, "generation": gen, "individual": j,
                "formula": formulas[j],
                "predicted_er": float(ers_safe[j]) if valid[j] else float("nan"),
                "sigma_conf": float(sigs_safe[j]) if valid[j] else float("nan"),
                "fitness": float(fitness[j]) if valid[j] else float("nan"),
                "in_ad": bool(in_ad[j]),
                "tolerance_factor": float(tolerance_factor(pop[j][0], pop[j][1], A_els, B_els)),
                "d0_fraction": float(d0_fraction(pop[j][1], B_els)),
                "charge_residual": float(charge_balance_residual(pop[j][0], pop[j][1], A_els, B_els)),
            }
            history.append(row)
            all_seen.append({**row,
                             "a_frac": pop[j][0].tolist(),
                             "b_frac": pop[j][1].tolist()})
            if log_writer is not None:
                log_writer.writerow(row)

        # Stats
        valid_fit = fitness[valid]
        if len(valid_fit):
            convergence["best_fit"].append(float(valid_fit.max()))
            convergence["mean_fit"].append(float(valid_fit.mean()))
            convergence["best_er"].append(float(ers_safe[valid].max()))
        else:
            convergence["best_fit"].append(-1e9)
            convergence["mean_fit"].append(-1e9)
            convergence["best_er"].append(-1e9)
        convergence["in_ad_count"].append(int(in_ad.sum()))

        # Selection / elitism
        order = np.argsort(-fitness)  # descending
        elite = [pop[i] for i in order[:ELITE]]
        new_pop = list(elite)

        # Children via crossover + mutation
        n_immigrants = max(1, int(IMMIGRATION_RATE * pop_size))
        while len(new_pop) < pop_size - n_immigrants:
            i1 = tournament_select(fitness, TOURN_K, rng)
            i2 = tournament_select(fitness, TOURN_K, rng)
            p1, p2 = pop[i1], pop[i2]
            if rng.random() < CROSS_RATE:
                child = crossover(p1, p2, A_els, B_els, d0_strict, rng)
            else:
                child = (p1[0].copy(), p1[1].copy())
            if rng.random() < MUT_RATE:
                child = mutate(child, A_els, B_els, d0_strict, rng)
            new_pop.append(child)
        # Diversity injection: re-seed bottom n_immigrants with fresh random valid individuals
        if n_immigrants > 0:
            A_imm, B_imm = sample_valid(n_immigrants, A_els, B_els, rng, d0_strict=d0_strict)
            for ii in range(n_immigrants):
                new_pop.append((A_imm[ii], B_imm[ii]))

        pop = new_pop[:pop_size]
        gen_t = time.time() - t_gen
        convergence["gen_time_s"].append(gen_t)
        if (gen + 1) % 5 == 0 or gen == 0:
            print(f"  gen {gen+1:3d}/{n_gen} | best_fit={convergence['best_fit'][-1]:7.2f}  "
                  f"best_er={convergence['best_er'][-1]:6.1f}  "
                  f"in_AD={convergence['in_ad_count'][-1]:3d}/{pop_size}  "
                  f"({gen_t:.1f}s)")
        if time.time() - t0 > time_budget_sec:
            print(f"  [{name}] time budget exceeded at gen {gen+1} → stopping early")
            break

    wall = time.time() - t0
    print(f"  [{name}] finished in {wall/60:.1f} min ({len(history)} evaluations)")

    # Final population: evaluate one more time for snapshot
    formulas = [vec_to_formula(a, b, A_els, B_els) for (a, b) in pop]
    ers, sigs, valid = predict_batch(formulas)
    final_pop = []
    for j in range(pop_size):
        if not valid[j]:
            continue
        a_pop, b_pop = pop[j]
        a_dict = {el: float(a_pop[i]) for i, el in enumerate(A_els) if a_pop[i] > 1e-3}
        b_dict = {el: float(b_pop[i]) for i, el in enumerate(B_els) if b_pop[i] > 1e-3}
        final_pop.append({
            "formula": formulas[j],
            "predicted_er": float(ers[j]),
            "sigma_conf": float(sigs[j]),
            "fitness": float(ers[j] - SIGMA_PENALTY * sigs[j]),
            "lower_90": float(ers[j] - Q90),
            "upper_90": float(ers[j] + Q90),
            "in_ad": bool(sigs[j] <= SIGMA_AD_CUTOFF),
            "tolerance_factor": float(tolerance_factor(a_pop, b_pop, A_els, B_els)),
            "d0_fraction": float(d0_fraction(b_pop, B_els)),
            "a_dominant": asite_family(a_dict),
            "a_frac": a_dict, "b_frac": b_dict,
        })

    return {
        "name": name,
        "A_elements": A_els,
        "B_elements": B_els,
        "d0_strict": d0_strict,
        "wall_clock_min": wall / 60,
        "n_generations_completed": len(convergence["best_fit"]),
        "convergence": convergence,
        "final_population": final_pop,
        "history": history,
    }


# ── Save & report ────────────────────────────────────────────────────────────
def save_results(name: str, out: dict, path: str):
    # Only persist what's JSON-friendly
    persistable = {
        "name": out["name"],
        "A_elements": out["A_elements"],
        "B_elements": out["B_elements"],
        "d0_strict": out["d0_strict"],
        "wall_clock_min": out["wall_clock_min"],
        "n_generations_completed": out["n_generations_completed"],
        "convergence": out["convergence"],
        "final_population": out["final_population"],
    }
    # Top-10 by fitness from final population
    fp = out["final_population"]
    fp_sorted = sorted(fp, key=lambda r: -r["fitness"])
    persistable["top10"] = fp_sorted[:10]
    # Top-50 best-ever from history
    hist = out["history"]
    hist_clean = [h for h in hist if h["predicted_er"] is not None
                  and not math.isnan(h["predicted_er"])]
    hist_clean.sort(key=lambda r: -r["fitness"])
    # Deduplicate by formula
    seen = set(); dedup = []
    for h in hist_clean:
        k = h["formula"]
        if k in seen: continue
        seen.add(k); dedup.append(h)
        if len(dedup) >= 50: break
    persistable["top50_history"] = dedup
    with open(path, "w") as f:
        json.dump(persistable, f, indent=2, default=float)
    print(f"  saved → {path}")


def main():
    grand_t0 = time.time()
    log_path = os.path.join(RES_DIR, "67_ga_log.csv")
    log_cols = ["ga", "generation", "individual", "formula",
                "predicted_er", "sigma_conf", "fitness", "in_ad",
                "tolerance_factor", "d0_fraction", "charge_residual"]
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_cols)
    log_writer.writeheader()

    runs = {}

    # GA-Pb-allowed (full search space, time-cap 1500s)
    A_pb = A_ELEMENTS_FULL
    B_pb = B_ELEMENTS_FULL
    runs["pb_allowed"] = run_ga(
        "pb_allowed", A_pb, B_pb, d0_strict=False,
        time_budget_sec=1500.0, log_writer=log_writer, seed=7,
    )
    save_results("pb_allowed", runs["pb_allowed"],
                 os.path.join(RES_DIR, "67_ga_pb_allowed.json"))

    # GA-Pb-free (Pb removed from A-site)
    A_pbfree = [el for el in A_ELEMENTS_FULL if el != "Pb"]
    runs["pb_free"] = run_ga(
        "pb_free", A_pbfree, B_pb, d0_strict=False,
        time_budget_sec=1500.0, log_writer=log_writer, seed=11,
    )
    save_results("pb_free", runs["pb_free"],
                 os.path.join(RES_DIR, "67_ga_pb_free.json"))

    # GA-d⁰-strict (B-site Ti/Nb/Ta/Zr only — full A-site)
    B_d0 = [el for el in B_ELEMENTS_FULL if el in D0_BSITE]
    runs["d0_strict"] = run_ga(
        "d0_strict", A_pb, B_d0, d0_strict=True,
        time_budget_sec=1500.0, log_writer=log_writer, seed=13,
    )
    save_results("d0_strict", runs["d0_strict"],
                 os.path.join(RES_DIR, "67_ga_d0_strict.json"))

    log_file.close()
    print(f"\nLog file: {log_path}")

    # Final summary
    total_min = (time.time() - grand_t0) / 60
    print(f"\n=== GA summary  (total wall {total_min:.1f} min) ===")
    for name, out in runs.items():
        fp = out["final_population"]
        if not fp:
            print(f"  {name}: NO valid individuals in final population")
            continue
        ers = [r["predicted_er"] for r in fp]
        sigs = [r["sigma_conf"] for r in fp]
        in_ad = sum(1 for r in fp if r["in_ad"])
        families = {}
        for r in fp:
            fam = r["a_dominant"]
            families[fam] = families.get(fam, 0) + 1
        print(f"  [{name}] wall={out['wall_clock_min']:.1f} min "
              f"| max_εr={max(ers):.1f}  mean_εr={np.mean(ers):.1f}  "
              f"in_AD={in_ad}/{len(fp)}  families={dict(sorted(families.items(), key=lambda kv: -kv[1]))}")
        best = sorted(fp, key=lambda r: -r["fitness"])[:5]
        for i, c in enumerate(best):
            print(f"     top{i+1}: {c['formula'][:60]:60s} "
                  f"εr={c['predicted_er']:6.1f}  σ={c['sigma_conf']:.2f}  "
                  f"in_AD={c['in_ad']}  A-dom={c['a_dominant']}")


if __name__ == "__main__":
    main()
