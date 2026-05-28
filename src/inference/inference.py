import os
RELEASE_ROOT = os.path.abspath(os.path.dirname(__file__))
while RELEASE_ROOT != os.path.dirname(RELEASE_ROOT) and not os.path.isdir(os.path.join(RELEASE_ROOT,'model_weights')):
    RELEASE_ROOT = os.path.dirname(RELEASE_ROOT)
"""
εxplorer inference engine — single-composition prediction for ABO₃ dielectric constant.

Strategy:
  1. If the formula matches a row in the training feature matrix → use the exact stored
     features (same as what the model saw during training → accurate predictions).
  2. Otherwise → compute all features from scratch including GII/BVVS/tBV/Di via
     Bond-Valence Sum theory (Brown & Altermatt 1985, Li et al. 2021 Chem. Mater.).
"""
import sys, os, json, math, pickle, warnings
warnings.filterwarnings("ignore")

# ── Resolve project paths ─────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))
_MCODE = os.path.join(_HERE, "..", "model_code")
if _MCODE not in sys.path:
    sys.path.insert(0, _MCODE)

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from feature_engineering import (compute_features, normalize_formula, parse_formula,
                                 BVS_R0_A, A_CHARGE, B_CHARGE)
from model_CMLTRPCNNv77 import CMLTRPCNN

# ── Constants ─────────────────────────────────────────────────────────────────
ASSETS      = os.path.join(RELEASE_ROOT, "model_weights")
FM_PATH     = os.path.join(RELEASE_ROOT, "data", "processed", "feature_matrix_v7.parquet")
DEVICE      = "cpu"
Q90         = 14.38   # 90% conformal quantile (calibrated on n=116 holdout)
GII_MAX     = 1.166   # 95th pct of non-zero GII in training set (n=1188)

D10_ELEMENTS = {"Zn","Cd","Hg","Ga","In","Tl","Cu","Ag","Au","Ge","Sn","Pb"}
RE_ELEMENTS  = {"La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Y","Sc"}

# B-site BVS R0 parameters — Brown & Altermatt (1985), 6-fold O coordination
BVS_B_PARAM = 0.37
BVS_R0_B = {
    "Ti":1.815,"Zr":1.928,"Nb":1.911,"Ta":1.920,"Hf":1.923,"Sn":1.905,
    "W":1.917,"Mo":1.907,"V":1.803,"Mg":1.636,"Zn":1.704,"Al":1.651,
    "Fe":1.759,"Mn":1.760,"Co":1.692,"Ni":1.654,"Cu":1.679,"Cr":1.724,
    "Sc":1.849,"In":1.902,"Ga":1.730,"Ge":1.739,"Si":1.624,"Ru":1.896,
    "Ir":1.879,"Re":1.910,"Os":1.911,
    "La":2.172,"Ce":2.152,"Pr":2.130,"Nd":2.105,"Sm":2.068,"Eu":2.054,
    "Gd":2.054,"Tb":2.038,"Dy":2.021,"Ho":2.007,"Er":1.994,"Yb":1.965,
    "Lu":1.958,"Y":2.014,
}


def calc_bvs_features(formula_str: str, a_pc: float):
    """
    Compute GII, BVVS, tBV, Di from formula + pseudo-cubic lattice parameter.
    Reference: Li et al. (2021) Chem. Mater.; Brown & Altermatt (1985).
    Returns (gii, bvvs, tbv, di) or (nan, nan, nan, nan) on failure.
    """
    nan4 = (float("nan"),) * 4
    if math.isnan(a_pc) or a_pc <= 0:
        return nan4
    try:
        norm   = normalize_formula(formula_str)
        a_site, b_site, _, _, _ = parse_formula(norm)
    except Exception:
        return nan4
    if not a_site or not b_site:
        return nan4

    a_tot = sum(a_site.values())
    b_tot = sum(b_site.values())

    R_AO = a_pc / math.sqrt(2)
    R_BO = a_pc / 2.0

    # A-site BVS
    if not all(el in BVS_R0_A for el in a_site):
        return nan4
    bvs_a = sum((n/a_tot)*12*math.exp((BVS_R0_A[el]-R_AO)/BVS_B_PARAM)
                for el, n in a_site.items())
    v_a = sum((n/a_tot)*A_CHARGE.get(el, float("nan")) for el, n in a_site.items())
    if math.isnan(v_a):
        return nan4

    # B-site BVS
    for el in b_site:
        if el not in BVS_R0_B and el not in BVS_R0_A:
            return nan4
    bvs_b = sum((n/b_tot)*6*math.exp((BVS_R0_B.get(el, BVS_R0_A.get(el, 0.0))-R_BO)/BVS_B_PARAM)
                for el, n in b_site.items())
    v_b_vals = [B_CHARGE.get(el, float("nan")) for el in b_site]
    if any(math.isnan(v) for v in v_b_vals):
        return nan4
    v_b = sum((n/b_tot)*B_CHARGE[el] for el, n in b_site.items())

    d_a = bvs_a - v_a
    d_b = bvs_b - v_b

    gii  = math.sqrt((d_a**2 + d_b**2) / 2)
    bvvs = d_a - d_b

    # tBV
    r0_a_avg = sum((n/a_tot)*BVS_R0_A[el] for el, n in a_site.items())
    r0_b_avg = sum((n/b_tot)*BVS_R0_B.get(el, BVS_R0_A.get(el, 0))
                   for el, n in b_site.items())
    v_a_cn, v_b_cn = v_a/12, v_b/6
    if v_a_cn <= 0 or v_b_cn <= 0 or r0_b_avg <= 0:
        return nan4
    r_ao_ideal = r0_a_avg - BVS_B_PARAM * math.log(v_a_cn)
    r_bo_ideal = r0_b_avg - BVS_B_PARAM * math.log(v_b_cn)
    if r_bo_ideal <= 0:
        return nan4
    tbv = r_ao_ideal / (math.sqrt(2) * r_bo_ideal)

    # Di
    if len(b_site) == 1:
        di = 0.0
    else:
        bvs_per_el = [6*math.exp((BVS_R0_B.get(el, BVS_R0_A.get(el, float("nan")))-R_BO)/BVS_B_PARAM)
                      for el in b_site]
        if any(math.isnan(v) for v in bvs_per_el):
            di = float("nan")
        else:
            mu = sum(bvs_per_el) / len(bvs_per_el)
            di = math.sqrt(sum((v-mu)**2 for v in bvs_per_el)/len(bvs_per_el)) / (mu + 1e-8)

    return gii, bvvs, tbv, di

_CACHE: dict = {}


# ── Model loading ─────────────────────────────────────────────────────────────
def _load():
    if _CACHE:
        return _CACHE

    # Model checkpoint
    ck = torch.load(os.path.join(ASSETS, "cmltrv77_final.pt"),
                    map_location="cpu", weights_only=False)
    hp = ck["hp"]
    si = ck["scaler_info"]

    # Scalers
    with open(os.path.join(ASSETS, "cmltrv77_scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)

    lst_cols  = si["lst_cols"]
    tilt_cols = si["tilt_cols"]
    res_cols  = si["res_cols"]
    n_lst, n_tilt, n_res = len(lst_cols), len(tilt_cols), len(res_cols)

    # Regime one-hot indices inside lst_cols (needed by model forward)
    regime_idx = [lst_cols.index(c) for c in
                  ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
                  if c in lst_cols]

    # Build 5-seed ensemble
    models = []
    for state in ck["models"]:
        m = CMLTRPCNN(
            n_lst=n_lst, n_tilt=n_tilt, n_res=n_res,
            trunk_hidden=hp["trunk_hidden"],
            n_trunk_blocks=hp["n_trunk_blocks"],
            lst_hidden=hp["lst_hidden"],
            tilt_hidden=hp["tilt_hidden"],
            res_hidden=hp["res_hidden"],
            residual_scale=hp["residual_scale"],
            dropout=0.0,
        )
        m.load_state_dict(state)
        m.eval()
        models.append(m)

    # Feature matrix for dataset lookup
    fm = pd.read_parquet(FM_PATH) if os.path.exists(FM_PATH) else None
    # Build formula → list of row indices (some formulas appear multiple times
    # with different sintering conditions — keep all so lookup can pick best)
    fm_index: dict = {}
    if fm is not None and "formula" in fm.columns:
        for idx, fml in enumerate(fm["formula"]):
            key = str(fml)
            fm_index.setdefault(key, []).append(idx)

    _CACHE.update(dict(
        models=models,
        sc_lst=scalers["sc_lst"],
        sc_tilt=scalers["sc_tilt"],
        sc_res=scalers["sc_res"],
        lst_cols=lst_cols,
        tilt_cols=tilt_cols,
        res_cols=res_cols,
        regime_idx=regime_idx,
        fm=fm,
        fm_index=fm_index,
    ))
    return _CACHE


# ── Feature helpers ───────────────────────────────────────────────────────────
def _b_site_flags(formula_str: str):
    try:
        norm = normalize_formula(formula_str)
        _, b_site, _, _, _ = parse_formula(norm)
        if not b_site:
            return 0.0, 0.0, 0.0
        b_tot = sum(b_site.values())
        d10 = sum(n for el, n in b_site.items() if el in D10_ELEMENTS)
        re  = sum(n for el, n in b_site.items() if el in RE_ELEMENTS)
        return float(d10/b_tot), float(re/b_tot), float(d10 > 0)
    except Exception:
        return 0.0, 0.0, 0.0


def _regime_label(t: float, soft_proxy: bool) -> str:
    if t is None or math.isnan(t):
        return "Unknown"
    if t > 0.985:
        return "Regime Ib — incipient ferroelectric" if soft_proxy \
               else "Regime Ia — paraelectric (untilted)"
    if t > 0.965:
        return "Regime II — antiphase tilts"
    return "Regime III — antiphase + in-phase tilts"


# ── Feature source 1: dataset lookup ─────────────────────────────────────────
def _features_from_matrix(norm_formula: str, state: dict,
                           st: float, ct: float) -> dict | None:
    """
    Return feature dict using the pre-computed row from feature_matrix_v7.parquet.
    When multiple rows exist for the same formula (different sintering conditions):
      - If user provides ST: pick the row whose T_sint_x_r_cov implies the closest ST
      - Otherwise: pick the row with the median index (most representative)
    Overrides ST/CT with user-supplied values if given.
    Returns None if formula not found.
    """
    fm_index = state["fm_index"]
    fm       = state["fm"]
    if fm is None or norm_formula not in fm_index:
        return None

    indices = fm_index[norm_formula]

    if len(indices) == 1:
        row = fm.iloc[indices[0]]
    elif st is not None:
        # Multiple rows = same composition sintered at different temperatures.
        # Recover each row's ST from T_sint_x_r_cov / r_cov, pick closest to user ST.
        best_idx = indices[len(indices) // 2]
        best_diff = float("inf")
        for idx in indices:
            r = fm.iloc[idx]
            tsrc = float(r.get("T_sint_x_r_cov", float("nan")))
            rcov = float(r.get("r_cov", float("nan")))
            if not math.isnan(tsrc) and not math.isnan(rcov) and rcov > 0:
                st_est = tsrc / rcov
                diff = abs(st_est - st)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
        row = fm.iloc[best_idx]
    else:
        # No ST given — use the middle row (median sintering condition)
        row = fm.iloc[indices[len(indices) // 2]]
    feats = row.to_dict()
    feats["_source"] = "dataset"

    # Override sintering/calcination if user gave them
    if st is not None:
        r_cov = feats.get("r_cov", float("nan"))
        feats["ST"] = st
        feats["T_sint_x_r_cov"] = st * r_cov if not math.isnan(r_cov) else feats.get("T_sint_x_r_cov", float("nan"))
        feats["Ba_x_T_sint_r_cov"] = feats.get("is_Ba", 0) * feats["T_sint_x_r_cov"]
        b_ent = feats.get("b_config_entropy", 0.0)
        feats["ordering_proxy"] = b_ent * (st / 1200.0)
    if ct is not None:
        r_cov = feats.get("r_cov", float("nan"))
        feats["CT"] = ct
        feats["T_calc_x_r_cov"] = ct * r_cov if not math.isnan(r_cov) else feats.get("T_calc_x_r_cov", float("nan"))

    return feats


# ── Feature source 2: compute from scratch ───────────────────────────────────
def _features_computed(formula: str, st: float, ct: float) -> dict:
    row = pd.Series({"Materials": formula, "ST": st, "CT": ct})
    feats = compute_features(row)

    if not feats.get("parse_ok", 0):
        return feats   # caller checks parse_ok

    r_cov = feats.get("r_cov", float("nan"))
    _nan = float("nan")
    _safe = lambda v: v if v is not None and not math.isnan(v) else None

    feats["CT"]      = ct if ct is not None else _nan
    feats["C_time"]  = _nan
    feats["ST"]      = st if st is not None else _nan
    feats["ST_time"] = _nan

    # has_sigma_CM: 1 when Clausius-Mossotti is computable (er_CM valid and > 0)
    er_cm_val = feats.get("er_CM", _nan)
    feats["has_sigma_CM"] = 1.0 if (not math.isnan(er_cm_val) and er_cm_val > 0) else 0.0

    feats["T_sint_x_r_cov"] = (st * r_cov) if (_safe(st) and _safe(r_cov)) else _nan
    feats["T_calc_x_r_cov"] = (ct * r_cov) if (_safe(ct) and _safe(r_cov)) else _nan
    feats["ST_norm_melt"]   = _nan
    feats["thermal_budget_sinter"] = _nan
    feats["thermal_budget_calc"]   = _nan

    is_Ba, is_Ca = feats.get("is_Ba", 0), feats.get("is_Ca", 0)
    is_Sr, is_RE = feats.get("is_Sr", 0), feats.get("is_RE", 0)
    t_f   = feats.get("tolerance_factor", _nan)
    chi_A = feats.get("chi_A",  _nan)
    alp_B = feats.get("alpha_B", _nan)
    tsrc  = feats.get("T_sint_x_r_cov", _nan)

    feats["Ba_x_T_sint_r_cov"] = is_Ba * (tsrc  if not math.isnan(tsrc)  else 0.0)
    feats["Ca_x_tol_factor"]   = is_Ca * (t_f   if not math.isnan(t_f)   else 0.0)
    feats["RE_x_chi_A"]        = is_RE * (chi_A if not math.isnan(chi_A) else 0.0)
    feats["Sr_x_alpha_B"]      = is_Sr * (alp_B if not math.isnan(alp_B) else 0.0)

    feats["additive_pct"]    = 0.0
    feats["has_additive"]    = 0
    feats["ST_time_missing"] = 1
    feats.setdefault("phase_transition", 0.0)
    feats["ST_time_x_phase"] = 0.0

    b_ent = feats.get("b_config_entropy", 0.0)
    feats["ordering_proxy"] = (b_ent * (st / 1200.0)) if _safe(st) else 0.0

    for cs in ["cs_cubic","cs_tetragonal","cs_orthorhombic","cs_rhombohedral",
               "cs_monoclinic","cs_hexagonal","cs_trigonal","cs_perovskite"]:
        feats.setdefault(cs, 0)

    d10, re_b, hd10 = _b_site_flags(formula)
    feats["d10_B_fraction"] = d10
    feats["RE_B_fraction"]  = re_b
    feats["has_d10_B"]      = hd10

    # Bond-valence features — computed from formula + pseudo-cubic a_pc
    a_pc = feats.get("a_pc", float("nan"))
    gii, bvvs, tbv, di = calc_bvs_features(formula, a_pc)
    feats["GII"]  = gii  if not math.isnan(gii)  else 0.0
    feats["BVVS"] = bvvs if not math.isnan(bvvs) else 0.0
    feats["tBV"]  = tbv  if not math.isnan(tbv)  else 0.0
    feats["Di"]   = di   if not math.isnan(di)   else 0.0

    feats["_source"] = "computed"
    return feats


# ── Main prediction function ──────────────────────────────────────────────────
def predict(formula: str, st: float = None, ct: float = None) -> dict:
    """
    Predict εr for a single ABO₃ composition.

    formula : e.g. "BaTiO3", "Ba0.5Sr0.5TiO3", "BaSr2YNb3O12"
    st      : sintering temperature °C  (optional)
    ct      : calcination temperature °C (optional)

    Returns dict with: pred, lower_90, upper_90, seed_std,
                       er_cm, delta_lst, delta_tilt, delta_res,
                       tolerance_factor, regime, parse_ok, source
    """
    state = _load()
    norm  = normalize_formula(formula.strip())

    # ── 1. Get features (dataset lookup preferred) ────────────────────────────
    feats = _features_from_matrix(norm, state, st, ct)
    if feats is None:
        feats = _features_computed(formula.strip(), st, ct)

    if not feats.get("parse_ok", feats.get("_source") == "dataset"):
        return {"error": f"Could not parse formula '{formula}'", "parse_ok": False}

    source = feats.get("_source", "computed")

    # ── 2. Assemble feature arrays ────────────────────────────────────────────
    def _arr(cols):
        return np.array([feats.get(c, float("nan")) for c in cols],
                        dtype=np.float32).reshape(1, -1)

    Xl = _arr(state["lst_cols"])
    Xt = _arr(state["tilt_cols"])
    Xr = _arr(state["res_cols"])

    Xl_s = np.nan_to_num(state["sc_lst"].transform(Xl),  nan=0.0)
    Xt_s = np.nan_to_num(state["sc_tilt"].transform(Xt), nan=0.0)
    Xr_s = np.nan_to_num(state["sc_res"].transform(Xr),  nan=0.0)

    er_cm_raw = feats.get("er_CM", float("nan"))
    if math.isnan(er_cm_raw) or er_cm_raw <= 0:
        er_cm_raw = 10.0   # fallback for unparseable CM
        has_cm_val = 0.0
        cm_valid = False
    else:
        has_cm_val = float(feats.get("has_sigma_CM", 1.0))
        cm_valid = True

    er_cm    = np.array([er_cm_raw],                               dtype=np.float32)
    has_cm   = np.array([has_cm_val],                              dtype=np.float32)
    cm_approx = np.array([feats.get("cm_approx_flag", 0.0)],       dtype=np.float32)
    gii_raw  = float(feats.get("GII", 0.0))
    gii_n    = np.array([np.clip(gii_raw / GII_MAX, 0.0, 1.0)],   dtype=np.float32)
    phase_tr = np.array([feats.get("phase_transition", 0.0)],      dtype=np.float32)

    # ── 3. Ensemble inference ─────────────────────────────────────────────────
    preds, ecm_l, dlst_l, dtilt_l, dres_l = [], [], [], [], []
    with torch.no_grad():
        for m in state["models"]:
            out = m(
                torch.tensor(Xl_s), torch.tensor(Xt_s), torch.tensor(Xr_s),
                torch.tensor(er_cm), torch.tensor(has_cm), torch.tensor(cm_approx),
                torch.tensor(gii_n), torch.tensor(phase_tr),
                state["regime_idx"],
            )
            preds.append(out["pred"].item())
            ecm_l.append(out["er_cm"].item())
            dlst_l.append(out["delta_lst"].item())
            dtilt_l.append(out["delta_tilt"].item())
            dres_l.append(out["delta_res"].item())

    pred     = float(np.mean(preds))
    seed_std = float(np.std(preds))

    # Guard against NaN in components (e.g. from NaN features slipping through)
    def _safe_mean(lst):
        v = np.mean([x for x in lst if not math.isnan(x)])
        return float(v) if not math.isnan(v) else 0.0

    er_cm_m  = _safe_mean(ecm_l)
    dlst_m   = _safe_mean(dlst_l)
    dtilt_m  = _safe_mean(dtilt_l)
    dres_m   = _safe_mean(dres_l)

    if math.isnan(pred):
        return {"error": "Prediction produced NaN — check formula or try simplified stoichiometry.",
                "parse_ok": False}

    t    = feats.get("tolerance_factor", float("nan"))
    soft = bool(feats.get("soft_mode_proxy", 0))

    return {
        "parse_ok":         True,
        "pred":             round(pred, 1),
        "lower_90":         round(max(1.0, pred - Q90), 1),
        "upper_90":         round(pred + Q90, 1),
        "seed_std":         round(seed_std, 2),
        "er_cm":            round(er_cm_m,  1),
        "delta_lst":        round(dlst_m,   1),
        "delta_tilt":       round(dtilt_m,  1),
        "delta_res":        round(dres_m,   1),
        "tolerance_factor": round(t, 4) if not math.isnan(t) else None,
        "regime":           _regime_label(t if not math.isnan(t) else float("nan"), soft),
        "soft_mode_proxy":  soft,
        "source":           source,   # "dataset" or "computed"
        "cm_valid":         cm_valid, # False when CM baseline could not be computed
    }
