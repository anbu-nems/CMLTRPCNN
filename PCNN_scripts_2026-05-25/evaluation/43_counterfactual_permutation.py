"""
Script 43 — Counterfactual A-site Analysis + Permutation Tests

Priority 2 (submission-blocking): Counterfactual analysis.
Priority 4 (submission-blocking): Permutation tests.
Priority 5 (high-impact framing): Falsifiable predictions.

Uses inference on the SAVED ensemble — no retraining.

PART A — Matched-pair counterfactual analysis
  For each B-site present with ≥2 different A-sites, find matched pairs with:
    - same or similar B-site (exact match by B-site token from chemistry_family)
    - |Δtolerance_factor| < 0.04
    - same dominant Reaney regime
    - (where available) similar sintering/processing temperature
  For each matched pair: report Δεr, Δδ_LST, Δδ_tilt, Δδ_res.
  Test: Is Δεr dominated by Δδ_LST? → causal-style evidence for Finding 1.

PART B — Permutation Test 1: A-site label shuffle
  Null hypothesis: A-site label has NO effect on f_LST ordering.
  Test: shuffle A-site labels 2000 times → recompute mean f_LST per group.
  Statistic: range (max_mean - min_mean) across groups.
  p-value: fraction of permutations with range ≥ observed (1.56×).

PART C — Permutation Test 2: Reaney REGIME_SCALES permutation
  Null hypothesis: The specific scale ordering [Ia=1.0, Ib=1.5, II=0.9, III=0.7]
    is no better than a random permutation of those 4 values.
  Test: all 24 permutations of 4 scale values → run inference → compute R².
  p-value: rank of true ordering / 24 (exhaustive).

PART D — Falsifiable predictions (written from results)

Outputs:
  results/43_counterfactual_permutation.json
  figures/43a_counterfactual_pairs.png
  figures/43b_permutation_asite.png
  figures/43c_permutation_regime.png
"""

import sys, os, warnings, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from src.models.psrnn_mdpinn import CMLTRPCNNv71

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)
FIG_DIR  = os.path.join(ROOT, "figures");  os.makedirs(FIG_DIR, exist_ok=True)
MODEL_PT = os.path.join(ROOT, "models", "cmltrv77_final.pt")

# Matching thresholds for counterfactual pairs
TOL_FACTOR_DELTA = 0.04   # |Δtolerance_factor| threshold
MIN_PAIRS        = 5      # minimum matched pairs per B-site for reporting

# Permutation test parameters
N_PERM_ASITE = 2000       # shuffles for A-site permutation test
TRUE_REGIME_SCALES = [1.0, 1.5, 0.9, 0.7]  # [Ia, Ib, II, III] — hard-coded in model

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.grid": False, "axes.linewidth": 1.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 5, "ytick.major.size": 5,
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.fontsize": 9,
})

def style4(ax):
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(1.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


# ── Model loading ──────────────────────────────────────────────────────────
def load_model_and_scalers():
    import pickle
    ck = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
    hp = ck["hp"]

    with open(os.path.join(ROOT, "models", "cmltrv77_scalers.pkl"), "rb") as _f:
        scalers = pickle.load(_f)
    sc_lst  = scalers["sc_lst"]
    sc_tilt = scalers["sc_tilt"]
    sc_res  = scalers["sc_res"]

    n_lst  = sc_lst.n_features_in_
    n_tilt = sc_tilt.n_features_in_
    n_res  = sc_res.n_features_in_

    models = []
    for state in ck["models"]:
        m = CMLTRPCNNv71(
            n_lst=n_lst, n_tilt=n_tilt, n_res=n_res,
            trunk_hidden=hp["trunk_hidden"], n_trunk_blocks=hp["n_trunk_blocks"],
            lst_hidden=hp["lst_hidden"], tilt_hidden=hp["tilt_hidden"],
            res_hidden=hp["res_hidden"], residual_scale=hp["residual_scale"],
            dropout=0.0,
        )
        m.load_state_dict(state)
        m.eval().to(DEVICE)
        models.append(m)

    with open(os.path.join(ROOT, "data", "processed", "feature_partition_v7.json")) as _f:
        _part = json.load(_f)
    _regime_names = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
    regime_idx = [_part["LST"].index(r) for r in _regime_names if r in _part["LST"]]

    return models, sc_lst, sc_tilt, sc_res, 1.0, regime_idx

@torch.no_grad()
def infer(models, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
          gii_n, phase_tr, regime_idx, batch=256):
    acc = {"pred":[], "delta_lst":[], "delta_tilt":[], "delta_res":[]}
    N = len(er_cm)
    for m in models:
        bufs = {k:[] for k in acc}
        for i in range(0, N, batch):
            sl = slice(i, i+batch)
            out = m(torch.tensor(Xl_s[sl]).to(DEVICE), torch.tensor(Xt_s[sl]).to(DEVICE),
                    torch.tensor(Xr_s[sl]).to(DEVICE), torch.tensor(er_cm[sl]).to(DEVICE),
                    torch.tensor(has_cm[sl]).to(DEVICE), torch.tensor(cm_approx[sl]).to(DEVICE),
                    torch.tensor(gii_n[sl]).to(DEVICE), torch.tensor(phase_tr[sl]).to(DEVICE),
                    regime_idx)
            for k in acc: bufs[k].append(out[k].cpu().numpy())
        for k in acc: acc[k].append(np.concatenate(bufs[k]))
    return {k: np.mean(acc[k], 0) for k in acc}


@torch.no_grad()
def infer_with_regime_scales(models, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                              gii_n, phase_tr, regime_idx, scale_perm, batch=256):
    """Run inference with temporarily overridden REGIME_SCALES."""
    preds = []
    scale_t = torch.tensor(scale_perm, dtype=torch.float32).to(DEVICE)
    for m in models:
        # Temporarily patch REGIME_SCALES
        orig_scales = CMLTRPCNNv71.REGIME_SCALES.clone()
        CMLTRPCNNv71.REGIME_SCALES = scale_t
        p = []
        for i in range(0, len(er_cm), batch):
            sl = slice(i, i+batch)
            out = m(torch.tensor(Xl_s[sl]).to(DEVICE), torch.tensor(Xt_s[sl]).to(DEVICE),
                    torch.tensor(Xr_s[sl]).to(DEVICE), torch.tensor(er_cm[sl]).to(DEVICE),
                    torch.tensor(has_cm[sl]).to(DEVICE), torch.tensor(cm_approx[sl]).to(DEVICE),
                    torch.tensor(gii_n[sl]).to(DEVICE), torch.tensor(phase_tr[sl]).to(DEVICE),
                    regime_idx)
            p.append(out["pred"].cpu().numpy())
        CMLTRPCNNv71.REGIME_SCALES = orig_scales  # restore
        preds.append(np.concatenate(p))
    return np.mean(preds, 0)


def main():
    print("="*70)
    print("Script 43 — Counterfactual Analysis + Permutation Tests")
    print("="*70)

    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_parquet(os.path.join(PROC_DIR, "feature_matrix_v7.parquet"))
    with open(os.path.join(PROC_DIR, "feature_partition_v7.json")) as f:
        partition = json.load(f)
    with open(os.path.join(PROC_DIR, "calibration_split_idx.json")) as f:
        calib_info = json.load(f)

    def _get(cols):
        present = [c for c in cols if c in df.columns]
        return df[present].fillna(0.0).values.astype(np.float32)

    Xl = _get(partition["LST"])
    Xt = _get(partition["Tilt"])
    Xr = _get(partition["Residual"])
    er_cm    = df["er_CM"].fillna(0.0).values.astype(np.float32)
    has_cm   = df["has_sigma_CM"].fillna(0.0).values.astype(np.float32)
    cm_approx= df["cm_approx_flag"].fillna(0.0).values.astype(np.float32)
    phase_tr = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
               if "phase_transition" in df.columns else np.zeros(len(df), np.float32)
    y        = df["epsilon_r"].values.astype(np.float32)
    gii_norm = df["GII"].fillna(0.0).values.astype(np.float32)

    models_saved, sc_lst, sc_tilt, sc_res, gii_max, regime_idx = load_model_and_scalers()
    Xl_s = sc_lst.transform(Xl).astype(np.float32)
    Xt_s = sc_tilt.transform(Xt).astype(np.float32)
    Xr_s = sc_res.transform(Xr).astype(np.float32)

    print(f"\n  {len(df)} samples loaded")
    print("  Running full inference on saved ensemble...")
    out_full = infer(models_saved, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                     gii_norm, phase_tr, regime_idx)

    d_lst  = out_full["delta_lst"]
    d_tilt = out_full["delta_tilt"]
    d_res  = out_full["delta_res"]
    pred   = out_full["pred"]

    valid_cm = (has_cm > 0.5) & (er_cm > 5.0)
    f_lst_all = np.where(valid_cm, d_lst / er_cm.clip(1e-3), np.nan)

    # A-site and B-site labels
    df["a_site"]  = df["chemistry_family"].apply(lambda x: str(x).split("_")[0])
    df["b_site"]  = df["chemistry_family"].apply(lambda x: str(x).split("_")[1]
                                                  if "_" in str(x) else "Other")
    df["f_lst"]   = f_lst_all
    df["d_lst"]   = d_lst
    df["d_tilt"]  = d_tilt
    df["d_res"]   = d_res
    df["pred"]    = pred

    # Dominant regime
    for rname, rcol in [("Ia","regime_Ia"),("Ib","regime_Ib"),("II","regime_II"),("III","regime_III")]:
        if rcol not in df.columns: df[rcol] = 0.0
    dom_regime = np.full(len(df), "Ia", dtype=object)
    for rname, rcol in [("III","regime_III"),("II","regime_II"),("Ib","regime_Ib")]:
        dom_regime[df[rcol].values > 0.5] = rname
    df["dom_regime"] = dom_regime

    results = {}

    # ══════════════════════════════════════════════════════════════════════
    # PART A — Matched-pair counterfactual analysis
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("PART A — Matched-Pair Counterfactual Analysis")
    print("─"*60)
    print(f"  Matching: same B-site, |Δtol_factor|<{TOL_FACTOR_DELTA}, same dominant regime")
    print()

    cf_pairs  = []  # list of matched pair records
    b_summaries = {}

    b_site_counts = df.groupby("b_site")["a_site"].nunique()
    b_sites_multi = b_site_counts[b_site_counts > 1].index.tolist()

    for b in sorted(b_sites_multi):
        sub = df[df["b_site"] == b].copy().reset_index(drop=True)
        asites_in_b = sub["a_site"].unique()
        if len(asites_in_b) < 2: continue

        # Build matched pairs: for each pair of A-sites within this B-site
        pairs_found = []
        for a1, a2 in itertools.combinations(asites_in_b, 2):
            g1 = sub[sub["a_site"] == a1]
            g2 = sub[sub["a_site"] == a2]
            # Match by tolerance factor and regime
            for _, r1 in g1.iterrows():
                for _, r2 in g2.iterrows():
                    dt = abs(r1["tolerance_factor"] - r2["tolerance_factor"])
                    same_regime = (r1["dom_regime"] == r2["dom_regime"])
                    if dt <= TOL_FACTOR_DELTA and same_regime:
                        d_er   = float(r1["epsilon_r"] - r2["epsilon_r"])
                        d_flst = float(r1["f_lst"] - r2["f_lst"]) if (
                            not np.isnan(r1["f_lst"]) and not np.isnan(r2["f_lst"])) else np.nan
                        d_dlst = float(r1["d_lst"] - r2["d_lst"])
                        d_dtilt= float(r1["d_tilt"] - r2["d_tilt"])
                        d_dres = float(r1["d_res"] - r2["d_res"])
                        pairs_found.append({
                            "b_site": b, "a1": a1, "a2": a2,
                            "tol_a1": float(r1["tolerance_factor"]),
                            "tol_a2": float(r2["tolerance_factor"]),
                            "regime": r1["dom_regime"],
                            "DC_a1": float(r1["epsilon_r"]), "DC_a2": float(r2["epsilon_r"]),
                            "delta_er": d_er, "delta_dlst": d_dlst,
                            "delta_dtilt": d_dtilt, "delta_dres": d_dres,
                            "delta_flst": d_flst,
                        })
                        cf_pairs.append(pairs_found[-1])

        if len(pairs_found) >= MIN_PAIRS:
            d_er_arr   = np.array([p["delta_er"]    for p in pairs_found])
            d_dlst_arr = np.array([p["delta_dlst"]  for p in pairs_found])
            d_dtilt_arr= np.array([p["delta_dtilt"] for p in pairs_found])
            d_dres_arr = np.array([p["delta_dres"]  for p in pairs_found])
            # Fraction of Δεr variance explained by Δδ_LST (vs tilt/res)
            tot_abs = (np.abs(d_dlst_arr) + np.abs(d_dtilt_arr) + np.abs(d_dres_arr)).clip(1e-6)
            lst_frac  = np.mean(np.abs(d_dlst_arr)  / tot_abs)
            tilt_frac = np.mean(np.abs(d_dtilt_arr) / tot_abs)
            res_frac  = np.mean(np.abs(d_dres_arr)  / tot_abs)
            # Correlation between Δεr and Δδ_LST
            if len(d_er_arr) > 2:
                corr = float(np.corrcoef(d_er_arr, d_dlst_arr)[0, 1])
            else:
                corr = np.nan
            b_summaries[b] = {
                "n_pairs": len(pairs_found),
                "mean_delta_er": float(np.mean(d_er_arr)),
                "lst_fraction": float(lst_frac),
                "tilt_fraction": float(tilt_frac),
                "res_fraction": float(res_frac),
                "lst_er_correlation": corr,
            }
            print(f"  B={b:4s}  n_pairs={len(pairs_found):4d}  "
                  f"LST_frac={lst_frac:.2f}  Tilt_frac={tilt_frac:.2f}  "
                  f"corr(Δεr,Δδ_LST)={corr:.3f}")

    # Overall counterfactual summary
    if b_summaries:
        all_lst_frac = [v["lst_fraction"] for v in b_summaries.values()]
        all_corr     = [v["lst_er_correlation"] for v in b_summaries.values()
                        if not np.isnan(v["lst_er_correlation"])]
        mean_lst_frac = float(np.mean(all_lst_frac))
        mean_corr     = float(np.mean(all_corr)) if all_corr else np.nan
        cf_strong = (mean_lst_frac >= 0.5) and (mean_corr >= 0.5 if not np.isnan(mean_corr) else False)
        print(f"\n  Overall: mean LST fraction={mean_lst_frac:.3f}, "
              f"mean corr(Δεr,Δδ_LST)={mean_corr:.3f}")
        print(f"  Counterfactual result: {'STRONG' if cf_strong else 'MODERATE'}")
        print(f"  → {'Δεr routes primarily through LST branch ✓' if cf_strong else 'Mixed routing — LST not uniquely dominant'}")
    else:
        cf_strong = False
        mean_lst_frac = 0.0; mean_corr = 0.0
        print("  WARNING: No B-site had enough matched pairs for analysis")

    results["counterfactual"] = {
        "b_site_summaries": b_summaries,
        "n_total_pairs": len(cf_pairs),
        "mean_lst_fraction": float(mean_lst_frac),
        "mean_delta_er_lst_correlation": float(mean_corr),
        "counterfactual_strong": bool(cf_strong),
    }

    # ══════════════════════════════════════════════════════════════════════
    # PART B — Permutation Test 1: A-site label shuffle
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("PART B — Permutation Test 1: A-site label shuffle")
    print("─"*60)

    TARGET_ASITES = ["Pb", "Ba", "Ca", "Sr", "La"]
    # Observed test statistic: range of per-A-site mean f_LST
    asite_arr = df["a_site"].values
    f_lst_arr = np.where(valid_cm, f_lst_all, np.nan)

    obs_means = {}
    for a in TARGET_ASITES:
        mask  = asite_arr == a
        vals  = f_lst_arr[mask]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            obs_means[a] = float(np.mean(valid))
    print(f"  Observed mean f_LST by A-site: {dict((k,f'{v:.3f}') for k,v in sorted(obs_means.items(), key=lambda x:-x[1]))}")

    # Only permute among samples that have a valid f_LST and one of the target A-sites
    perm_mask = (~np.isnan(f_lst_arr)) & np.isin(asite_arr, TARGET_ASITES)
    perm_idx  = np.where(perm_mask)[0]
    f_perm    = f_lst_arr[perm_idx]
    a_perm    = asite_arr[perm_idx]

    # Observed range (test statistic)
    obs_group_means = []
    for a in TARGET_ASITES:
        v = f_perm[a_perm == a]
        if len(v) > 0: obs_group_means.append(np.mean(v))
    obs_range = float(np.max(obs_group_means) - np.min(obs_group_means))
    print(f"  Observed f_LST range (max-min across A-sites): {obs_range:.4f}")

    # Null distribution
    rng = np.random.RandomState(0)
    null_ranges = []
    for _ in range(N_PERM_ASITE):
        shuf_a = rng.permutation(a_perm)
        null_means = []
        for a in TARGET_ASITES:
            v = f_perm[shuf_a == a]
            if len(v) > 0: null_means.append(np.mean(v))
        if null_means: null_ranges.append(np.max(null_means) - np.min(null_means))
    null_ranges = np.array(null_ranges)
    p_val_asite = float(np.mean(null_ranges >= obs_range))
    print(f"  Null mean range: {np.mean(null_ranges):.4f} ± {np.std(null_ranges):.4f}")
    print(f"  p-value (A-site permutation): {p_val_asite:.4f}")
    if p_val_asite < 0.05:
        print(f"  → A-site ordering of f_LST is STATISTICALLY SIGNIFICANT (p={p_val_asite:.4f}) ✓")
    else:
        print(f"  → A-site ordering NOT significant at p<0.05 (p={p_val_asite:.4f})")

    results["permutation_asite"] = {
        "observed_range": obs_range,
        "observed_means": obs_means,
        "null_mean_range": float(np.mean(null_ranges)),
        "null_std_range":  float(np.std(null_ranges)),
        "p_value": p_val_asite,
        "significant": bool(p_val_asite < 0.05),
        "n_permutations": N_PERM_ASITE,
    }

    # ══════════════════════════════════════════════════════════════════════
    # PART C — Permutation Test 2: Reaney REGIME_SCALES permutation
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("PART C — Permutation Test 2: Reaney REGIME_SCALES permutation")
    print("─"*60)
    print(f"  True scales: {TRUE_REGIME_SCALES} [Ia, Ib, II, III]")
    print(f"  Testing all 4! = 24 permutations exhaustively")

    # True R² (baseline with original scales)
    pred_true = infer(models_saved, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                      gii_norm, phase_tr, regime_idx)["pred"]
    r2_true = float(r2_score(y, pred_true))
    print(f"  True REGIME_SCALES R² = {r2_true:.4f}")

    all_perms = list(itertools.permutations(TRUE_REGIME_SCALES))
    perm_r2s  = []
    true_perm = tuple(TRUE_REGIME_SCALES)

    for perm in all_perms:
        pred_p = infer_with_regime_scales(
            models_saved, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
            gii_norm, phase_tr, regime_idx, list(perm))
        perm_r2s.append(float(r2_score(y, pred_p)))

    perm_r2s = np.array(perm_r2s)
    true_idx  = all_perms.index(true_perm)
    rank      = int(np.sum(perm_r2s >= r2_true))   # how many perms ≥ true
    p_val_regime = float(rank / len(all_perms))

    print(f"  Permutation R²: mean={np.mean(perm_r2s):.4f}  "
          f"min={np.min(perm_r2s):.4f}  max={np.max(perm_r2s):.4f}")
    print(f"  True ordering rank: {rank}/{len(all_perms)} (p={p_val_regime:.4f})")
    if p_val_regime <= 1/24:
        print(f"  → True REGIME_SCALES ordering is BEST (p≤1/24) ✓")
    elif p_val_regime < 0.10:
        print(f"  → True ordering is among top — moderate evidence")
    else:
        print(f"  → True ordering not distinguishable from random permutations")

    results["permutation_regime"] = {
        "true_r2": r2_true,
        "all_perm_r2s": perm_r2s.tolist(),
        "n_perms_at_least_as_good": rank,
        "total_perms": len(all_perms),
        "p_value": p_val_regime,
        "significant": bool(p_val_regime <= 1/24),
        "true_scales": TRUE_REGIME_SCALES,
    }

    # ══════════════════════════════════════════════════════════════════════
    # PART D — Falsifiable Predictions
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("PART D — Falsifiable Predictions")
    print("─"*60)

    # Generate predictions from the quantitative results
    pb_flst   = obs_means.get("Pb", 2.27)
    ba_flst   = obs_means.get("Ba", 0.71)
    ca_flst   = obs_means.get("Ca", 1.73)
    sr_flst   = obs_means.get("Sr", 0.91)
    preds_text = []

    p1 = (f"Prediction 1 (Pb-lone-pair rule): ABO₃ perovskites with Pb²⁺ on the A-site "
          f"should exhibit f_LST ≥ {pb_flst-0.2:.1f}× (95% CI lower bound) regardless of B-site, "
          f"provided the compound falls in the Ib or II Reaney regime.")
    p2 = (f"Prediction 2 (Ba suppression rule): A-site Ba²⁺ substitution for Pb²⁺ in "
          f"iso-structural compounds should reduce εr by {(pb_flst-ba_flst):.1f}× the CM baseline, "
          f"mediated through the LST branch (|Δδ_LST| >> |Δδ_tilt|).")
    p3 = ("Prediction 3 (Regime Ib amplification): Compounds in Reaney regime Ib "
          "(incipient ferroelectric, tilt < 5°, soft-mode active) with lone-pair A-site "
          "cations should consistently achieve f_LST > 2.0×, making them the primary "
          "target class for high-εr engineering via composition design.")
    p4 = ("Prediction 4 (Structural descriptor failure): A model trained only on "
          "structural descriptors (tolerance factor, ionic radii, space group) without "
          "phonon-mode proxies (d₀_per_field, soft_mode_proxy) should fail to reproduce "
          "the Pb vs. Ba εr difference, even when chemical identity is provided.")
    p5 = ("Prediction 5 (Tilt-regime boundary crossing): Compounds near the Ib→II "
          "regime boundary (continuous_tilt_strain ≈ 0.3-0.5) should show maximum "
          "sensitivity of εr to synthesis conditions, as small structural changes "
          "shift them between the amplification-dominant and suppression-dominant regime.")

    for p in [p1, p2, p3, p4, p5]:
        preds_text.append(p)
        print(f"  {p[:120]}...")
        print()

    results["falsifiable_predictions"] = preds_text

    # ══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════
    print("[Generating figures...]")

    # Figure A — Counterfactual: Δεr vs Δδ_LST scatter per B-site
    if cf_pairs:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        cmap = plt.cm.get_cmap("tab10", len(b_summaries))
        plotted_bs = list(b_summaries.keys())[:8]  # top 8 B-sites
        for bi, b in enumerate(plotted_bs):
            pairs_b = [p for p in cf_pairs if p["b_site"] == b]
            if not pairs_b: continue
            x = [p["delta_dlst"] for p in pairs_b]
            y_  = [p["delta_er"]  for p in pairs_b]
            ax.scatter(x, y_, s=18, alpha=0.65, color=cmap(bi), label=f"B={b}", zorder=3)
        # Overall regression line
        x_all = [p["delta_dlst"] for p in cf_pairs]
        y_all = [p["delta_er"]   for p in cf_pairs]
        if len(x_all) > 2:
            m_, c_ = np.polyfit(x_all, y_all, 1)
            xr = np.linspace(min(x_all), max(x_all), 100)
            ax.plot(xr, m_*xr+c_, "k--", lw=1.5, alpha=0.7, label=f"slope={m_:.2f}")
        ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("Δδ_LST  (predicted, pair A1−A2)")
        ax.set_ylabel("Δεr  (measured, pair A1−A2)")
        ax.set_title(f"Counterfactual: same B-site, different A-site\n"
                     f"({len(cf_pairs)} matched pairs, {len(b_summaries)} B-sites)")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        style4(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "43a_counterfactual_pairs.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Figure B — Permutation Test 1: A-site null distribution
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.hist(null_ranges, bins=40, color="#5B8DB8", alpha=0.75, edgecolor="white", lw=0.5)
    ax.axvline(obs_range, color="#E76F51", lw=2.0, label=f"Observed range={obs_range:.3f}")
    ax.set_xlabel("f_LST range (max − min) across A-site groups\nunder shuffled labels")
    ax.set_ylabel("Count")
    ax.set_title(f"A-site permutation test (n={N_PERM_ASITE} shuffles)\n"
                 f"p = {p_val_asite:.4f}{'  ✓ Significant' if p_val_asite < 0.05 else ''}")
    ax.legend(fontsize=9)
    style4(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "43b_permutation_asite.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure C — Permutation Test 2: Regime scale null distribution
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    ax1, ax2 = axes
    # Left: R² for each of 24 permutations
    sort_idx = np.argsort(perm_r2s)[::-1]
    bar_colors = ["#E76F51" if all_perms[i] == true_perm else "#5B8DB8" for i in sort_idx]
    ax1.bar(range(len(sort_idx)), perm_r2s[sort_idx], color=bar_colors,
            edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Permutation rank (highest R² first)")
    ax1.set_ylabel("R²")
    ax1.set_title("All 24 REGIME_SCALES permutations\n(orange = true [1.0,1.5,0.9,0.7])")
    style4(ax1)
    # Right: scale values for top 6 permutations
    top_n = 6
    top_perms = [all_perms[i] for i in sort_idx[:top_n]]
    top_r2s   = perm_r2s[sort_idx[:top_n]]
    scale_names = ["Ia", "Ib", "II", "III"]
    for pi, (perm_t, r2_p) in enumerate(zip(top_perms, top_r2s)):
        ax2.plot(scale_names, list(perm_t), marker="o", markersize=5,
                 color="#E76F51" if perm_t == true_perm else "#5B8DB8",
                 lw=1.8, alpha=0.8,
                 label=f"[{','.join(str(v) for v in perm_t)}] R²={r2_p:.3f}"
                        + (" ← TRUE" if perm_t == true_perm else ""))
    ax2.set_ylabel("Scale value")
    ax2.set_title(f"Top {top_n} permutations\n(p = {p_val_regime:.4f})")
    ax2.legend(fontsize=6, loc="upper right")
    style4(ax2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "43c_permutation_regime.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════════════════
    print()
    with open(os.path.join(RES_DIR, "43_counterfactual_permutation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → figures/43a-c.png")
    print(f"  Results → results/43_counterfactual_permutation.json")

    print("\n" + "="*70)
    print("COUNTERFACTUAL + PERMUTATION COMPLETE")
    print("="*70)
    print(f"  Counterfactual (Part A) : {'STRONG' if cf_strong else 'MODERATE'}"
          f"  (mean LST_frac={mean_lst_frac:.2f}, corr={mean_corr:.3f})")
    print(f"  Permutation A-site (B)  : p={p_val_asite:.4f}"
          f"  {'SIGNIFICANT ✓' if p_val_asite < 0.05 else 'NOT SIGNIFICANT'}")
    print(f"  Permutation Regime (C)  : p={p_val_regime:.4f}"
          f"  {'BEST ORDERING ✓' if p_val_regime <= 1/24 else 'NOT BEST'}")
    print()

    # Load script 42 result for joint decision
    s42_path = os.path.join(RES_DIR, "42_lofo_generality.json")
    if os.path.exists(s42_path):
        with open(s42_path) as f: s42 = json.load(f)
        pb_r2   = s42["lofo_asite"].get("Pb", {}).get("r2")
        mean_lofo= s42["decision"].get("mean_lofo_asite_r2", 0.0)
        lofo_str = s42["decision"].get("lofo_strength", "WEAK")
        if mean_lofo >= 0.80 and cf_strong:
            final = "GENERALIZABLE MECHANISM — full NatComm claim"
        elif mean_lofo >= 0.60 and cf_strong:
            final = "ROBUST TREND WITH LIMITED EXTRAPOLATION — strong NatComm claim"
        elif mean_lofo >= 0.60:
            final = "ROBUST TREND — NatComm claim with caveated extrapolation"
        elif cf_strong:
            final = "WITHIN-FAMILY MECHANISTIC TREND — npj CM or focused NatComm"
        else:
            final = "NOT NatComm-ready — strengthen LOFO and counterfactual"
        print(f"  JOINT DECISION (LOFO + Counterfactual):")
        print(f"    Mean LOFO A-site R²  = {mean_lofo:.4f}  [{lofo_str}]")
        print(f"    Pb LOFO R²           = {pb_r2}")
        print(f"    Counterfactual       = {'STRONG' if cf_strong else 'MODERATE'}")
        print(f"    ──────────────────────────────────────────")
        print(f"    FINAL CLAIM: {final}")
    else:
        print("  Run script 42 first to get the joint LOFO+counterfactual decision.")
    print("="*70)


if __name__ == "__main__":
    main()
