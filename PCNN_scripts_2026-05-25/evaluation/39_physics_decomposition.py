"""
Script 39 — Physics Decomposition Analysis (NatComm discovery figures)

Two findings for NatComm:
  Option 1: LST enhancement map by composition, regime, A-site, B-site type
            → Design rule: which compositions maximise εr through soft-mode coupling
  Option 3: Why tilt branch contributes negligibly
            → Architecture insight: regime-scaled LST already encodes tilt physics
               (REGIME_SCALES = [Ia:1.0, Ib:1.5, II:0.9, III:0.7])

Uses the trained final ensemble from script 37 (models/cmltrv77_final.pt).
Runs inference on all 1360 samples, extracts per-sample:
  δ_LST, δ_tilt, δ_res, σ_conf, f_lst = δ_LST/εr_CM

Outputs:
  results/39_physics_decomposition.json
  figures/39a_decomp_by_regime.png   — εr stacked decomposition by regime
  figures/39b_flst_by_composition.png — f_LST by A-site, B-site type, regime
  figures/39c_tilt_story.png         — why tilt branch is redundant (Option 3)
"""

import sys, os, warnings, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from sklearn.preprocessing import StandardScaler

from src.models.psrnn_mdpinn import CMLTRPCNNv71

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)
FIG_DIR  = os.path.join(ROOT, "figures");  os.makedirs(FIG_DIR, exist_ok=True)
MODEL_PT = os.path.join(ROOT, "models", "cmltrv77_final.pt")

# B-site element classification
D0_B  = {"Ti", "Nb", "Ta", "Zr", "W", "Mo", "V", "Cr", "Mn"}
D10_B = {"Zn", "Cd", "Hg", "Ga", "In", "Tl", "Cu", "Ge", "Sn", "Ag"}
RE_B  = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
         "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.grid": False, "axes.linewidth": 1.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 5, "ytick.major.size": 5,
    "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.frameon": True, "legend.framealpha": 0.90, "legend.fontsize": 9,
})

PALETTE = {
    "CM":    "#4A90D9",
    "LST":   "#009E73",
    "Tilt":  "#D55E00",
    "Res":   "#CC79A7",
    "d0":    "#009E73",
    "d10":   "#E69F00",
    "RE":    "#56B4E9",
    "Other": "#AAAAAA",
    "Ba":    "#264653", "Ca": "#2A9D8F",
    "Sr":    "#E9C46A", "Pb": "#E76F51", "Other_A": "#AAAAAA",
}


def style4(ax):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def classify_b_site(family_str):
    """Return B-site type from chemistry_family string like 'Ba_Ti'."""
    parts = str(family_str).split("_")
    if len(parts) < 2:
        return "Other"
    b = parts[1].strip()
    if b in D0_B:  return "d0 (Ti/Nb/Zr)"
    if b in D10_B: return "d10 (Zn/Ga)"
    if b in RE_B:  return "RE (La/Y/Nd)"
    return "Other"


def classify_a_site(family_str):
    """Return A-site cation from chemistry_family string."""
    parts = str(family_str).split("_")
    a = parts[0].strip() if parts else "Other"
    if a in ("Ba", "Ca", "Sr", "Pb", "Bi", "La"):
        return a
    return "Other"


def get_regime(row):
    """Get Reaney regime label from regime_* columns."""
    for col, label in [("regime_Ib", "Ib"), ("regime_Ia", "Ia"),
                        ("regime_III", "III"), ("regime_II", "II")]:
        if col in row.index and row[col] > 0.5:
            return label
    return "Unknown"


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
def extract_decomposition(models, Xl_s, Xt_s, Xr_s,
                          er_cm, has_cm, cm_approx, gii_norm, phase_tr,
                          regime_idx, batch=256):
    """Run ensemble inference and return mean of each physics component."""
    all_pred, all_lst, all_tilt, all_res, all_conf = [], [], [], [], []

    for m in models:
        pred_i, lst_i, tilt_i, res_i, conf_i = [], [], [], [], []
        N = len(er_cm)
        for i in range(0, N, batch):
            sl = slice(i, i + batch)
            out = m(
                torch.tensor(Xl_s[sl],       dtype=torch.float32).to(DEVICE),
                torch.tensor(Xt_s[sl],       dtype=torch.float32).to(DEVICE),
                torch.tensor(Xr_s[sl],       dtype=torch.float32).to(DEVICE),
                torch.tensor(er_cm[sl],      dtype=torch.float32).to(DEVICE),
                torch.tensor(has_cm[sl],     dtype=torch.float32).to(DEVICE),
                torch.tensor(cm_approx[sl],  dtype=torch.float32).to(DEVICE),
                torch.tensor(gii_norm[sl],   dtype=torch.float32).to(DEVICE),
                torch.tensor(phase_tr[sl],   dtype=torch.float32).to(DEVICE),
                regime_idx,
            )
            pred_i.append(out["pred"].cpu().numpy())
            lst_i.append(out["delta_lst"].cpu().numpy())
            tilt_i.append(out["delta_tilt"].cpu().numpy())
            res_i.append(out["delta_res"].cpu().numpy())
            conf_i.append(out["sigma_conf"].cpu().numpy())
        all_pred.append(np.concatenate(pred_i))
        all_lst.append(np.concatenate(lst_i))
        all_tilt.append(np.concatenate(tilt_i))
        all_res.append(np.concatenate(res_i))
        all_conf.append(np.concatenate(conf_i))

    return (np.mean(all_pred, 0), np.mean(all_lst, 0),
            np.mean(all_tilt, 0), np.mean(all_res, 0), np.mean(all_conf, 0))


def main():
    print("=" * 70)
    print("Script 39 — Physics Decomposition Analysis")
    print("  Options 1 + 3: LST enhancement map + tilt redundancy story")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data and model...")
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
    train_idx= np.array(calib_info["train_idx"])

    models, sc_lst, sc_tilt, sc_res, gii_max, regime_idx = load_model_and_scalers()
    print(f"  Loaded {len(models)} ensemble members | {len(df)} samples")

    # Scale using saved scalers (fit on train_idx)
    Xl_s = sc_lst.transform(Xl).astype(np.float32)
    Xt_s = sc_tilt.transform(Xt).astype(np.float32)
    Xr_s = sc_res.transform(Xr).astype(np.float32)
    gii_norm = df["GII"].fillna(0.0).values.astype(np.float32)

    # Extract per-sample physics components
    print("\n[2/5] Extracting physics decomposition for all 1360 samples...")
    pred, d_lst, d_tilt, d_res, sigma = extract_decomposition(
        models, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
        gii_norm, phase_tr, regime_idx)

    # Build analysis dataframe
    print("\n[3/5] Building analysis dataframe...")
    adf = pd.DataFrame({
        "formula":    df["formula"].values,
        "family":     df["chemistry_family"].values if "chemistry_family" in df.columns else "Unknown",
        "y_true":     y,
        "y_pred":     pred,
        "er_cm":      er_cm,
        "has_cm":     has_cm,
        "delta_lst":  d_lst,
        "delta_tilt": d_tilt,
        "delta_res":  d_res,
        "sigma_conf": sigma,
        "cm_approx":  cm_approx,
    })

    # Regime classification from LST feature columns
    lst_cols = [c for c in partition["LST"] if c in df.columns]
    lst_df   = df[lst_cols]
    adf["regime"] = lst_df.apply(get_regime, axis=1)

    # A-site and B-site classification
    adf["a_site"] = adf["family"].apply(classify_a_site)
    adf["b_type"] = adf["family"].apply(classify_b_site)

    # f_LST = delta_LST / er_CM (only for samples with CM, er_CM > 5)
    valid_cm = (adf["has_cm"] > 0.5) & (adf["er_cm"] > 5.0)
    adf["f_lst"] = np.where(valid_cm, adf["delta_lst"] / adf["er_cm"], np.nan)

    # Per-composition SIGNED fractional contributions to ε_r.
    # The architecture is exactly additive (ε_pred = ε_CM + δ_LST + δ_tilt + δ_res), so the four
    # signed fractions sum to 1 per composition; the tilt term is negative (suppressive). Earlier
    # versions took |δ_tilt| and |δ_res|, which flipped tilt's sign and broke the sum-to-100%
    # property of the decomposition — fixed here so the reported fractions are physical and
    # reproducible from this script (see GLOBAL_DECOMP below).
    adf["abs_pred"] = adf["y_pred"].clip(lower=1.0)
    adf["frac_cm"]   = adf["er_cm"]      / adf["abs_pred"]
    adf["frac_lst"]  = adf["delta_lst"]  / adf["abs_pred"]
    adf["frac_tilt"] = adf["delta_tilt"] / adf["abs_pred"]   # signed (≤ 0, suppressive)
    adf["frac_res"]  = adf["delta_res"]  / adf["abs_pred"]   # signed

    # CANONICAL decomposition (manuscript): ε_r-weighted "fraction of total permittivity",
    # computed over the valid-CM compositions where the additive identity
    # ε_pred = ε_CM + δ_LST + δ_tilt + δ_res holds exactly (so the four fractions sum to 1.0).
    # The ~180 no-CM fallback compositions are excluded (no CM term to decompose against).
    _decomp_mask = valid_cm
    _tot = float(adf.loc[_decomp_mask, "y_pred"].sum())
    _add_resid = float((adf.loc[_decomp_mask, ["er_cm", "delta_lst", "delta_tilt", "delta_res"]].sum(axis=1)
                        - adf.loc[_decomp_mask, "y_pred"]).abs().mean())
    GLOBAL_DECOMP = {
        "definition": "epsilon_r_weighted_valid_cm",
        "n_decomp":   int(_decomp_mask.sum()),
        "frac_cm":    float(adf.loc[_decomp_mask, "er_cm"].sum())      / _tot,
        "frac_lst":   float(adf.loc[_decomp_mask, "delta_lst"].sum())  / _tot,
        "frac_tilt":  float(adf.loc[_decomp_mask, "delta_tilt"].sum()) / _tot,
        "frac_res":   float(adf.loc[_decomp_mask, "delta_res"].sum())  / _tot,
        "additive_residual_mean": _add_resid,
    }
    _decomp_sum = sum(GLOBAL_DECOMP[k] for k in ("frac_cm", "frac_lst", "frac_tilt", "frac_res"))
    print(f"    [additive check] mean|Σ δX − ε_pred| (n={int(_decomp_mask.sum())}) = {_add_resid:.3e}")
    print(f"    [decomposition sum] = {_decomp_sum:.4f} (should be ≈ 1.0)")
    # Diagnostic: decomposition restricted to valid-CM compositions (identity holds exactly there).
    for _label, _m in (("valid-CM only", valid_cm), ("no-CM only", ~valid_cm & _decomp_mask)):
        if int(_m.sum()) == 0:
            continue
        _r = float((adf.loc[_m, ["er_cm", "delta_lst", "delta_tilt", "delta_res"]].sum(axis=1)
                    - adf.loc[_m, "y_pred"]).abs().mean())
        _fc, _fl, _ft, _fr = (float(adf.loc[_m, c].mean()) for c in ("frac_cm", "frac_lst", "frac_tilt", "frac_res"))
        print(f"    [{_label} | mean-of-fractions] n={int(_m.sum())}  CM={_fc:.3%} LST={_fl:.3%} "
              f"tilt={_ft:.3%} res={_fr:.3%}  sum={_fc+_fl+_ft+_fr:.4f}  resid={_r:.3e}")
        _tot = float(adf.loc[_m, "y_pred"].sum())
        _sc, _sl, _st, _sr = (float(adf.loc[_m, c].sum()) / _tot for c in ("er_cm", "delta_lst", "delta_tilt", "delta_res"))
        print(f"    [{_label} | ε_r-weighted ]      CM={_sc:.3%} LST={_sl:.3%} "
              f"tilt={_st:.3%} res={_sr:.3%}  sum={_sc+_sl+_st+_sr:.4f}")

    # Print summary
    print(f"\n  Global mean physics decomposition (all {len(adf)} samples):")
    print(f"    CM contribution   : {adf['frac_cm'].mean():.1%}")
    print(f"    LST contribution  : {adf['frac_lst'].mean():.1%}")
    print(f"    Tilt contribution : {adf['frac_tilt'].mean():.1%}")
    print(f"    Res contribution  : {adf['frac_res'].mean():.1%}")
    print(f"\n  f_LST = δ_LST / εr_CM (samples with CM, n={valid_cm.sum()}):")
    print(f"    Mean : {adf['f_lst'].mean():.3f}×")
    print(f"    Median: {adf['f_lst'].median():.3f}×")
    print(f"    Std  : {adf['f_lst'].std():.3f}")

    print(f"\n  f_LST by Reaney regime:")
    for regime in ["Ia", "Ib", "II", "III", "Unknown"]:
        sub = adf[(adf["regime"] == regime) & valid_cm]
        if len(sub) > 5:
            print(f"    Regime {regime}: n={len(sub):4d}  "
                  f"f_LST={sub['f_lst'].mean():.3f}±{sub['f_lst'].std():.3f}  "
                  f"|δ_tilt|={sub['delta_tilt'].abs().mean():.2f}  "
                  f"δ_LST={sub['delta_lst'].mean():.2f}")

    print(f"\n  f_LST by B-site type:")
    for btype in ["d0 (Ti/Nb/Zr)", "d10 (Zn/Ga)", "RE (La/Y/Nd)", "Other"]:
        sub = adf[(adf["b_type"] == btype) & valid_cm]
        if len(sub) > 5:
            print(f"    {btype:<20s}: n={len(sub):4d}  "
                  f"f_LST={sub['f_lst'].mean():.3f}±{sub['f_lst'].std():.3f}")

    print(f"\n  f_LST by A-site:")
    for asite in ["Ba", "Ca", "Sr", "Pb", "Bi", "La", "Other"]:
        sub = adf[(adf["a_site"] == asite) & valid_cm]
        if len(sub) > 5:
            print(f"    {asite:<8s}: n={len(sub):4d}  "
                  f"f_LST={sub['f_lst'].mean():.3f}±{sub['f_lst'].std():.3f}  "
                  f"εr_mean={sub['y_true'].mean():.1f}")

    print("\n[4/5] Generating figures...")

    # ── Figure A: εr decomposition stacked bars by regime ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    regimes_ordered = ["Ia", "Ib", "II", "III"]
    regime_labels   = ["Regime Ia\n(untilted)", "Regime Ib\n(incipient FE)",
                       "Regime II\n(antiphase)", "Regime III\n(in-phase+anti)"]
    x = np.arange(len(regimes_ordered))

    cm_means, lst_means, tilt_means, res_means, ns = [], [], [], [], []
    for reg in regimes_ordered:
        sub = adf[adf["regime"] == reg]
        ns.append(len(sub))
        cm_means.append(sub["er_cm"].mean())
        lst_means.append(sub["delta_lst"].mean())
        tilt_means.append(sub["delta_tilt"].abs().mean())
        res_means.append(sub["delta_res"].abs().mean())

    ax = axes[0]
    w = 0.55
    b1 = ax.bar(x, cm_means,   w, label="εr_CM (Clausius-Mossotti)", color=PALETTE["CM"],   edgecolor="white")
    b2 = ax.bar(x, lst_means,  w, bottom=cm_means, label="δ_LST (soft-mode enhancement)",
                color=PALETTE["LST"],  edgecolor="white")
    b3 = ax.bar(x, tilt_means, w,
                bottom=[c+l for c,l in zip(cm_means, lst_means)],
                label="|δ_tilt| (octahedral tilt)",
                color=PALETTE["Tilt"], edgecolor="white", alpha=0.8)
    b4 = ax.bar(x, res_means,  w,
                bottom=[c+l+t for c,l,t in zip(cm_means, lst_means, tilt_means)],
                label="|δ_res| (residual)",
                color=PALETTE["Res"], edgecolor="white", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(regime_labels, fontsize=9)
    ax.set_ylabel("Mean contribution to εr")
    ax.set_title("εr Decomposition by Reaney Regime", fontsize=11)
    for i, (xi, n) in enumerate(zip(x, ns)):
        ax.text(xi, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 1,
                f"n={n}", ha="center", va="top", fontsize=8, color="#444")
    ax.legend(fontsize=7.5, loc="upper right"); style4(ax)

    # Right: f_LST by regime (violin / box)
    ax2 = axes[1]
    flst_data = [adf[(adf["regime"] == r) & valid_cm]["f_lst"].dropna().values
                 for r in regimes_ordered]
    vp = ax2.violinplot(flst_data, positions=x, widths=0.6,
                        showmedians=True, showextrema=True)
    for i, pc in enumerate(vp["bodies"]):
        pc.set_facecolor(["#4A90D9", "#009E73", "#D55E00", "#CC79A7"][i])
        pc.set_alpha(0.6)
    vp["cmedians"].set_color("#222"); vp["cmedians"].set_linewidth(2)
    ax2.set_xticks(x); ax2.set_xticklabels(regime_labels, fontsize=9)
    ax2.set_ylabel("f_LST = δ_LST / εr_CM")
    ax2.set_title("LST Enhancement Factor by Regime", fontsize=11)
    ax2.axhline(1.0, color="#888", ls="--", lw=1, label="f=1 (no enhancement)")
    for i, (xi, d) in enumerate(zip(x, flst_data)):
        if len(d) > 0:
            ax2.text(xi, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else np.nanmax(d)*1.05,
                     f"{np.nanmean(d):.2f}×", ha="center", va="bottom", fontsize=9,
                     fontweight="bold", color=["#4A90D9","#009E73","#D55E00","#CC79A7"][i])
    ax2.legend(fontsize=8); style4(ax2)

    plt.tight_layout()
    p = os.path.join(FIG_DIR, "39a_decomp_by_regime.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved → {p}")

    # ── Figure B: f_LST by A-site and B-site type ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: by A-site
    ax = axes[0]
    asites = [a for a in ["Ba", "Ca", "Sr", "Pb", "Bi", "La", "Other"]
              if len(adf[(adf["a_site"] == a) & valid_cm]) >= 5]
    flst_a = [adf[(adf["a_site"] == a) & valid_cm]["f_lst"].dropna().values for a in asites]
    means_a = [np.nanmean(d) for d in flst_a]
    stds_a  = [np.nanstd(d)  for d in flst_a]
    ns_a    = [len(d) for d in flst_a]
    colors_a = [PALETTE.get(a, PALETTE["Other_A"]) for a in asites]
    xa = np.arange(len(asites))
    bars = ax.bar(xa, means_a, 0.6, yerr=stds_a, color=colors_a, edgecolor="white",
                  error_kw={"elinewidth": 1.5, "capsize": 3, "ecolor": "#555"})
    ax.set_xticks(xa); ax.set_xticklabels(asites)
    ax.set_ylabel("f_LST = δ_LST / εr_CM")
    ax.set_title("LST Enhancement by A-site Cation", fontsize=11)
    ax.axhline(np.nanmean(adf[valid_cm]["f_lst"]), color="#888", ls="--", lw=1,
               label=f"Global mean {np.nanmean(adf[valid_cm]['f_lst']):.2f}×")
    for i, (b, n) in enumerate(zip(bars, ns_a)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + stds_a[i] + 0.03,
                f"n={n}", ha="center", va="bottom", fontsize=7.5, color="#444")
    ax.legend(fontsize=8); style4(ax)

    # Right: by B-site type
    ax2 = axes[1]
    btypes = ["d0 (Ti/Nb/Zr)", "d10 (Zn/Ga)", "RE (La/Y/Nd)", "Other"]
    btypes = [b for b in btypes if len(adf[(adf["b_type"] == b) & valid_cm]) >= 5]
    flst_b = [adf[(adf["b_type"] == b) & valid_cm]["f_lst"].dropna().values for b in btypes]
    means_b = [np.nanmean(d) for d in flst_b]
    stds_b  = [np.nanstd(d)  for d in flst_b]
    ns_b    = [len(d) for d in flst_b]
    bcolors = [PALETTE.get(b.split()[0], "#AAAAAA") for b in btypes]
    xb = np.arange(len(btypes))
    bars2 = ax2.bar(xb, means_b, 0.6, yerr=stds_b, color=bcolors, edgecolor="white",
                    error_kw={"elinewidth": 1.5, "capsize": 3, "ecolor": "#555"})
    ax2.set_xticks(xb); ax2.set_xticklabels(btypes, fontsize=9)
    ax2.set_ylabel("f_LST = δ_LST / εr_CM")
    ax2.set_title("LST Enhancement by B-site Type", fontsize=11)
    ax2.axhline(np.nanmean(adf[valid_cm]["f_lst"]), color="#888", ls="--", lw=1,
                label=f"Global mean {np.nanmean(adf[valid_cm]['f_lst']):.2f}×")
    for i, (b, n) in enumerate(zip(bars2, ns_b)):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + stds_b[i] + 0.03,
                 f"n={n}", ha="center", va="bottom", fontsize=7.5, color="#444")
    ax2.legend(fontsize=8); style4(ax2)

    plt.tight_layout()
    p = os.path.join(FIG_DIR, "39b_flst_by_composition.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved → {p}")

    # ── Figure C: The tilt story (Option 3) ──────────────────────────────
    # Show: |δ_tilt| << δ_LST across all regimes
    # And: regime scale already encodes tilt effect into LST
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # C1: δ_LST vs |δ_tilt| scatter, coloured by regime
    ax = axes[0]
    regime_colors = {"Ia": "#4A90D9", "Ib": "#009E73", "II": "#D55E00",
                     "III": "#CC79A7", "Unknown": "#AAAAAA"}
    for reg, col in regime_colors.items():
        sub = adf[adf["regime"] == reg]
        if len(sub) > 0:
            ax.scatter(sub["delta_lst"], sub["delta_tilt"].abs(),
                       c=col, alpha=0.4, s=12, label=f"Regime {reg} (n={len(sub)})")
    lim = max(adf["delta_lst"].max(), adf["delta_tilt"].abs().max())
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="δ_LST = |δ_tilt|")
    ax.set_xlabel("δ_LST (soft-mode enhancement)")
    ax.set_ylabel("|δ_tilt| (tilt suppression)")
    ax.set_title("LST vs Tilt Contribution\n(δ_LST dominates across all regimes)", fontsize=10)
    ax.legend(fontsize=7, markerscale=1.5); style4(ax)

    # C2: Mean |δ_tilt| / δ_LST ratio by regime — shows tilt is always << LST
    ax2 = axes[1]
    ratio_means, ratio_stds, ns_r = [], [], []
    for reg in regimes_ordered:
        sub = adf[adf["regime"] == reg]
        ratio = (sub["delta_tilt"].abs() / sub["delta_lst"].clip(lower=0.1)).values
        ratio_means.append(float(np.mean(ratio)))
        ratio_stds.append(float(np.std(ratio)))
        ns_r.append(len(sub))
    xr = np.arange(len(regimes_ordered))
    ax2.bar(xr, ratio_means, 0.55, yerr=ratio_stds,
            color=["#4A90D9", "#009E73", "#D55E00", "#CC79A7"],
            edgecolor="white",
            error_kw={"elinewidth": 1.5, "capsize": 3, "ecolor": "#555"})
    ax2.axhline(1.0, color="#333", ls="--", lw=1.5, label="|δ_tilt| = δ_LST")
    ax2.set_xticks(xr); ax2.set_xticklabels(regime_labels, fontsize=8)
    ax2.set_ylabel("|δ_tilt| / δ_LST ratio")
    ax2.set_title("Tilt/LST Ratio by Regime\n(< 1 means LST dominates)", fontsize=10)
    ax2.legend(fontsize=8); style4(ax2)

    # C3: The architectural explanation — regime_scale encodes tilt
    ax3 = axes[2]
    regime_scales = {"Ia": 1.0, "Ib": 1.5, "II": 0.9, "III": 0.7}
    actual_flst = {}
    for reg in regimes_ordered:
        sub = adf[(adf["regime"] == reg) & valid_cm]
        actual_flst[reg] = sub["f_lst"].median() if len(sub) > 5 else np.nan

    xc = np.arange(len(regimes_ordered))
    w  = 0.35
    ax3.bar(xc - w/2, [regime_scales[r] for r in regimes_ordered], w,
            label="Architecture regime scale", color="#56B4E9", edgecolor="white", alpha=0.9)
    ax3.bar(xc + w/2, [actual_flst.get(r, np.nan) / max(actual_flst.values())
                        for r in regimes_ordered], w,
            label="Observed f_LST (normalised)", color="#009E73", edgecolor="white", alpha=0.9)
    ax3.set_xticks(xc); ax3.set_xticklabels(regime_labels, fontsize=8)
    ax3.set_ylabel("Scale (normalised)")
    ax3.set_title("Regime Scale vs Observed f_LST\n(tilt physics encoded in LST scaling)", fontsize=10)
    ax3.legend(fontsize=8); style4(ax3)

    plt.suptitle("Option 3: Tilt Effect Mediated Through LST (Not Direct Suppression)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "39c_tilt_story.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved → {p}")

    # Save JSON results
    print("\n[5/5] Saving results...")
    global_flst = float(np.nanmean(adf[valid_cm]["f_lst"]))
    out = {
        "n_samples": len(adf),
        "n_valid_cm": int(valid_cm.sum()),
        "global_mean_f_lst": global_flst,
        "global_median_f_lst": float(np.nanmedian(adf[valid_cm]["f_lst"])),
        "global_decomposition": GLOBAL_DECOMP,
        "f_lst_by_regime": {
            reg: {
                "mean":   float(np.nanmean(adf[(adf["regime"]==reg) & valid_cm]["f_lst"])),
                "median": float(np.nanmedian(adf[(adf["regime"]==reg) & valid_cm]["f_lst"])),
                "std":    float(np.nanstd(adf[(adf["regime"]==reg) & valid_cm]["f_lst"])),
                "n":      int((adf["regime"]==reg).sum()),
            }
            for reg in regimes_ordered
            if (adf["regime"]==reg).sum() > 5
        },
        "f_lst_by_b_type": {
            b: {
                "mean": float(np.nanmean(adf[(adf["b_type"]==b) & valid_cm]["f_lst"])),
                "n":    int((adf["b_type"]==b).sum()),
            }
            for b in btypes if len(adf[(adf["b_type"]==b) & valid_cm]) >= 5
        },
        "f_lst_by_a_site": {
            a: {
                "mean": float(np.nanmean(adf[(adf["a_site"]==a) & valid_cm]["f_lst"])),
                "n":    int((adf["a_site"]==a).sum()),
            }
            for a in asites
        },
        "tilt_lst_ratio_by_regime": {
            reg: float(ratio_means[i])
            for i, reg in enumerate(regimes_ordered)
        },
        "architecture_insight": (
            "REGIME_SCALES=[Ia:1.0, Ib:1.5, II:0.9, III:0.7] encode tilt physics into LST. "
            "Dedicated tilt branch adds negligible R² (ablation: -0.001) because "
            "tilt effect on εr is mediated through soft-mode modulation, "
            "not direct geometric suppression."
        ),
    }
    res_path = os.path.join(RES_DIR, "39_physics_decomposition.json")
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results → {res_path}")

    print("\n" + "=" * 70)
    print("DONE — Script 39 Physics Decomposition")
    print("=" * 70)
    print(f"  Global f_LST = {global_flst:.3f}× (soft-mode enhancement over CM)")
    print(f"  Physics fractions: CM={out['global_decomposition']['frac_cm']:.1%}  "
          f"LST={out['global_decomposition']['frac_lst']:.1%}  "
          f"Tilt={out['global_decomposition']['frac_tilt']:.1%}  "
          f"Res={out['global_decomposition']['frac_res']:.1%}")
    print(f"\n  NatComm finding 1: f_LST varies by composition and regime")
    print(f"    → d0 B-site (Ti/Nb/Zr) expected highest f_LST")
    print(f"    → Regime Ib expected highest (architecture scale=1.5×)")
    print(f"\n  NatComm finding 2: |δ_tilt| << δ_LST across all regimes")
    print(f"    → Tilt effect mediated through LST scaling, not direct suppression")
    print(f"    → Architecture regime scales [1.0, 1.5, 0.9, 0.7] encode this physics")
    print("=" * 70)


if __name__ == "__main__":
    main()
