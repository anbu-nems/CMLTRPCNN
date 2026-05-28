"""
Script 45 — Composition Engineering Map: Soft-Mode Amplification Design Atlas

Constructs a 2D design atlas for ABO₃ microwave ceramics:
  X-axis : A-site chemistry (ordered by lone-pair activity: Pb → Ba)
  Y-axis : Reaney structural regime (Ia → III, increasing tilt severity)
  Color  : Mean soft-mode amplification factor f_LST = δ_LST / εr_CM

LOFO risk overlay (from script 42):
  HIGH risk  (R²<0.30) → dense cross-hatching
  MODERATE   (R²<0.65) → light hatching
  LOW risk   (R²≥0.65) → no hatching

Outputs:
  results/45_composition_engineering_map.json
  figures/45_design_atlas.png  — main 2-panel hero figure
"""

import sys, os, warnings, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator
from sklearn.preprocessing import StandardScaler

from src.models.psrnn_mdpinn import CMLTRPCNNv71

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)
FIG_DIR  = os.path.join(ROOT, "figures");  os.makedirs(FIG_DIR, exist_ok=True)
MODEL_PT = os.path.join(ROOT, "models", "cmltrv77_final.pt")

# A-site ordering: descending lone-pair / f_LST activity
ASITE_ORDER   = ["Pb", "Ca", "La", "Sr", "Ba"]
ASITE_LABELS  = {
    "Pb": "Pb²⁺\n(lone pair)",
    "Ca": "Ca²⁺\n(small 2+)",
    "La": "La³⁺\n(rare earth)",
    "Sr": "Sr²⁺\n(mid 2+)",
    "Ba": "Ba²⁺\n(large 2+)",
}
# Reaney regime ordering: Ia (untilted) → III (heavily tilted)
REGIME_ORDER  = ["Ia", "Ib", "II", "III"]
REGIME_LABELS = {
    "Ia": "Ia  untilted\nparaelectric",
    "Ib": "Ib  incipient FE\nsoft-mode active",
    "II": "II  antiphase\ntilt moderate",
    "III": "III  in-phase\ntilt severe",
}

# LOFO R² from script 42 — used for extrapolation risk overlay
LOFO_R2 = {
    "asite":  {"Pb":-0.947, "Ba":-0.096, "Ca":0.685, "Sr":0.555, "La":0.652},
    "regime": {"Ia":-4.431, "Ib":-0.189, "II":0.685, "III":0.605},
}

def cell_risk(a, reg):
    """Returns 'HIGH', 'MODERATE', or 'LOW' for a given (A-site, regime) cell."""
    r_a   = LOFO_R2["asite"].get(a, 0.5)
    r_reg = LOFO_R2["regime"].get(reg, 0.5)
    worst = min(r_a, r_reg)
    if worst < 0.30:  return "HIGH"
    if worst < 0.65:  return "MODERATE"
    return "LOW"


# ── Model ─────────────────────────────────────────────────────────────────
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
def infer(models, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx, gii_n, phase_tr,
          regime_idx, batch=256):
    acc = {"pred":[], "delta_lst":[], "delta_tilt":[], "delta_res":[]}
    N   = len(er_cm)
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


def main():
    print("="*70)
    print("Script 45 — Composition Engineering Map")
    print("="*70)

    # ── Data ───────────────────────────────────────────────────────────────
    df = pd.read_parquet(os.path.join(PROC_DIR,"feature_matrix_v7.parquet"))
    with open(os.path.join(PROC_DIR,"feature_partition_v7.json")) as f: partition=json.load(f)

    def _get(cols):
        present = [c for c in cols if c in df.columns]
        return df[present].fillna(0.0).values.astype(np.float32)

    Xl = _get(partition["LST"]); Xt = _get(partition["Tilt"]); Xr = _get(partition["Residual"])
    er_cm     = df["er_CM"].fillna(0.0).values.astype(np.float32)
    has_cm    = df["has_sigma_CM"].fillna(0.0).values.astype(np.float32)
    cm_approx = df["cm_approx_flag"].fillna(0.0).values.astype(np.float32)
    phase_tr  = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
                if "phase_transition" in df.columns else np.zeros(len(df),np.float32)
    y         = df["epsilon_r"].values.astype(np.float32)
    gii_norm  = np.zeros(len(df), np.float32)

    models_saved, sc_lst, sc_tilt, sc_res, gii_max, regime_idx = load_model_and_scalers()
    Xl_s = sc_lst.transform(Xl).astype(np.float32)
    Xt_s = sc_tilt.transform(Xt).astype(np.float32)
    Xr_s = sc_res.transform(Xr).astype(np.float32)

    print("  Running inference on saved ensemble...")
    out = infer(models_saved, Xl_s, Xt_s, Xr_s, er_cm, has_cm, cm_approx,
                gii_norm, phase_tr, regime_idx)
    d_lst  = out["delta_lst"]; d_tilt = out["delta_tilt"]
    d_res  = out["delta_res"]; pred   = out["pred"]

    valid_cm = (has_cm > 0.5) & (er_cm > 5.0)
    em_cm    = er_cm.clip(1e-3)

    # Amplification factors
    f_lst  = np.where(valid_cm, d_lst  / em_cm, np.nan)   # soft-mode
    f_tilt = np.where(valid_cm, d_tilt / em_cm, np.nan)   # tilt suppression (negative)
    f_tot  = np.where(valid_cm, pred   / em_cm, np.nan)   # total predicted
    emp_f  = np.where(valid_cm, y      / em_cm, np.nan)   # experimental (ground truth)

    # Labels
    df["a_site"] = df["chemistry_family"].apply(lambda x: str(x).split("_")[0])
    dom_regime = np.full(len(df), "Ia", dtype=object)
    for rn, rc in [("III","regime_III"),("II","regime_II"),("Ib","regime_Ib")]:
        if rc in df.columns: dom_regime[df[rc].values > 0.5] = rn
    df["dom_regime"] = dom_regime

    # ── Build cell statistics ──────────────────────────────────────────────
    print("\n  Building A-site × Regime cell statistics:")
    nA, nR = len(ASITE_ORDER), len(REGIME_ORDER)

    f_lst_grid  = np.full((nR, nA), np.nan)  # rows=regime, cols=asite
    f_tilt_grid = np.full((nR, nA), np.nan)
    emp_grid    = np.full((nR, nA), np.nan)
    n_grid      = np.zeros((nR, nA), dtype=int)
    std_grid    = np.full((nR, nA), np.nan)

    cell_data = {}
    for ri, reg in enumerate(REGIME_ORDER):
        for ai, a in enumerate(ASITE_ORDER):
            mask = (valid_cm &
                    (df["a_site"].values == a) &
                    (df["dom_regime"].values == reg))
            n = int(mask.sum())
            n_grid[ri, ai] = n
            if n < 3:
                print(f"    {a:3}-{reg:3}: n={n:3d}  (too sparse, skip)")
                continue
            fl_vals = f_lst[mask][~np.isnan(f_lst[mask])]
            ft_vals = f_tilt[mask][~np.isnan(f_tilt[mask])]
            em_vals = emp_f[mask][~np.isnan(emp_f[mask])]
            if len(fl_vals) > 0:
                f_lst_grid[ri, ai]  = float(np.mean(fl_vals))
                std_grid[ri, ai]    = float(np.std(fl_vals))
            if len(ft_vals) > 0:
                f_tilt_grid[ri, ai] = float(np.mean(ft_vals))
            if len(em_vals) > 0:
                emp_grid[ri, ai]    = float(np.mean(em_vals))
            cell_data[f"{a}_{reg}"] = {
                "n": n, "f_lst": float(f_lst_grid[ri,ai]),
                "f_tilt": float(f_tilt_grid[ri,ai]) if not np.isnan(f_tilt_grid[ri,ai]) else None,
                "emp_f": float(emp_grid[ri,ai]) if not np.isnan(emp_grid[ri,ai]) else None,
                "std": float(std_grid[ri,ai]),
                "risk": cell_risk(a, reg),
            }
            emp_str = f"{emp_grid[ri,ai]:.2f}" if not np.isnan(emp_grid[ri,ai]) else "nan"
            print(f"    {a:3}-{reg:3}: n={n:3d}  f_lst={f_lst_grid[ri,ai]:.3f}  "
                  f"emp={emp_str}  risk={cell_risk(a, reg)}")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE — 2-panel design atlas
    # ══════════════════════════════════════════════════════════════════════
    print("\n[Generating design atlas figure...]")

    fig = plt.figure(figsize=(14, 9))
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 0.06],
                           hspace=0.35, wspace=0.38)

    ax_lst  = fig.add_subplot(gs[0, 0])   # f_LST heatmap
    ax_emp  = fig.add_subplot(gs[0, 1])   # experimental amplification heatmap
    ax_cbar1= fig.add_subplot(gs[1, 0])   # colorbar for f_LST
    ax_cbar2= fig.add_subplot(gs[1, 1])   # colorbar for experimental

    # ── Panel 1: f_LST (soft-mode amplification) ──────────────────────────
    vmin, vmax = 0.4, 2.8
    norm1  = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    cmap1  = plt.cm.RdYlGn  # red=suppressed, yellow=neutral, green=enhanced

    im1 = ax_lst.imshow(f_lst_grid, aspect="auto", cmap=cmap1, norm=norm1,
                        origin="upper", interpolation="nearest")

    # Hatching for risk zones
    for ri, reg in enumerate(REGIME_ORDER):
        for ai, a in enumerate(ASITE_ORDER):
            risk = cell_risk(a, reg)
            n    = n_grid[ri, ai]
            fl   = f_lst_grid[ri, ai]

            if np.isnan(fl):
                # Empty cell — gray fill
                rect = mpatches.FancyBboxPatch((ai-0.5, ri-0.5), 1.0, 1.0,
                    boxstyle="square,pad=0", linewidth=1.2,
                    facecolor="#CCCCCC", edgecolor="white", alpha=0.7)
                ax_lst.add_patch(rect)
                ax_lst.text(ai, ri, "n.d.", ha="center", va="center",
                            fontsize=8, color="#888888", style="italic")
                continue

            # Risk hatching overlay
            if risk == "HIGH":
                hatch_pat, hatch_alpha = "///", 0.40
                ax_lst.add_patch(mpatches.FancyBboxPatch(
                    (ai-0.49, ri-0.49), 0.98, 0.98,
                    boxstyle="square,pad=0", linewidth=0,
                    facecolor="none", edgecolor="#333333",
                    hatch=hatch_pat, alpha=hatch_alpha))
            elif risk == "MODERATE":
                ax_lst.add_patch(mpatches.FancyBboxPatch(
                    (ai-0.49, ri-0.49), 0.98, 0.98,
                    boxstyle="square,pad=0", linewidth=0,
                    facecolor="none", edgecolor="#555555",
                    hatch="...", alpha=0.30))

            # Cell annotations: f_LST value (top) and n (bottom)
            text_color = "white" if (fl > 2.0 or fl < 0.7) else "black"
            ax_lst.text(ai, ri-0.15, f"{fl:.2f}×",
                        ha="center", va="center", fontsize=10, fontweight="bold",
                        color=text_color)
            ax_lst.text(ai, ri+0.22, f"n={n}",
                        ha="center", va="center", fontsize=7.5, color=text_color, alpha=0.85)

    # Grid lines
    ax_lst.set_xticks(np.arange(nA)); ax_lst.set_yticks(np.arange(nR))
    ax_lst.set_xticklabels([ASITE_LABELS[a] for a in ASITE_ORDER], fontsize=8.5)
    ax_lst.set_yticklabels([REGIME_LABELS[r] for r in REGIME_ORDER], fontsize=8.5)
    ax_lst.set_xlabel("A-site Chemistry  (decreasing lone-pair activity →)", fontsize=10)
    ax_lst.set_ylabel("Reaney Structural Regime  (increasing tilt severity ↓)", fontsize=10)
    ax_lst.set_title("Predicted Soft-Mode Amplification  f_LST = δ_LST / εr_CM",
                     fontsize=11, fontweight="bold", pad=10)
    for spine in ax_lst.spines.values():
        spine.set_visible(True); spine.set_linewidth(1.5)

    # Region annotation arrows
    # High amplification zone: Pb-Ib (ai=0, ri=1)
    if not np.isnan(f_lst_grid[1, 0]):
        ax_lst.annotate("MAX AMP ZONE\nPb lone-pair × soft mode",
                        xy=(0, 1), xytext=(1.3, 0.2),
                        fontsize=8.5, fontweight="bold", color="#1a5c1a",
                        arrowprops=dict(arrowstyle="->", color="#1a5c1a", lw=1.4),
                        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#1a5c1a", alpha=0.85))

    # Suppressed zone: Ba-III (ai=4, ri=3)
    if not np.isnan(f_lst_grid[3, 4]):
        ax_lst.annotate("SUPPRESSED ZONE\nBa + in-phase tilt",
                        xy=(4, 3), xytext=(3.1, 3.7),
                        fontsize=8.5, fontweight="bold", color="#8b1a00",
                        arrowprops=dict(arrowstyle="->", color="#8b1a00", lw=1.4),
                        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0", ec="#8b1a00", alpha=0.85))

    # Design sweet spot: Ca-II
    if not np.isnan(f_lst_grid[2, 1]):
        ax_lst.annotate("RELIABLE\nDESIGN ZONE",
                        xy=(1, 2), xytext=(2.5, 1.5),
                        fontsize=8, color="#2563EB",
                        arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.2),
                        bbox=dict(boxstyle="round,pad=0.25", fc="#e8f0ff", ec="#2563EB", alpha=0.85))

    # ── Panel 2: Experimental amplification (ground truth from literature) ─
    vmin2, vmax2 = 0.6, 4.5
    norm2  = TwoSlopeNorm(vmin=vmin2, vcenter=1.5, vmax=vmax2)
    cmap2  = plt.cm.PuOr_r

    im2 = ax_emp.imshow(emp_grid, aspect="auto", cmap=cmap2, norm=norm2,
                        origin="upper", interpolation="nearest")

    for ri, reg in enumerate(REGIME_ORDER):
        for ai, a in enumerate(ASITE_ORDER):
            ev   = emp_grid[ri, ai]
            n    = n_grid[ri, ai]
            risk = cell_risk(a, reg)

            if np.isnan(ev):
                rect = mpatches.FancyBboxPatch((ai-0.5, ri-0.5), 1.0, 1.0,
                    boxstyle="square,pad=0", linewidth=1.2,
                    facecolor="#CCCCCC", edgecolor="white", alpha=0.7)
                ax_emp.add_patch(rect)
                ax_emp.text(ai, ri, "n.d.", ha="center", va="center",
                            fontsize=8, color="#888888", style="italic")
                continue

            if risk == "HIGH":
                ax_emp.add_patch(mpatches.FancyBboxPatch(
                    (ai-0.49, ri-0.49), 0.98, 0.98,
                    boxstyle="square,pad=0", linewidth=0,
                    facecolor="none", edgecolor="#333333",
                    hatch="///", alpha=0.40))
            elif risk == "MODERATE":
                ax_emp.add_patch(mpatches.FancyBboxPatch(
                    (ai-0.49, ri-0.49), 0.98, 0.98,
                    boxstyle="square,pad=0", linewidth=0,
                    facecolor="none", edgecolor="#555555",
                    hatch="...", alpha=0.30))

            text_color = "white" if ev > 3.5 else "black"
            ax_emp.text(ai, ri-0.15, f"{ev:.2f}×",
                        ha="center", va="center", fontsize=10, fontweight="bold",
                        color=text_color)
            ax_emp.text(ai, ri+0.22, f"n={n}",
                        ha="center", va="center", fontsize=7.5, color=text_color, alpha=0.85)

    ax_emp.set_xticks(np.arange(nA)); ax_emp.set_yticks(np.arange(nR))
    ax_emp.set_xticklabels([ASITE_LABELS[a] for a in ASITE_ORDER], fontsize=8.5)
    ax_emp.set_yticklabels([REGIME_LABELS[r] for r in REGIME_ORDER], fontsize=8.5)
    ax_emp.set_xlabel("A-site Chemistry  (decreasing lone-pair activity →)", fontsize=10)
    ax_emp.set_ylabel("Reaney Structural Regime", fontsize=10)
    ax_emp.set_title("Experimental Amplification  (εr_meas / εr_CM)\nfrom 1360 published dielectric measurements",
                     fontsize=11, fontweight="bold", pad=10)
    for spine in ax_emp.spines.values():
        spine.set_visible(True); spine.set_linewidth(1.5)

    # ── Colorbars ──────────────────────────────────────────────────────────
    plt.colorbar(im1, cax=ax_cbar1, orientation="horizontal", label="f_LST  (model predicted)")
    plt.colorbar(im2, cax=ax_cbar2, orientation="horizontal", label="εr_meas / εr_CM  (experimental)")

    # ── Legend for risk hatching ───────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor="white", edgecolor="#333333", hatch="///",
                       label="High extrapolation risk (LOFO R²<0.30)"),
        mpatches.Patch(facecolor="white", edgecolor="#555555", hatch="...",
                       label="Moderate risk (LOFO 0.30≤R²<0.65)"),
        mpatches.Patch(facecolor="white", edgecolor="black",
                       label="Reliable interpolation zone (LOFO R²≥0.65)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=9, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04),
               title="Extrapolation reliability (from LOFO analysis)",
               title_fontsize=9)

    fig.suptitle(
        "ABO₃ Dielectric Design Atlas: Soft-Mode Amplification Landscape\n"
        "Left: Physics-constrained model predictions  |  "
        "Right: Experimentally observed amplification",
        fontsize=12, fontweight="bold", y=1.01)

    fig.savefig(os.path.join(FIG_DIR,"45_design_atlas.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Save JSON results ─────────────────────────────────────────────────
    results = {
        "cell_data": cell_data,
        "lofo_r2": LOFO_R2,
        "asite_order": ASITE_ORDER,
        "regime_order": REGIME_ORDER,
        "design_rules": [
            "Maximum amplification: Pb²⁺ + Regime Ib — lone-pair × soft-mode synergy",
            "Reliable high-amplification: Ca²⁺ + Regime Ib or II — good balance of performance and data reliability",
            "Moderate amplification: La³⁺ systems — processability advantage over Pb",
            "Suppressed amplification: Ba²⁺ or Sr²⁺ + Regime III — dominated by tilt suppression",
            "High extrapolation risk: Pb and Ba cells, and all Ia/Ib regime cells — require experimental validation",
            "Reliable design zone: Ca/Sr/La × Regime II/III — sufficient training data for confident predictions",
        ],
        "grid_f_lst": [[float(f_lst_grid[ri,ai]) if not np.isnan(f_lst_grid[ri,ai]) else None
                        for ai in range(nA)] for ri in range(nR)],
        "grid_empirical": [[float(emp_grid[ri,ai]) if not np.isnan(emp_grid[ri,ai]) else None
                            for ai in range(nA)] for ri in range(nR)],
        "grid_n": [[int(n_grid[ri,ai]) for ai in range(nA)] for ri in range(nR)],
    }
    with open(os.path.join(RES_DIR,"45_composition_engineering_map.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved → figures/45_design_atlas.png")
    print(f"  Results → results/45_composition_engineering_map.json")

    print("\n" + "="*70)
    print("COMPOSITION ENGINEERING MAP COMPLETE")
    print("="*70)
    print("\n  f_LST Design Atlas Summary:")
    print(f"  {'':4}  " + "  ".join(f"{a:>7}" for a in ASITE_ORDER))
    for ri, reg in enumerate(REGIME_ORDER):
        row = []
        for ai, a in enumerate(ASITE_ORDER):
            v = f_lst_grid[ri, ai]
            row.append(f"{v:7.3f}" if not np.isnan(v) else "    n.d.")
        print(f"  {reg:4}  {'  '.join(row)}")

    print("\n  Experimental Amplification (εr_meas/εr_CM) Summary:")
    print(f"  {'':4}  " + "  ".join(f"{a:>7}" for a in ASITE_ORDER))
    for ri, reg in enumerate(REGIME_ORDER):
        row = []
        for ai, a in enumerate(ASITE_ORDER):
            v = emp_grid[ri, ai]
            row.append(f"{v:7.3f}" if not np.isnan(v) else "    n.d.")
        print(f"  {reg:4}  {'  '.join(row)}")

    print("\n  Design Rules:")
    for i, rule in enumerate(results["design_rules"], 1):
        print(f"  {i}. {rule}")
    print("="*70)


if __name__ == "__main__":
    main()
