"""
Script 59 — Physics Law Combinatorial Ablation (2³ factorial design)

Three physical laws embedded in CMLTRPCNNv7.7:
  Law I   — Clausius-Mossotti (CM):       electrostatic anchor εr_CM
  Law II  — Lyddane-Sachs-Teller (LST):   soft-mode enhancement δ_LST ≥ 0
  Law III — Reaney/Goldschmidt Tilt:      octahedral suppression δ_tilt ≤ 0

8 conditions (2³):
  000 — No physics (data-driven residual only)
  I   — CM only
  II  — LST only
  III — Tilt only
  I+II   — CM + LST
  I+III  — CM + Tilt
  II+III — LST + Tilt (no CM anchor)
  I+II+III — Full model (all three laws)

Outputs: results/59_physics_law_ablation.json
         NC figures/comparison/04_physics_law_ablation.csv
         NC figures/comparison/fig_physics_laws_*.pdf/png
"""
import sys, os, json, warnings, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error

DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT     = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR  = os.path.join(ROOT, "results")
COMP_DIR = "/Users/anbu/Desktop/NC figures/comparison"

from src.models.psrnn_mdpinn import CMLTRPCNNv71

# ── Load data & checkpoint ────────────────────────────────────────────────────
df        = pd.read_parquet(os.path.join(PROC_DIR, "feature_matrix_v7.parquet"))
partition = json.load(open(os.path.join(PROC_DIR, "feature_partition_v7.json")))
calib     = json.load(open(os.path.join(PROC_DIR, "calibration_split_idx.json")))

def get(cols):
    present = [c for c in cols if c in df.columns]
    return df[present].fillna(0.0).values.astype(np.float32), present

Xl, lcols = get(partition["LST"])
Xt, tcols = get(partition["Tilt"])
Xr, rcols = get(partition["Residual"])

y         = df["epsilon_r"].values.astype(np.float32)
er_cm     = df["er_CM"].fillna(0.0).values.astype(np.float32)
has_cm    = df["has_sigma_CM"].fillna(0.0).values.astype(np.float32)
cm_approx = df["cm_approx_flag"].fillna(0.0).values.astype(np.float32)
gii_norm  = np.zeros(len(df), np.float32)
phase_tr  = df["phase_transition"].fillna(0.0).values.astype(np.float32) \
            if "phase_transition" in df.columns else np.zeros(len(df), np.float32)

train_idx  = np.array(calib["train_idx"])
calib_idx  = np.array(calib["calib_idx"])

ckpt = torch.load(os.path.join(ROOT, "models", "cmltrv77_final.pt"),
                  map_location="cpu", weights_only=False)
hp   = ckpt["hp"]

with open(os.path.join(ROOT, "models", "cmltrv77_scalers.pkl"), "rb") as f:
    scalers = pickle.load(f)

Xl_s = scalers["sc_lst"].transform(Xl).astype(np.float32)
Xt_s = scalers["sc_tilt"].transform(Xt).astype(np.float32)
Xr_s = scalers["sc_res"].transform(Xr).astype(np.float32)

regime_names = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
regime_idx   = [lcols.index(r) for r in regime_names if r in lcols]

# ── Modified forward pass ─────────────────────────────────────────────────────
# Training mean used as baseline anchor when CM is disabled
TRAIN_MEAN_Y = float(y[train_idx].mean())

def predict_with_laws(use_cm, use_lst, use_tilt, idx):
    """
    Run ensemble prediction with specified physics laws active/disabled.
    Reconstructs prediction post-hoc from model component outputs:
      pred = base + [delta_lst] + [delta_tilt] + [delta_res]
    where base = has_cm-gated er_cm  (if use_cm)
               = training mean y     (if not use_cm)

    use_cm  : bool — whether CM anchor (Law I) is used
    use_lst : bool — whether LST branch δ_LST (Law II) is active
    use_tilt: bool — whether Tilt branch δ_tilt (Law III) is active
    """
    preds = []
    for sd in ckpt["models"]:
        m = CMLTRPCNNv71(
            n_lst=Xl.shape[1], n_tilt=Xt.shape[1], n_res=Xr.shape[1],
            trunk_hidden=hp["trunk_hidden"], n_trunk_blocks=hp["n_trunk_blocks"],
            lst_hidden=hp["lst_hidden"], tilt_hidden=hp["tilt_hidden"],
            res_hidden=hp["res_hidden"], dropout=0.0,
            residual_scale=hp["residual_scale"]
        )
        m.load_state_dict(sd)
        m.eval().to(DEVICE)

        with torch.no_grad():
            xl_  = torch.tensor(Xl_s[idx]).to(DEVICE)
            xt_  = torch.tensor(Xt_s[idx]).to(DEVICE)
            xr_  = torch.tensor(Xr_s[idx]).to(DEVICE)
            ecm_ = torch.tensor(er_cm[idx]).to(DEVICE)
            hcm_ = torch.tensor(has_cm[idx]).to(DEVICE)
            cma_ = torch.tensor(cm_approx[idx]).to(DEVICE)
            g_   = torch.tensor(gii_norm[idx]).to(DEVICE)
            pt_  = torch.tensor(phase_tr[idx]).to(DEVICE)

            # Full forward to get all branch components
            out = m(xl_, xt_, xr_, ecm_, hcm_, cma_, g_, pt_, regime_idx)

        # Reconstruct prediction using only active laws
        # out["er_cm"] is the has_cm-gated CM anchor from the model
        if use_cm:
            base = out["er_cm"].cpu()
        else:
            base = torch.full((len(idx),), TRAIN_MEAN_Y)

        dlst  = out["delta_lst"].cpu()  if use_lst  else torch.zeros(len(idx))
        dtilt = out["delta_tilt"].cpu() if use_tilt else torch.zeros(len(idx))
        dres  = out["delta_res"].cpu()

        pred = (base + dlst + dtilt + dres).clamp(1, 600)
        preds.append(pred.cpu().numpy())
    return np.mean(preds, axis=0)

# ── Run all 8 conditions ──────────────────────────────────────────────────────
conditions = [
    ("No physics\n(residual only)", False, False, False, "000"),
    ("Law I only\n(CM)",            True,  False, False, "I"),
    ("Law II only\n(LST)",          False, True,  False, "II"),
    ("Law III only\n(Tilt)",        False, False, True,  "III"),
    ("Law I + II\n(CM + LST)",      True,  True,  False, "I+II"),
    ("Law I + III\n(CM + Tilt)",    True,  False, True,  "I+III"),
    ("Law II + III\n(LST + Tilt)",  False, True,  True,  "II+III"),
    ("All Laws\n(I + II + III)",    True,  True,  True,  "I+II+III"),
]

print(f"{'='*65}")
print(f"  Physics Law Combinatorial Ablation (2³ = 8 conditions)")
print(f"  Evaluated on held-out test set (n={len(calib_idx)})")
print(f"{'='*65}")
print(f"  {'Condition':<28} {'CM':^5} {'LST':^5} {'Tilt':^5} {'R²':>8} {'MAE':>8}")
print(f"  {'-'*62}")

results = {}
for label, cm, lst, tilt, code in conditions:
    p = predict_with_laws(cm, lst, tilt, calib_idx)
    r2  = float(r2_score(y[calib_idx], p))
    mae = float(mean_absolute_error(y[calib_idx], p))
    flag = " ◄ full model" if code == "I+II+III" else ""
    short = label.replace("\n"," ")
    print(f"  {short:<28} {'✓' if cm else '✗':^5} {'✓' if lst else '✗':^5} {'✓' if tilt else '✗':^5} {r2:>8.4f} {mae:>8.3f}{flag}")
    results[code] = {
        "label": label.replace("\n"," "),
        "use_cm": cm, "use_lst": lst, "use_tilt": tilt,
        "r2": round(r2, 4), "mae": round(mae, 3)
    }

print(f"{'='*65}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
out = {
    "description": "Physics law combinatorial ablation — 2³ factorial design",
    "laws": {
        "I":   "Clausius-Mossotti — electrostatic polarizability anchor (εr_CM)",
        "II":  "Lyddane-Sachs-Teller — soft-mode enhancement (δ_LST ≥ 0)",
        "III": "Reaney/Goldschmidt tilt — octahedral suppression (δ_tilt ≤ 0)"
    },
    "n_test": int(len(calib_idx)),
    "conditions": results
}
json.dump(out, open(os.path.join(RES_DIR, "59_physics_law_ablation.json"), "w"), indent=2)
print(f"\nSaved: results/59_physics_law_ablation.json")

# ── Save CSV ──────────────────────────────────────────────────────────────────
rows = []
for code, v in results.items():
    rows.append({
        "condition": code,
        "label": v["label"],
        "use_cm":   int(v["use_cm"]),
        "use_lst":  int(v["use_lst"]),
        "use_tilt": int(v["use_tilt"]),
        "test_r2":  v["r2"],
        "test_mae": v["mae"],
    })
csv_path = os.path.join(COMP_DIR, "04_physics_law_ablation.csv")
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"Saved: comparison/04_physics_law_ablation.csv")

# ── Generate Figures ───────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mplcm

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linestyle": "--",
})

codes  = [c[4] for c in conditions]
labels_short = ["None\n(000)", "CM\n(I)", "LST\n(II)", "Tilt\n(III)",
                "CM+LST\n(I+II)", "CM+Tilt\n(I+III)", "LST+Tilt\n(II+III)", "All\n(I+II+III)"]
r2s    = [results[c]["r2"]  for c in codes]
maes   = [results[c]["mae"] for c in codes]

# Color by number of active laws
n_laws = [sum([results[c]["use_cm"], results[c]["use_lst"], results[c]["use_tilt"]]) for c in codes]
palette = {0: "#B0BEC5", 1: "#5E81AC", 2: "#A3BE8C", 3: "#E76F51"}
bar_colors = [palette[n] for n in n_laws]
bar_colors[-1] = "#E76F51"  # full model always coral

# ── Figure 1: R² bar chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 3.5))
x = np.arange(len(codes))
bars = ax.bar(x, r2s, color=bar_colors, edgecolor="white", linewidth=0.6,
              width=0.65, zorder=3)

# Value labels
for bar, val in zip(bars, r2s):
    ypos = val + 0.02 if val >= 0 else val - 0.08
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{val:.3f}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            color="#222" if val > 0 else "#c00")

# Law contribution arrows
ax.annotate("", xy=(4, r2s[4]), xytext=(1, r2s[1]),
            arrowprops=dict(arrowstyle="->", color="#5E81AC", lw=1.2))
ax.annotate("", xy=(4, r2s[4]), xytext=(2, r2s[2]),
            arrowprops=dict(arrowstyle="->", color="#A3BE8C", lw=1.2))

ax.axhline(0, color="black", linewidth=0.7, zorder=2)
ax.axhline(r2s[-1], color="#E76F51", linewidth=1.0, linestyle="--",
           alpha=0.5, label=f"Full model R²={r2s[-1]:.3f}")

ax.set_xticks(x)
ax.set_xticklabels(labels_short, fontsize=8)
ax.set_ylabel("Test R² (held-out, n=123)")
ax.set_ylim(-1.35, 1.08)
ax.set_title("Physics Law Combinatorial Ablation — 2³ Factorial Design\n"
             "Laws: I = Clausius-Mossotti  |  II = Lyddane-Sachs-Teller  |  III = Reaney Tilt")

legend_patches = [
    mpatches.Patch(color=palette[0], label="0 laws (data-driven)"),
    mpatches.Patch(color=palette[1], label="1 law"),
    mpatches.Patch(color=palette[2], label="2 laws"),
    mpatches.Patch(color=palette[3], label="3 laws (full PCNN)"),
]
ax.legend(handles=legend_patches, loc="lower right", ncol=2, fontsize=7.5)

plt.tight_layout()
fig.savefig(os.path.join(COMP_DIR, "fig_physics_laws_r2.pdf"))
fig.savefig(os.path.join(COMP_DIR, "fig_physics_laws_r2.png"), dpi=300)
plt.close()
print("Saved fig_physics_laws_r2")

# ── Figure 2: Heatmap of R² vs Law combinations ───────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))

# Build 2x4 grid: rows = CM (0/1), cols = (LST, Tilt) combinations
combos  = [(0,0), (1,0), (0,1), (1,1)]
combo_labels = ["No LST,\nNo Tilt", "LST only", "Tilt only", "LST + Tilt"]
cm_labels = ["No CM\n(Law I off)", "With CM\n(Law I on)"]

matrix = np.zeros((2, 4))
for ci, (use_lst, use_tilt) in enumerate(combos):
    for ri, use_cm in enumerate([False, True]):
        code_match = [c for c in codes
                      if results[c]["use_cm"] == use_cm
                      and results[c]["use_lst"] == bool(use_lst)
                      and results[c]["use_tilt"] == bool(use_tilt)]
        if code_match:
            matrix[ri, ci] = results[code_match[0]]["r2"]

vmax = max(r2s)
vmin = min(r2s)
im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="auto")

for i in range(2):
    for j in range(4):
        val = matrix[i, j]
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white" if abs(val) > 0.6 else "#222")

ax.set_xticks(range(4))
ax.set_xticklabels(combo_labels, fontsize=8.5)
ax.set_yticks(range(2))
ax.set_yticklabels(cm_labels, fontsize=8.5)
ax.set_title("Test R² Heatmap — Physics Law Interactions\n"
             "Law I: CM  |  Law II: LST  |  Law III: Tilt", fontsize=9.5)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Test R²", fontsize=8.5)

plt.tight_layout()
fig.savefig(os.path.join(COMP_DIR, "fig_physics_laws_heatmap.pdf"))
fig.savefig(os.path.join(COMP_DIR, "fig_physics_laws_heatmap.png"), dpi=300)
plt.close()
print("Saved fig_physics_laws_heatmap")

# ── Figure 3: Marginal contribution of each law ───────────────────────────────
# Marginal: mean R² gain when adding each law across all contexts
r2_none   = results["000"]["r2"]
r2_I      = results["I"]["r2"]
r2_II     = results["II"]["r2"]
r2_III    = results["III"]["r2"]
r2_full   = results["I+II+III"]["r2"]

# Shapley-inspired: average marginal contribution
# Law I marginal: avg of [R(I)-R(0), R(I+II)-R(II), R(I+III)-R(III), R(I+II+III)-R(II+III)]
marg_I   = np.mean([results["I"]["r2"]       - results["000"]["r2"],
                    results["I+II"]["r2"]    - results["II"]["r2"],
                    results["I+III"]["r2"]   - results["III"]["r2"],
                    results["I+II+III"]["r2"]- results["II+III"]["r2"]])

marg_II  = np.mean([results["II"]["r2"]      - results["000"]["r2"],
                    results["I+II"]["r2"]    - results["I"]["r2"],
                    results["II+III"]["r2"]  - results["III"]["r2"],
                    results["I+II+III"]["r2"]- results["I+III"]["r2"]])

marg_III = np.mean([results["III"]["r2"]     - results["000"]["r2"],
                    results["I+III"]["r2"]   - results["I"]["r2"],
                    results["II+III"]["r2"]  - results["II"]["r2"],
                    results["I+II+III"]["r2"]- results["I+II"]["r2"]])

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))

# Panel A — Marginal contributions (Shapley-style)
ax = axes[0]
law_names = ["Law I\n(Clausius-Mossotti)", "Law II\n(Lyddane-Sachs-Teller)", "Law III\n(Reaney Tilt)"]
marginals = [marg_I, marg_II, marg_III]
law_colors = ["#5E81AC", "#A3BE8C", "#EBCB8B"]
bars = ax.bar(law_names, marginals, color=law_colors, edgecolor="white",
              linewidth=0.6, width=0.5, zorder=3)
for bar, val in zip(bars, marginals):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.005 if val >= 0 else val - 0.02,
            f"ΔR²={val:+.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylabel("Average Marginal ΔR²")
ax.set_title("(a)  Shapley-Style Marginal Contribution\nof Each Physical Law")
ax.set_ylim(min(marginals)-0.15, max(marginals)+0.08)

# Panel B — Staircase: building up from 0 to full model
ax = axes[1]
stair_labels = ["None\n(data only)", "+CM\n(Law I)", "+LST\n(Law II)", "+Tilt\n(Law III)"]
stair_vals   = [results["000"]["r2"], results["I"]["r2"],
                results["I+II"]["r2"], results["I+II+III"]["r2"]]
stair_colors = [palette[0], palette[1], palette[2], palette[3]]
x = np.arange(len(stair_labels))
bars = ax.bar(x, stair_vals, color=stair_colors, edgecolor="white",
              linewidth=0.6, width=0.55, zorder=3)
for i in range(1, len(stair_vals)):
    delta = stair_vals[i] - stair_vals[i-1]
    ax.annotate("", xy=(i, stair_vals[i]), xytext=(i, stair_vals[i-1]),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.text(i + 0.28, (stair_vals[i]+stair_vals[i-1])/2,
            f"+{delta:.3f}", va="center", fontsize=8, color="#333")
for bar, val in zip(bars, stair_vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            max(val, 0)+0.02, f"{val:.3f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(stair_labels, fontsize=8.5)
ax.set_ylabel("Test R²")
ax.set_title("(b)  Cumulative R² as Laws Added Sequentially")
ax.set_ylim(-0.2, 1.05)

plt.tight_layout()
fig.savefig(os.path.join(COMP_DIR, "fig_physics_laws_contribution.pdf"))
fig.savefig(os.path.join(COMP_DIR, "fig_physics_laws_contribution.png"), dpi=300)
plt.close()
print("Saved fig_physics_laws_contribution")

# Print marginal contributions
print(f"\n{'='*50}")
print(f"  Shapley-style marginal R² contribution per law")
print(f"{'='*50}")
print(f"  Law I   (CM):  ΔR² = {marg_I:+.4f}")
print(f"  Law II  (LST): ΔR² = {marg_II:+.4f}")
print(f"  Law III (Tilt):ΔR² = {marg_III:+.4f}")
print(f"{'='*50}")

# Save marginals
results["_marginal_contributions"] = {
    "Law_I_CM":    round(marg_I, 4),
    "Law_II_LST":  round(marg_II, 4),
    "Law_III_Tilt":round(marg_III, 4),
    "note": "Shapley-style: average marginal R² gain across all contexts"
}
json.dump(out | {"marginal_contributions": results["_marginal_contributions"]},
          open(os.path.join(RES_DIR, "59_physics_law_ablation.json"), "w"), indent=2)
print(f"\nAll figures saved to: {COMP_DIR}")
