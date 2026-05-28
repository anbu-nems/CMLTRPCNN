"""
Phase 1 — Script 03: Residual Target Computation
Computes Δεr = DC - εr_CM, runs physics sanity checks, saves residual_target.csv.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

IN_CSV  = os.path.join(os.path.dirname(__file__), "..", "data", "feature_matrix.csv")
OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "residual_target.csv")
RES_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

df = pd.read_csv(IN_CSV)
print(f"Loaded feature matrix: {len(df)} rows")

# ── Compute residual target ───────────────────────────────────────────────────
df["delta_er"] = df["DC"] - df["er_CM"]
valid_mask = df["delta_er"].notna() & df["er_CM"].notna() & df["DC"].notna()
print(f"Valid Δεr: {valid_mask.sum()} / {len(df)}")

# ── Physics sanity checks (CLAUDE.md requirement) ─────────────────────────────
# These MUST pass before any model training.
SANITY_CHECKS = {
    "SrTiO3":  {"er_CM_target": 40.0,  "er_CM_tol": 5.0},
    "LaAlO3":  {"er_CM_target": 18.0,  "er_CM_tol": 4.0},
    "BaTiO3":  {"er_CM_target": 35.0,  "er_CM_tol": 6.0},
}

print("\n── Physics sanity checks ──")
all_passed = True
for formula, check in SANITY_CHECKS.items():
    matches = df[df["formula"].str.strip() == formula]
    if len(matches) == 0:
        # Try partial match
        matches = df[df["formula"].str.contains(formula.replace("3", ""), case=False, na=False)]
    if len(matches) == 0:
        print(f"  {formula:12s}  NOT FOUND in dataset — computing from scratch")
        from src.features.feature_engineering import compute_features
        row = pd.Series({"Materials": formula})
        feats = compute_features(row)
        er_cm_val = feats.get("er_CM", float("nan"))
    else:
        er_cm_val = matches["er_CM"].iloc[0]

    target  = check["er_CM_target"]
    tol     = check["er_CM_tol"]
    passed  = abs(er_cm_val - target) <= tol if not np.isnan(er_cm_val) else False
    status  = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"  {formula:12s}  εr_CM={er_cm_val:.1f}  (expected ≈{target}±{tol})  [{status}]")

if not all_passed:
    print("\nWARNING: Some sanity checks failed — review polarizability/radii tables.")
else:
    print("\nAll sanity checks passed.")

# ── Δεr statistics ────────────────────────────────────────────────────────────
df_valid = df[valid_mask].copy()
print(f"\n── Δεr statistics (n={len(df_valid)}) ──")
print(f"  mean  : {df_valid['delta_er'].mean():.2f}")
print(f"  std   : {df_valid['delta_er'].std():.2f}")
print(f"  min   : {df_valid['delta_er'].min():.2f}")
print(f"  max   : {df_valid['delta_er'].max():.2f}")
print(f"  Δεr > 0: {(df_valid['delta_er'] > 0).sum()} ({100*(df_valid['delta_er'] > 0).mean():.1f}%)")
print(f"  Δεr < 0: {(df_valid['delta_er'] < 0).sum()} ({100*(df_valid['delta_er'] < 0).mean():.1f}%)")

# ── Δεr by regime ─────────────────────────────────────────────────────────────
print("\n── Mean Δεr by Reaney regime ──")
for regime in ["Ia", "Ib", "II", "III"]:
    sub = df_valid[df_valid["reaney_regime"] == regime]
    if len(sub):
        print(f"  Regime {regime}: n={len(sub):4d}  mean Δεr={sub['delta_er'].mean():+.1f}"
              f"  std={sub['delta_er'].std():.1f}")

# ── Δεr by soft-mode proxy ────────────────────────────────────────────────────
print("\n── Mean Δεr by soft-mode proxy ──")
for flag, label in [(1, "True (soft-mode)"), (0, "False")]:
    sub = df_valid[df_valid["d0_B_polarizable_A"] == flag]
    if len(sub):
        print(f"  d0_B_polarizable_A={label}: n={len(sub):4d}  mean Δεr={sub['delta_er'].mean():+.1f}")

# ── Figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. DC vs εr_CM parity plot
axes[0].scatter(df_valid["er_CM"], df_valid["DC"], alpha=0.4, s=15, color="#2c7bb6")
lims = [0, df_valid[["DC", "er_CM"]].max().max() * 1.05]
axes[0].plot(lims, lims, "k--", lw=1, label="y=x")
axes[0].set_xlabel("εr_CM (Clausius-Mossotti)")
axes[0].set_ylabel("εr measured (DC)")
axes[0].set_title("CM baseline parity")
axes[0].legend()

# 2. Δεr distribution
axes[1].hist(df_valid["delta_er"], bins=40, color="#d7191c", edgecolor="white")
axes[1].axvline(0, color="k", lw=1.5, ls="--")
axes[1].set_xlabel("Δεr = εr_meas − εr_CM")
axes[1].set_title("Residual distribution")

# 3. Δεr vs tolerance factor, coloured by soft-mode
sc_colors = df_valid["d0_B_polarizable_A"].map({1: "#d7191c", 0: "#2c7bb6"})
axes[2].scatter(df_valid["tolerance_factor"], df_valid["delta_er"],
                c=sc_colors, alpha=0.4, s=15)
axes[2].axhline(0, color="k", lw=1, ls="--")
axes[2].set_xlabel("Tolerance factor t")
axes[2].set_ylabel("Δεr")
axes[2].set_title("Δεr vs t (red=soft-mode)")

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_residual_target.png"), dpi=300)
plt.close()
print(f"\nFigure saved → figures/03_residual_target.png")

# ── Save ─────────────────────────────────────────────────────────────────────
df.to_csv(OUT_CSV, index=False)
print(f"Residual target saved → data/residual_target.csv")

metrics = {
    "n_valid": int(valid_mask.sum()),
    "delta_er_mean": float(df_valid["delta_er"].mean()),
    "delta_er_std": float(df_valid["delta_er"].std()),
    "delta_er_min": float(df_valid["delta_er"].min()),
    "delta_er_max": float(df_valid["delta_er"].max()),
    "pct_positive_delta": float(100 * (df_valid["delta_er"] > 0).mean()),
    "sanity_checks_all_passed": all_passed,
}
with open(os.path.join(RES_DIR, "03_residual_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✓ Script 03 complete — residual_target.csv written")
print(f"  valid={valid_mask.sum()}  Δεr_mean={df_valid['delta_er'].mean():.1f}  sanity_ok={all_passed}")
