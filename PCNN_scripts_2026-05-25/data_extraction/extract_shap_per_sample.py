import os

# --- self-contained release root (auto-injected) ---
RELEASE_ROOT = os.path.abspath(os.path.dirname(__file__))
while RELEASE_ROOT != os.path.dirname(RELEASE_ROOT) and not os.path.isdir(os.path.join(RELEASE_ROOT, 'model_weights')):
    RELEASE_ROOT = os.path.dirname(RELEASE_ROOT)
# ----------------------------------------------------

"""
Convert the saved SHAP .npy file into a tidy per-sample CSV.

Input:  results/60_shap_values.npy + 60_supplementary_analysis.json
Output: all_figures/extracted_data/shap_per_sample.csv
        all_figures/extracted_data/shap_feature_importance.csv
"""
import os, json
import numpy as np
import pandas as pd



OUT_DIR    = os.path.join(RELEASE_ROOT, "extracted_data")
os.makedirs(OUT_DIR, exist_ok=True)
PIML       = RELEASE_ROOT
SHAP_NPY   = f"{PIML}/results/60_shap_values.npy"
SUPP_JSON  = f"{PIML}/results/60_supplementary_analysis.json"
FM_PATH    = f"{PIML}/data/processed/feature_matrix_v7.parquet"

shap = np.load(SHAP_NPY, allow_pickle=True)
if isinstance(shap, np.ndarray) and shap.dtype == object and shap.shape == ():
    shap = shap.item()
print(f"SHAP loaded — type={type(shap).__name__}", end=" ")
if hasattr(shap, "shape"):
    print(f"shape={shap.shape}")
else:
    print(f"keys={list(shap.keys())[:6] if hasattr(shap,'keys') else 'N/A'}")

with open(SUPP_JSON) as f:
    supp = json.load(f)

shap_meta = supp.get("shap", {})
feat_names = shap_meta.get("feature_names")
if feat_names is None:
    fm = pd.read_parquet(FM_PATH)
    feat_names = [c for c in fm.columns if c not in
                  ("epsilon_r", "formula", "chemistry_family", "DC")]

# Coerce to ndarray
if isinstance(shap, dict):
    values = shap.get("values", shap.get("shap_values"))
    if values is None:
        # Try first array key
        values = next(iter(shap.values()))
    shap_arr = np.asarray(values)
else:
    shap_arr = np.asarray(shap)
print(f"  SHAP matrix shape: {shap_arr.shape}")
n_samples, n_features = shap_arr.shape
print(f"  n_features_supplied = {len(feat_names)}")
if len(feat_names) != n_features:
    print("  WARNING: feature name count mismatches SHAP columns — using generic names")
    feat_names = [f"f_{i}" for i in range(n_features)]

# Per-sample tidy table: long format
sample_idx = np.repeat(np.arange(n_samples), n_features)
feat_col   = np.tile(np.arange(n_features), n_samples)
records = pd.DataFrame({
    "sample_idx":   sample_idx,
    "feature_name": np.array(feat_names)[feat_col],
    "shap_value":   shap_arr.flatten(),
})
records.to_csv(f"{OUT_DIR}/shap_per_sample.csv", index=False)
print(f"Saved: {OUT_DIR}/shap_per_sample.csv  ({len(records)} rows)")

# Feature-level importance (|SHAP| mean per feature)
imp = pd.DataFrame({
    "feature":           feat_names,
    "mean_abs_shap":     np.abs(shap_arr).mean(axis=0),
    "mean_signed_shap":  shap_arr.mean(axis=0),
    "std_shap":          shap_arr.std(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
imp.to_csv(f"{OUT_DIR}/shap_feature_importance.csv", index=False)
print(f"Saved: {OUT_DIR}/shap_feature_importance.csv  ({len(imp)} features)")

print("\nTop 10 features by |SHAP|:")
print(imp.head(10).to_string(index=False))
