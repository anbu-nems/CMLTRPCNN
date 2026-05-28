#!/usr/bin/env python
"""
68 — Re-check GA inverse-design candidates under the REAL applicability domain.

The GA (script 67) recorded `in_ad` by testing the ENSEMBLE STD (≈3-5 ε_r units)
against SIGMA_AD_CUTOFF=0.35 — but 0.35 is the threshold for the model's
*sigma_conf head* (a sigmoid in [0,1]; script 49). So the GA's in_ad flag is a
scale bug and the candidates' true AD status was never computed.

This script reuses the exact inference/featurization pipeline (the same modules the
GA used) and, for each GA optimum, computes:
  • the REAL dual AD (49): layer1 = out['sigma_conf'] < 0.35 ; layer2 = KDE density gate
  • the mechanistic decomposition ε_CM / δ_LST / δ_tilt / δ_res and f_LST
  • nearest-neighbour distance to the training set (independence / not-memorised check)
  • the conformal lower bound (Q90 = 14.38)

→ results/68_ga_candidate_ad_recheck_v77.json
"""
import importlib.util, os, json, math
import numpy as np, torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

ROOT = "/Users/anbu/Desktop/PIML/piml_ceramic"
Q90 = 14.38
SIGMA_CONF_THRESHOLD = 0.35   # real AD layer-1 threshold (script 49)
KDE_PERCENTILE, PCA_COMPONENTS = 5, 10

# ── reuse the GA module's exact featurization (importlib: name starts with a digit)
spec = importlib.util.spec_from_file_location(
    "ga67", os.path.join(ROOT, "scripts", "67_physics_ga_inverse_design.py"))
ga = importlib.util.module_from_spec(spec); spec.loader.exec_module(ga)
state = ga._inf._load()
sc_lst, sc_tilt, sc_res = state["sc_lst"], state["sc_tilt"], state["sc_res"]
lst_cols, tilt_cols, res_cols = state["lst_cols"], state["tilt_cols"], state["res_cols"]
regime_idx, models = state["regime_idx"], state["models"]
GII_MAX = ga.GII_MAX


def featurize(formula):
    feats = ga._build_feature_row_dict(formula)
    if feats is None:
        return None
    def row(cols):
        return np.array([[feats.get(c, np.nan) if feats.get(c, None) is not None else np.nan
                          for c in cols]], dtype=np.float32)
    Xl, Xt, Xr = row(lst_cols), row(tilt_cols), row(res_cols)
    er = feats.get("er_CM", np.nan)
    er_cm = np.array([10.0 if (er is None or math.isnan(er) or er <= 0) else er], np.float32)
    has_cm = np.array([0.0 if (er is None or math.isnan(er) or er <= 0)
                       else float(feats.get("has_sigma_CM", 1.0))], np.float32)
    cm_approx = np.array([float(feats.get("cm_approx_flag", 0.0))], np.float32)
    gii_n = np.array([float(np.clip(float(feats.get("GII", 0.0)) / GII_MAX, 0, 1))], np.float32)
    phase_tr = np.array([float(feats.get("phase_transition", 0.0))], np.float32)
    return Xl, Xt, Xr, er_cm, has_cm, cm_approx, gii_n, phase_tr


def infer(Xl, Xt, Xr, er_cm, has_cm, cm_approx, gii_n, phase_tr):
    Xl_s = np.nan_to_num(sc_lst.transform(Xl), nan=0.0).astype(np.float32)
    Xt_s = np.nan_to_num(sc_tilt.transform(Xt), nan=0.0).astype(np.float32)
    Xr_s = np.nan_to_num(sc_res.transform(Xr), nan=0.0).astype(np.float32)
    t = lambda a: torch.from_numpy(a)
    preds, sigs, dlst, dtilt, dres = [], [], [], [], []
    with torch.no_grad():
        for m in models:
            o = m(t(Xl_s), t(Xt_s), t(Xr_s), t(er_cm), t(has_cm), t(cm_approx),
                  t(gii_n), t(phase_tr), regime_idx)
            preds.append(float(o["pred"][0]))
            sigs.append(float(o["sigma_conf"][0]))
            dlst.append(float(o["delta_lst"][0])); dtilt.append(float(o["delta_tilt"][0]))
            dres.append(float(o["delta_res"][0]))
    return (np.mean(preds), np.std(preds), np.mean(sigs),
            np.mean(dlst), np.mean(dtilt), np.mean(dres), np.hstack([Xl_s, Xt_s, Xr_s])[0])


# ── build the training KDE + scaled cloud (reproduce script 49 layer-2) ──────────
df = pd.read_parquet(os.path.join(ROOT, "data/processed/feature_matrix_v7.parquet"))
part = json.load(open(os.path.join(ROOT, "data/processed/feature_partition_v7.json")))
split = json.load(open(os.path.join(ROOT, "data/processed/calibration_split_idx.json")))
train_idx = np.array(split["train_idx"])
def get(cols): return df[[c for c in cols if c in df.columns]].fillna(0.0).values.astype(np.float32)
Xtr = np.hstack([np.nan_to_num(sc_lst.transform(get(part["LST"])), nan=0.0),
                 np.nan_to_num(sc_tilt.transform(get(part["Tilt"])), nan=0.0),
                 np.nan_to_num(sc_res.transform(get(part["Residual"])), nan=0.0)]).astype(np.float32)
pca = PCA(n_components=PCA_COMPONENTS, random_state=42); Xpca = pca.fit_transform(Xtr)
kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(Xpca[train_idx])
thr = np.percentile(kde.score_samples(Xpca[train_idx]), KDE_PERCENTILE)
# training nearest-neighbour spacing (median) for context
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=2).fit(Xtr[train_idx])
train_nn = nn.kneighbors(Xtr[train_idx])[0][:, 1]
train_nn_median, train_nn_p95 = float(np.median(train_nn)), float(np.percentile(train_nn, 95))

# ── candidates: distinct best per GA run ─────────────────────────────────────────
runs = {"Pb-free": "67_ga_pb_free.json", "d0-strict": "67_ga_d0_strict.json",
        "Pb-allowed": "67_ga_pb_allowed.json"}
out = {"AD_definition": "real dual AD (script 49): layer1 sigma_conf head < 0.35 AND layer2 KDE density > p5",
       "Q90": Q90, "train_nn_dist_median": round(train_nn_median, 4),
       "train_nn_dist_p95": round(train_nn_p95, 4), "candidates": {}}

for run, fn in runs.items():
    d = json.load(open(os.path.join(ROOT, "results", fn)))
    formula = d["top10"][0]["formula"]
    feo = featurize(formula)
    if feo is None:
        out["candidates"][run] = {"formula": formula, "error": "featurization failed"}; continue
    pred, estd, sconf, dl, dt, dr, xvec = infer(*feo)
    logdens = float(kde.score_samples(pca.transform(xvec.reshape(1, -1)))[0])
    nndist = float(NearestNeighbors(n_neighbors=1).fit(Xtr[train_idx]).kneighbors(xvec.reshape(1, -1))[0][0, 0])
    l1 = bool(sconf < SIGMA_CONF_THRESHOLD); l2 = bool(logdens > thr)
    er_cm = float(feo[3][0])
    out["candidates"][run] = {
        "formula": formula,
        "predicted_er": round(pred, 1), "conformal_lower_90": round(pred - Q90, 1),
        "ensemble_std": round(estd, 2),
        "sigma_conf_head": round(sconf, 3), "layer1_physics_in": l1,
        "kde_log_density": round(logdens, 2), "kde_threshold": round(float(thr), 2), "layer2_kde_in": l2,
        "IN_AD_real": bool(l1 and l2),
        "nn_dist_to_train": round(nndist, 3),
        "er_cm": round(er_cm, 1), "delta_lst": round(dl, 1), "delta_tilt": round(dt, 2), "delta_res": round(dr, 2),
        "f_lst": round(dl / er_cm, 2) if er_cm > 0 else None,
    }

with open(os.path.join(ROOT, "results", "68_ga_candidate_ad_recheck_v77.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
