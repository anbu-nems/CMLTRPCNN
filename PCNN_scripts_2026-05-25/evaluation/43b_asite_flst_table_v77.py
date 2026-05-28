#!/usr/bin/env python
"""
43b — Canonical A-site soft-mode amplification table (v7.7).

Emits the per-A-site and per-(A-site x Reaney-regime) soft-mode amplification
factor used in Results/captions/manuscript:

    f_LST = delta_LST / eps_CM      (soft-mode contribution as a multiple of the
                                     electronic Clausius-Mossotti baseline)

computed over the *valid-CM* population (the only set where eps_CM is defined).
Source = the committed v7.7 per-sample inference artifact
(open_source_release/extracted_data/decomposition_per_sample.csv), whose global
A-site means reproduce scripts/43_counterfactual_permutation.py exactly
(Pb 2.28 / Ca 1.66 / La 1.13 / Sr 0.91 / Ba 0.66; observed range 1.622, p=0.0).

This file exists so every A-site number quoted in the manuscript traces to a
results JSON, including the per-regime breakdown that script 43 does not store.
"""
import json, os, re
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(os.path.expanduser("~"),
                   "Desktop/NC figures/open_source_release/extracted_data/decomposition_per_sample.csv")
OUT = os.path.join(ROOT, "results", "43b_asite_flst_table_v77.json")

FAMILIES = ["Pb", "Ca", "La", "Sr", "Ba"]


def primary_a_site(formula: str) -> str:
    """First cation token of an ABO3 formula = dominant A-site (recovers Pb/La
    that the coarse `a_site` column folds into 'Other')."""
    m = re.findall(r"[A-Z][a-z]?", str(formula))
    return m[0] if m else "?"


def main():
    df = pd.read_csv(CSV)
    valid = df[df["has_cm"] == True].copy()           # noqa: E712  (csv stores bool)
    valid["a_site"] = valid["formula"].apply(primary_a_site)
    valid["f_lst"] = valid["delta_lst"] / valid["er_cm"]

    out = {
        "definition": "f_LST = delta_LST / er_cm  (soft-mode amplification factor)",
        "population": "valid-CM (has_cm == True)",
        "n_valid_cm": int(len(valid)),
        "source_csv": "open_source_release/extracted_data/decomposition_per_sample.csv",
        "reproduces": "scripts/43_counterfactual_permutation.py permutation_asite.observed_means",
        "by_a_site": {},
        "by_a_site_regime": {},
    }

    for a in FAMILIES:
        s = valid[valid["a_site"] == a]
        out["by_a_site"][a] = {"n": int(len(s)), "f_lst_mean": round(float(s["f_lst"].mean()), 3)}

    for a in FAMILIES:
        out["by_a_site_regime"][a] = {}
        for r in ["Ia", "Ib", "II", "III"]:
            s = valid[(valid["a_site"] == a) & (valid["regime"] == r)]
            if len(s):
                out["by_a_site_regime"][a][r] = {
                    "n": int(len(s)),
                    "f_lst_mean": round(float(s["f_lst"].mean()), 3),
                }

    # span across (A-site x regime) cells with n >= 5, for the manuscript "span" claim
    cells = [(a, r, v["f_lst_mean"], v["n"])
             for a, rr in out["by_a_site_regime"].items()
             for r, v in rr.items() if v["n"] >= 5]
    hi = max(cells, key=lambda c: c[2])
    lo = min(cells, key=lambda c: c[2])
    out["span_high"] = {"a_site": hi[0], "regime": hi[1], "f_lst": hi[2], "n": hi[3]}
    out["span_low"] = {"a_site": lo[0], "regime": lo[1], "f_lst": lo[2], "n": lo[3]}
    out["span_ratio"] = round(hi[2] / lo[2], 1)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
