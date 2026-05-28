#!/usr/bin/env python3
"""
run_all.py — end-to-end reproduction pipeline for CMLTRPCNN v7.7 (PCNN).

Runs the full pipeline in dependency order. By DEFAULT it reuses the provided
trained checkpoint (model_weights/cmltrv77_final.pt) and reproduces every
downstream result; pass --train to retrain the 5-seed ensemble from scratch.

Usage
-----
  python run_all.py                  # reproduce all results from the checkpoint
  python run_all.py --train          # retrain the 5-seed ensemble first, then reproduce
  python run_all.py --list           # list the pipeline stages and exit
  python run_all.py --only conformal # run a single stage
  python run_all.py --from baselines # run from a given stage onward
  python run_all.py --dry-run        # print the commands without executing

Outputs land in results/, extracted_data/, and figures_output/.
Requires the environment in requirements.txt (Python >= 3.10, PyTorch >= 2.0).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

_PYPATH = [str(ROOT), str(ROOT/'src')] + [str(p) for p in (ROOT/'src').glob('*') if p.is_dir()]
if os.environ.get('PYTHONPATH'): _PYPATH.append(os.environ['PYTHONPATH'])
ENV = {**os.environ, 'PYTHONPATH': os.pathsep.join(_PYPATH)}

# (key, description, script-path-relative-to-ROOT, train_only)
STAGES = [
    ("features",      "Feature engineering — build the 103-feature matrix",        "src/features/feature_engineering.py",          False),
    ("train",         "Train the 5-seed PCNN ensemble -> model_weights/",          "src/training/train_final_model.py",            True),
    ("decompose",     "Per-composition decomposition (CM / LST / tilt / res)",     "src/analysis/extract_per_sample_data.py",      False),
    ("cv",            "Formula-split and Strat-GSS cross-validation predictions",  "src/analysis/gen_cv_predictions.py",           False),
    ("lofo",          "Leave-one-family-out (A-site and regime) generalisation",   "scripts/42_lofo_canonical_with_csv.py",        False),
    ("conformal",     "Split conformal prediction intervals",                      "scripts/54_conformal_prediction.py",           False),
    ("baselines",     "Seven-baseline comparison (CM, Ridge, RF, XGB, CatBoost, MLP, PIRNN)", "scripts/57_baseline_comparison.py",   False),
    ("pirnn",         "PIRNN (no-partition) fair comparison",                      "scripts/56_pirnn_fair_comparison.py",          False),
    ("ablation",      "Branch ablation (LST / tilt / residual)",                   "scripts/53_ablation_v77.py",                   False),
    ("ad",            "Two-layer applicability domain (physics + density gates)",  "scripts/49_ad_v77.py",                         False),
    ("counterfactual","A-site counterfactual permutation (B-site independence)",   "scripts/43_counterfactual_permutation.py",     False),
    ("shap",          "SHAP feature attribution",                                  "src/analysis/extract_shap_per_sample.py",      False),
    ("screening",     "Uncertainty-aware Pb-free virtual screening",               "scripts/61_virtual_screening.py",              False),
    ("inverse_design","Physics-constrained genetic-algorithm inverse design",      "scripts/67_physics_ga_inverse_design.py",      False),
]


def main():
    ap = argparse.ArgumentParser(description="End-to-end CMLTRPCNN v7.7 reproduction pipeline.")
    ap.add_argument("--train", action="store_true",
                    help="retrain the 5-seed ensemble (otherwise reuse the provided checkpoint)")
    ap.add_argument("--only", metavar="STAGE", help="run only this stage")
    ap.add_argument("--from", dest="from_stage", metavar="STAGE", help="run from this stage onward")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--dry-run", action="store_true", help="print commands without executing")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going if a stage fails (default: stop)")
    a = ap.parse_args()

    # Drop the training stage unless --train (the checkpoint is shipped).
    stages = [s for s in STAGES if (a.train or not s[3])]

    if a.list:
        for k, d, p, _ in stages:
            mark = "" if (ROOT / p).exists() else "   [MISSING from release]"
            print(f"  {k:16} {d}{mark}")
        return

    if a.only:
        stages = [s for s in stages if s[0] == a.only]
        if not stages:
            sys.exit(f"unknown stage: {a.only}  (use --list)")
    elif a.from_stage:
        names = [s[0] for s in stages]
        if a.from_stage not in names:
            sys.exit(f"unknown stage: {a.from_stage}  (use --list)")
        stages = stages[names.index(a.from_stage):]

    print("=" * 72)
    print("CMLTRPCNN v7.7 (PCNN) — full reproduction pipeline")
    print("=" * 72)
    print(f"mode: {'RETRAIN ensemble' if a.train else 'reuse provided checkpoint'}   "
          f"| stages to run: {len(stages)}\n")

    failures = []
    for i, (k, desc, relpath, _) in enumerate(stages, 1):
        script = ROOT / relpath
        print(f"[{i}/{len(stages)}] {k}: {desc}")
        if not script.exists():
            print(f"   ! script missing from release: {relpath} — skipping\n")
            failures.append(k)
            continue
        if a.dry_run:
            print(f"   $ {PY} {relpath}\n")
            continue
        t0 = time.time()
        rc = subprocess.run([PY, str(script)], cwd=ROOT, env=ENV).returncode
        dt = time.time() - t0
        if rc != 0:
            print(f"   FAILED (exit {rc}) after {dt:.0f}s\n")
            failures.append(k)
            if not a.continue_on_error:
                sys.exit(rc)
        else:
            print(f"   done ({dt:.0f}s)\n")

    print("=" * 72)
    if failures:
        print(f"Pipeline finished with issues in: {', '.join(failures)}")
    else:
        print("Pipeline complete. Outputs in results/, extracted_data/, figures_output/.")


if __name__ == "__main__":
    main()
