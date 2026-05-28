"""
predict.py — Single-composition εr prediction using CMLTRPCNNv7.7

Usage:
    python predict.py "BaTiO3"
    python predict.py "Ba0.5Sr0.5TiO3" --st 1300
    python predict.py "Ba0.5Sr0.5TiO3" --st 1300 --ct 1100

Output:
    Predicted εr with 90% conformal interval and branch decomposition.
"""
import sys, os, argparse

# Add model_code to path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "model_code"))
sys.path.insert(0, os.path.join(_HERE, "model_code", "physics"))

# Patch asset paths to point to this folder
import inference as _inf
_inf.ASSETS  = os.path.join(_HERE, "model")
_inf.FM_PATH = os.path.join(_HERE, "model", "feature_matrix_v7.parquet")

from inference import predict


def main():
    parser = argparse.ArgumentParser(
        description="Predict dielectric constant (εr) for an ABO₃ perovskite ceramic."
    )
    parser.add_argument("formula", type=str,
                        help='Perovskite formula, e.g. "BaTiO3" or "Ba0.5Sr0.5TiO3"')
    parser.add_argument("--st", type=float, default=None,
                        help="Sintering temperature (°C), optional")
    parser.add_argument("--ct", type=float, default=None,
                        help="Calcination temperature (°C), optional")
    args = parser.parse_args()

    print(f"\nFormula : {args.formula}")
    if args.st:
        print(f"ST      : {args.st} °C")
    if args.ct:
        print(f"CT      : {args.ct} °C")
    print()

    result = predict(args.formula, st=args.st, ct=args.ct)

    if not result.get("parse_ok"):
        print(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)

    pred  = result["pred"]
    lo    = result["lower_90"]
    hi    = result["upper_90"]
    std   = result["seed_std"]
    er_cm = result["er_cm"]
    dlst  = result["delta_lst"]
    dtilt = result["delta_tilt"]
    dres  = result["delta_res"]
    tf    = result["tolerance_factor"]
    regime = result["regime"]
    source = result["source"]

    print(f"  Predicted εr        : {pred:.1f}  [{lo:.1f}, {hi:.1f}]  (90% CI)")
    print(f"  Ensemble std        : ±{std:.2f}")
    print()
    print(f"  Branch decomposition:")
    print(f"    εr_CM  (CM anchor): {er_cm:.1f}")
    print(f"    δ_LST  (≥0)       : +{dlst:.1f}")
    print(f"    δ_tilt (≤0)       : {dtilt:.1f}")
    print(f"    δ_res  (gated)    : {dres:+.1f}")
    print(f"    ─────────────────────")
    print(f"    Total             : {er_cm + dlst + dtilt + dres:.1f}")
    print()
    if tf:
        print(f"  Tolerance factor    : {tf:.4f}")
    print(f"  Reaney regime       : {regime}")
    print(f"  Feature source      : {source}")
    print()


if __name__ == "__main__":
    main()
