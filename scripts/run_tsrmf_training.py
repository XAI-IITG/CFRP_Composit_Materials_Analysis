"""
TS-RMF End-to-End Training Script
==================================

Runs the complete Time-Sliced Relational Memory Fusion pipeline:
  1. Converts existing pickle data -> NPZ (via data_adapter)
  2. Runs grouped-split experiment with plain_transformer baseline
  3. Runs grouped-split experiment with ts_rmf model
  4. Prints comparative results

Usage:
    python scripts/run_tsrmf_training.py
    python scripts/run_tsrmf_training.py --epochs 50 --n-splits 3

Dependencies:
    pip install rdflib torch scikit-learn numpy
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# Provide a clean output for training explicitly
warnings.filterwarnings("ignore")

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_adapter import convert_pickle_to_npz
from grouped_split_experiment import run_grouped_experiment, auto_device


def main():
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="End-to-end TS-RMF training: data conversion + grouped experiment"
    )
    parser.add_argument("--models-dir", type=str, default=None, help="Path to saved_models dir")
    parser.add_argument("--ttl", type=str, default=None, help="Path to populated ontology TTL")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-stage", type=float, default=0.5)
    parser.add_argument("--lambda-kg", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-baseline", action="store_true", help="Skip plain transformer baseline")
    parser.add_argument("--out", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    models_dir = Path(args.models_dir) if args.models_dir else project_root / "outputs" / "saved_models"
    ttl_path = args.ttl if args.ttl else str(project_root / "data" / "ontology" / "cfrp_ontology_populated.ttl")
    npz_path = project_root / "data" / "processed" / "cfrp_windows.npz"
    target_scaler_path = str(models_dir / "target_scaler_combined.pkl")
    results_path = Path(args.out) if args.out else models_dir / "tsrmf_results.json"
    checkpoint_dir = models_dir / "tsrmf_checkpoints"
    device = args.device if args.device else auto_device()

    print("=" * 70)
    print("TS-RMF End-to-End Pipeline")
    print("=" * 70)
    print(f"  Device:      {device}")
    print(f"  TTL:         {ttl_path}")
    print(f"  Models dir:  {models_dir}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Splits:      {args.n_splits}")
    print(f"  Batch size:  {args.batch_size}")

    # ========================================================================
    # Step 1: Convert pickle -> NPZ
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Data Conversion (Pickle -> NPZ)")
    print("=" * 70)
    convert_pickle_to_npz(models_dir, npz_path, verbose=True)

    # Common experiment kwargs
    exp_kwargs = dict(
        dataset_path=str(npz_path),
        ttl_path=ttl_path,
        n_splits=args.n_splits,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_stage=args.lambda_stage,
        lambda_kg=args.lambda_kg,
        random_state=args.random_state,
        device=device,
        target_scaler_path=target_scaler_path,
    )

    all_results = {}

    # ========================================================================
    # Step 2: Plain Transformer Baseline
    # ========================================================================
    if not args.skip_baseline:
        print("\n" + "=" * 70)
        print("STEP 2: Plain Transformer Baseline")
        print("=" * 70)
        baseline_summary = run_grouped_experiment(
            model_name="plain_transformer",
            checkpoint_dir=str(checkpoint_dir),
            **exp_kwargs,
        )
        all_results["plain_transformer"] = baseline_summary
        print(f"\n  Baseline MAE:  {baseline_summary['mean_test_mae']:,.0f} +/- {baseline_summary['std_test_mae']:,.0f} cycles")
        print(f"  Baseline RMSE: {baseline_summary['mean_test_rmse']:,.0f} +/- {baseline_summary['std_test_rmse']:,.0f} cycles")

    # ========================================================================
    # Step 3: TS-RMF Model
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Time-Sliced Relational Memory Fusion (TS-RMF)")
    print("=" * 70)
    tsrmf_summary = run_grouped_experiment(
        model_name="ts_rmf",
        checkpoint_dir=str(checkpoint_dir),
        **exp_kwargs,
    )
    all_results["ts_rmf"] = tsrmf_summary
    print(f"\n  TS-RMF MAE:  {tsrmf_summary['mean_test_mae']:,.0f} +/- {tsrmf_summary['std_test_mae']:,.0f} cycles")
    print(f"  TS-RMF RMSE: {tsrmf_summary['mean_test_rmse']:,.0f} +/- {tsrmf_summary['std_test_rmse']:,.0f} cycles")

    # ========================================================================
    # Step 4: Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\n  {'Model':<30} {'MAE (cycles)':>18} {'RMSE (cycles)':>18}")
    print(f"  {'-' * 66}")

    if "plain_transformer" in all_results:
        b = all_results["plain_transformer"]
        print(f"  {'Plain Transformer':<30} {b['mean_test_mae']:>10,.0f} +/- {b['std_test_mae']:<7,.0f} {b['mean_test_rmse']:>10,.0f} +/- {b['std_test_rmse']:<7,.0f}")

    t = all_results["ts_rmf"]
    print(f"  {'TS-RMF':<30} {t['mean_test_mae']:>10,.0f} +/- {t['std_test_mae']:<7,.0f} {t['mean_test_rmse']:>10,.0f} +/- {t['std_test_rmse']:<7,.0f}")

    if "plain_transformer" in all_results:
        improvement = all_results["plain_transformer"]["mean_test_mae"] - all_results["ts_rmf"]["mean_test_mae"]
        if improvement > 0:
            pct = (improvement / all_results["plain_transformer"]["mean_test_mae"]) * 100
            print(f"\n  [OK] TS-RMF improved MAE by {improvement:,.0f} cycles ({pct:.1f}%)")
        else:
            print(f"\n  [!] TS-RMF MAE was {-improvement:,.0f} cycles higher than baseline")

    # Save results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n  [OK] Results saved to: {results_path}")

    # Per-fold detail for TS-RMF
    print("\n  TS-RMF per-fold breakdown:")
    for fold in tsrmf_summary["folds"]:
        print(f"    Fold {fold['fold']}: MAE={fold['test_mae']:,.0f} cycles, RMSE={fold['test_rmse']:,.0f} cycles, specimens={fold['test_specimens']}")
        if fold.get("per_specimen_mae"):
            for sid, mae in fold["per_specimen_mae"].items():
                print(f"      {sid}: MAE={mae:,.0f} cycles")

    print("\n" + "=" * 70)
    print("TS-RMF PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
