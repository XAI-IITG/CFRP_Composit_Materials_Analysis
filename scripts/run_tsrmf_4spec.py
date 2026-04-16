"""
TS-RMF 4-Specimen Experiment with Fractional RUL
==================================================

Runs the grouped-split experiment on the 4-specimen dataset with
fractional RUL targets (0 = end of life, 1 = start of life).

Steps:
    1. Build 4-specimen NPZ dataset (if not already built)
    2. Run plain transformer baseline (4-fold leave-one-out)
    3. Run TS-RMF model (4-fold leave-one-out)
    4. Compare results

Usage:
    python scripts/run_tsrmf_4spec.py
    python scripts/run_tsrmf_4spec.py --epochs 50
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grouped_split_experiment import run_grouped_experiment, auto_device


def main():
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="4-specimen TS-RMF experiment with fractional RUL"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-stage", type=float, default=0.5)
    parser.add_argument("--lambda-kg", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-build", action="store_true", help="Skip dataset building if NPZ already exists")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    models_dir = project_root / "outputs" / "saved_models"
    ttl_path = str(project_root / "data" / "ontology" / "cfrp_ontology_populated.ttl")
    npz_path = project_root / "data" / "processed" / "cfrp_windows_4spec.npz"
    max_life_path = project_root / "data" / "processed" / "specimen_max_lives.json"
    results_path = Path(args.out) if args.out else models_dir / "tsrmf_results_4spec.json"
    checkpoint_dir = models_dir / "tsrmf_checkpoints_4spec"
    device = args.device or auto_device()

    print("=" * 70)
    print("TS-RMF 4-Specimen Experiment (Fractional RUL)")
    print("=" * 70)
    print(f"  Device:      {device}")
    print(f"  TTL:         {ttl_path}")
    print(f"  Dataset:     {npz_path}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Splits:      {args.n_splits}")
    print(f"  Batch size:  {args.batch_size}")

    # ========================================================================
    # Step 1: Build dataset if needed
    # ========================================================================
    if not args.skip_build or not npz_path.exists():
        print("\n" + "=" * 70)
        print("STEP 1: Building 4-Specimen Dataset")
        print("=" * 70)
        from build_4spec_dataset import main as build_main
        sys.argv = ["build_4spec_dataset.py"]  # Reset argv for sub-script
        build_main()
    else:
        print(f"\n  [OK] Using existing dataset: {npz_path}")

    # Load max-life map for cycle-space interpretation
    max_life_map = {}
    if max_life_path.exists():
        max_life_map = json.loads(max_life_path.read_text(encoding="utf-8"))
        print(f"  Specimen max lives: {max_life_map}")

    # Common experiment kwargs
    # NOTE: No target_scaler_path — targets are fractional RUL [0,1]
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
        target_scaler_path=None,  # Fractional RUL, no inverse scaling needed
    )

    all_results = {}

    # ========================================================================
    # Step 2: Plain Transformer Baseline
    # ========================================================================
    if not args.skip_baseline:
        print("\n" + "=" * 70)
        print("STEP 2: Plain Transformer Baseline (4-fold)")
        print("=" * 70)
        baseline_summary = run_grouped_experiment(
            model_name="plain_transformer",
            checkpoint_dir=str(checkpoint_dir),
            **exp_kwargs,
        )
        all_results["plain_transformer"] = baseline_summary
        print(f"\n  Baseline MAE:  {baseline_summary['mean_test_mae']:.4f} +/- "
              f"{baseline_summary['std_test_mae']:.4f} (fraction of life)")
        print(f"  Baseline RMSE: {baseline_summary['mean_test_rmse']:.4f} +/- "
              f"{baseline_summary['std_test_rmse']:.4f} (fraction of life)")

    # ========================================================================
    # Step 3: TS-RMF Model
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Time-Sliced Relational Memory Fusion (4-fold)")
    print("=" * 70)
    tsrmf_summary = run_grouped_experiment(
        model_name="ts_rmf",
        checkpoint_dir=str(checkpoint_dir),
        **exp_kwargs,
    )
    all_results["ts_rmf"] = tsrmf_summary
    print(f"\n  TS-RMF MAE:  {tsrmf_summary['mean_test_mae']:.4f} +/- "
          f"{tsrmf_summary['std_test_mae']:.4f} (fraction of life)")
    print(f"  TS-RMF RMSE: {tsrmf_summary['mean_test_rmse']:.4f} +/- "
          f"{tsrmf_summary['std_test_rmse']:.4f} (fraction of life)")

    # ========================================================================
    # Step 4: Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\n  {'Model':<30} {'MAE (frac)':<20} {'RMSE (frac)':<20}")
    print(f"  {'-' * 68}")

    if "plain_transformer" in all_results:
        b = all_results["plain_transformer"]
        print(f"  {'Plain Transformer':<30} "
              f"{b['mean_test_mae']:.4f} +/- {b['std_test_mae']:.4f}   "
              f"{b['mean_test_rmse']:.4f} +/- {b['std_test_rmse']:.4f}")

    t = all_results["ts_rmf"]
    print(f"  {'TS-RMF':<30} "
          f"{t['mean_test_mae']:.4f} +/- {t['std_test_mae']:.4f}   "
          f"{t['mean_test_rmse']:.4f} +/- {t['std_test_rmse']:.4f}")

    if "plain_transformer" in all_results:
        improvement = all_results["plain_transformer"]["mean_test_mae"] - t["mean_test_mae"]
        if improvement > 0:
            pct = (improvement / all_results["plain_transformer"]["mean_test_mae"]) * 100
            print(f"\n  [OK] TS-RMF improved MAE by {improvement:.4f} ({pct:.1f}%)")
        else:
            print(f"\n  [!] TS-RMF MAE was {-improvement:.4f} higher than baseline")

    # Per-fold breakdown with cycle-space interpretation
    print("\n" + "-" * 70)
    print("  Per-fold Breakdown (with cycle-space interpretation):")
    print("-" * 70)

    for model_name, summary in all_results.items():
        print(f"\n  {model_name}:")
        for fold in summary["folds"]:
            frac_mae = fold["test_mae"]
            specimens = fold["test_specimens"]
            print(f"    Fold {fold['fold']}: MAE = {frac_mae:.4f} (frac), specimens = {specimens}")

            if fold.get("per_specimen_mae"):
                for sid, mae_frac in fold["per_specimen_mae"].items():
                    ml = max_life_map.get(sid, None)
                    if ml:
                        cycles_mae = mae_frac * ml
                        print(f"      {sid}: MAE = {mae_frac:.4f} frac = {cycles_mae:,.0f} cycles "
                              f"(of {ml:,} total life)")
                    else:
                        print(f"      {sid}: MAE = {mae_frac:.4f} frac")

    # Save results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n  [OK] Results saved to: {results_path}")

    # Load old 3-spec results for comparison if available
    old_results_path = models_dir / "tsrmf_results_3spec.json"
    if old_results_path.exists():
        print("\n" + "=" * 70)
        print("COMPARISON WITH OLD 3-SPECIMEN RESULTS")
        print("=" * 70)
        old = json.loads(old_results_path.read_text(encoding="utf-8"))
        if "plain_transformer" in old:
            print(f"\n  Old Plain Transformer (3-spec, absolute cycles):")
            print(f"    MAE = {old['plain_transformer']['mean_test_mae']:,.0f} cycles")
        if "ts_rmf" in old:
            print(f"  Old TS-RMF (3-spec, absolute cycles):")
            print(f"    MAE = {old['ts_rmf']['mean_test_mae']:,.0f} cycles")
        print(f"\n  New (4-spec, fractional RUL):")
        if "plain_transformer" in all_results:
            print(f"    Plain Transformer MAE = {all_results['plain_transformer']['mean_test_mae']:.4f} fraction")
        print(f"    TS-RMF MAE = {all_results['ts_rmf']['mean_test_mae']:.4f} fraction")
        print(f"\n  Note: Old results in cycles, new in fraction. Compare direction of improvement.")

    print("\n" + "=" * 70)
    print("4-SPECIMEN EXPERIMENT COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
