"""
Data Adapter: Pickle -> NPZ for Time-Sliced Relational KG-RUL Pipeline
======================================================================

Bridges the existing preprocessed CFRP data (pickle format with pre-split
train/val/test) to the unified NPZ format expected by the TS-RMF grouped-
split experiment runner.

Key operations:
  1. Loads preprocessed_data_combined.pkl and feature_scaler_combined.pkl
  2. Inverse-transforms the 'cycles' column to recover raw cycle counts
  3. Recombines train/val/test into a single array (grouped splitting re-splits)
  4. Saves as .npz with keys: X, y_rul, specimen_ids, current_cycles

Usage:
    python scripts/data_adapter.py
    python scripts/data_adapter.py --models-dir outputs/saved_models --out data/processed/cfrp_windows.npz
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np


CYCLES_FEATURE_INDEX = 14  # 'cycles' column in the 16-feature vector


def load_pickle_data(models_dir: Path) -> Tuple[dict, object]:
    """Load the preprocessed data pickle and feature scaler."""
    data_path = models_dir / "preprocessed_data_combined.pkl"
    scaler_path = models_dir / "feature_scaler_combined.pkl"

    if not data_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found at {data_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Feature scaler not found at {scaler_path}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return data, scaler


def recover_raw_cycles(X_scaled: np.ndarray, scaler) -> np.ndarray:
    """
    Inverse-transform only the cycles column to recover raw cycle counts.

    The feature scaler is a StandardScaler, so:
        raw = scaled * scale + mean

    We extract the last timestep's cycles value as the window's current_cycle.
    """
    cycles_mean = scaler.mean_[CYCLES_FEATURE_INDEX]
    cycles_scale = scaler.scale_[CYCLES_FEATURE_INDEX]

    # Last timestep of each window gives the most recent cycle count
    cycles_scaled = X_scaled[:, -1, CYCLES_FEATURE_INDEX]
    cycles_raw = (cycles_scaled * cycles_scale + cycles_mean).astype(np.int64)

    # Clamp to non-negative
    cycles_raw = np.maximum(cycles_raw, 0)

    return cycles_raw


def combine_splits(data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recombine train/val/test splits into a single unified dataset."""
    X = np.concatenate([data["X_train"], data["X_val"], data["X_test"]], axis=0)
    y_rul = np.concatenate([data["y_train"], data["y_val"], data["y_test"]], axis=0)
    specimen_ids = np.concatenate(
        [data["specimen_ids_train"], data["specimen_ids_val"], data["specimen_ids_test"]],
        axis=0,
    )

    return X.astype(np.float32), y_rul.astype(np.float32), specimen_ids.astype(str)


def convert_pickle_to_npz(
    models_dir: str | Path,
    output_path: str | Path,
    verbose: bool = True,
) -> Path:
    """
    Full conversion pipeline: pickle -> NPZ.

    Returns the path to the saved NPZ file.
    """
    models_dir = Path(models_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Data Adapter: Pickle -> NPZ")
        print("=" * 60)

    # Step 1: Load
    data, scaler = load_pickle_data(models_dir)
    if verbose:
        print(f"\n  Loaded preprocessed data:")
        print(f"    Train: {data['X_train'].shape[0]} samples")
        print(f"    Val:   {data['X_val'].shape[0]} samples")
        print(f"    Test:  {data['X_test'].shape[0]} samples")
        print(f"    Features: {data['n_features']}, Seq length: {data['sequence_length']}")

    # Step 2: Recover raw cycles from the scaled data
    # We need the raw cycles BEFORE recombining (scaler was fit on all data)
    X_all_scaled = np.concatenate(
        [data["X_train"], data["X_val"], data["X_test"]], axis=0
    )
    current_cycles = recover_raw_cycles(X_all_scaled, scaler)

    if verbose:
        print(f"\n  Recovered raw cycle counts:")
        print(f"    Range: [{current_cycles.min():,} -- {current_cycles.max():,}]")
        print(f"    Mean:  {current_cycles.mean():,.0f}")

    # Step 3: Combine splits
    X, y_rul, specimen_ids = combine_splits(data)

    if verbose:
        print(f"\n  Combined dataset:")
        print(f"    X:              {X.shape}")
        print(f"    y_rul:          {y_rul.shape}")
        print(f"    specimen_ids:   {specimen_ids.shape} -- unique: {np.unique(specimen_ids)}")
        print(f"    current_cycles: {current_cycles.shape}")

    # Step 4: Save as NPZ
    np.savez(
        output_path,
        X=X,
        y_rul=y_rul,
        specimen_ids=specimen_ids,
        current_cycles=current_cycles,
    )

    if verbose:
        print(f"\n  [OK] Saved to: {output_path}")
        print(f"    File size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert CFRP preprocessed pickle data to NPZ for TS-RMF pipeline"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Path to saved_models directory (default: outputs/saved_models)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output NPZ path (default: data/processed/cfrp_windows.npz)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    models_dir = Path(args.models_dir) if args.models_dir else project_root / "outputs" / "saved_models"
    output_path = Path(args.out) if args.out else project_root / "data" / "processed" / "cfrp_windows.npz"

    convert_pickle_to_npz(models_dir, output_path)


if __name__ == "__main__":
    main()
