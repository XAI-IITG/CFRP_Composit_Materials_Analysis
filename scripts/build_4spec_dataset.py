"""
Build 4-Specimen Dataset with Fractional RUL
=============================================

Processes raw PZT .mat files for specimens S11, S12, S18, S19 (all Layup1)
and creates an NPZ dataset with fractional RUL targets.

Fractional RUL = (max_cycles - current_cycles) / max_cycles
    -> Always in [0, 1] regardless of specimen total life
    -> Removes distribution shift between specimens

Output:
    data/processed/cfrp_windows_4spec.npz
        X:              (N, seq_len, 16) float32 - feature sequences
        y_rul:          (N,) float32              - fractional RUL [0, 1]
        specimen_ids:   (N,) str                  - specimen IDs
        current_cycles: (N,) int64                - raw cycle counts
        max_lives:      (N,) int64                - per-specimen max cycle count

Usage:
    python scripts/build_4spec_dataset.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io
from scipy import signal as sp_signal

warnings.filterwarnings("ignore")


# ==========================================================================
# Feature Extraction (replicated from layup1_combined.ipynb)
# ==========================================================================

def load_pzt_data(mat_file_path: Path) -> Optional[dict]:
    """Load PZT sensor data from .mat file."""
    try:
        mat_data = scipy.io.loadmat(str(mat_file_path), struct_as_record=False, squeeze_me=True)
        coupon = mat_data["coupon"]
        cycles = getattr(coupon, "cycles", None)
        load = getattr(coupon, "load", None)
        condition = getattr(coupon, "condition", None)

        path_data_obj = getattr(coupon, "path_data", None)
        path_data = []

        if path_data_obj is not None:
            if isinstance(path_data_obj, np.ndarray):
                for path in path_data_obj:
                    path_dict = {
                        "actuator": getattr(path, "actuator", None),
                        "sensor": getattr(path, "sensor", None),
                        "frequency": getattr(path, "frequency", None),
                        "amplitude": getattr(path, "amplitude", None),
                        "gain": getattr(path, "gain", None),
                        "sampling_rate": getattr(path, "sampling_rate", None),
                        "signal_actuator": np.asarray(getattr(path, "signal_actuator", None)).squeeze()
                            if getattr(path, "signal_actuator", None) is not None else None,
                        "signal_sensor": np.asarray(getattr(path, "signal_sensor", None)).squeeze()
                            if getattr(path, "signal_sensor", None) is not None else None,
                    }
                    path_data.append(path_dict)
            else:
                path_dict = {
                    "actuator": getattr(path_data_obj, "actuator", None),
                    "sensor": getattr(path_data_obj, "sensor", None),
                    "frequency": getattr(path_data_obj, "frequency", None),
                    "amplitude": getattr(path_data_obj, "amplitude", None),
                    "gain": getattr(path_data_obj, "gain", None),
                    "sampling_rate": getattr(path_data_obj, "sampling_rate", None),
                    "signal_actuator": np.asarray(getattr(path_data_obj, "signal_actuator", None)).squeeze()
                        if getattr(path_data_obj, "signal_actuator", None) is not None else None,
                    "signal_sensor": np.asarray(getattr(path_data_obj, "signal_sensor", None)).squeeze()
                        if getattr(path_data_obj, "signal_sensor", None) is not None else None,
                }
                path_data.append(path_dict)

        return {"cycles": cycles, "load": load, "condition": condition, "path_data": path_data}
    except Exception as e:
        return None


def load_strain_data(mat_file_path: Path) -> Optional[dict]:
    """Load strain gauge data from .mat file."""
    try:
        mat_data = scipy.io.loadmat(str(mat_file_path), squeeze_me=True)
        import re

        available = {}
        for k in mat_data.keys():
            if not k.startswith("__") and "strain" in k.lower():
                available[k] = np.array(mat_data[k]).squeeze()

        if not available:
            return None

        strain_data = {f"strain{i}": None for i in range(1, 5)}
        for i in range(1, 5):
            name = f"strain{i}"
            if name in available:
                strain_data[name] = available[name]

        def _nat_key(k):
            m = re.search(r"(\d+)\s*$", k)
            return int(m.group(1)) if m else int(1e9)

        remaining = [k for k in sorted(available.keys(), key=_nat_key)
                     if k.lower() not in ["strain1", "strain2", "strain3", "strain4"]]
        idx = 1
        for k in remaining:
            while idx <= 4 and strain_data[f"strain{idx}"] is not None:
                idx += 1
            if idx > 4:
                break
            strain_data[f"strain{idx}"] = available[k]
            idx += 1

        return strain_data
    except Exception:
        return None


def extract_pzt_features(pzt_data: dict) -> List[float]:
    """Extract 9 PZT features from raw signals."""
    if pzt_data is None or "path_data" not in pzt_data:
        return [0.0] * 9

    path_data = pzt_data["path_data"]
    n_paths = len(path_data)
    if n_paths == 0:
        return [0.0] * 9

    psd_values, phase_values, energy_values, rms_values, peak_frequencies = [], [], [], [], []

    for path in path_data:
        sig = path.get("signal_sensor")
        fs = path.get("sampling_rate")
        if sig is None:
            continue
        sig = np.asarray(sig).squeeze()
        if sig.size == 0:
            continue

        rms_values.append(float(np.sqrt(np.mean(sig ** 2))))
        energy_values.append(float(np.sum(sig ** 2)))

        try:
            sig_centered = sig - np.mean(sig)
            fft = np.fft.rfft(sig_centered)
            psd = np.abs(fft) ** 2
            psd_values.append(float(np.mean(psd)))

            freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs if fs and fs > 0 else 1.0)
            if len(psd) > 0:
                peak_frequencies.append(float(freqs[np.argmax(psd)]))

            phase_values.append(float(np.mean(np.abs(np.angle(fft)))))
        except Exception:
            pass

    return [
        np.mean(psd_values) if psd_values else 0.0,
        np.std(psd_values) if psd_values else 0.0,
        np.mean(phase_values) if phase_values else 0.0,
        np.std(phase_values) if phase_values else 0.0,
        np.mean(energy_values) if energy_values else 0.0,
        np.std(energy_values) if energy_values else 0.0,
        np.mean(rms_values) if rms_values else 0.0,
        np.mean(peak_frequencies) if peak_frequencies else 0.0,
        float(n_paths),
    ]


def extract_strain_features(strain_data: Optional[dict]) -> List[float]:
    """Extract 4 strain features."""
    if strain_data is None:
        return [0.0, 0.0, 0.0, 0.0]

    channel_rms, channel_amplitude = [], []
    for i in range(1, 5):
        sig = strain_data.get(f"strain{i}")
        if sig is None:
            continue
        try:
            sig_array = np.asarray(sig).squeeze()
            if sig_array.size == 0:
                continue
            channel_rms.append(float(np.sqrt(np.mean(sig_array ** 2))))
            channel_amplitude.append(float(np.ptp(sig_array)))
        except Exception:
            continue

    if not channel_rms:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        float(np.mean(channel_rms)),
        float(np.std(channel_rms)),
        float(np.mean(channel_amplitude)),
        float(len(channel_rms)),
    ]


def compute_stiffness_degradation(cycles: int, max_cycles: int) -> float:
    """Power-law stiffness degradation model."""
    if max_cycles == 0:
        return 0.0
    normalized = cycles / max_cycles
    return max(0.0, min(1.0, 1.0 * (1 - normalized ** 1.5)))


def parse_pzt_filename(filename: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Parse PZT filename: L1S11_1000_1_1.mat -> (1000, 1, 1)."""
    parts = filename.replace(".mat", "").split("_")
    try:
        if len(parts) >= 4:
            return int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        pass
    return None, None, None


# ==========================================================================
# Specimen Loading
# ==========================================================================

SPECIMENS = {
    "S11": {"name": "L1_S11_F", "max_cycles": 223963},
    "S12": {"name": "L1_S12_F", "max_cycles": None},
    "S18": {"name": "L1_S18_F", "max_cycles": None},
    "S19": {"name": "L1_S19_F", "max_cycles": None},
}

INPUT_FEATURE_NAMES = [
    "avg_delta_psd", "std_delta_psd", "avg_delta_tof", "std_delta_tof",
    "avg_scatter_energy", "std_scatter_energy", "avg_rms", "avg_peak_frequency",
    "n_pzt_paths",
    "mean_strain_rms", "std_strain_rms", "mean_strain_amplitude", "n_active_channels",
    "stiffness_degradation",
    "cycles", "normalized_cycles",
]


def load_specimen(spec_id: str, spec_info: dict, base_dir: Path) -> Optional[dict]:
    """Load and extract features for a single specimen."""
    spec_dir = base_dir / spec_info["name"]
    pzt_dir = spec_dir / "PZT-data"
    strain_dir = spec_dir / "StrainData"

    if not pzt_dir.exists():
        print(f"    [!] PZT directory not found: {pzt_dir}")
        return None

    pzt_files = sorted(pzt_dir.glob("*.mat"))
    print(f"    Found {len(pzt_files)} PZT files")

    records = []
    for pzt_file in pzt_files:
        cycles, load_val, condition = parse_pzt_filename(pzt_file.name)
        if cycles is None:
            continue

        pzt_data = load_pzt_data(pzt_file)
        if pzt_data is None:
            continue

        actual_cycles = pzt_data.get("cycles", cycles)
        if actual_cycles is None:
            actual_cycles = cycles

        pzt_features = extract_pzt_features(pzt_data)

        # Try to load corresponding strain data
        strain_file = strain_dir / pzt_file.name
        strain_features = [0.0, 0.0, 0.0, 0.0]
        if strain_file.exists():
            sd = load_strain_data(strain_file)
            if sd is not None:
                strain_features = extract_strain_features(sd)

        records.append({
            "cycles": int(actual_cycles),
            "pzt_features": pzt_features,
            "strain_features": strain_features,
        })

    if not records:
        return None

    # Determine max_cycles
    max_cycles = spec_info["max_cycles"]
    if max_cycles is None:
        max_cycles = max(r["cycles"] for r in records)

    print(f"    [OK] {len(records)} records, cycle range: [0 .. {max_cycles:,}]")

    return {"records": records, "max_cycles": max_cycles}


def build_combined_dataframe(
    specimens_data: Dict[str, dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build combined feature array with fractional RUL targets."""

    all_features = []
    all_rul = []
    all_ids = []
    all_cycles = []
    all_max_lives = []

    for spec_id, data in sorted(specimens_data.items()):
        max_cycles = data["max_cycles"]
        # Sort records by cycle count
        records = sorted(data["records"], key=lambda r: r["cycles"])

        for rec in records:
            cyc = rec["cycles"]
            # 9 PZT + 4 strain + stiffness + cycles + normalized_cycles = 16 features
            stiffness = compute_stiffness_degradation(cyc, max_cycles)
            norm_cycles = cyc / max_cycles if max_cycles > 0 else 0.0

            features = rec["pzt_features"] + rec["strain_features"] + [stiffness, float(cyc), norm_cycles]
            all_features.append(features)

            # FRACTIONAL RUL: always in [0, 1]
            fractional_rul = (max_cycles - cyc) / max_cycles if max_cycles > 0 else 0.0
            all_rul.append(fractional_rul)
            all_ids.append(spec_id)
            all_cycles.append(cyc)
            all_max_lives.append(max_cycles)

    return (
        np.array(all_features, dtype=np.float32),
        np.array(all_rul, dtype=np.float32),
        np.array(all_ids, dtype=str),
        np.array(all_cycles, dtype=np.int64),
        np.array(all_max_lives, dtype=np.int64),
    )


def create_sequences(
    features: np.ndarray,
    y_rul: np.ndarray,
    specimen_ids: np.ndarray,
    current_cycles: np.ndarray,
    max_lives: np.ndarray,
    sequence_length: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding-window sequences per specimen."""
    X_seqs, y_seqs, id_seqs, cyc_seqs, ml_seqs = [], [], [], [], []

    for spec_id in sorted(set(specimen_ids)):
        mask = specimen_ids == spec_id
        spec_features = features[mask]
        spec_rul = y_rul[mask]
        spec_cycles = current_cycles[mask]
        spec_ml = max_lives[mask]

        for i in range(len(spec_features) - sequence_length + 1):
            X_seqs.append(spec_features[i : i + sequence_length])
            y_seqs.append(spec_rul[i + sequence_length - 1])
            id_seqs.append(spec_id)
            cyc_seqs.append(spec_cycles[i + sequence_length - 1])
            ml_seqs.append(spec_ml[i + sequence_length - 1])

    return (
        np.array(X_seqs, dtype=np.float32),
        np.array(y_seqs, dtype=np.float32),
        np.array(id_seqs, dtype=str),
        np.array(cyc_seqs, dtype=np.int64),
        np.array(ml_seqs, dtype=np.int64),
    )


def main():
    parser = argparse.ArgumentParser(description="Build 4-specimen dataset with fractional RUL")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to raw Layup1 data")
    parser.add_argument("--out", type=str, default=None, help="Output NPZ path")
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence length")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data" / "raw" / "Layup1"
    output_path = Path(args.out) if args.out else project_root / "data" / "processed" / "cfrp_windows_4spec.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Build 4-Specimen Dataset with Fractional RUL")
    print("=" * 70)
    print(f"  Data dir:  {data_dir}")
    print(f"  Output:    {output_path}")
    print(f"  Seq length: {args.seq_len}")

    # Step 1: Load all specimens
    print("\n" + "=" * 70)
    print("STEP 1: Load Raw Data")
    print("=" * 70)

    specimens_data = {}
    for spec_id, spec_info in SPECIMENS.items():
        print(f"\n  {spec_id} ({spec_info['name']}):")
        result = load_specimen(spec_id, spec_info, data_dir)
        if result is not None:
            specimens_data[spec_id] = result
        else:
            print(f"    [!] Failed to load {spec_id}")

    print(f"\n  [OK] Loaded {len(specimens_data)} / {len(SPECIMENS)} specimens")

    # Step 2: Build combined features with fractional RUL
    print("\n" + "=" * 70)
    print("STEP 2: Extract Features & Compute Fractional RUL")
    print("=" * 70)

    features, y_rul, specimen_ids, current_cycles, max_lives = build_combined_dataframe(specimens_data)
    print(f"  Total samples: {len(features)}")
    print(f"  Features: {features.shape[1]}")
    print(f"  y_rul (fractional): min={y_rul.min():.4f}, max={y_rul.max():.4f}")

    print("\n  Per-specimen summary:")
    for sid in sorted(set(specimen_ids)):
        mask = specimen_ids == sid
        ml = max_lives[mask][0]
        yr = y_rul[mask]
        print(f"    {sid}: {mask.sum()} samples, max_life={ml:,} cycles, "
              f"frac_RUL=[{yr.min():.4f} .. {yr.max():.4f}]")

    # Step 3: Create sequences
    print("\n" + "=" * 70)
    print("STEP 3: Create Temporal Sequences")
    print("=" * 70)

    X, y, sids, cycs, mls = create_sequences(
        features, y_rul, specimen_ids, current_cycles, max_lives,
        sequence_length=args.seq_len,
    )

    print(f"  X shape:           {X.shape}")
    print(f"  y_rul shape:       {y.shape}")
    print(f"  specimen_ids:      {np.unique(sids)}")
    print(f"  y_rul range:       [{y.min():.4f}, {y.max():.4f}]")

    print("\n  Per-specimen sequences:")
    for sid in sorted(set(sids)):
        mask = sids == sid
        ml = mls[mask][0]
        print(f"    {sid}: {mask.sum()} sequences, max_life={ml:,} cycles")

    # Step 4: Save
    print("\n" + "=" * 70)
    print("STEP 4: Save NPZ")
    print("=" * 70)

    np.savez(
        output_path,
        X=X,
        y_rul=y,
        specimen_ids=sids,
        current_cycles=cycs,
        max_lives=mls,
    )

    print(f"  [OK] Saved to: {output_path}")
    print(f"    File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Save specimen max-life mapping for later interpretation
    max_life_map = {sid: int(specimens_data[sid]["max_cycles"]) for sid in specimens_data}
    ml_path = output_path.parent / "specimen_max_lives.json"
    import json
    ml_path.write_text(json.dumps(max_life_map, indent=2), encoding="utf-8")
    print(f"  [OK] Max-life map: {ml_path}")
    print(f"    {max_life_map}")

    print("\n" + "=" * 70)
    print("DATASET BUILD COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
