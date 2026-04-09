"""
Populate CFRP Ontology with Experimental Dataset

Reads Excel logbooks, strain .mat files, and PZT metadata from the Layup1
dataset and generates a populated RDF/OWL ontology in Turtle format.

Usage:
    python scripts/populate_ontology.py

Output:
    data/ontology/cfrp_ontology_populated.ttl
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.io

warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Layup1')
SCHEMA_PATH = os.path.join(PROJECT_ROOT, 'data', 'ontology', 'cfrp_ontology.ttl')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'ontology', 'cfrp_ontology_populated.ttl')

# Specimen configurations
SPECIMENS = {
    'L1S11': {
        'dir': os.path.join(DATA_DIR, 'L1_S11_F'),
        'logbook': 'L1S11.xlsx',
        'prefix': 'L1_S11',
    },
    'L1S12': {
        'dir': os.path.join(DATA_DIR, 'L1_S12_F'),
        'logbook': 'L1S12.xlsx',
        'prefix': 'L1_S12',
    },
    'L1S18': {
        'dir': os.path.join(DATA_DIR, 'L1_S18_F'),
        'logbook': 'L1S18.xlsx',
        'prefix': 'L1_S18',
    },
    'L1S19': {
        'dir': os.path.join(DATA_DIR, 'L1_S19_F'),
        'logbook': 'L1S19.xlsx',
        'prefix': 'L1_S19',
    },
}

STRAIN_TYPES = {'A': 'Axial', 'M': 'Membrane', 'S': 'Shear'}


def escape_ttl_string(s):
    """Escape a string for Turtle format."""
    if not isinstance(s, str):
        s = str(s)
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', '')


def safe_uri(name):
    """Make a name safe for use as a URI fragment."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))


def extract_strain_stats(mat_path):
    """Extract statistical features from a strain .mat file.
    
    Returns dict with per-gauge stats, or None if file cannot be read.
    """
    try:
        data = scipy.io.loadmat(mat_path)
    except Exception as e:
        print(f"  WARNING: Could not load {mat_path}: {e}")
        return None

    stats = {}
    for gauge in ['strain1', 'strain2', 'strain3', 'strain4']:
        if gauge not in data:
            continue
        arr = data[gauge].flatten().astype(float)
        stats[gauge] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'max': float(np.max(arr)),
            'min': float(np.min(arr)),
            'rms': float(np.sqrt(np.mean(arr ** 2))),
            'range': float(np.max(arr) - np.min(arr)),
            'n_points': len(arr),
        }
    return stats


def read_logbook(specimen_id, config):
    """Read an Excel logbook and extract the checkpoint-to-cycle mapping.
    
    Returns list of dicts with keys: mts_base, cycles, date, load, bc, remarks
    """
    logbook_path = os.path.join(config['dir'], config['logbook'])
    if not os.path.exists(logbook_path):
        print(f"  WARNING: Logbook not found: {logbook_path}")
        return []

    df = pd.read_excel(logbook_path)

    # Identify columns
    mts_col = None
    cycles_col = None
    bc_col = None
    load_col = None
    date_col = None
    remarks_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if 'mts' in col_lower and 'file' in col_lower:
            mts_col = col
        elif 'cycle' in col_lower:
            cycles_col = col
        elif 'bound' in col_lower or 'condition' in col_lower:
            bc_col = col
        elif 'load' in col_lower:
            load_col = col
        elif 'date' in col_lower:
            date_col = col
        elif 'remark' in col_lower:
            remarks_col = col

    if not mts_col or not cycles_col:
        print(f"  WARNING: Could not identify MTS/Cycles columns in {logbook_path}")
        print(f"  Available columns: {df.columns.tolist()}")
        return []

    # Extract unique checkpoints
    mask = df[mts_col].notna() & (df[mts_col].astype(str).str.len() > 3)
    subset = df.loc[mask].copy()

    checkpoints = []
    seen_bases = set()

    for _, row in subset.iterrows():
        mts_raw = str(row[mts_col]).strip().strip("'")
        # Extract base name (before _STRAIN)
        base = mts_raw.split(',')[0].split('_STRAIN')[0].strip()

        if not base.startswith(config['prefix']):
            continue
        if base in seen_bases:
            continue
        seen_bases.add(base)

        cycles = int(row[cycles_col]) if pd.notna(row[cycles_col]) else 0
        load_val = float(row[load_col]) if load_col and pd.notna(row.get(load_col)) else 0.0
        bc_val = str(row[bc_col]).strip() if bc_col and pd.notna(row.get(bc_col)) else ''
        date_val = str(row[date_col]) if date_col and pd.notna(row.get(date_col)) else ''
        remarks_val = str(row[remarks_col]).strip() if remarks_col and pd.notna(row.get(remarks_col)) else ''

        checkpoints.append({
            'mts_base': base,
            'cycles': cycles,
            'date': date_val[:10] if date_val else '',
            'load': load_val,
            'bc': bc_val,
            'remarks': remarks_val,
        })

    return checkpoints


def extract_damage_from_remarks(remarks):
    """Parse researcher remarks to extract crack count and delamination info."""
    if not remarks or remarks == 'nan' or len(remarks) < 3:
        return None

    info = {'text': remarks, 'crack_count': None, 'has_delamination': False}

    # Extract crack count
    crack_match = re.search(r'(\d+)\s*cracks?\s*observed', remarks, re.IGNORECASE)
    if not crack_match:
        crack_match = re.search(r'(\d+)\s*(?:H\s*)?cracks?', remarks, re.IGNORECASE)
    if crack_match:
        info['crack_count'] = int(crack_match.group(1))

    # Check for delamination
    if 'delam' in remarks.lower():
        info['has_delamination'] = True

    return info


def find_strain_files(strain_dir, mts_base, prefix):
    """Find STRAIN_A/M/S files that correspond to a given MTS base checkpoint.
    
    The STRAIN files sometimes have the same number as the MTS base,
    or the next number. We try both.
    """
    if not os.path.exists(strain_dir):
        return {}

    # Extract the file number from mts_base (e.g., L1_S11_F07 -> 7)
    m = re.match(r'.*_([FS])(\d+)$', mts_base)
    if not m:
        return {}

    test_type, num_str = m.groups()
    num = int(num_str)

    results = {}
    # Try the same number and next number
    for try_num in [num, num + 1]:
        for stype in ['A', 'M', 'S']:
            fname = f"{prefix}_{test_type}{try_num:02d}_STRAIN_{stype}_DAT.mat"
            fpath = os.path.join(strain_dir, fname)
            if os.path.exists(fpath):
                results[stype] = {
                    'path': fpath,
                    'filename': fname,
                }

    return results


def generate_triples(specimen_id, config):
    """Generate Turtle triples for one specimen."""
    triples = []
    triples.append(f"\n# {'='*70}")
    triples.append(f"# Experimental Data for {specimen_id}")
    triples.append(f"# {'='*70}\n")

    # Read logbook
    checkpoints = read_logbook(specimen_id, config)
    if not checkpoints:
        print(f"  No checkpoints found for {specimen_id}")
        return triples

    strain_dir = os.path.join(config['dir'], 'StrainData')
    checkpoint_count = 0
    measurement_count = 0
    observation_count = 0

    for cp in checkpoints:
        mts_base = cp['mts_base']
        cycles = cp['cycles']
        cp_uri = safe_uri(f"{specimen_id}_cp_{cycles}_{mts_base.split('_')[-1]}")

        # --- FatigueCheckpoint ---
        triples.append(f"ex:{cp_uri} a ex:FatigueCheckpoint ;")
        triples.append(f'    ex:atCycleCount {cycles} ;')
        triples.append(f'    ex:mtsFileName "{escape_ttl_string(mts_base)}" ;')
        if cp['date']:
            triples.append(f'    ex:testDate "{escape_ttl_string(cp["date"])}" ;')
        if cp['load']:
            triples.append(f'    ex:loadKips "{cp["load"]:.3f}"^^xsd:float ;')
        triples.append(f'    rdfs:label "Checkpoint {mts_base} at {cycles} cycles" .')
        triples.append(f"")

        # Link coupon to checkpoint
        triples.append(f"ex:{specimen_id} ex:hasCheckpoint ex:{cp_uri} .")
        triples.append(f"")
        checkpoint_count += 1

        # --- DamageObservation ---
        damage_info = extract_damage_from_remarks(cp['remarks'])
        if damage_info:
            obs_uri = safe_uri(f"{specimen_id}_damage_{cycles}")
            triples.append(f"ex:{obs_uri} a ex:DamageObservation ;")
            triples.append(f'    ex:observationText "{escape_ttl_string(damage_info["text"])}" ;')
            if damage_info['crack_count'] is not None:
                triples.append(f'    ex:crackCount {damage_info["crack_count"]} ;')
            triples.append(f'    ex:hasDelamination {"true" if damage_info["has_delamination"] else "false"} .')
            triples.append(f"ex:{cp_uri} ex:hasDamageObservation ex:{obs_uri} .")
            triples.append(f"")
            observation_count += 1

        # --- StrainMeasurements ---
        strain_files = find_strain_files(strain_dir, mts_base, config['prefix'])

        for stype_code, stype_info in strain_files.items():
            stype_name = STRAIN_TYPES[stype_code]
            stats = extract_strain_stats(stype_info['path'])
            if not stats:
                continue

            for gauge_id, gauge_stats in stats.items():
                meas_uri = safe_uri(
                    f"{specimen_id}_strain_{cycles}_{stype_code}_{gauge_id}"
                )
                triples.append(f"ex:{meas_uri} a ex:StrainMeasurement ;")
                triples.append(f"    ex:hasStrainType ex:{stype_name} ;")
                triples.append(f'    ex:gaugeID "{gauge_id}" ;')
                triples.append(f'    ex:sourceFile "{escape_ttl_string(stype_info["filename"])}" ;')
                triples.append(f'    ex:numDataPoints {gauge_stats["n_points"]} ;')
                triples.append(f'    ex:strainMean "{gauge_stats["mean"]:.10f}"^^xsd:float ;')
                triples.append(f'    ex:strainStd "{gauge_stats["std"]:.10f}"^^xsd:float ;')
                triples.append(f'    ex:strainMax "{gauge_stats["max"]:.10f}"^^xsd:float ;')
                triples.append(f'    ex:strainMin "{gauge_stats["min"]:.10f}"^^xsd:float ;')
                triples.append(f'    ex:strainRMS "{gauge_stats["rms"]:.10f}"^^xsd:float ;')
                triples.append(f'    ex:strainRange "{gauge_stats["range"]:.10f}"^^xsd:float .')
                triples.append(f"ex:{cp_uri} ex:hasMeasurement ex:{meas_uri} .")
                triples.append(f"")
                measurement_count += 1

    print(f"  {specimen_id}: {checkpoint_count} checkpoints, "
          f"{measurement_count} measurements, {observation_count} damage observations")

    return triples


def extract_pzt_damage_observations(specimen_id, config):
    """Extract damage observations from PZT data files' comment fields."""
    pzt_dir = os.path.join(config['dir'], 'PZT-data')
    if not os.path.exists(pzt_dir):
        return []

    triples = []
    observations = {}  # {cycles: comment}

    for f in sorted(os.listdir(pzt_dir)):
        if not f.endswith('.mat'):
            continue
        try:
            d = scipy.io.loadmat(
                os.path.join(pzt_dir, f),
                struct_as_record=False, squeeze_me=True
            )
            coupon = d.get('coupon')
            if coupon is None:
                continue

            comment = str(getattr(coupon, 'comment', ''))
            cycles = int(getattr(coupon, 'cycles', 0))

            if (comment and comment != '[]' and comment != 'nan'
                    and len(comment) > 3 and comment != '\n'
                    and cycles not in observations):
                observations[cycles] = comment
        except Exception:
            continue

    if observations:
        triples.append(f"\n# PZT-derived damage observations for {specimen_id}")
        for cycles, comment in sorted(observations.items()):
            damage_info = extract_damage_from_remarks(comment)
            if not damage_info:
                continue

            obs_uri = safe_uri(f"{specimen_id}_pzt_damage_{cycles}")
            cp_uri = safe_uri(f"{specimen_id}_cp_{cycles}")

            # Only add if we don't already have an observation at this cycle
            triples.append(f"ex:{obs_uri} a ex:DamageObservation ;")
            triples.append(f'    ex:observationText "{escape_ttl_string(damage_info["text"])}" ;')
            if damage_info['crack_count'] is not None:
                triples.append(f'    ex:crackCount {damage_info["crack_count"]} ;')
            triples.append(f'    ex:hasDelamination {"true" if damage_info["has_delamination"] else "false"} .')
            triples.append(f"")

        print(f"  {specimen_id}: {len(observations)} PZT damage observations")

    return triples


def main():
    """Main entry point."""
    print("=" * 60)
    print("CFRP Ontology Population")
    print("=" * 60)
    print(f"Schema: {SCHEMA_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    # Read the original schema
    if not os.path.exists(SCHEMA_PATH):
        print(f"ERROR: Schema file not found: {SCHEMA_PATH}")
        sys.exit(1)

    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        schema_content = f.read()

    # Generate data triples for each specimen
    all_triples = []
    all_triples.append("\n\n# " + "=" * 70)
    all_triples.append("# POPULATED EXPERIMENTAL DATA")
    all_triples.append("# Generated by scripts/populate_ontology.py")
    all_triples.append("# " + "=" * 70)

    for specimen_id, config in SPECIMENS.items():
        print(f"\nProcessing {specimen_id}...")
        if not os.path.exists(config['dir']):
            print(f"  SKIP: Directory not found: {config['dir']}")
            continue

        # Main strain + logbook data
        triples = generate_triples(specimen_id, config)
        all_triples.extend(triples)

        # PZT damage observations
        pzt_triples = extract_pzt_damage_observations(specimen_id, config)
        all_triples.extend(pzt_triples)

    # Write output
    output_content = schema_content + "\n" + "\n".join(all_triples) + "\n"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(output_content)

    print(f"\n{'=' * 60}")
    print(f"Output written to: {OUTPUT_PATH}")

    # Validate with rdflib
    try:
        from rdflib import Graph
        g = Graph()
        g.parse(OUTPUT_PATH, format='turtle')
        print(f"Validation: PASSED — {len(g)} triples loaded successfully")
    except ImportError:
        print("rdflib not installed — skipping validation")
    except Exception as e:
        print(f"Validation: FAILED — {e}")

    print("=" * 60)


if __name__ == '__main__':
    main()
