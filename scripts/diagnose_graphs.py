"""Quick diagnostic: check cycle distribution and graph sizes."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from time_sliced_graph_builder import TimeSlicedCFRPKG

project_root = Path(__file__).resolve().parent.parent
npz = np.load(project_root / "data" / "processed" / "cfrp_windows.npz", allow_pickle=True)
cyc = npz["current_cycles"]
sids = npz["specimen_ids"]

print("=== Cycle Distribution ===")
print(f"  Range: [{cyc.min()}, {cyc.max()}]")
for p in [0, 25, 50, 75, 100]:
    print(f"  {p}th percentile: {int(np.percentile(cyc, p))}")

print("\n=== Per-Specimen ===")
for s in sorted(set(sids)):
    mask = sids == s
    sc = cyc[mask]
    print(f"  {s}: n={mask.sum()}, cycles=[{sc.min()}, {sc.max()}]")

print("\n=== Graph Sizes at Sample Cycle Counts ===")
ttl_path = str(project_root / "data" / "ontology" / "cfrp_ontology_populated.ttl")
kg = TimeSlicedCFRPKG(ttl_path)

# Check graphs at various cycle counts for each specimen
for sid in sorted(set(sids)):
    mask = sids == sid
    sc = cyc[mask]
    test_cycles = [int(sc.min()), int(np.median(sc)), int(sc.max())]
    for c in test_cycles:
        try:
            g = kg.build_sample_graph(sid, c)
            print(f"  {sid} @ cycle={c:>7}: nodes={g.node_count:>3}, edges={g.edge_index.shape[1]:>4}, memory={g.memory_node_indices.shape[0]:>3}")
        except Exception as e:
            print(f"  {sid} @ cycle={c:>7}: ERROR: {e}")
