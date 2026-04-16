from __future__ import annotations

"""
Leakage-safe time-sliced graph builder for the CFRP ontology.

This module extracts a per-sample subgraph G_{i,<=c} from the populated CFRP ontology,
where c is the cycle count at the end of the sequence window. Only ontology facts
available up to that cycle are included for dynamic checkpoint-linked entities.
"""

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Sequence, Tuple

import rdflib
import torch

EX_NS = "http://example.org/cfrp/"

STATIC_RELATIONS = {
    "hasLayup",
    "hasMaterial",
    "hasFeature",
    "hasBoundaryCondition",
    "susceptibleTo",
    "correlatesWith",
    "precedes",
    "causesProgression",
    "extractedFrom",
    "derivedFrom",
}

DYNAMIC_EXPANSION_RELATIONS = {
    "hasMeasurement",
    "hasDamageObservation",
    "measuredUnder",
    "hasStrainType",
}

DYNAMIC_NODE_TYPES = {
    "FatigueCheckpoint",
    "StrainMeasurement",
    "DamageObservation",
}

COUPON_PREFIXES = ("L1", "L2", "L3")


@dataclass
class GraphBatch:
    node_features: torch.Tensor
    node_type_ids: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    target_node_idx: int
    memory_node_indices: torch.Tensor
    relation_count: int
    node_count: int
    node_names: List[str]


class TimeSlicedCFRPKG:
    def __init__(self, ttl_path: str, ex_ns: str = EX_NS):
        self.ttl_path = ttl_path
        self.ex_ns = ex_ns

        self.graph = rdflib.Graph()
        self.graph.parse(ttl_path, format="turtle")

        self.entity_triples: List[Tuple[str, str, str]] = []
        self.literal_map: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.type_map: Dict[str, str] = {}
        self.checkpoint_cycle: Dict[str, int] = {}
        self.coupon_checkpoints: Dict[str, List[str]] = defaultdict(list)
        self.coupon_ids: set[str] = set()

        self.out_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.in_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._sample_cache: Dict[Tuple[str, int], GraphBatch] = {}
        self._static_cache: Dict[str, set[str]] = {}

        self._parse_graph()

        self.relations = sorted({r for _, r, _ in self.entity_triples})
        self.rel2id = {r: i for i, r in enumerate(self.relations)}

        self.node_types = sorted({t for t in self.type_map.values()}) + ["Unknown"]
        self.node_type2id = {t: i for i, t in enumerate(self.node_types)}

        self.numeric_feature_keys = sorted({k for attrs in self.literal_map.values() for k in attrs.keys()})
        self.num_feat_dim = len(self.numeric_feature_keys)

    def _short(self, uri: str) -> str:
        if uri.startswith(self.ex_ns):
            return uri[len(self.ex_ns):]
        if "#" in uri:
            return uri.split("#")[-1]
        return uri.rsplit("/", 1)[-1]

    def _parse_graph(self) -> None:
        for s, p, o in self.graph:
            s_name = self._short(str(s))
            p_name = self._short(str(p))

            if isinstance(o, rdflib.URIRef):
                o_name = self._short(str(o))
                self.entity_triples.append((s_name, p_name, o_name))
                self.out_edges[s_name].append((p_name, o_name))
                self.in_edges[o_name].append((p_name, s_name))
                if p_name == "type":
                    self.type_map[s_name] = o_name
                    if o_name == "Coupon":
                        self.coupon_ids.add(s_name)
                if p_name == "hasCheckpoint":
                    self.coupon_checkpoints[s_name].append(o_name)
            elif isinstance(o, rdflib.Literal):
                try:
                    numeric_value = float(o)
                    self.literal_map[s_name][p_name] = numeric_value
                    if p_name == "atCycleCount":
                        self.checkpoint_cycle[s_name] = int(round(numeric_value))
                except Exception:
                    continue

    def resolve_coupon(self, specimen_id: str) -> str:
        candidates = [specimen_id]
        if not specimen_id.startswith("L"):
            candidates.extend(f"{prefix}{specimen_id}" for prefix in COUPON_PREFIXES)
        for cand in candidates:
            if cand in self.coupon_ids:
                return cand
        raise KeyError(f"Could not resolve specimen_id={specimen_id!r} to ontology coupon")

    def build_static_neighbourhood(self, coupon_id: str) -> set[str]:
        if coupon_id in self._static_cache:
            return set(self._static_cache[coupon_id])

        allowed = {coupon_id}
        q = deque([coupon_id])
        while q:
            node = q.popleft()
            for rel, nbr in self.out_edges.get(node, []):
                if rel in STATIC_RELATIONS and nbr not in allowed:
                    allowed.add(nbr)
                    q.append(nbr)
            for rel, nbr in self.in_edges.get(node, []):
                if rel in STATIC_RELATIONS and nbr not in allowed:
                    allowed.add(nbr)
                    q.append(nbr)
        self._static_cache[coupon_id] = set(allowed)
        return allowed

    def build_dynamic_neighbourhood(self, coupon_id: str, current_cycle: int) -> set[str]:
        allowed: set[str] = set()
        q = deque()
        for cp in self.coupon_checkpoints.get(coupon_id, []):
            cyc = self.checkpoint_cycle.get(cp)
            if cyc is not None and cyc <= current_cycle:
                allowed.add(cp)
                q.append(cp)

        while q:
            node = q.popleft()
            for rel, nbr in self.out_edges.get(node, []):
                if rel in DYNAMIC_EXPANSION_RELATIONS and nbr not in allowed:
                    allowed.add(nbr)
                    q.append(nbr)
            for rel, nbr in self.in_edges.get(node, []):
                if rel in DYNAMIC_EXPANSION_RELATIONS and nbr not in allowed:
                    allowed.add(nbr)
                    q.append(nbr)
        return allowed

    def _allowed_nodes(self, coupon_id: str, current_cycle: int) -> set[str]:
        allowed = self.build_static_neighbourhood(coupon_id)
        allowed.update(self.build_dynamic_neighbourhood(coupon_id, current_cycle))
        return allowed

    def build_sample_graph(self, specimen_id: str, current_cycle: int) -> GraphBatch:
        coupon_id = self.resolve_coupon(specimen_id)
        cache_key = (coupon_id, int(current_cycle))
        if cache_key in self._sample_cache:
            return self._sample_cache[cache_key]

        allowed = self._allowed_nodes(coupon_id, int(current_cycle))
        triples = []
        for s, r, o in self.entity_triples:
            if r == "type":
                continue
            if r in {"influencesRUL", "basedOnCoupon"}:
                continue
            if s in allowed and o in allowed:
                triples.append((s, r, o))

        nodes = sorted({coupon_id} | {s for s, _, _ in triples} | {o for _, _, o in triples})
        node2id = {n: i for i, n in enumerate(nodes)}

        node_features = torch.zeros((len(nodes), self.num_feat_dim), dtype=torch.float32)
        node_type_ids = torch.zeros(len(nodes), dtype=torch.long)
        for n, idx in node2id.items():
            attrs = self.literal_map.get(n, {})
            for j, key in enumerate(self.numeric_feature_keys):
                if key in attrs:
                    node_features[idx, j] = float(attrs[key])
            ntype = self.type_map.get(n, "Unknown")
            node_type_ids[idx] = self.node_type2id.get(ntype, self.node_type2id["Unknown"])

        rel2id_local = dict(self.rel2id)
        src, dst, etype = [], [], []
        for s, r, o in triples:
            rid = rel2id_local.setdefault(r, len(rel2id_local))
            src.append(node2id[s]); dst.append(node2id[o]); etype.append(rid)
            rev_r = f"rev_{r}"
            rev_id = rel2id_local.setdefault(rev_r, len(rel2id_local))
            src.append(node2id[o]); dst.append(node2id[s]); etype.append(rev_id)

        if src:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_type = torch.tensor(etype, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros((0,), dtype=torch.long)

        memory_nodes = []
        for n in nodes:
            ntype = self.type_map.get(n, "Unknown")
            if n == coupon_id or ntype in DYNAMIC_NODE_TYPES or ntype in {"Layup", "Feature", "BoundaryCondition", "Material"}:
                memory_nodes.append(node2id[n])
        if not memory_nodes:
            memory_nodes = [node2id[coupon_id]]

        batch = GraphBatch(
            node_features=node_features,
            node_type_ids=node_type_ids,
            edge_index=edge_index,
            edge_type=edge_type,
            target_node_idx=node2id[coupon_id],
            memory_node_indices=torch.tensor(memory_nodes, dtype=torch.long),
            relation_count=len(rel2id_local),
            node_count=len(nodes),
            node_names=nodes,
        )
        self._sample_cache[cache_key] = batch
        return batch

    def summarize_visible_checkpoint_counts(self, specimen_ids: Sequence[str], current_cycles: Sequence[int]) -> List[Tuple[str, int, int]]:
        rows = []
        for sid, cyc in zip(specimen_ids, current_cycles):
            coupon_id = self.resolve_coupon(sid)
            visible = self.build_dynamic_neighbourhood(coupon_id, int(cyc))
            n_cp = sum(1 for n in visible if self.type_map.get(n) == "FatigueCheckpoint")
            rows.append((sid, int(cyc), n_cp))
        return rows


def collate_graph_batches(builder: TimeSlicedCFRPKG, specimen_ids: Sequence[str], current_cycles: Sequence[int]) -> List[GraphBatch]:
    return [builder.build_sample_graph(sid, cyc) for sid, cyc in zip(specimen_ids, current_cycles)]
