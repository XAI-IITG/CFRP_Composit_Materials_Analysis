
from __future__ import annotations

"""
Time-Sliced Relational KG Fusion for CFRP RUL
=============================================

A reference implementation of a novel idea for the CFRP KG branch:

1. Build a time-sliced, leakage-safe subgraph G_{i,<=c} for each sequence window.
2. Encode the multi-relational subgraph with an R-GCN style encoder.
3. Optionally regularize node/relation embeddings with a TransE loss.
4. Fuse graph memory into the Transformer sequence encoder with cross-attention.
5. Predict both RUL and degradation stage.

Dependencies:
    pip install rdflib torch
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict

import math
import random

import rdflib
import torch
import torch.nn as nn
import torch.nn.functional as F


EX_NS = "http://example.org/cfrp/"


@dataclass
class SampleRequest:
    specimen_id: str
    current_cycle: int


@dataclass
class GraphBatch:
    node_features: torch.Tensor       # [N, node_feat_dim]
    node_type_ids: torch.Tensor       # [N]
    edge_index: torch.Tensor          # [2, E]
    edge_type: torch.Tensor           # [E]
    target_node_idx: int
    memory_node_indices: torch.Tensor # [M]
    relation_count: int
    node_count: int


class TimeSlicedCFRPKG:
    """
    Parses the populated TTL ontology and emits time-sliced, relation-aware graphs.

    Important design choice:
    only include checkpoints with atCycleCount <= current_cycle to avoid future leakage.
    """

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

        self._parse_graph()
        self.relations = sorted({r for _, r, _ in self.entity_triples})
        self.rel2id = {r: i for i, r in enumerate(self.relations)}

        self.node_types = sorted({t for t in self.type_map.values()}) + ["Unknown"]
        self.node_type2id = {t: i for i, t in enumerate(self.node_types)}

        self.numeric_feature_keys = sorted(
            {
                key
                for attrs in self.literal_map.values()
                for key in attrs.keys()
            }
        )
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
                if p_name == "type":
                    self.type_map[s_name] = o_name
                if p_name == "hasCheckpoint":
                    self.coupon_checkpoints[s_name].append(o_name)

            elif isinstance(o, rdflib.Literal):
                # Keep numeric literals as node attributes when possible.
                try:
                    numeric_value = float(o)
                    self.literal_map[s_name][p_name] = numeric_value
                    if p_name == "atCycleCount":
                        self.checkpoint_cycle[s_name] = int(round(numeric_value))
                except Exception:
                    continue

    def _coupon_candidates(self, specimen_id: str) -> List[str]:
        if specimen_id.startswith("L"):
            return [specimen_id]
        return [f"{prefix}{specimen_id}" for prefix in ("L1", "L2", "L3")]

    def resolve_coupon(self, specimen_id: str) -> str:
        for candidate in self._coupon_candidates(specimen_id):
            if candidate in self.type_map:
                return candidate
        if specimen_id in self.type_map:
            return specimen_id
        raise KeyError(f"Could not resolve specimen_id={specimen_id!r} in ontology")

    def _allowed_nodes(self, coupon_id: str, current_cycle: int) -> set[str]:
        allowed = {coupon_id}

        # Include coupon-adjacent static structure.
        for s, r, o in self.entity_triples:
            if s == coupon_id or o == coupon_id:
                allowed.add(s)
                allowed.add(o)

        # Include only checkpoints available up to current_cycle.
        for cp in self.coupon_checkpoints.get(coupon_id, []):
            cycle = self.checkpoint_cycle.get(cp, None)
            if cycle is not None and cycle <= current_cycle:
                allowed.add(cp)

        # Expand one hop around allowed checkpoints to pull in measurements/damage observations.
        changed = True
        while changed:
            changed = False
            for s, r, o in self.entity_triples:
                if s in allowed and o not in allowed:
                    # only expand from already allowed dynamic nodes or coupon
                    if s == coupon_id or s in self.checkpoint_cycle or self.type_map.get(s, "") in (
                        "FatigueCheckpoint",
                        "DamageObservation",
                        "StrainMeasurement",
                    ):
                        allowed.add(o)
                        changed = True
                if o in allowed and s not in allowed:
                    if o == coupon_id or o in self.checkpoint_cycle or self.type_map.get(o, "") in (
                        "FatigueCheckpoint",
                        "DamageObservation",
                        "StrainMeasurement",
                    ):
                        allowed.add(s)
                        changed = True

        return allowed

    def build_sample_graph(self, specimen_id: str, current_cycle: int) -> GraphBatch:
        coupon_id = self.resolve_coupon(specimen_id)
        allowed = self._allowed_nodes(coupon_id, current_cycle)

        triples = [
            (s, r, o)
            for (s, r, o) in self.entity_triples
            if s in allowed and o in allowed and r != "type"
        ]

        nodes = sorted({coupon_id} | {s for s, _, _ in triples} | {o for _, _, o in triples})
        node2id = {n: i for i, n in enumerate(nodes)}

        # Node features: numeric literals in fixed order.
        x_num = torch.zeros(len(nodes), self.num_feat_dim, dtype=torch.float32)
        node_type_ids = torch.zeros(len(nodes), dtype=torch.long)

        for node, idx in node2id.items():
            attrs = self.literal_map.get(node, {})
            for j, key in enumerate(self.numeric_feature_keys):
                if key in attrs:
                    x_num[idx, j] = float(attrs[key])

            node_type = self.type_map.get(node, "Unknown")
            node_type_ids[idx] = self.node_type2id.get(node_type, self.node_type2id["Unknown"])

        # Build directed edges and relation ids.
        edges_src, edges_dst, edge_types = [], [], []
        for s, r, o in triples:
            rid = self.rel2id[r]
            edges_src.append(node2id[s])
            edges_dst.append(node2id[o])
            edge_types.append(rid)
            # explicit reverse edge
            rev_name = f"rev_{r}"
            if rev_name not in self.rel2id:
                self.rel2id[rev_name] = len(self.rel2id)
            edges_src.append(node2id[o])
            edges_dst.append(node2id[s])
            edge_types.append(self.rel2id[rev_name])

        if edges_src:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros((0,), dtype=torch.long)

        # Memory nodes: coupon + checkpoints + measurements + damage observations visible so far.
        memory_nodes = []
        for node in nodes:
            node_type = self.type_map.get(node, "Unknown")
            if (
                node == coupon_id
                or node_type in {"FatigueCheckpoint", "StrainMeasurement", "DamageObservation", "Layup"}
            ):
                memory_nodes.append(node2id[node])
        if not memory_nodes:
            memory_nodes = [node2id[coupon_id]]

        return GraphBatch(
            node_features=x_num,
            node_type_ids=node_type_ids,
            edge_index=edge_index,
            edge_type=edge_type,
            target_node_idx=node2id[coupon_id],
            memory_node_indices=torch.tensor(memory_nodes, dtype=torch.long),
            relation_count=len(self.rel2id),
            node_count=len(nodes),
        )


class RelGraphConvLayer(nn.Module):
    """
    Small R-GCN style layer implemented in plain PyTorch.

    h_v^{l+1} = sigma( W0 h_v^l + sum_r mean_{u in N_r(v)} W_r h_u^l )
    """

    def __init__(self, hidden_dim: int, num_relations: int, dropout: float = 0.1):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.rel_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_relations)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        out = self.self_linear(h)

        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            for rid, linear in enumerate(self.rel_linears):
                mask = edge_type == rid
                if mask.sum() == 0:
                    continue
                rel_src = src[mask]
                rel_dst = dst[mask]
                msgs = linear(h[rel_src])
                agg = torch.zeros_like(out)
                agg.index_add_(0, rel_dst, msgs)

                deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
                deg.index_add_(0, rel_dst, torch.ones_like(rel_dst, dtype=h.dtype))
                deg = deg.clamp_min(1.0).unsqueeze(-1)
                out = out + agg / deg

        out = self.norm(F.relu(out))
        out = self.dropout(out)
        return out


class TemporalRelationalEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        num_node_types: int,
        hidden_dim: int,
        num_relations: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.type_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.feat_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [RelGraphConvLayer(hidden_dim, num_relations, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, batch: GraphBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.feat_proj(batch.node_features) + self.type_embedding(batch.node_type_ids)
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.edge_type)

        target = h[batch.target_node_idx]                    # [H]
        memory = h[batch.memory_node_indices]                # [M, H]
        return target, memory


class CrossAttentionFusion(nn.Module):
    """
    Fuse graph memory into time-series states with cross-attention.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_states: torch.Tensor, graph_memory: torch.Tensor) -> torch.Tensor:
        # seq_states: [B, T, H], graph_memory: [B, M, H]
        fused, _ = self.attn(query=seq_states, key=graph_memory, value=graph_memory)
        return self.norm(seq_states + self.dropout(fused))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TimeSlicedRelationalKGRUL(nn.Module):
    """
    Novel model:
      - time-series encoder on sensor windows
      - time-sliced relational KG encoder
      - graph-to-sequence cross-attention fusion
      - multi-task outputs (RUL + stage)
    """

    def __init__(
        self,
        input_dim: int,
        node_feat_dim: int,
        num_node_types: int,
        num_relations: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        num_stages: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.graph_encoder = TemporalRelationalEncoder(
            node_feat_dim=node_feat_dim,
            num_node_types=num_node_types,
            hidden_dim=d_model,
            num_relations=num_relations,
            num_layers=2,
            dropout=dropout,
        )

        self.fusion = CrossAttentionFusion(hidden_dim=d_model, num_heads=max(1, nhead // 2), dropout=dropout)

        self.rul_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.stage_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_stages),
        )

        # relation embeddings for optional TransE-style regularization
        self.rel_embeddings = nn.Embedding(num_relations, d_model)

    def forward(self, x_seq: torch.Tensor, graph_batches: Sequence[GraphBatch]) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_seq: [B, T, D]
            graph_batches: length-B sequence of GraphBatch objects
        """
        bsz, seq_len, _ = x_seq.shape
        z = self.input_proj(x_seq)
        z = self.pos_enc(z)
        z = self.seq_encoder(z)

        target_list = []
        memory_list = []
        max_memory = 1
        for gb in graph_batches:
            target, memory = self.graph_encoder(gb)
            target_list.append(target)
            memory_list.append(memory)
            max_memory = max(max_memory, memory.size(0))

        # pad graph memories in batch
        memory_batch = []
        for memory in memory_list:
            if memory.size(0) < max_memory:
                pad = torch.zeros(max_memory - memory.size(0), memory.size(1), device=memory.device, dtype=memory.dtype)
                memory = torch.cat([memory, pad], dim=0)
            memory_batch.append(memory)
        memory_batch = torch.stack(memory_batch, dim=0)  # [B, M, H]
        target_batch = torch.stack(target_list, dim=0)   # [B, H]

        z_fused = self.fusion(z, memory_batch)
        seq_summary = z_fused.mean(dim=1)

        joint = torch.cat([seq_summary, target_batch], dim=-1)
        rul = self.rul_head(joint).squeeze(-1)
        stage_logits = self.stage_head(joint)

        return {
            "rul": rul,
            "stage_logits": stage_logits,
            "seq_summary": seq_summary,
            "graph_summary": target_batch,
        }

    def transe_loss(
        self,
        graph_batches: Sequence[GraphBatch],
        margin: float = 1.0,
        num_negative: int = 2,
    ) -> torch.Tensor:
        """
        Optional relation-regularization term:
            ||h + r - t||_2 should be low for positive triples
            and high for corrupted triples.
        """
        losses = []
        for gb in graph_batches:
            target, _ = self.graph_encoder(gb)
            # We need all node embeddings; recompute directly.
            h_nodes = self.graph_encoder.feat_proj(gb.node_features) + self.graph_encoder.type_embedding(gb.node_type_ids)
            for layer in self.graph_encoder.layers:
                h_nodes = layer(h_nodes, gb.edge_index, gb.edge_type)

            if gb.edge_index.numel() == 0:
                continue

            src = gb.edge_index[0]
            dst = gb.edge_index[1]
            rel = gb.edge_type

            pos_score = torch.norm(h_nodes[src] + self.rel_embeddings(rel) - h_nodes[dst], p=2, dim=-1)

            n_nodes = h_nodes.size(0)
            for _ in range(num_negative):
                corrupt_dst = torch.randint(0, n_nodes, size=dst.shape, device=dst.device)
                neg_score = torch.norm(h_nodes[src] + self.rel_embeddings(rel) - h_nodes[corrupt_dst], p=2, dim=-1)
                losses.append(F.relu(margin + pos_score - neg_score).mean())

        if not losses:
            return torch.tensor(0.0, device=self.rel_embeddings.weight.device)
        return torch.stack(losses).mean()


def multitask_loss(
    outputs: Dict[str, torch.Tensor],
    y_rul: torch.Tensor,
    y_stage: torch.Tensor,
    model: TimeSlicedRelationalKGRUL,
    graph_batches: Sequence[GraphBatch],
    lambda_stage: float = 0.5,
    lambda_kg: float = 0.1,
) -> Dict[str, torch.Tensor]:
    rul_loss = F.smooth_l1_loss(outputs["rul"], y_rul)
    stage_loss = F.cross_entropy(outputs["stage_logits"], y_stage)
    kg_loss = model.transe_loss(graph_batches)

    total = rul_loss + lambda_stage * stage_loss + lambda_kg * kg_loss
    return {
        "total": total,
        "rul_loss": rul_loss,
        "stage_loss": stage_loss,
        "kg_loss": kg_loss,
    }


def stage_from_rul(
    y: torch.Tensor,
    tau1: float,
    tau2: float,
    tau3: float,
) -> torch.Tensor:
    """
    Maps RUL to 4 ordered stages:
      0 = Critical, 1 = Progressive, 2 = Early, 3 = Healthy
    """
    out = torch.zeros_like(y, dtype=torch.long)
    out = torch.where(y > tau1, torch.ones_like(out), out)
    out = torch.where(y > tau2, torch.full_like(out, 2), out)
    out = torch.where(y > tau3, torch.full_like(out, 3), out)
    return out


def collate_graph_batches(
    kg: TimeSlicedCFRPKG,
    specimen_ids: Sequence[str],
    current_cycles: Sequence[int],
) -> List[GraphBatch]:
    return [kg.build_sample_graph(sid, cyc) for sid, cyc in zip(specimen_ids, current_cycles)]


def demo_batch() -> Tuple[torch.Tensor, List[str], List[int]]:
    """
    Small synthetic batch for smoke testing.
    """
    x = torch.randn(2, 10, 16)
    specimen_ids = ["S11", "S12"]
    cycles = [10000, 20000]
    return x, specimen_ids, cycles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl", type=str, required=True, help="Path to populated CFRP ontology TTL")
    args = parser.parse_args()

    kg = TimeSlicedCFRPKG(args.ttl)
    x_seq, specimen_ids, cycles = demo_batch()
    graph_batches = collate_graph_batches(kg, specimen_ids, cycles)

    model = TimeSlicedRelationalKGRUL(
        input_dim=x_seq.size(-1),
        node_feat_dim=kg.num_feat_dim,
        num_node_types=len(kg.node_type2id),
        num_relations=max(gb.relation_count for gb in graph_batches),
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_stages=4,
    )

    y_rul = torch.tensor([120.0, 80.0], dtype=torch.float32)
    y_stage = stage_from_rul(y_rul, tau1=50.0, tau2=100.0, tau3=150.0)

    out = model(x_seq, graph_batches)
    losses = multitask_loss(out, y_rul, y_stage, model, graph_batches)

    print("rul:", out["rul"].detach())
    print("stage logits:", out["stage_logits"].shape)
    print({k: float(v.detach()) for k, v in losses.items()})
