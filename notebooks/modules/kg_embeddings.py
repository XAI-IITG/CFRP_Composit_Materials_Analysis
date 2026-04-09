"""
KG Embedding Generator for CFRP RUL Prediction
=================================================
Parses the populated CFRP ontology TTL file, builds a NetworkX graph
from structural/ontological triples, runs a self-contained Node2Vec
implementation to produce static embeddings, and composes per-coupon
embedding vectors for model input enrichment.

Embedding Strategy (v2 — Neighbourhood Aggregation):
  Instead of concatenating fixed global entities (which produced 60%
  identical dimensions across all coupons), we aggregate each coupon's
  actual graph neighbourhood:
    1. Coupon's own embedding      — unique per coupon
    2. Mean of 1-hop neighbours    — captures direct connections
    3. Mean of 2-hop neighbours    — captures broader structural context
    4. Layup embedding             — structural family context
  Result: 4 × embed_dim dimensions, all unique per coupon.

Dependencies: rdflib, networkx, numpy (all already available)
"""

import numpy as np
import networkx as nx
import rdflib
from pathlib import Path
import random
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Predicates that encode structural/ontological relationships
# ============================================================================

STRUCTURAL_PREDICATES = {
    # Core ontological relationships
    'http://example.org/cfrp/hasLayup',
    'http://example.org/cfrp/hasMaterial',
    'http://example.org/cfrp/hasFeature',
    'http://example.org/cfrp/hasBoundaryCondition',
    'http://example.org/cfrp/susceptibleTo',
    'http://example.org/cfrp/correlatesWith',
    'http://example.org/cfrp/precedes',
    'http://example.org/cfrp/causesProgression',
    'http://example.org/cfrp/extractedFrom',
    'http://example.org/cfrp/derivedFrom',
    'http://example.org/cfrp/influencesRUL',
    'http://example.org/cfrp/basedOnCoupon',
    # Per-coupon experimental data connections (NEW — enables unique per-coupon
    # neighbourhoods by connecting coupons to their checkpoints, measurements,
    # damage observations, and boundary/strain-type nodes)
    'http://example.org/cfrp/hasCheckpoint',
    'http://example.org/cfrp/hasMeasurement',
    'http://example.org/cfrp/hasDamageObservation',
    'http://example.org/cfrp/measuredUnder',
    'http://example.org/cfrp/hasStrainType',
    # RDF/RDFS type hierarchy
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
    'http://www.w3.org/2000/01/rdf-schema#subClassOf',
}

# Only keep type triples for key classes (not every measurement)
KEY_RDF_TYPES = {
    'Coupon', 'Layup', 'Material', 'DamageMode', 'Feature',
    'MatrixCrack', 'Delamination', 'Fracture', 'FatigueTest',
    'BoundaryCondition', 'LambWaveMode', 'RULPrediction',
    'FatigueCheckpoint', 'StrainMeasurement', 'DamageObservation',
    'StrainType',
}

COUPON_IDS = [
    'L1S11', 'L1S12', 'L1S17', 'L1S18', 'L1S19',
    'L2S11', 'L2S12', 'L2S17', 'L2S18', 'L2S20',
    'L3S17', 'L3S18', 'L3S19', 'L3S20',
]

EX_NS = 'http://example.org/cfrp/'
RDF_TYPE = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'


# ============================================================================
# Lightweight Node2Vec with Skip-Gram (numpy-only)
# ============================================================================

class SimpleNode2Vec:
    """Lightweight Node2Vec using only numpy. Optimized for small graphs."""

    def __init__(self, graph, dimensions=16, walk_length=20, num_walks=50,
                 p=1.0, q=1.0, window=5, epochs=3, lr=0.01, seed=42):
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window = window
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

        self.nodes = list(graph.nodes())
        self.node2idx = {n: i for i, n in enumerate(self.nodes)}
        self.n_nodes = len(self.nodes)

        random.seed(seed)
        np.random.seed(seed)

        # Precompute neighbor lists as numpy arrays for faster sampling
        self._neighbors = {}
        for node in self.nodes:
            nbrs = list(graph.neighbors(node))
            self._neighbors[node] = nbrs

    def _random_walk(self, start):
        """Generate a biased random walk."""
        walk = [start]
        for _ in range(self.walk_length - 1):
            cur = walk[-1]
            nbrs = self._neighbors.get(cur, [])
            if not nbrs:
                break
            if len(walk) < 2:
                walk.append(random.choice(nbrs))
            else:
                prev = walk[-2]
                # Biased sampling
                weights = []
                for nbr in nbrs:
                    if nbr == prev:
                        weights.append(1.0 / self.p)
                    elif self.graph.has_edge(nbr, prev):
                        weights.append(1.0)
                    else:
                        weights.append(1.0 / self.q)
                total = sum(weights)
                weights = [w / total for w in weights]
                walk.append(random.choices(nbrs, weights=weights, k=1)[0])
        return walk

    def _generate_walks(self):
        """Generate all random walks."""
        walks = []
        nodes = list(self.nodes)
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._random_walk(node))
        print(f"  Generated {len(walks)} walks (len={self.walk_length})")
        return walks

    def _train_skipgram(self, walks):
        """Train Skip-Gram with negative sampling."""
        n = self.n_nodes
        d = self.dimensions
        n_neg = 5

        # Initialize embeddings (small init to avoid divergence)
        W = np.random.randn(n, d).astype(np.float32) * 0.01
        C = np.random.randn(n, d).astype(np.float32) * 0.01

        # Build frequency table for negative sampling
        freq = np.ones(n, dtype=np.float64)
        for walk in walks:
            for node in walk:
                freq[self.node2idx[node]] += 1
        neg_probs = freq ** 0.75
        neg_probs /= neg_probs.sum()

        # Pre-sample negative indices for speed
        total_neg_samples = len(walks) * self.walk_length * self.window * 2 * n_neg
        total_neg_samples = min(total_neg_samples, 5_000_000)
        neg_pool = np.random.choice(n, size=total_neg_samples, p=neg_probs)
        neg_ptr = 0

        lr = self.lr
        max_grad = 5.0  # Gradient clipping threshold

        for epoch in range(self.epochs):
            for walk in walks:
                indices = [self.node2idx[node] for node in walk]
                wlen = len(indices)

                for i in range(wlen):
                    center = indices[i]
                    start = max(0, i - self.window)
                    end = min(wlen, i + self.window + 1)

                    # Accumulate gradients for center word
                    grad_center = np.zeros(d, dtype=np.float32)

                    for j in range(start, end):
                        if i == j:
                            continue
                        ctx = indices[j]

                        # Positive update (use temp vars to avoid feedback)
                        dot = np.dot(W[center], C[ctx])
                        dot = np.clip(dot, -6, 6)
                        sig = 1.0 / (1.0 + np.exp(-dot))
                        g = lr * (1.0 - sig)
                        g = np.clip(g, -max_grad, max_grad)
                        
                        grad_w = g * C[ctx]
                        grad_c = g * W[center]
                        grad_center += grad_w
                        C[ctx] += grad_c

                        # Negative updates
                        if neg_ptr + n_neg > len(neg_pool):
                            neg_ptr = 0
                        neg_ids = neg_pool[neg_ptr:neg_ptr + n_neg]
                        neg_ptr += n_neg

                        dots = W[center] @ C[neg_ids].T
                        dots = np.clip(dots, -6, 6)
                        sigs = 1.0 / (1.0 + np.exp(-dots))
                        neg_grads = np.clip(-lr * sigs, -max_grad, max_grad)

                        grad_center += neg_grads @ C[neg_ids]
                        C[neg_ids] += np.outer(neg_grads, W[center])

                    W[center] += grad_center

            lr *= 0.8
            # Check for NaN
            if np.any(np.isnan(W)):
                print(f"  ⚠ NaN detected at epoch {epoch+1}, reinitializing...")
                W = np.random.randn(n, d).astype(np.float32) * 0.01
                C = np.random.randn(n, d).astype(np.float32) * 0.01
                lr = self.lr * 0.5
                continue
            print(f"  Epoch {epoch+1}/{self.epochs} done (max|W|={np.abs(W).max():.4f})")

        return W

    def fit(self):
        """Run Node2Vec pipeline."""
        walks = self._generate_walks()
        W = self._train_skipgram(walks)

        result = {}
        for node, idx in self.node2idx.items():
            result[node] = W[idx]
        return result


# ============================================================================
# KG Embedding Generator
# ============================================================================

class KGEmbeddingGenerator:
    """
    Generates static Node2Vec embeddings from the CFRP Knowledge Graph.
    
    Embedding Composition (v2 — Neighbourhood Aggregation):
      For each coupon, we concatenate:
        [coupon_emb | mean_1hop_emb | mean_2hop_emb | layup_emb]
      This produces 4 × embed_dim dimensions, all unique per coupon.
    
    Usage:
        gen = KGEmbeddingGenerator('path/to/cfrp_ontology_populated.ttl')
        gen.fit()
        emb = gen.get_coupon_embedding('L1S11')  # np.array shape (kg_embed_dim,)
    """

    # Number of components in the composed embedding
    N_COMPONENTS = 4  # coupon + 1-hop mean + 2-hop mean + layup

    def __init__(self, ttl_path, embed_dim=16, walk_length=20, num_walks=50,
                 p=1.0, q=1.0, seed=42):
        self.ttl_path = Path(ttl_path)
        self.embed_dim = embed_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.seed = seed

        self.graph = None
        self.embeddings = {}
        self._fitted = False
        self.kg_embed_dim = self.N_COMPONENTS * embed_dim  # 4 * 16 = 64

    def _parse_kg(self):
        """Parse TTL file and extract structural-level triples."""
        g = rdflib.Graph()
        g.parse(str(self.ttl_path), format='turtle')

        triples = []
        for s, p, o in g:
            # Skip literal objects
            if isinstance(o, rdflib.Literal):
                continue

            pred_str = str(p)
            if pred_str not in STRUCTURAL_PREDICATES:
                continue

            subj = self._uri_to_name(str(s))
            obj = self._uri_to_name(str(o))

            if not subj or not obj:
                continue

            # For rdf:type triples, only keep key structural types
            if pred_str == RDF_TYPE:
                if obj not in KEY_RDF_TYPES:
                    continue

            triples.append((subj, obj, pred_str.split('/')[-1].split('#')[-1]))

        print(f"  Extracted {len(triples)} structural triples from KG")
        return triples

    def _uri_to_name(self, uri):
        """Convert URI to short name."""
        if uri.startswith(EX_NS):
            return uri[len(EX_NS):]
        if '#' in uri:
            return uri.split('#')[-1]
        if '/' in uri:
            return uri.split('/')[-1]
        return uri

    def _build_graph(self, triples):
        """Build NetworkX graph from triples."""
        G = nx.Graph()
        for subj, obj, pred in triples:
            G.add_edge(subj, obj, relation=pred)
        print(f"  Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def fit(self):
        """Parse KG, build graph, train Node2Vec embeddings."""
        print("KG Embedding Generation:")
        triples = self._parse_kg()
        self.graph = self._build_graph(triples)

        n2v = SimpleNode2Vec(
            self.graph,
            dimensions=self.embed_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            seed=self.seed
        )
        self.embeddings = n2v.fit()
        self._fitted = True
        print(f"  ✓ KG embeddings ready ({len(self.embeddings)} nodes, dim={self.embed_dim})")

        # Print per-coupon neighbourhood sizes for diagnostics
        print(f"\n  Per-coupon graph neighbourhood sizes:")
        for cid in COUPON_IDS:
            if cid in self.graph:
                n1 = len(list(self.graph.neighbors(cid)))
                n2_set = set()
                for nb in self.graph.neighbors(cid):
                    n2_set.update(self.graph.neighbors(nb))
                n2_set.discard(cid)
                print(f"    {cid}: 1-hop={n1}, 2-hop={len(n2_set)}")
            else:
                print(f"    {cid}: NOT in graph")

        return self

    def _get_entity_embedding(self, entity_name):
        """Get embedding for a single entity, or zeros if not found."""
        if entity_name in self.embeddings:
            return self.embeddings[entity_name].astype(np.float32)
        return np.zeros(self.embed_dim, dtype=np.float32)

    def get_coupon_embedding(self, coupon_id):
        """
        Compose per-coupon KG embedding using neighbourhood aggregation.
        
        Components (4 × embed_dim):
          1. Coupon's own embedding           — unique per coupon
          2. Mean of 1-hop neighbour embeddings — direct connections
          3. Mean of 2-hop neighbour embeddings — broader context
          4. Layup embedding                   — structural family
        
        This replaces the old approach of concatenating fixed global
        damage-mode entities (which were identical across all coupons).
        """
        assert self._fitted, "Must call .fit() first"

        mapped_id = self._map_specimen_to_coupon(coupon_id)
        layup_id = self._get_layup_for_coupon(mapped_id)

        # 1. Coupon's own embedding
        emb_coupon = self._get_entity_embedding(mapped_id)

        # 2. Mean of 1-hop neighbours
        if mapped_id in self.graph:
            neighbors_1hop = list(self.graph.neighbors(mapped_id))
            if neighbors_1hop:
                emb_1hop = np.mean(
                    [self._get_entity_embedding(n) for n in neighbors_1hop],
                    axis=0
                ).astype(np.float32)
            else:
                emb_1hop = np.zeros(self.embed_dim, dtype=np.float32)

            # 3. Mean of 2-hop neighbours (excluding the coupon itself)
            neighbors_2hop = set()
            for n in neighbors_1hop:
                neighbors_2hop.update(self.graph.neighbors(n))
            neighbors_2hop.discard(mapped_id)
            # Also exclude direct 1-hop to get truly 2nd-order context
            neighbors_2hop -= set(neighbors_1hop)

            if neighbors_2hop:
                emb_2hop = np.mean(
                    [self._get_entity_embedding(n) for n in neighbors_2hop],
                    axis=0
                ).astype(np.float32)
            else:
                emb_2hop = np.zeros(self.embed_dim, dtype=np.float32)
        else:
            emb_1hop = np.zeros(self.embed_dim, dtype=np.float32)
            emb_2hop = np.zeros(self.embed_dim, dtype=np.float32)

        # 4. Layup embedding
        emb_layup = self._get_entity_embedding(layup_id)

        parts = [emb_coupon, emb_1hop, emb_2hop, emb_layup]
        return np.concatenate(parts).astype(np.float32)

    def _map_specimen_to_coupon(self, specimen_id):
        """Map notebook specimen IDs (e.g. 'S11') to KG coupon IDs (e.g. 'L1S11')."""
        if specimen_id in COUPON_IDS:
            return specimen_id
        for prefix in ['L1', 'L2', 'L3']:
            candidate = prefix + specimen_id
            if candidate in COUPON_IDS:
                return candidate
        return specimen_id

    def _get_layup_for_coupon(self, coupon_id):
        """Determine layup ID from coupon ID."""
        if coupon_id.startswith('L1'):
            return 'Layup1'
        elif coupon_id.startswith('L2'):
            return 'Layup2'
        elif coupon_id.startswith('L3'):
            return 'Layup3'
        return 'Layup1'

    def get_all_coupon_embeddings(self):
        """Get embeddings for all known coupons."""
        return {cid: self.get_coupon_embedding(cid) for cid in COUPON_IDS}

    def get_graph_stats(self):
        """Return summary statistics."""
        if self.graph is None:
            return {}
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_embeddings': len(self.embeddings),
            'embed_dim_per_entity': self.embed_dim,
            'kg_embed_dim': self.kg_embed_dim,
            'composition': 'neighbourhood_aggregation_v2',
            'components': 'coupon + 1hop_mean + 2hop_mean + layup',
        }


def generate_kg_embeddings(ttl_path, embed_dim=16, **kwargs):
    """One-call convenience function. Returns a fitted KGEmbeddingGenerator."""
    gen = KGEmbeddingGenerator(ttl_path, embed_dim=embed_dim, **kwargs)
    gen.fit()
    return gen


if __name__ == '__main__':
    import sys
    ttl_path = sys.argv[1] if len(sys.argv) > 1 else 'data/ontology/cfrp_ontology_populated.ttl'

    gen = generate_kg_embeddings(ttl_path)

    print("\nGraph Stats:")
    for k, v in gen.get_graph_stats().items():
        print(f"  {k}: {v}")

    print("\nCoupon Embeddings:")
    for cid in COUPON_IDS:
        emb = gen.get_coupon_embedding(cid)
        print(f"  {cid}: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")

    # Show that embeddings are now unique per coupon
    print("\nEmbedding uniqueness check (norms for same-layup coupons):")
    for group, ids in [('L1', ['L1S11', 'L1S12', 'L1S17']),
                        ('L2', ['L2S11', 'L2S12', 'L2S17']),
                        ('L3', ['L3S17', 'L3S18', 'L3S19'])]:
        embs = [gen.get_coupon_embedding(cid) for cid in ids]
        diffs = []
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                diffs.append(np.linalg.norm(embs[i] - embs[j]))
        print(f"  {group} group: mean pairwise dist = {np.mean(diffs):.4f}")

    print("\nSpecimen ID Mapping Test:")
    for sid in ['S11', 'S12', 'S19']:
        emb = gen.get_coupon_embedding(sid)
        print(f"  {sid} -> shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
