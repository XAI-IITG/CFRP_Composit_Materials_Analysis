"""
Signal Temporal Logic (STL) Rule Extraction for CFRP Composite Analysis

Uses STL robustness signatures (min/max over time intervals) to create
features that can be formally interpreted as temporal logic predicates:

    min(x[a:b+1]) > τ   ↔  G_[a,b](x > τ)   "x is ALWAYS above τ during [a,b]"
    max(x[a:b+1]) > τ   ↔  F_[a,b](x > τ)   "x EVENTUALLY exceeds τ during [a,b]"
    min(x[a:b+1]) ≤ τ   ↔  F_[a,b](x ≤ τ)   "x EVENTUALLY drops to ≤ τ"
    max(x[a:b+1]) ≤ τ   ↔  G_[a,b](x ≤ τ)   "x is ALWAYS below τ"

This is an ADD-ON to the existing temporal_rules.py module, not a replacement.

Classes:
    STLFeatureExtractor  – min/max over configurable time intervals
    STLRuleFit           – RuleFit on STL-augmented features
    STLRuleQueryEngine   – Query engine with STL-aware input handling
    STLRuleTranslator    – Converts rules into formal STL notation + English

Usage:
    from modules import STLRuleFit, STLRuleQueryEngine, STLRuleTranslator

    stl_rulefit = STLRuleFit(
        model=transformer_model,
        feature_names=feature_names,
        augment_features=["avg_scatter_energy", "avg_delta_tof", "avg_rms"],
        seq_len=10,
    )
    stl_rulefit.fit(X_train, y_train_actual, stage_labels)

    stl_qe = STLRuleQueryEngine(stl_rulefit, stage_boundaries)
    result = stl_qe.stl_why_query(sample)
"""

import numpy as np
import pandas as pd
import torch

from .rulefit import RuleFit
from .query_engine import RuleQueryEngine


# ═══════════════════════════════════════════════════════════════════════ #
#  STL Feature Extractor                                                 #
# ═══════════════════════════════════════════════════════════════════════ #

class STLFeatureExtractor:
    """
    Computes Signal Temporal Logic robustness features from 3D sequences.

    For each selected feature and time interval, computes:
        min  → supports G (Globally / "Always") predicates
        max  → supports F (Finally / "Eventually") predicates

    Features are concatenated with the last-timestep snapshot.

    Feature naming convention:
        min_{start}_{end}__{base_feature}   (e.g. min_0_4__avg_scatter_energy)
        max_{start}_{end}__{base_feature}   (e.g. max_5_9__avg_rms)
    """

    def __init__(
        self,
        base_feature_names,
        augment_features=None,
        intervals=None,
        seq_len=10,
    ):
        """
        Args:
            base_feature_names: list[str] of original 16 feature names.
            augment_features:   Subset of features to compute STL for.
                                None = all features (not recommended).
            intervals:          Dict of {name: (start, end)} time intervals.
                                None = auto-split into early/late/full.
            seq_len:            Sequence length (default 10).
        """
        self.base_feature_names = list(base_feature_names)
        self.n_base = len(self.base_feature_names)
        self.seq_len = seq_len

        # Which features to augment
        if augment_features is None:
            self.augment_indices = list(range(self.n_base))
            self.augment_names = list(self.base_feature_names)
        else:
            self.augment_indices = [
                self.base_feature_names.index(f) for f in augment_features
            ]
            self.augment_names = list(augment_features)

        # Time intervals
        if intervals is None:
            mid = seq_len // 2
            self.intervals = {
                "early": (0, mid - 1),       # first half
                "late": (mid, seq_len - 1),   # second half
                "full": (0, seq_len - 1),     # entire window
            }
        else:
            self.intervals = dict(intervals)

        # Build augmented feature name list
        self._augmented_names = list(self.base_feature_names)
        for feat in self.augment_names:
            for _interval_name, (start, end) in sorted(self.intervals.items()):
                self._augmented_names.append(f"min_{start}_{end}__{feat}")
                self._augmented_names.append(f"max_{start}_{end}__{feat}")

    @property
    def augmented_feature_names(self):
        return list(self._augmented_names)

    @property
    def n_augmented(self):
        return len(self._augmented_names)

    @property
    def n_stl(self):
        return len(self.augment_names) * len(self.intervals) * 2  # min + max

    def transform(self, X_3d):
        """
        Convert 3D sequences to 2D with STL robustness features appended.

        Args:
            X_3d: shape (n, seq_len, n_base) or (seq_len, n_base).

        Returns:
            np.ndarray of shape (n, n_base + n_stl).
        """
        X = np.asarray(X_3d, dtype=np.float64)
        if X.ndim == 2:
            X = X[np.newaxis, :, :]
        if X.ndim != 3 or X.shape[2] != self.n_base:
            raise ValueError(
                f"Expected shape (n, seq, {self.n_base}), got {X.shape}"
            )

        base_2d = X[:, -1, :]  # last-timestep snapshot
        parts = []

        for fi in self.augment_indices:
            series = X[:, :, fi]  # (n, seq_len)
            for _name, (start, end) in sorted(self.intervals.items()):
                interval = series[:, start : end + 1]  # inclusive end
                parts.append(np.min(interval, axis=1))   # G robustness
                parts.append(np.max(interval, axis=1))   # F robustness

        if parts:
            return np.concatenate([base_2d, np.column_stack(parts)], axis=1)
        return base_2d


# ═══════════════════════════════════════════════════════════════════════ #
#  STL Rule Translator                                                   #
# ═══════════════════════════════════════════════════════════════════════ #

class STLRuleTranslator:
    """
    Translates STL-encoded feature conditions into:
      1. Formal STL notation:  G_[0,4](scatter_energy > 0.50)
      2. English:  "scatter energy was consistently above 0.50 during
                    the early phase (steps 0–4)"
    """

    # ── Human-readable feature labels ─────────────────────────────────
    FEATURE_LABELS = {
        "stiffness_degradation": "stiffness",
        "avg_scatter_energy": "scatter energy",
        "std_scatter_energy": "scatter energy variability",
        "avg_delta_tof": "ultrasonic velocity change (ΔToF)",
        "std_delta_tof": "ΔToF variability",
        "avg_delta_psd": "power spectral density change (ΔPSD)",
        "std_delta_psd": "ΔPSD variability",
        "avg_rms": "acoustic emission (RMS)",
        "avg_peak_frequency": "peak frequency",
        "normalized_cycles": "fatigue life fraction",
        "cycles": "loading cycles",
        "avg_strain_rms": "strain RMS",
        "avg_strain_amplitude": "strain amplitude",
        "avg_intensity": "optical intensity",
        "avg_entropy": "signal entropy",
    }

    # ── Parsing ───────────────────────────────────────────────────────

    @classmethod
    def is_stl_feature(cls, feature_name):
        """Check if a feature name is an STL robustness feature."""
        return (
            feature_name.startswith("min_") or feature_name.startswith("max_")
        ) and "__" in feature_name

    @classmethod
    def parse(cls, feature_name):
        """
        Parse 'min_0_4__avg_scatter_energy' →
        ('min', 0, 4, 'avg_scatter_energy')   or None if not STL.
        """
        if not cls.is_stl_feature(feature_name):
            return None
        parts = feature_name.split("__", 1)
        if len(parts) != 2:
            return None
        prefix, base_feat = parts
        tokens = prefix.split("_")
        if len(tokens) != 3:
            return None
        try:
            return (tokens[0], int(tokens[1]), int(tokens[2]), base_feat)
        except ValueError:
            return None

    # ── STL Formal Notation ───────────────────────────────────────────

    @classmethod
    def to_stl(cls, feature_name, operator, threshold):
        """
        Convert a condition into formal STL notation.

        Mapping:
            feature:  min_a_b   operator: >   →  G_[a,b](feat > τ)
            feature:  min_a_b   operator: ≤   →  F_[a,b](feat ≤ τ)
            feature:  max_a_b   operator: >   →  F_[a,b](feat > τ)
            feature:  max_a_b   operator: ≤   →  G_[a,b](feat ≤ τ)
        """
        parsed = cls.parse(feature_name)
        if parsed is None:
            return f"{feature_name} {operator} {threshold:.4f}"

        rob, start, end, base = parsed

        if rob == "min":
            stl_op = "G" if operator == ">" else "F"
        else:  # max
            stl_op = "F" if operator == ">" else "G"

        return f"{stl_op}_[{start},{end}]({base} {operator} {threshold:.4f})"

    # ── English Translation ───────────────────────────────────────────

    @classmethod
    def get_zscore_adjective(cls, val):
        """Map a standardized Z-score to a descriptive adjective."""
        if val >= 2.0: return "exceptionally high (≥ +2.0σ)"
        if val >= 1.0: return "significantly elevated (≥ +1.0σ)"
        if val >= 0.5: return "moderately elevated (≥ +0.5σ)"
        if val > -0.5: return "near baseline levels (~ 0σ)"
        if val > -1.0: return "moderately reduced (≤ -0.5σ)"
        if val > -2.0: return "significantly dropped (≤ -1.0σ)"
        return "exceptionally low (≤ -2.0σ)"

    @classmethod
    def to_english(cls, feature_name, operator, threshold):
        """Convert a condition into natural English with qualitative terms."""
        parsed = cls.parse(feature_name)
        
        # Get qualitative description of the threshold
        desc = cls.get_zscore_adjective(threshold)

        # ── Static (non-STL) feature ──
        if parsed is None:
            label = cls.FEATURE_LABELS.get(feature_name, feature_name)
            if operator == ">":
                return f"{label} is {desc}"
            return f"{label} is {desc}"

        rob, start, end, base = parsed
        label = cls.FEATURE_LABELS.get(base, base)

        # Phase label
        if end - start + 1 <= 5 and start == 0:
            phase = "early phase"
        elif start >= 5:
            phase = "recent phase"
        else:
            phase = "entire window"

        steps = f"steps {start}–{end}"

        # ── min feature + > threshold → G: Always above ──
        if rob == "min" and operator == ">":
            return (
                f"{label} was consistently maintained at a {desc} level "
                f"during the {phase} ({steps})"
            )

        # ── min feature + ≤ threshold → F: Eventually dropped ──
        if rob == "min" and operator in ("<=", "<"):
            return (
                f"{label} fell to a {desc} level at some point "
                f"during the {phase} ({steps})"
            )

        # ── max feature + > threshold → F: Eventually spiked ──
        if rob == "max" and operator == ">":
            return (
                f"{label} reached a {desc} level at some point "
                f"during the {phase} ({steps})"
            )

        # ── max feature + ≤ threshold → G: Always below ──
        if rob == "max" and operator in ("<=", "<"):
            return (
                f"{label} was restricted to a {desc} level throughout "
                f"the {phase} ({steps})"
            )

        return f"{feature_name} {operator} {threshold:.4f}"

    # ── Translate a full rule ─────────────────────────────────────────

    @classmethod
    def translate_rule(cls, reason):
        """
        Translate one reason dict (from why_query / stl_why_query)
        into both STL and English strings.

        Returns dict with keys:
            stl_rule, english_explanation, conditions (list of dicts)
        """
        stl_parts = []
        eng_parts = []
        conditions = []

        for ev in reason.get("evidence", []):
            feat = ev["feature"]
            op = ev["operator"]
            thresh = ev["threshold"]

            stl_str = cls.to_stl(feat, op, thresh)
            eng_str = cls.to_english(feat, op, thresh)

            stl_parts.append(stl_str)
            eng_parts.append(eng_str)

            conditions.append({
                **ev,
                "stl_notation": stl_str,
                "english": eng_str,
            })

        return {
            "stl_rule": " ∧ ".join(stl_parts),
            "english_explanation": "; AND ".join(eng_parts),
            "conditions": conditions,
        }


# ═══════════════════════════════════════════════════════════════════════ #
#  STL RuleFit                                                           #
# ═══════════════════════════════════════════════════════════════════════ #

class STLRuleFit(RuleFit):
    """
    RuleFit with STL robustness feature augmentation.

    Model predictions use original 3D sequences; the tree ensemble
    and extracted rules operate on base features + min/max interval
    features that carry formal STL semantics.
    """

    def __init__(
        self,
        model,
        feature_names,
        augment_features=None,
        intervals=None,
        seq_len=10,
        n_estimators=100,
        max_depth=3,
        tree_size=4,
        memory_par=0.01,
        random_state=42,
        target_scaler=None,
        stage_boundaries=None,
    ):
        """
        Additional args (vs RuleFit):
            augment_features: list of base feature names for STL
                              augmentation (or None for all).
            intervals:        Dict {name: (start, end)} or None for
                              automatic early/late/full split.
            seq_len:          Sequence length (default 10).
        """
        self.stl_extractor = STLFeatureExtractor(
            base_feature_names=feature_names,
            augment_features=augment_features,
            intervals=intervals,
            seq_len=seq_len,
        )
        self._base_feature_names = list(feature_names)

        super().__init__(
            model=model,
            feature_names=self.stl_extractor.augmented_feature_names,
            n_estimators=n_estimators,
            max_depth=max_depth,
            tree_size=tree_size,
            memory_par=memory_par,
            random_state=random_state,
            target_scaler=target_scaler,
            stage_boundaries=stage_boundaries,
        )

    @property
    def augmented_feature_names(self):
        return self.stl_extractor.augmented_feature_names

    # ── override fit ──────────────────────────────────────────────────

    def fit(self, X, y_true, stage_labels):
        print("\n" + "=" * 80)
        print("STL RULEFIT — RULE EXTRACTION")
        print("=" * 80)

        # 1 — model predictions from original 3D sequences
        print("\n1️⃣  Getting model predictions ...")
        y_pred = self._get_predictions(X)
        print(
            f"   ✓ {len(y_pred)} predictions  "
            f"[{y_pred.min():.0f}, {y_pred.max():.0f}] cycles"
        )

        # 2 — compute STL robustness features
        print("\n2️⃣  Computing STL robustness features ...")
        X_aug = self.stl_extractor.transform(X)
        print(f"   Base features:  {self.stl_extractor.n_base}")
        print(f"   STL features:   {self.stl_extractor.n_stl}")
        print(f"   Total features: {X_aug.shape[1]}")
        for name, (s, e) in sorted(self.stl_extractor.intervals.items()):
            print(f"   Interval '{name}': steps [{s}, {e}]")

        # 3 — tree ensemble on augmented features
        print("\n3️⃣  Training tree ensemble ...")
        self.rule_ensemble = self._train_tree_ensemble(X_aug, y_pred)
        print(f"   ✓ {self.n_estimators} trees")

        # 4 — extract candidate rules
        print("\n4️⃣  Extracting rules ...")
        self.all_rules = self._extract_rules_from_ensemble(X_aug)
        print(f"   ✓ {len(self.all_rules)} candidate rules")

        # 5 — rule feature matrix
        print("\n5️⃣  Building rule feature matrix ...")
        rule_features = self._create_rule_features(X_aug, self.all_rules)
        print(f"   ✓ shape {rule_features.shape}")

        # 6 — Lasso selection
        print("\n6️⃣  Selecting rules (Lasso) ...")
        self.selected_rule_indices, self.selected_rule_importances = (
            self._select_rules_lasso(rule_features, y_pred)
        )
        print(f"   ✓ {len(self.selected_rule_indices)} important rules")

        # 7 — categorise by stage
        print("\n7️⃣  Categorising rules ...")
        self.rules = self._categorize_rules(
            self.selected_rule_indices,
            self.selected_rule_importances,
            X_aug,
            y_pred,
            stage_labels,
        )

        # Count rules containing STL predicates
        n_stl = sum(
            1
            for r in self.rules
            if any(
                STLRuleTranslator.is_stl_feature(f)
                for f, _, _ in r["conditions"]
            )
        )

        print(f"   ✓ {len(self.rules)} final rules")
        print(f"   ✓ {n_stl} rules include STL temporal predicates")

        print("\n" + "=" * 80)
        print(f"✅  STL RULEFIT COMPLETE: {len(self.rules)} rules")
        print(f"    ({n_stl} with formal temporal logic conditions)")
        print("=" * 80)
        return self

    def to_augmented_2d(self, X_3d):
        """Public convenience: 3D → STL-augmented 2D."""
        return self.stl_extractor.transform(X_3d)


# ═══════════════════════════════════════════════════════════════════════ #
#  STL Rule Query Engine                                                 #
# ═══════════════════════════════════════════════════════════════════════ #

class STLRuleQueryEngine(RuleQueryEngine):
    """
    RuleQueryEngine subclass that:
      • Transparently computes STL features from 3D input sequences
      • Provides stl_why_query() with dual output: conditional + English

    All parent query methods (why, why_not, what_if, counterfactual, etc.)
    work as before; 3D inputs are automatically augmented.
    """

    def __init__(self, stl_rulefit, stage_boundaries=None):
        if not isinstance(stl_rulefit, STLRuleFit):
            raise TypeError("Expected an STLRuleFit instance.")

        self.stl_extractor = stl_rulefit.stl_extractor
        self._base_names = list(stl_rulefit._base_feature_names)

        super().__init__(
            rulefit=stl_rulefit,
            feature_names=stl_rulefit.augmented_feature_names,
            stage_boundaries=stage_boundaries,
        )
        print(
            f"✓ STLRuleQueryEngine ready  "
            f"({len(self.feature_names)} features: "
            f"{self.stl_extractor.n_base} base + "
            f"{self.stl_extractor.n_stl} STL)"
        )

    # ── override: 3D → STL-augmented 2D ──────────────────────────────

    def _to_last_timestep(self, X, single_sample=False):
        X = np.asarray(X)

        if X.ndim == 1:
            if X.shape[0] == len(self.feature_names):
                return X.reshape(1, -1)
            raise ValueError(
                f"1D input length {X.shape[0]} ≠ "
                f"{len(self.feature_names)} augmented features."
            )

        if X.ndim == 2:
            if X.shape[1] == len(self.feature_names):
                if single_sample and X.shape[0] > 1:
                    return X[-1:, :]
                return X
            if X.shape[1] == self.stl_extractor.n_base:
                return self.stl_extractor.transform(X[np.newaxis, :, :])
            raise ValueError(
                f"2D input has {X.shape[1]} cols; expected "
                f"{len(self.feature_names)} (augmented) or "
                f"{self.stl_extractor.n_base} (base)."
            )

        if X.ndim == 3:
            if X.shape[2] != self.stl_extractor.n_base:
                raise ValueError(
                    f"3D input has {X.shape[2]} features, "
                    f"expected {self.stl_extractor.n_base}."
                )
            return self.stl_extractor.transform(X)

        raise ValueError(f"Unsupported ndim={X.ndim}.")

    # ── override: black-box fallback uses original 3D ─────────────────

    def _black_box_prediction(self, sample):
        sample = np.asarray(sample)
        if sample.ndim == 1:
            return None
        n_base = self.stl_extractor.n_base
        if sample.ndim == 2:
            if sample.shape[1] != n_base:
                return None
            sample = sample[np.newaxis, :, :]
        if sample.ndim != 3 or sample.shape[2] != n_base:
            return None
        pred = self.rulefit._get_predictions(sample)
        return float(pred[0])

    # ── override: feature changes restricted to base ──────────────────

    def _apply_feature_changes(self, sample, feature_changes):
        for name in feature_changes:
            if name not in self._base_names:
                raise ValueError(
                    f"Cannot directly set STL feature '{name}'. "
                    f"Modify the underlying base feature instead."
                )

        sample_arr = np.asarray(sample).copy()

        if sample_arr.ndim == 3:
            for name, value in feature_changes.items():
                idx = self._base_names.index(name)
                sample_arr[0, -1, idx] = float(value)
            return sample_arr

        if sample_arr.ndim == 2:
            for name, value in feature_changes.items():
                idx = self._base_names.index(name)
                sample_arr[-1, idx] = float(value)
            return sample_arr

        if sample_arr.ndim == 1:
            for name, value in feature_changes.items():
                idx = self._base_names.index(name)
                sample_arr[idx] = float(value)
            return sample_arr

        raise ValueError(f"Unsupported ndim={sample_arr.ndim}.")

    # ── STL-enhanced queries ──────────────────────────────────────────

    def stl_why_query(self, sample, top_k=5, min_importance=0.0):
        """
        Enhanced why-query that augments each matched rule with:
          • stl_notation:  formal STL string (e.g. G_[0,4](feat > τ))
          • english:       natural English explanation

        Returns the same dict as why_query, with extra fields per reason.
        """
        why = self.why_query(sample, top_k=top_k, min_importance=min_importance)

        for reason in why.get("reasons", []):
            translation = STLRuleTranslator.translate_rule(reason)
            reason["stl_rule"] = translation["stl_rule"]
            reason["english_explanation"] = translation["english_explanation"]

            # Also augment each evidence item
            for ev, cond in zip(
                reason["evidence"], translation["conditions"]
            ):
                ev["stl_notation"] = cond["stl_notation"]
                ev["english"] = cond["english"]

        return why

    def stl_why_not_query(self, sample, target_stage, top_k=3):
        """
        Enhanced why-not query with STL translations for failed conditions.
        """
        result = self.why_not_query(sample, target_stage, top_k=top_k)

        for candidate in result.get("candidate_rules", []):
            for cond in candidate.get("failed_conditions", []):
                feat = cond["feature"]
                op = cond["required_operator"]
                thresh = cond["required_threshold"]
                cond["stl_notation"] = STLRuleTranslator.to_stl(feat, op, thresh)
                cond["english"] = STLRuleTranslator.to_english(feat, op, thresh)

        return result
