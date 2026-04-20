"""
Temporal-Aware Rule Extraction for CFRP Composite Analysis

Extends RuleFit and RuleQueryEngine to incorporate temporal features
computed from input sequences (delta, slope, volatility, trend strength).
This allows rules like:
    "IF slope_stiffness_degradation < -0.05 AND delta_avg_scatter_energy > 0.3 → Critical"

Usage:
    from modules import TemporalRuleFit, TemporalRuleQueryEngine

    trulefit = TemporalRuleFit(
        model=transformer_model,
        feature_names=feature_names,
        temporal_stats=["delta", "window_delta", "slope", "volatility"],
        augment_features=None,    # None = all features; or pass a subset
        ...                       # same params as RuleFit
    )
    trulefit.fit(X, y_actual, stage_labels)

    tquery_engine = TemporalRuleQueryEngine(trulefit, stage_boundaries)
    tquery_engine.point_query(sample_3d)   # sample_3d: [1, seq, n_features]
"""

import numpy as np
import pandas as pd
import torch

from .rulefit import RuleFit
from .query_engine import RuleQueryEngine


# ═══════════════════════════════════════════════════════════════════════ #
#  Temporal Feature Extractor                                            #
# ═══════════════════════════════════════════════════════════════════════ #

class TemporalFeatureExtractor:
    """
    Computes temporal statistics from 3D time-series data and concatenates
    them with the last-timestep snapshot to produce an augmented 2D matrix.

    Supported statistics (per feature):
        delta          – last-step change: x[t] − x[t−1]
        window_delta   – total change over the window: x[-1] − x[0]
        slope          – OLS slope over the window
        volatility     – std of consecutive differences
        trend_strength – R² of the linear fit (0 = noisy, 1 = smooth trend)
    """

    AVAILABLE_STATS = [
        "delta",
        "window_delta",
        "slope",
        "volatility",
        "trend_strength",
    ]

    def __init__(self, base_feature_names, stats=None, augment_features=None):
        """
        Args:
            base_feature_names: list[str] of original feature names.
            stats:  Which temporal statistics to compute. Default is
                    ["delta", "window_delta", "slope", "volatility"].
            augment_features: Subset of base features to augment.
                              ``None`` means augment every feature.
        """
        self.base_feature_names = list(base_feature_names)
        self.n_base = len(self.base_feature_names)

        self.stats = list(stats or ["delta", "window_delta", "slope", "volatility"])
        bad = [s for s in self.stats if s not in self.AVAILABLE_STATS]
        if bad:
            raise ValueError(f"Unknown stat(s): {bad}. Choose from {self.AVAILABLE_STATS}")

        if augment_features is None:
            self.augment_indices = list(range(self.n_base))
            self.augment_names = list(self.base_feature_names)
        else:
            self.augment_indices = [self.base_feature_names.index(f) for f in augment_features]
            self.augment_names = list(augment_features)

        # Build augmented name list: [base_features..., temporal_features...]
        self._augmented_names = list(self.base_feature_names)
        for feat in self.augment_names:
            for stat in self.stats:
                self._augmented_names.append(f"{stat}__{feat}")

    # ── properties ────────────────────────────────────────────────────
    @property
    def augmented_feature_names(self):
        return list(self._augmented_names)

    @property
    def n_augmented(self):
        return len(self._augmented_names)

    @property
    def n_temporal(self):
        return len(self.augment_names) * len(self.stats)

    # ── core transform ────────────────────────────────────────────────
    def transform(self, X_3d):
        """
        Convert 3D sequences to 2D augmented feature matrix.

        Args:
            X_3d: array of shape (n, seq_len, n_base)
                  or (seq_len, n_base) for a single sample.

        Returns:
            np.ndarray of shape (n, n_base + n_temporal).
        """
        X = np.asarray(X_3d, dtype=np.float64)
        if X.ndim == 2:
            X = X[np.newaxis, :, :]
        if X.ndim != 3 or X.shape[2] != self.n_base:
            raise ValueError(
                f"Expected shape (n, seq, {self.n_base}), got {X.shape}"
            )

        n, seq_len, _ = X.shape
        base_2d = X[:, -1, :]  # last-timestep snapshot
        parts = []

        for fi in self.augment_indices:
            series = X[:, :, fi]  # (n, seq_len)
            for stat in self.stats:
                parts.append(self._compute_stat(series, stat, seq_len, n))

        if parts:
            return np.concatenate([base_2d, np.column_stack(parts)], axis=1)
        return base_2d

    # ── individual stat functions ─────────────────────────────────────
    @staticmethod
    def _compute_stat(series, stat, seq_len, n):
        """Return (n,) array of the requested statistic."""
        if stat == "delta":
            if seq_len >= 2:
                return series[:, -1] - series[:, -2]
            return np.zeros(n)

        if stat == "window_delta":
            return series[:, -1] - series[:, 0]

        if stat == "slope":
            t = np.arange(seq_len, dtype=np.float64)
            t_mean = t.mean()
            t_var = np.sum((t - t_mean) ** 2)
            if t_var < 1e-12:
                return np.zeros(n)
            s_mean = series.mean(axis=1, keepdims=True)
            return np.sum(
                (t[np.newaxis, :] - t_mean) * (series - s_mean), axis=1
            ) / t_var

        if stat == "volatility":
            if seq_len >= 2:
                return np.std(np.diff(series, axis=1), axis=1)
            return np.zeros(n)

        if stat == "trend_strength":
            if seq_len < 3:
                return np.zeros(n)
            t = np.arange(seq_len, dtype=np.float64)
            t_mean = t.mean()
            t_var = np.sum((t - t_mean) ** 2)
            if t_var < 1e-12:
                return np.zeros(n)
            s_mean = series.mean(axis=1, keepdims=True)
            slope = np.sum(
                (t[np.newaxis, :] - t_mean) * (series - s_mean), axis=1
            ) / t_var
            intercept = series.mean(axis=1) - slope * t_mean
            pred = slope[:, np.newaxis] * t[np.newaxis, :] + intercept[:, np.newaxis]
            ss_res = np.sum((series - pred) ** 2, axis=1)
            ss_tot = np.sum((series - s_mean) ** 2, axis=1)
            return np.clip(
                np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0), 0, 1
            )

        raise ValueError(f"Unknown stat: {stat}")


# ═══════════════════════════════════════════════════════════════════════ #
#  Temporal RuleFit                                                      #
# ═══════════════════════════════════════════════════════════════════════ #

class TemporalRuleFit(RuleFit):
    """
    RuleFit with automatic temporal feature augmentation.

    The underlying model prediction still uses the original 3D sequence,
    but the tree ensemble (and thus the extracted rules) operate on
    *augmented* features that include temporal statistics.
    """

    def __init__(
        self,
        model,
        feature_names,
        temporal_stats=None,
        augment_features=None,
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
            temporal_stats:   list of stats, e.g. ["delta", "slope", "volatility"].
            augment_features: list of base feature names to augment, or None for all.
        """
        self.temporal_extractor = TemporalFeatureExtractor(
            base_feature_names=feature_names,
            stats=temporal_stats,
            augment_features=augment_features,
        )
        self._base_feature_names = list(feature_names)

        # Parent init with AUGMENTED feature names
        super().__init__(
            model=model,
            feature_names=self.temporal_extractor.augmented_feature_names,
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
        return self.temporal_extractor.augmented_feature_names

    # ── override fit ──────────────────────────────────────────────────
    def fit(self, X, y_true, stage_labels):
        print("\n" + "=" * 80)
        print("TEMPORAL RULEFIT — RULE EXTRACTION")
        print("=" * 80)

        # 1 — model predictions from original 3D sequences
        print("\n1️⃣  Getting model predictions ...")
        y_pred = self._get_predictions(X)
        print(f"   ✓ {len(y_pred)} predictions  [{y_pred.min():.0f}, {y_pred.max():.0f}] cycles")

        # 2 — compute augmented 2D features
        print("\n2️⃣  Computing temporal features ...")
        X_aug = self.temporal_extractor.transform(X)
        print(f"   Base features:     {self.temporal_extractor.n_base}")
        print(f"   Temporal features: {self.temporal_extractor.n_temporal}")
        print(f"   Total features:    {X_aug.shape[1]}")
        print(f"   Stats:             {self.temporal_extractor.stats}")

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

        # count rules that reference temporal features
        n_temporal_rules = sum(
            1
            for r in self.rules
            if any(f not in self._base_feature_names for f, _, _ in r["conditions"])
        )

        print(f"   ✓ {len(self.rules)} final rules")
        print(f"   ✓ {n_temporal_rules} rules include temporal conditions")

        print("\n" + "=" * 80)
        print(f"✅  TEMPORAL RULEFIT COMPLETE: {len(self.rules)} rules")
        print(f"    ({n_temporal_rules} with temporal conditions)")
        print("=" * 80)
        return self

    # ── helpers for the query engine ──────────────────────────────────
    def to_augmented_2d(self, X_3d):
        """Public convenience: 3D → augmented 2D."""
        return self.temporal_extractor.transform(X_3d)


# ═══════════════════════════════════════════════════════════════════════ #
#  Temporal Rule Query Engine                                            #
# ═══════════════════════════════════════════════════════════════════════ #

class TemporalRuleQueryEngine(RuleQueryEngine):
    """
    RuleQueryEngine subclass that transparently computes temporal
    features when a 3D sequence is passed.

    All query methods (point_query, why_query, why_not_query, what_if_query,
    counterfactual_query, etc.) work exactly as in the parent class, but
    3D inputs are automatically augmented before rule matching.
    """

    def __init__(self, temporal_rulefit, stage_boundaries=None):
        if not isinstance(temporal_rulefit, TemporalRuleFit):
            raise TypeError("Expected a TemporalRuleFit instance.")

        self.temporal_extractor = temporal_rulefit.temporal_extractor
        self._base_names = list(temporal_rulefit._base_feature_names)

        super().__init__(
            rulefit=temporal_rulefit,
            feature_names=temporal_rulefit.augmented_feature_names,
            stage_boundaries=stage_boundaries,
        )
        print("✓ TemporalRuleQueryEngine ready  "
              f"({len(self.feature_names)} features: "
              f"{self.temporal_extractor.n_base} base + "
              f"{self.temporal_extractor.n_temporal} temporal)")

    # ── override: convert 3D → augmented 2D ──────────────────────────
    def _to_last_timestep(self, X, single_sample=False):
        X = np.asarray(X)

        # 1D — assume already an augmented row
        if X.ndim == 1:
            if X.shape[0] == len(self.feature_names):
                return X.reshape(1, -1)
            raise ValueError(
                f"1D input length {X.shape[0]} ≠ {len(self.feature_names)} augmented features."
            )

        # 2D — either (n, n_augmented) or (seq_len, n_base)
        if X.ndim == 2:
            if X.shape[1] == len(self.feature_names):
                # already augmented 2D
                if single_sample and X.shape[0] > 1:
                    return X[-1:, :]
                return X
            if X.shape[1] == self.temporal_extractor.n_base:
                # single sequence: (seq_len, n_base) → treat as 1 sample
                return self.temporal_extractor.transform(X[np.newaxis, :, :])
            raise ValueError(
                f"2D input has {X.shape[1]} cols; expected "
                f"{len(self.feature_names)} (augmented) or "
                f"{self.temporal_extractor.n_base} (base)."
            )

        # 3D — (n, seq_len, n_base) → augment via extractor
        if X.ndim == 3:
            if X.shape[2] != self.temporal_extractor.n_base:
                raise ValueError(
                    f"3D input has {X.shape[2]} features, "
                    f"expected {self.temporal_extractor.n_base}."
                )
            return self.temporal_extractor.transform(X)

        raise ValueError(f"Unsupported input ndim={X.ndim}.")

    # ── override: black-box fallback uses original 3D ─────────────────
    def _black_box_prediction(self, sample):
        sample = np.asarray(sample)
        if sample.ndim == 1:
            return None
        n_base = self.temporal_extractor.n_base
        if sample.ndim == 2:
            if sample.shape[1] != n_base:
                return None
            sample = sample[np.newaxis, :, :]
        if sample.ndim != 3 or sample.shape[2] != n_base:
            return None
        pred = self.rulefit._get_predictions(sample)
        return float(pred[0])

    # ── override: feature changes only on base features ───────────────
    def _apply_feature_changes(self, sample, feature_changes):
        """
        Modify base features in the original 3D sample.
        Temporal features that reference those base features are
        automatically recomputed when _to_last_timestep is called.
        """
        for name in feature_changes:
            if name not in self._base_names:
                raise ValueError(
                    f"Cannot directly set temporal feature '{name}'. "
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
