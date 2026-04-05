"""
Quantitative XAI Benchmarking Framework

Metrics implemented:
  - Faithfulness: Do identified features actually matter to the model?
  - Stability:    Are explanations robust to small input perturbations?
  - Compactness:  How concise are the explanations?
  - Concordance:  Do RuleFit and SHAP agree on feature importance?

Usage:
    benchmark = XAIBenchmark(model, feature_names, device, target_scaler)
    results   = benchmark.run_full_benchmark(X_test, query_engine, shap_explainer)
"""

import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


class XAIBenchmark:
    """
    Quantitative benchmarking framework for comparing XAI methods
    (RuleFit-based RuleQueryEngine vs SHAP) on sequence models.
    """

    def __init__(self, model, feature_names, device=None, target_scaler=None):
        """
        Args:
            model:          Trained PyTorch model (e.g. TransformerRULPredictor).
            feature_names:  List[str] of input feature names.
            device:         torch.device or str.
            target_scaler:  sklearn scaler used to inverse-transform predictions
                            to actual RUL cycles (optional but recommended).
        """
        self.model = model
        self.feature_names = list(feature_names)
        self.n_features = len(feature_names)
        self.target_scaler = target_scaler
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _predict(self, X):
        """Return model predictions in actual RUL cycles (1-D array)."""
        self.model.eval()
        X_arr = np.asarray(X)
        if X_arr.ndim == 2:
            X_arr = X_arr[np.newaxis, :, :]
        with torch.no_grad():
            t = torch.FloatTensor(X_arr).to(self.device)
            preds = self.model(t).detach().cpu().numpy().flatten()
        if self.target_scaler is not None:
            preds = self.target_scaler.inverse_transform(
                preds.reshape(-1, 1)
            ).flatten()
        return preds

    @staticmethod
    def _jaccard(set_a, set_b):
        if not set_a and not set_b:
            return 1.0
        union = set_a | set_b
        return len(set_a & set_b) / max(len(union), 1)

    def _ensure_3d(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[np.newaxis, :, :]
        return X

    def _sample_indices(self, n_total, max_samples):
        if max_samples is not None and max_samples < n_total:
            return np.linspace(0, n_total - 1, max_samples, dtype=int)
        return np.arange(n_total)

    # ------------------------------------------------------------------ #
    #  Feature ranking helpers                                             #
    # ------------------------------------------------------------------ #

    def _rulefit_top_features(self, query_engine, sample, top_k=5):
        """
        Get top-k features from RuleFit why-query, scored by cumulative
        rule importance across all matched rules.
        """
        why = query_engine.why_query(sample, top_k=50, min_importance=0.0)
        scores = {}
        for reason in why.get("reasons", []):
            imp = reason["importance"]
            for ev in reason["evidence"]:
                fname = ev["feature"]
                scores[fname] = scores.get(fname, 0.0) + imp
        ranked = sorted(scores, key=lambda f: -scores[f])
        return ranked[:top_k]

    def _shap_top_features(self, shap_explainer, sample, top_k=5):
        pairs = shap_explainer.get_feature_importances(sample, top_k=top_k)
        return [name for name, _ in pairs]

    def _rulefit_full_ranking(self, query_engine, sample):
        """Return {feature: score} dict covering all features."""
        why = query_engine.why_query(sample, top_k=100, min_importance=0.0)
        scores = {f: 0.0 for f in self.feature_names}
        for reason in why.get("reasons", []):
            imp = reason["importance"]
            for ev in reason["evidence"]:
                scores[ev["feature"]] = scores.get(ev["feature"], 0.0) + imp
        return scores

    # ================================================================== #
    #  1) FAITHFULNESS                                                     #
    # ================================================================== #

    def faithfulness_single(
        self,
        sample,
        top_features,
        perturbation="zero",
        n_random_baselines=5,
    ):
        """
        Faithfulness for one sample.

        Idea:
          • Perturb top-k important features → big prediction change = faithful.
          • Perturb k *random* features → serves as baseline.
          • Ratio > 1 means explanation captures genuinely important features.

        Args:
            sample:             np.array, shape [1, seq_len, n_features].
            top_features:       List[str] of features to test.
            perturbation:       'zero' | 'noise' — how to perturb.
            n_random_baselines: How many random feature-sets to average over.

        Returns:
            dict with faithfulness_ratio, pred_drop_top, pred_drop_random.
        """
        sample = self._ensure_3d(sample)
        original_pred = self._predict(sample)[0]

        top_indices = [
            self.feature_names.index(f)
            for f in top_features
            if f in self.feature_names
        ]
        k = len(top_indices)
        if k == 0:
            return {
                "faithfulness_ratio": np.nan,
                "pred_drop_top": 0.0,
                "pred_drop_random": 0.0,
                "original_pred": float(original_pred),
            }

        # Perturb top features across the whole sequence
        perturbed_top = sample.copy()
        self._apply_perturbation(perturbed_top, top_indices, perturbation)
        pred_drop_top = abs(original_pred - self._predict(perturbed_top)[0])

        # Random baselines
        all_idx = list(range(self.n_features))
        remaining = [i for i in all_idx if i not in top_indices]
        random_drops = []
        for _ in range(n_random_baselines):
            pool = remaining if len(remaining) >= k else all_idx
            rand_idx = list(
                np.random.choice(pool, size=min(k, len(pool)), replace=False)
            )
            perturbed_rand = sample.copy()
            self._apply_perturbation(perturbed_rand, rand_idx, perturbation)
            random_drops.append(
                abs(original_pred - self._predict(perturbed_rand)[0])
            )
        pred_drop_random = float(np.mean(random_drops)) if random_drops else 1e-8

        ratio = pred_drop_top / max(pred_drop_random, 1e-8)
        return {
            "faithfulness_ratio": float(ratio),
            "pred_drop_top": float(pred_drop_top),
            "pred_drop_random": float(pred_drop_random),
            "original_pred": float(original_pred),
        }

    def _apply_perturbation(self, arr, feat_indices, mode):
        """In-place perturbation of ``arr[0, :, feat_indices]``."""
        for idx in feat_indices:
            col = arr[0, :, idx]
            if mode == "zero":
                arr[0, :, idx] = 0.0
            elif mode == "noise":
                std = np.std(col) + 1e-8
                arr[0, :, idx] = col + np.random.normal(0, std * 2, size=col.shape)
            else:
                arr[0, :, idx] = 0.0  # fallback

    def faithfulness_batch(
        self,
        X,
        query_engine=None,
        shap_explainer=None,
        top_k=5,
        perturbation="zero",
        max_samples=None,
        n_random_baselines=5,
    ):
        """
        Compute faithfulness over many test samples for RuleFit and/or SHAP.

        Returns:
            summary dict, detailed DataFrame.
        """
        X = self._ensure_3d(X)
        indices = self._sample_indices(X.shape[0], max_samples)

        rows = []
        for cnt, idx in enumerate(indices):
            sample = X[idx : idx + 1]
            row = {"sample_idx": int(idx)}

            if query_engine is not None:
                top_r = self._rulefit_top_features(query_engine, sample, top_k)
                if top_r:
                    f = self.faithfulness_single(
                        sample, top_r, perturbation, n_random_baselines
                    )
                    row["rulefit_faithfulness"] = f["faithfulness_ratio"]
                    row["rulefit_pred_drop"] = f["pred_drop_top"]
                    row["rulefit_pred_drop_random"] = f["pred_drop_random"]
                    row["rulefit_top_features"] = ", ".join(top_r)
                else:
                    row["rulefit_faithfulness"] = np.nan
                    row["rulefit_pred_drop"] = 0.0
                    row["rulefit_pred_drop_random"] = 0.0
                    row["rulefit_top_features"] = ""

            if shap_explainer is not None:
                top_s = self._shap_top_features(shap_explainer, sample, top_k)
                if top_s:
                    f = self.faithfulness_single(
                        sample, top_s, perturbation, n_random_baselines
                    )
                    row["shap_faithfulness"] = f["faithfulness_ratio"]
                    row["shap_pred_drop"] = f["pred_drop_top"]
                    row["shap_pred_drop_random"] = f["pred_drop_random"]
                    row["shap_top_features"] = ", ".join(top_s)
                else:
                    row["shap_faithfulness"] = np.nan
                    row["shap_pred_drop"] = 0.0
                    row["shap_pred_drop_random"] = 0.0
                    row["shap_top_features"] = ""

            rows.append(row)

        df = pd.DataFrame(rows)

        summary = {
            "n_samples": len(df),
            "perturbation": perturbation,
            "top_k": top_k,
        }
        if query_engine is not None and "rulefit_faithfulness" in df.columns:
            summary["rulefit_mean_faithfulness"] = float(
                df["rulefit_faithfulness"].mean()
            )
            summary["rulefit_median_faithfulness"] = float(
                df["rulefit_faithfulness"].median()
            )
            summary["rulefit_mean_pred_drop"] = float(
                df["rulefit_pred_drop"].mean()
            )
        if shap_explainer is not None and "shap_faithfulness" in df.columns:
            summary["shap_mean_faithfulness"] = float(
                df["shap_faithfulness"].mean()
            )
            summary["shap_median_faithfulness"] = float(
                df["shap_faithfulness"].median()
            )
            summary["shap_mean_pred_drop"] = float(df["shap_pred_drop"].mean())

        return summary, df

    # ================================================================== #
    #  2) STABILITY                                                        #
    # ================================================================== #

    def stability_single_rulefit(
        self,
        query_engine,
        sample,
        noise_std=0.05,
        n_repeats=10,
        top_k=5,
    ):
        """
        Rule-explanation stability for one sample.

        Add Gaussian noise → re-run rule matching → measure Jaccard overlap
        of matched rule-IDs, top feature-sets, and stage consistency.
        """
        sample = self._ensure_3d(sample)

        base_result = query_engine.point_query(
            sample, top_k=top_k * 3, min_importance=0.0, return_all_matches=True
        )
        base_rules = (
            set(base_result["top_rules"]["rule_id"].tolist())
            if not base_result["top_rules"].empty
            else set()
        )
        base_feats = set(
            self._rulefit_top_features(query_engine, sample, top_k)
        )
        base_stage = base_result["predicted_stage"]

        rule_jac, feat_jac, stage_agree = [], [], []

        for _ in range(n_repeats):
            noise = np.random.normal(0, noise_std, size=sample.shape)
            noisy = sample + noise

            nr = query_engine.point_query(
                noisy, top_k=top_k * 3, min_importance=0.0, return_all_matches=True
            )
            noisy_rules = (
                set(nr["top_rules"]["rule_id"].tolist())
                if not nr["top_rules"].empty
                else set()
            )
            noisy_feats = set(
                self._rulefit_top_features(query_engine, noisy, top_k)
            )

            rule_jac.append(self._jaccard(base_rules, noisy_rules))
            feat_jac.append(self._jaccard(base_feats, noisy_feats))
            stage_agree.append(1.0 if nr["predicted_stage"] == base_stage else 0.0)

        return {
            "rule_stability": float(np.mean(rule_jac)),
            "feature_stability": float(np.mean(feat_jac)),
            "stage_stability": float(np.mean(stage_agree)),
            "rule_stability_std": float(np.std(rule_jac)),
            "feature_stability_std": float(np.std(feat_jac)),
            "n_base_rules": len(base_rules),
        }

    def stability_single_shap(
        self,
        shap_explainer,
        sample,
        noise_std=0.05,
        n_repeats=10,
        top_k=5,
    ):
        """SHAP explanation stability for one sample."""
        sample = self._ensure_3d(sample)
        base = set(self._shap_top_features(shap_explainer, sample, top_k))
        jacs = []
        for _ in range(n_repeats):
            noisy = sample + np.random.normal(0, noise_std, size=sample.shape)
            noisy_feats = set(
                self._shap_top_features(shap_explainer, noisy, top_k)
            )
            jacs.append(self._jaccard(base, noisy_feats))
        return {
            "feature_stability": float(np.mean(jacs)),
            "feature_stability_std": float(np.std(jacs)),
        }

    def stability_batch(
        self,
        X,
        query_engine=None,
        shap_explainer=None,
        noise_std=0.05,
        n_repeats=10,
        top_k=5,
        max_samples=None,
    ):
        """Compute stability across test samples for both methods."""
        X = self._ensure_3d(X)
        indices = self._sample_indices(X.shape[0], max_samples)

        rows = []
        for idx in indices:
            sample = X[idx : idx + 1]
            row = {"sample_idx": int(idx)}

            if query_engine is not None:
                s = self.stability_single_rulefit(
                    query_engine, sample, noise_std, n_repeats, top_k
                )
                row["rulefit_rule_stability"] = s["rule_stability"]
                row["rulefit_feature_stability"] = s["feature_stability"]
                row["rulefit_stage_stability"] = s["stage_stability"]

            if shap_explainer is not None:
                s = self.stability_single_shap(
                    shap_explainer, sample, noise_std, n_repeats, top_k
                )
                row["shap_feature_stability"] = s["feature_stability"]

            rows.append(row)

        df = pd.DataFrame(rows)
        summary = {
            "noise_std": noise_std,
            "n_repeats": n_repeats,
            "top_k": top_k,
            "n_samples": len(df),
        }
        if query_engine is not None and "rulefit_rule_stability" in df.columns:
            summary["rulefit_mean_rule_stability"] = float(
                df["rulefit_rule_stability"].mean()
            )
            summary["rulefit_mean_feature_stability"] = float(
                df["rulefit_feature_stability"].mean()
            )
            summary["rulefit_mean_stage_stability"] = float(
                df["rulefit_stage_stability"].mean()
            )
        if shap_explainer is not None and "shap_feature_stability" in df.columns:
            summary["shap_mean_feature_stability"] = float(
                df["shap_feature_stability"].mean()
            )
        return summary, df

    # ================================================================== #
    #  3) COMPACTNESS                                                      #
    # ================================================================== #

    def compactness(self, X, query_engine, top_k=10, min_importance=0.0):
        """
        Measure how concise rule explanations are.

        Returns dict with avg rules/sample, avg conditions/rule,
        unique features used, feature coverage ratio.
        """
        X = self._ensure_3d(X)
        n_matched, n_conds = [], []

        for idx in range(X.shape[0]):
            r = query_engine.point_query(
                X[idx : idx + 1], top_k=top_k, min_importance=min_importance
            )
            n_matched.append(r["n_matched_rules"])
            if (
                not r["top_rules"].empty
                and "specificity" in r["top_rules"].columns
            ):
                n_conds.extend(r["top_rules"]["specificity"].tolist())

        total_rules = len(query_engine.rules_df)
        feats_in_rules = set()
        for _, rule in query_engine.rules_df.iterrows():
            for feat, _, _ in rule["conditions"]:
                feats_in_rules.add(feat)

        return {
            "avg_matched_rules_per_sample": float(np.mean(n_matched)),
            "median_matched_rules_per_sample": float(np.median(n_matched)),
            "avg_conditions_per_rule": (
                float(np.mean(n_conds)) if n_conds else 0.0
            ),
            "total_rules": total_rules,
            "unique_features_in_rules": len(feats_in_rules),
            "feature_coverage": len(feats_in_rules) / max(self.n_features, 1),
        }

    # ================================================================== #
    #  4) CONCORDANCE  (RuleFit vs SHAP)                                   #
    # ================================================================== #

    def concordance_single(self, query_engine, shap_explainer, sample):
        """
        Spearman rank-correlation and top-k overlap between RuleFit and SHAP
        feature rankings for one sample.
        """
        # SHAP ranking
        shap_dict = shap_explainer.get_full_importance_vector(sample)
        shap_order = sorted(shap_dict, key=lambda f: -shap_dict[f])

        # RuleFit ranking
        rule_scores = self._rulefit_full_ranking(query_engine, sample)
        rule_order = sorted(rule_scores, key=lambda f: -rule_scores[f])

        # Spearman over ALL features
        shap_ranks = {f: i for i, f in enumerate(shap_order)}
        rule_ranks = {f: i for i, f in enumerate(rule_order)}
        common = sorted(set(shap_ranks) & set(rule_ranks))
        if len(common) < 3:
            return {
                "spearman": np.nan,
                "spearman_p": 1.0,
                "top3_overlap": 0.0,
                "top5_overlap": 0.0,
            }

        corr, pval = spearmanr(
            [shap_ranks[f] for f in common],
            [rule_ranks[f] for f in common],
        )

        t3 = len(set(shap_order[:3]) & set(rule_order[:3])) / 3.0
        t5 = len(set(shap_order[:5]) & set(rule_order[:5])) / 5.0

        return {
            "spearman": float(corr) if np.isfinite(corr) else 0.0,
            "spearman_p": float(pval) if np.isfinite(pval) else 1.0,
            "top3_overlap": float(t3),
            "top5_overlap": float(t5),
        }

    def concordance_batch(
        self, X, query_engine, shap_explainer, max_samples=None
    ):
        X = self._ensure_3d(X)
        indices = self._sample_indices(X.shape[0], max_samples)
        rows = []
        for idx in indices:
            c = self.concordance_single(
                query_engine, shap_explainer, X[idx : idx + 1]
            )
            c["sample_idx"] = int(idx)
            rows.append(c)
        df = pd.DataFrame(rows)
        summary = {
            "mean_spearman": float(df["spearman"].mean()),
            "mean_top3_overlap": float(df["top3_overlap"].mean()),
            "mean_top5_overlap": float(df["top5_overlap"].mean()),
            "n_samples": len(df),
        }
        return summary, df

    # ================================================================== #
    #  5) FULL BENCHMARK RUNNER                                            #
    # ================================================================== #

    def run_full_benchmark(
        self,
        X,
        query_engine,
        shap_explainer=None,
        top_k=5,
        noise_std=0.05,
        perturbation="zero",
        n_stability_repeats=10,
        n_random_baselines=5,
        max_samples=None,
    ):
        """
        Run all benchmarks and print a comparison table.

        Args:
            X:                   Test data [n, seq_len, n_features].
            query_engine:        Fitted RuleQueryEngine.
            shap_explainer:      SHAPBenchmarkExplainer (optional).
            top_k:               Number of top features to compare.
            noise_std:           Gaussian noise σ for stability.
            perturbation:        'zero' or 'noise' for faithfulness.
            n_stability_repeats: Perturbation repeats per sample.
            n_random_baselines:  Random-feature baselines for faithfulness.
            max_samples:         Cap number of samples (for speed).

        Returns:
            dict with keys: faithfulness, stability, compactness,
            concordance (if SHAP provided), comparison_table.
        """
        print("=" * 80)
        print("XAI BENCHMARK SUITE")
        print("=" * 80)

        out = {}

        # ── Faithfulness ──────────────────────────────────────────────
        print("\n📊 [1/4] Computing Faithfulness ...")
        t0 = time.time()
        faith_sum, faith_df = self.faithfulness_batch(
            X,
            query_engine=query_engine,
            shap_explainer=shap_explainer,
            top_k=top_k,
            perturbation=perturbation,
            max_samples=max_samples,
            n_random_baselines=n_random_baselines,
        )
        print(f"   ✓ done in {time.time() - t0:.1f}s")
        self._print_dict(faith_sum)
        out["faithfulness"] = {"summary": faith_sum, "details": faith_df}

        # ── Stability ─────────────────────────────────────────────────
        print("\n📊 [2/4] Computing Stability ...")
        t0 = time.time()
        stab_sum, stab_df = self.stability_batch(
            X,
            query_engine=query_engine,
            shap_explainer=shap_explainer,
            noise_std=noise_std,
            n_repeats=n_stability_repeats,
            top_k=top_k,
            max_samples=max_samples,
        )
        print(f"   ✓ done in {time.time() - t0:.1f}s")
        self._print_dict(stab_sum)
        out["stability"] = {"summary": stab_sum, "details": stab_df}

        # ── Compactness ───────────────────────────────────────────────
        print("\n📊 [3/4] Computing Compactness ...")
        compact = self.compactness(X, query_engine, top_k=top_k)
        self._print_dict(compact)
        out["compactness"] = compact

        # ── Concordance ───────────────────────────────────────────────
        if shap_explainer is not None:
            print("\n📊 [4/4] Computing Concordance (RuleFit vs SHAP) ...")
            t0 = time.time()
            conc_sum, conc_df = self.concordance_batch(
                X, query_engine, shap_explainer, max_samples=max_samples
            )
            print(f"   ✓ done in {time.time() - t0:.1f}s")
            self._print_dict(conc_sum)
            out["concordance"] = {"summary": conc_sum, "details": conc_df}
        else:
            print("\n📊 [4/4] Concordance skipped (no SHAP explainer).")

        # ── Comparison table ──────────────────────────────────────────
        table = self._build_comparison_table(out)
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)
        print(table.to_string(index=False))
        print("\n" + "=" * 80)
        print("✅ BENCHMARK COMPLETE")
        print("=" * 80)

        out["comparison_table"] = table
        return out

    # ------------------------------------------------------------------ #
    #  Pretty-printing                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _print_dict(d):
        for k, v in d.items():
            if isinstance(v, float):
                print(f"   {k}: {v:.4f}")

    def _fmt(self, val, fmt=".4f"):
        if isinstance(val, float) and np.isfinite(val):
            return f"{val:{fmt}}"
        return "N/A"

    def _build_comparison_table(self, results):
        faith = results.get("faithfulness", {}).get("summary", {})
        stab = results.get("stability", {}).get("summary", {})
        comp = results.get("compactness", {})
        conc = results.get("concordance", {}).get("summary", {})

        rows = [
            {
                "Metric": "Faithfulness (mean ratio)",
                "RuleFit": self._fmt(faith.get("rulefit_mean_faithfulness")),
                "SHAP": self._fmt(faith.get("shap_mean_faithfulness")),
            },
            {
                "Metric": "Faithfulness (median ratio)",
                "RuleFit": self._fmt(faith.get("rulefit_median_faithfulness")),
                "SHAP": self._fmt(faith.get("shap_median_faithfulness")),
            },
            {
                "Metric": "Avg Pred Drop (cycles)",
                "RuleFit": self._fmt(faith.get("rulefit_mean_pred_drop"), ".0f"),
                "SHAP": self._fmt(faith.get("shap_mean_pred_drop"), ".0f"),
            },
            {
                "Metric": "Feature Stability (Jaccard)",
                "RuleFit": self._fmt(stab.get("rulefit_mean_feature_stability")),
                "SHAP": self._fmt(stab.get("shap_mean_feature_stability")),
            },
            {
                "Metric": "Rule Stability (Jaccard)",
                "RuleFit": self._fmt(stab.get("rulefit_mean_rule_stability")),
                "SHAP": "—",
            },
            {
                "Metric": "Stage Stability",
                "RuleFit": self._fmt(stab.get("rulefit_mean_stage_stability")),
                "SHAP": "—",
            },
            {
                "Metric": "Avg Rules / Sample",
                "RuleFit": self._fmt(comp.get("avg_matched_rules_per_sample"), ".2f"),
                "SHAP": "— (continuous)",
            },
            {
                "Metric": "Avg Conditions / Rule",
                "RuleFit": self._fmt(comp.get("avg_conditions_per_rule"), ".2f"),
                "SHAP": "—",
            },
            {
                "Metric": "Feature Coverage",
                "RuleFit": self._fmt(
                    comp.get("feature_coverage", 0) * 100 if comp.get("feature_coverage") is not None else None,
                    ".1f",
                )
                + "%"
                if comp.get("feature_coverage") is not None
                else "N/A",
                "SHAP": "100%",
            },
        ]

        if conc:
            rows.append(
                {
                    "Metric": "Spearman Rank Corr",
                    "RuleFit": self._fmt(conc.get("mean_spearman")),
                    "SHAP": "(reference)",
                }
            )
            rows.append(
                {
                    "Metric": "Top-5 Feature Overlap",
                    "RuleFit": self._fmt(
                        conc.get("mean_top5_overlap", 0) * 100, ".1f"
                    )
                    + "%",
                    "SHAP": "(reference)",
                }
            )

        return pd.DataFrame(rows)
