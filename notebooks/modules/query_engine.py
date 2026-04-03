import numpy as np
import pandas as pd


class RuleQueryEngine:
    """
    Query layer on top of a fitted RuleFit model.

    Supports:
    - point_query: what do the extracted rules say for this sample?
    - why_query: why was that answer produced?
    - evaluate: coverage and rule-based approximation quality
    """

    def __init__(self, rulefit, feature_names, stage_boundaries=None):
        self.rulefit = rulefit
        self.feature_names = list(feature_names)
        self.stage_boundaries = stage_boundaries or {
            "Early": 130000,
            "Mid": 80000,
        }

        self.rules_df = self.rulefit.get_rules_dataframe().copy()
        if self.rules_df.empty:
            raise ValueError("RuleFit has no extracted rules. Call fit() first.")

        if "conditions" not in self.rules_df.columns:
            raise ValueError("Rules DataFrame must contain a 'conditions' column.")

        if "specificity" not in self.rules_df.columns:
            self.rules_df["specificity"] = self.rules_df["conditions"].apply(len)

    def _stage_from_rul(self, rul_value):
        if rul_value is None or np.isnan(rul_value):
            return "Unknown"
        if rul_value >= self.stage_boundaries["Early"]:
            return "Early"
        if rul_value >= self.stage_boundaries["Mid"]:
            return "Mid"
        return "Late"

    def _to_last_timestep(self, X, single_sample=False):
        X = np.asarray(X)

        if X.ndim == 1:
            if X.shape[0] != len(self.feature_names):
                raise ValueError("1D input must have one value per feature.")
            return X.reshape(1, -1)

        if X.ndim == 2:
            if X.shape[1] != len(self.feature_names):
                raise ValueError("2D input must have shape (n_samples, n_features) or (seq_len, n_features).")
            if single_sample and X.shape[0] > 1:
                return X[-1:, :]
            return X

        if X.ndim == 3:
            if X.shape[2] != len(self.feature_names):
                raise ValueError("3D input must have shape (n_samples, seq_len, n_features).")
            return X[:, -1, :]

        raise ValueError("Unsupported input shape.")

    def _rule_matches_row(self, row, conditions):
        for feature_name, operator, threshold in conditions:
            feature_idx = self.feature_names.index(feature_name)
            value = row[feature_idx]

            if operator == "<=" and not (value <= threshold):
                return False
            if operator == ">" and not (value > threshold):
                return False

        return True

    def _condition_evidence(self, row, conditions):
        evidence = []

        for feature_name, operator, threshold in conditions:
            feature_idx = self.feature_names.index(feature_name)
            actual_value = float(row[feature_idx])

            if operator == "<=":
                margin = threshold - actual_value
                satisfied = actual_value <= threshold
            else:
                margin = actual_value - threshold
                satisfied = actual_value > threshold

            evidence.append(
                {
                    "feature": feature_name,
                    "operator": operator,
                    "threshold": float(threshold),
                    "actual_value": actual_value,
                    "margin": float(margin),
                    "satisfied": bool(satisfied),
                }
            )

        return evidence

    def _rank_rules(self, matched_df):
        if matched_df.empty:
            return matched_df

        ranked = matched_df.copy()
        ranked["ranking_score"] = (
            ranked["importance"] * (1.0 + 0.15 * ranked["specificity"])
        )
        ranked = ranked.sort_values(
            by=["ranking_score", "importance", "specificity", "coverage"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        return ranked

    def _black_box_prediction(self, sample):
        sample = np.asarray(sample)

        if sample.ndim == 1:
            return None

        if sample.ndim == 2:
            if sample.shape[1] != len(self.feature_names):
                return None
            sample = sample[np.newaxis, :, :]

        if sample.ndim != 3:
            return None

        pred = self.rulefit._get_predictions(sample)
        return float(pred[0])

    def match_rules(
        self,
        sample,
        stage=None,
        prediction=None,
        min_importance=0.0,
        top_k=None,
    ):
        sample_2d = self._to_last_timestep(sample, single_sample=True)
        if sample_2d.shape[0] != 1:
            raise ValueError("match_rules expects a single sample.")

        row = sample_2d[0]

        matched_rows = []
        for _, rule in self.rules_df.iterrows():
            if stage is not None and rule["stage"] != stage:
                continue
            if prediction is not None and rule["prediction"] != prediction:
                continue
            if float(rule["importance"]) < min_importance:
                continue
            if self._rule_matches_row(row, rule["conditions"]):
                matched_rows.append(rule.to_dict())

        matched_df = pd.DataFrame(matched_rows)
        if matched_df.empty:
            return matched_df

        matched_df = self._rank_rules(matched_df)

        if top_k is not None:
            matched_df = matched_df.head(top_k).reset_index(drop=True)

        return matched_df

    def point_query(
        self,
        sample,
        top_k=5,
        min_importance=0.0,
        return_all_matches=False,
    ):
        sample_2d = self._to_last_timestep(sample, single_sample=True)
        row = sample_2d[0]

        matched_df = self.match_rules(
            sample=sample,
            min_importance=min_importance,
            top_k=None,
        )

        feature_snapshot = {
            feature_name: float(row[idx])
            for idx, feature_name in enumerate(self.feature_names)
        }

        if matched_df.empty:
            black_box_rul = self._black_box_prediction(sample)
            return {
                "predicted_rul": black_box_rul,
                "predicted_stage": self._stage_from_rul(black_box_rul),
                "prediction_label": "No matched rules",
                "confidence": 0.0,
                "n_matched_rules": 0,
                "top_rules": pd.DataFrame(),
                "all_matched_rules": pd.DataFrame(),
                "feature_snapshot": feature_snapshot,
            }

        weights = matched_df["importance"].values
        avg_ruls = matched_df["avg_rul"].values

        if np.sum(weights) <= 0:
            predicted_rul = float(np.mean(avg_ruls))
        else:
            predicted_rul = float(np.average(avg_ruls, weights=weights))

        predicted_stage = self._stage_from_rul(predicted_rul)
        prediction_label = matched_df["prediction"].value_counts().idxmax()

        importance_mass = float(
            matched_df["importance"].sum() / max(self.rules_df["importance"].sum(), 1e-12)
        )
        agreement = float(
            matched_df["prediction"].value_counts(normalize=True).iloc[0]
        )
        confidence = float(min(1.0, 0.7 * importance_mass + 0.3 * agreement))

        top_rules = matched_df.head(top_k).copy().reset_index(drop=True)

        return {
            "predicted_rul": predicted_rul,
            "predicted_stage": predicted_stage,
            "prediction_label": prediction_label,
            "confidence": confidence,
            "n_matched_rules": int(len(matched_df)),
            "top_rules": top_rules,
            "all_matched_rules": matched_df if return_all_matches else pd.DataFrame(),
            "feature_snapshot": feature_snapshot,
        }

    def why_query(self, sample, top_k=3, min_importance=0.0):
        point_result = self.point_query(
            sample=sample,
            top_k=top_k,
            min_importance=min_importance,
            return_all_matches=True,
        )

        if point_result["n_matched_rules"] == 0:
            return {
                "summary": "No extracted RuleFit rule matched this sample. Use the black-box prediction directly or loosen the rule selection threshold.",
                "predicted_rul": point_result["predicted_rul"],
                "predicted_stage": point_result["predicted_stage"],
                "confidence": point_result["confidence"],
                "reasons": [],
            }

        sample_2d = self._to_last_timestep(sample, single_sample=True)
        row = sample_2d[0]

        reasons = []
        for _, rule in point_result["top_rules"].iterrows():
            evidence = self._condition_evidence(row, rule["conditions"])
            reasons.append(
                {
                    "rule_id": rule["rule_id"],
                    "stage": rule["stage"],
                    "prediction": rule["prediction"],
                    "importance": float(rule["importance"]),
                    "avg_rul": float(rule["avg_rul"]),
                    "coverage": float(rule["coverage"]),
                    "specificity": int(rule["specificity"]),
                    "condition_str": rule["condition_str"],
                    "evidence": evidence,
                }
            )

        strongest = reasons[0]
        summary = (
            f"Predicted stage: {point_result['predicted_stage']} with estimated RUL "
            f"{point_result['predicted_rul']:.0f} cycles. "
            f"{point_result['n_matched_rules']} rules matched this sample. "
            f"The strongest supporting rule is {strongest['rule_id']}, which predicts "
            f"{strongest['prediction']} behavior and average RUL {strongest['avg_rul']:.0f} cycles."
        )

        return {
            "summary": summary,
            "predicted_rul": point_result["predicted_rul"],
            "predicted_stage": point_result["predicted_stage"],
            "prediction_label": point_result["prediction_label"],
            "confidence": point_result["confidence"],
            "reasons": reasons,
        }
    
    def why_not_query(
        self,
        sample,
        target_stage=None,
        target_prediction=None,
        top_k=3,
        min_importance=0.0,
    ):
        """
        Explain why a sample did NOT receive a requested stage/prediction.

        Example:
        why_not_query(sample, target_stage="Early")
        why_not_query(sample, target_prediction="Normal")
        """
        if target_stage is None and target_prediction is None:
            raise ValueError("Provide target_stage and/or target_prediction.")

        point_result = self.point_query(
            sample=sample,
            top_k=top_k,
            min_importance=min_importance,
            return_all_matches=True,
        )

        sample_2d = self._to_last_timestep(sample, single_sample=True)
        row = sample_2d[0]

        target_rules = self._candidate_rules_for_target(
            target_stage=target_stage,
            target_prediction=target_prediction,
            min_importance=min_importance,
        )

        if target_rules.empty:
            return {
                "summary": "No candidate rules exist for the requested target outcome.",
                "actual_prediction": point_result["prediction_label"],
                "actual_stage": point_result["predicted_stage"],
                "target_stage": target_stage,
                "target_prediction": target_prediction,
                "candidate_rules": [],
            }

        explanations = []
        for _, rule in target_rules.head(top_k).iterrows():
            failed_conditions, satisfied_conditions = self._rule_failure_details(
                row, rule["conditions"]
            )

            explanations.append(
                {
                    "rule_id": rule["rule_id"],
                    "target_stage": rule["stage"],
                    "target_prediction": rule["prediction"],
                    "importance": float(rule["importance"]),
                    "avg_rul": float(rule["avg_rul"]),
                    "coverage": float(rule["coverage"]),
                    "condition_str": rule["condition_str"],
                    "failed_conditions": failed_conditions,
                    "satisfied_conditions": satisfied_conditions,
                    "n_failed_conditions": len(failed_conditions),
                }
            )

        explanations = sorted(
            explanations,
            key=lambda x: (x["n_failed_conditions"], -x["importance"])
        )

        if explanations:
            best = explanations[0]
            summary = (
                f"The sample was not assigned to target outcome "
                f"(stage={target_stage}, prediction={target_prediction}) because the "
                f"closest supporting rule {best['rule_id']} failed "
                f"{best['n_failed_conditions']} condition(s)."
            )
        else:
            summary = "No near-miss rules were found for the requested target outcome."

        return {
            "summary": summary,
            "actual_prediction": point_result["prediction_label"],
            "actual_stage": point_result["predicted_stage"],
            "actual_rul": point_result["predicted_rul"],
            "target_stage": target_stage,
            "target_prediction": target_prediction,
            "candidate_rules": explanations,
        }

    def what_if_query(
        self,
        sample,
        feature_changes,
        top_k=5,
        min_importance=0.0,
    ):
        """
        Apply user-specified feature changes and compare the rule-based result.

        Example:
        what_if_query(sample, {"stiffness_degradation": 0.15})
        """
        original_result = self.point_query(
            sample=sample,
            top_k=top_k,
            min_importance=min_importance,
            return_all_matches=True,
        )

        modified_sample = self._apply_feature_changes(sample, feature_changes)

        modified_result = self.point_query(
            sample=modified_sample,
            top_k=top_k,
            min_importance=min_importance,
            return_all_matches=True,
        )

        delta_rul = None
        if (
            original_result["predicted_rul"] is not None
            and modified_result["predicted_rul"] is not None
        ):
            delta_rul = float(
                modified_result["predicted_rul"] - original_result["predicted_rul"]
            )

        summary = (
            f"Original stage={original_result['predicted_stage']}, "
            f"modified stage={modified_result['predicted_stage']}. "
            f"Original RUL={original_result['predicted_rul']:.0f} cycles, "
            f"modified RUL={modified_result['predicted_rul']:.0f} cycles."
        )

        return {
            "summary": summary,
            "feature_changes": feature_changes,
            "original": original_result,
            "modified": modified_result,
            "delta_rul": delta_rul,
            "stage_changed": original_result["predicted_stage"] != modified_result["predicted_stage"],
            "prediction_changed": original_result["prediction_label"] != modified_result["prediction_label"],
        }

    def counterfactual_query(
        self,
        sample,
        target_stage=None,
        target_prediction=None,
        top_k=10,
        min_importance=0.0,
    ):
        """
        Find a small rule-based change set that could move the sample toward a target outcome.

        Strategy:
        - Look at strong rules for the requested target
        - Find the rule with smallest unmet condition set
        - Suggest feature values that satisfy that rule
        """
        if target_stage is None and target_prediction is None:
            raise ValueError("Provide target_stage and/or target_prediction.")

        sample_2d = self._to_last_timestep(sample, single_sample=True)
        row = sample_2d[0]

        current_result = self.point_query(
            sample=sample,
            top_k=top_k,
            min_importance=min_importance,
            return_all_matches=True,
        )

        candidate_rules = self._candidate_rules_for_target(
            target_stage=target_stage,
            target_prediction=target_prediction,
            min_importance=min_importance,
        )

        if candidate_rules.empty:
            return {
                "summary": "No rules are available for the requested counterfactual target.",
                "current_result": current_result,
                "counterfactual_found": False,
            }

        candidates = []
        for _, rule in candidate_rules.head(top_k).iterrows():
            failed_conditions, satisfied_conditions = self._rule_failure_details(
                row, rule["conditions"]
            )

            proposed_changes = {}
            total_change_cost = 0.0

            for item in failed_conditions:
                feature_name = item["feature"]
                proposed_value = item["suggested_value"]
                proposed_changes[feature_name] = proposed_value
                total_change_cost += abs(proposed_value - item["actual_value"])

            candidates.append(
                {
                    "rule_id": rule["rule_id"],
                    "target_stage": rule["stage"],
                    "target_prediction": rule["prediction"],
                    "importance": float(rule["importance"]),
                    "avg_rul": float(rule["avg_rul"]),
                    "coverage": float(rule["coverage"]),
                    "condition_str": rule["condition_str"],
                    "n_failed_conditions": len(failed_conditions),
                    "failed_conditions": failed_conditions,
                    "proposed_changes": proposed_changes,
                    "total_change_cost": float(total_change_cost),
                }
            )

        candidates = sorted(
            candidates,
            key=lambda x: (
                x["n_failed_conditions"],
                x["total_change_cost"],
                -x["importance"],
            )
        )

        best = candidates[0]

        modified_sample = self._apply_feature_changes(sample, best["proposed_changes"])
        modified_result = self.point_query(
            sample=modified_sample,
            top_k=top_k,
            min_importance=min_importance,
            return_all_matches=True,
        )

        summary = (
            f"Closest counterfactual for target outcome "
            f"(stage={target_stage}, prediction={target_prediction}) is to modify "
            f"{len(best['proposed_changes'])} feature(s). "
            f"Suggested rule: {best['rule_id']}."
        )

        return {
            "summary": summary,
            "current_result": current_result,
            "target_stage": target_stage,
            "target_prediction": target_prediction,
            "counterfactual_found": True,
            "best_counterfactual": best,
            "modified_result": modified_result,
            "all_candidates": candidates,
        }

    def global_feature_influence_query(
        self,
        stage=None,
        prediction=None,
        top_k=10,
        min_importance=0.0,
        weight_by="importance",
    ):
        """
        Global/cohort feature influence from extracted rules.

        Example:
            global_feature_influence_query(stage="Late")
            global_feature_influence_query(stage="Late", prediction="Critical")
        """
        df = self.rules_df.copy()

        if stage is not None:
            df = df[df["stage"] == stage]
        if prediction is not None:
            df = df[df["prediction"] == prediction]

        df = df[df["importance"] >= float(min_importance)]

        if df.empty:
            return {
                "summary": "No rules found for the requested filters.",
                "stage": stage,
                "prediction": prediction,
                "n_rules": 0,
                "feature_ranking": pd.DataFrame(),
            }

        feature_stats = {}

        for _, rule in df.iterrows():
            rule_importance = float(rule["importance"])
            rule_coverage = float(rule.get("coverage", 0.0))
            rule_specificity = int(rule.get("specificity", len(rule["conditions"])))

            if weight_by == "importance_x_coverage":
                rule_weight = rule_importance * max(rule_coverage, 1e-12)
            else:
                rule_weight = rule_importance

            for feat, op, thr in rule["conditions"]:
                if feat not in feature_stats:
                    feature_stats[feat] = {
                        "feature": feat,
                        "weighted_score": 0.0,
                        "rule_count": 0,
                        "avg_coverage_num": 0.0,
                        "avg_specificity_num": 0.0,
                        "example_conditions": set(),
                    }

                feature_stats[feat]["weighted_score"] += rule_weight
                feature_stats[feat]["rule_count"] += 1
                feature_stats[feat]["avg_coverage_num"] += rule_coverage
                feature_stats[feat]["avg_specificity_num"] += rule_specificity
                feature_stats[feat]["example_conditions"].add(f"{feat} {op} {float(thr):.6f}")

        rows = []
        for feat, s in feature_stats.items():
            cnt = max(s["rule_count"], 1)
            rows.append(
                {
                    "feature": feat,
                    "weighted_score": float(s["weighted_score"]),
                    "rule_count": int(s["rule_count"]),
                    "avg_rule_coverage": float(s["avg_coverage_num"] / cnt),
                    "avg_rule_specificity": float(s["avg_specificity_num"] / cnt),
                    "example_conditions": " | ".join(list(s["example_conditions"])[:3]),
                }
            )

        ranking_df = pd.DataFrame(rows).sort_values(
            by=["weighted_score", "rule_count", "avg_rule_coverage"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        ranking_df["normalized_score"] = (
            ranking_df["weighted_score"] / max(ranking_df["weighted_score"].max(), 1e-12)
        )

        top_df = ranking_df.head(top_k).copy()

        summary = (
            f"Computed global feature influence from {len(df)} rules "
            f"(stage={stage}, prediction={prediction})."
        )

        return {
            "summary": summary,
            "stage": stage,
            "prediction": prediction,
            "n_rules": int(len(df)),
            "feature_ranking": top_df,
            "all_feature_ranking": ranking_df,
        }

    def cohort_pattern_query(
        self,
        X,
        feature_name,
        operator=">",
        threshold=None,
        quantile=None,
        y_true=None,
        top_k_features=8,
        top_k_rules=5,
        min_importance=0.0,
    ):
        """
        Cohort-pattern query:
        Compare samples with high/low feature values against the complement and
        return associated degradation patterns.

        Example:
            cohort_pattern_query(X_test_pool, "avg_delta_tof", operator=">", quantile=0.9, y_true=y_test_actual_holdout)
            cohort_pattern_query(X_test_pool, "avg_delta_tof", operator=">", threshold=0.6)
        """
        if feature_name not in self.feature_names:
            raise KeyError(f"Unknown feature: {feature_name}")

        X_2d = self._to_last_timestep(X, single_sample=False)
        feat_idx = self.feature_names.index(feature_name)
        feat_vals = X_2d[:, feat_idx]

        if threshold is None:
            if quantile is None:
                raise ValueError("Provide either threshold or quantile.")
            if not (0.0 < quantile < 1.0):
                raise ValueError("quantile must be in (0, 1).")
            threshold = float(np.quantile(feat_vals, quantile))
        else:
            threshold = float(threshold)

        def _build_mask(values, op, thr):
            if op == ">":
                return values > thr
            if op == ">=":
                return values >= thr
            if op == "<":
                return values < thr
            if op == "<=":
                return values <= thr
            if op == "==":
                return values == thr
            raise ValueError(f"Unsupported operator: {op}")

        cohort_mask = _build_mask(feat_vals, operator, threshold)
        comp_mask = ~cohort_mask

        n_total = len(feat_vals)
        n_cohort = int(cohort_mask.sum())
        n_comp = int(comp_mask.sum())

        if n_cohort == 0 or n_comp == 0:
            return {
                "summary": "Cohort or complement is empty; adjust threshold/operator.",
                "feature_name": feature_name,
                "operator": operator,
                "threshold": threshold,
                "n_total": int(n_total),
                "n_cohort": n_cohort,
                "n_complement": n_comp,
                "feature_differences": pd.DataFrame(),
                "supporting_rules": pd.DataFrame(),
            }

        cohort_X = X_2d[cohort_mask]
        comp_X = X_2d[comp_mask]

        cohort_means = cohort_X.mean(axis=0)
        comp_means = comp_X.mean(axis=0)
        delta = cohort_means - comp_means

        diff_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "cohort_mean": cohort_means,
                "complement_mean": comp_means,
                "delta": delta,
                "abs_delta": np.abs(delta),
            }
        ).sort_values(by="abs_delta", ascending=False).reset_index(drop=True)

        top_diff_df = diff_df.head(top_k_features).copy()

        # Rules associated with this feature
        candidate_rules = []
        for _, rule in self.rules_df.iterrows():
            if float(rule["importance"]) < float(min_importance):
                continue
            cond_feats = [c[0] for c in rule["conditions"]]
            if feature_name in cond_feats:
                candidate_rules.append(rule.to_dict())

        rules_df = pd.DataFrame(candidate_rules)
        if not rules_df.empty:
            rules_df = self._rank_rules(rules_df).head(top_k_rules).reset_index(drop=True)

        # Optional stage pattern using provided y_true RUL in cycles
        stage_pattern = None
        if y_true is not None:
            y_true = np.asarray(y_true).reshape(-1)
            if len(y_true) != n_total:
                raise ValueError("y_true length must match number of samples in X.")

            cohort_stages = pd.Series([self._stage_from_rul(v) for v in y_true[cohort_mask]])
            comp_stages = pd.Series([self._stage_from_rul(v) for v in y_true[comp_mask]])

            cohort_dist = cohort_stages.value_counts(normalize=True)
            comp_dist = comp_stages.value_counts(normalize=True)

            all_stages = sorted(set(cohort_dist.index).union(set(comp_dist.index)))
            stage_rows = []
            for s in all_stages:
                c = float(cohort_dist.get(s, 0.0))
                k = float(comp_dist.get(s, 0.0))
                stage_rows.append(
                    {
                        "stage": s,
                        "cohort_fraction": c,
                        "complement_fraction": k,
                        "delta_fraction": c - k,
                    }
                )
            stage_pattern = pd.DataFrame(stage_rows).sort_values(
                by="delta_fraction", ascending=False
            ).reset_index(drop=True)

        summary = (
            f"Cohort query for {feature_name} {operator} {threshold:.6f}: "
            f"{n_cohort}/{n_total} samples ({100.0*n_cohort/max(n_total,1):.1f}%)."
        )

        return {
            "summary": summary,
            "feature_name": feature_name,
            "operator": operator,
            "threshold": float(threshold),
            "n_total": int(n_total),
            "n_cohort": int(n_cohort),
            "n_complement": int(n_comp),
            "feature_differences": top_diff_df,
            "all_feature_differences": diff_df,
            "supporting_rules": rules_df if not rules_df.empty else pd.DataFrame(),
            "stage_pattern": stage_pattern,
        }

    def _feature_dict_to_row(self, feature_updates, base_row=None):
        if base_row is None:
            row = np.zeros(len(self.feature_names), dtype=float)
        else:
            row = np.array(base_row, dtype=float).copy()

        for feature_name, value in feature_updates.items():
            if feature_name not in self.feature_names:
                raise KeyError(f"Unknown feature: {feature_name}")
            feature_idx = self.feature_names.index(feature_name)
            row[feature_idx] = float(value)

        return row

    def _rule_failure_details(self, row, conditions):
        failed = []
        satisfied = []

        for feature_name, operator, threshold in conditions:
            feature_idx = self.feature_names.index(feature_name)
            actual_value = float(row[feature_idx])

            if operator == "<=":
                is_satisfied = actual_value <= threshold
                distance_to_satisfy = max(0.0, actual_value - threshold)
                target_value = float(threshold)
            else:
                is_satisfied = actual_value > threshold
                distance_to_satisfy = max(0.0, threshold - actual_value + 1e-6)
                target_value = float(threshold + 1e-6)

            item = {
                "feature": feature_name,
                "operator": operator,
                "threshold": float(threshold),
                "actual_value": actual_value,
                "satisfied": bool(is_satisfied),
                "distance_to_satisfy": float(distance_to_satisfy),
                "suggested_value": target_value,
            }

            if is_satisfied:
                satisfied.append(item)
            else:
                failed.append(item)

        return failed, satisfied

    def _apply_feature_changes(self, sample, feature_changes):
        sample_arr = np.asarray(sample).copy()

        if sample_arr.ndim == 1:
            updated = sample_arr.copy()
            for feature_name, value in feature_changes.items():
                feature_idx = self.feature_names.index(feature_name)
                updated[feature_idx] = float(value)
            return updated

        if sample_arr.ndim == 2:
            updated = sample_arr.copy()
            for feature_name, value in feature_changes.items():
                feature_idx = self.feature_names.index(feature_name)
                updated[-1, feature_idx] = float(value)
            return updated

        if sample_arr.ndim == 3:
            updated = sample_arr.copy()
            for feature_name, value in feature_changes.items():
                feature_idx = self.feature_names.index(feature_name)
                updated[0, -1, feature_idx] = float(value)
            return updated

        raise ValueError("Unsupported sample shape for modification.")

    def _candidate_rules_for_target(self, target_stage=None, target_prediction=None, min_importance=0.0):
        candidate_df = self.rules_df.copy()

        if target_stage is not None:
            candidate_df = candidate_df[candidate_df["stage"] == target_stage]

        if target_prediction is not None:
            candidate_df = candidate_df[candidate_df["prediction"] == target_prediction]

        candidate_df = candidate_df[candidate_df["importance"] >= min_importance]

        if candidate_df.empty:
            return candidate_df

        return self._rank_rules(candidate_df)
    
    def evaluate(self, X, y_true=None, min_importance=0.0):
        X_2d = self._to_last_timestep(X, single_sample=False)

        records = []
        for idx in range(X_2d.shape[0]):
            sample = X_2d[idx]
            result = self.point_query(sample, min_importance=min_importance)

            record = {
                "sample_idx": idx,
                "predicted_rul_rulefit": result["predicted_rul"],
                "predicted_stage_rulefit": result["predicted_stage"],
                "prediction_label_rulefit": result["prediction_label"],
                "confidence": result["confidence"],
                "n_matched_rules": result["n_matched_rules"],
                "covered": result["n_matched_rules"] > 0,
            }

            if y_true is not None:
                true_rul = float(y_true[idx])
                record["actual_rul"] = true_rul
                if result["predicted_rul"] is not None:
                    record["abs_error"] = abs(result["predicted_rul"] - true_rul)
                else:
                    record["abs_error"] = np.nan

            records.append(record)

        results_df = pd.DataFrame(records)

        summary = {
            "coverage_rate": float(results_df["covered"].mean()),
            "avg_matched_rules": float(results_df["n_matched_rules"].mean()),
            "avg_confidence": float(results_df["confidence"].mean()),
        }

        if y_true is not None and results_df["predicted_rul_rulefit"].notna().any():
            valid = results_df.dropna(subset=["predicted_rul_rulefit", "actual_rul"])
            errors = valid["predicted_rul_rulefit"].values - valid["actual_rul"].values
            summary["mae"] = float(np.mean(np.abs(errors)))
            summary["rmse"] = float(np.sqrt(np.mean(errors ** 2)))

        return summary, results_df