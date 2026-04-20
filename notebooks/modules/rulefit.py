import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.tree import _tree
from typing import List, Dict, Tuple

class RuleFit:
    """
    RuleFit algorithm for extracting interpretable rules from black-box models
    
    Based on: Friedman & Popescu (2008) "Predictive Learning via Rule Ensembles"
    
    Strategy:
    1. Train ensemble of decision trees on model predictions
    2. Extract all rules (paths) from trees
    3. Use Lasso to select most important rules
    4. Return ranked interpretable rules
    """
    
    def __init__(
        self,
        model,
        feature_names,
        n_estimators=100,
        max_depth=3,
        tree_size=4,
        memory_par=0.01,
        random_state=42,
        target_scaler=None,
        stage_boundaries=None,
    ):
        """
        Initialize RuleFit

        Args:
            model: Trained PyTorch model to extract rules from
            feature_names: List of feature names
            n_estimators: Number of trees in ensemble
            max_depth: Maximum depth of each tree
            tree_size: Average number of terminal nodes in trees
            memory_par: Lasso regularization parameter (higher = fewer rules)
            random_state: Random seed
            target_scaler: Optional scaler used to inverse-transform model outputs
            stage_boundaries: Dict like {"Early": 130000, "Mid": 80000}
        """
        self.model = model
        self.feature_names = feature_names
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.tree_size = tree_size
        self.memory_par = memory_par
        self.random_state = random_state
        self.target_scaler = target_scaler
        self.stage_boundaries = stage_boundaries or {
            "Early": 130000,
            "Mid": 80000,
        }

        self.device = next(model.parameters()).device

        self.rules = []
        self.all_rules = []
        self.rule_ensemble = None
        self.lasso_model = None
        self.linear_coefs = None
        self.selected_rule_indices = []
        self.selected_rule_importances = []
        
    def fit(self, X, y_true, stage_labels):
        """
        Fit RuleFit: extract rules and select important ones

        Args:
            X: Input features (n_samples, seq_length, n_features) - NumPy array
            y_true: True target values (actual RUL cycles)
            stage_labels: Degradation stage for each sample

        Returns:
            self
        """
        print("\n" + "=" * 80)
        print("RULEFIT RULE EXTRACTION")
        print("=" * 80)

        print("\n1️⃣ Getting model predictions...")
        y_pred = self._get_predictions(X)

        X_2d = X[:, -1, :]

        print(f"   ✓ Got {len(y_pred)} predictions")
        print(f"   Prediction range: [{y_pred.min():.0f}, {y_pred.max():.0f}] cycles")

        print("\n2️⃣ Training tree ensemble...")
        self.rule_ensemble = self._train_tree_ensemble(X_2d, y_pred)
        print(f"   ✓ Trained {self.n_estimators} trees")

        print("\n3️⃣ Extracting rules from trees...")
        self.all_rules = self._extract_rules_from_ensemble(X_2d)
        print(f"   ✓ Extracted {len(self.all_rules)} candidate rules")

        print("\n4️⃣ Creating rule feature matrix...")
        rule_features = self._create_rule_features(X_2d, self.all_rules)
        print(f"   ✓ Created feature matrix: {rule_features.shape}")

        print("\n5️⃣ Selecting important rules with Lasso...")
        self.selected_rule_indices, self.selected_rule_importances = self._select_rules_lasso(
            rule_features, y_pred
        )
        print(f"   ✓ Selected {len(self.selected_rule_indices)} important rules")

        print("\n6️⃣ Categorizing rules by stage...")
        self.rules = self._categorize_rules(
            self.selected_rule_indices,
            self.selected_rule_importances,
            X_2d,
            y_pred,
            stage_labels,
        )
        print(f"   ✓ Categorized {len(self.rules)} final rules")

        print("\n" + "=" * 80)
        print(f"✅ RULEFIT COMPLETE: {len(self.rules)} rules extracted")
        print("=" * 80)

        return self
    
    def _get_predictions(self, X):
        """Get model predictions and convert them to actual RUL cycles when needed."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_pred = self.model(X_tensor).detach().cpu().numpy().reshape(-1, 1)

        if self.target_scaler is not None:
            y_pred = self.target_scaler.inverse_transform(y_pred)

        return y_pred.flatten()
    
    def _prediction_from_rul(self, avg_rul):
        if avg_rul < self.stage_boundaries["Mid"]:
            return "Critical"
        if avg_rul < self.stage_boundaries["Early"]:
            return "Warning"
        return "Normal"


    def _rule_specificity(self, conditions):
        return len(conditions)
    
    def _train_tree_ensemble(self, X, y):
        """Train gradient boosting ensemble"""
        ensemble = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.01,
            subsample=0.5,  # Bagging for diversity
            max_features='sqrt',
            random_state=self.random_state,
            verbose=0
        )
        ensemble.fit(X, y)
        return ensemble
    
    def _extract_rules_from_ensemble(self, X):
        """Extract all rules from all trees in ensemble"""
        all_rules = []
        
        for tree_idx, tree in enumerate(self.rule_ensemble.estimators_):
            tree_estimator = tree[0]  # GBM wraps trees
            tree_rules = self._extract_rules_from_tree(tree_estimator.tree_, tree_idx)
            all_rules.extend(tree_rules)
        
        return all_rules
    
    def _extract_rules_from_tree(self, tree, tree_idx):
        """Extract all paths (rules) from a single tree"""
        rules = []
        
        def recurse(node, conditions, depth):
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                # Internal node
                feature = self.feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                
                # Left branch: feature <= threshold
                left_cond = conditions + [(feature, '<=', threshold)]
                recurse(tree.children_left[node], left_cond, depth + 1)
                
                # Right branch: feature > threshold
                right_cond = conditions + [(feature, '>', threshold)]
                recurse(tree.children_right[node], right_cond, depth + 1)
            else:
                # Leaf node - create rule
                if len(conditions) > 0:  # Ignore root-only rules
                    rule = {
                        'tree_id': tree_idx,
                        'conditions': conditions,
                        'support': tree.n_node_samples[node],
                        'prediction_value': tree.value[node][0, 0]
                    }
                    rules.append(rule)
        
        recurse(0, [], 0)
        return rules
    
    def _create_rule_features(self, X, rules):
        """
        Create binary feature matrix where each column is whether a rule applies
        
        Args:
            X: Feature matrix (n_samples, n_features)
            rules: List of rule dictionaries
            
        Returns:
            Binary matrix (n_samples, n_rules)
        """
        n_samples = X.shape[0]
        n_rules = len(rules)
        rule_features = np.zeros((n_samples, n_rules))
        
        for rule_idx, rule in enumerate(rules):
            # Check which samples satisfy this rule
            mask = np.ones(n_samples, dtype=bool)
            
            for feature_name, operator, threshold in rule['conditions']:
                feature_idx = self.feature_names.index(feature_name)
                
                if operator == '<=':
                    mask &= (X[:, feature_idx] <= threshold)
                else:  # '>'
                    mask &= (X[:, feature_idx] > threshold)
            
            rule_features[:, rule_idx] = mask.astype(float)
        
        return rule_features
    
    def _select_rules_lasso(self, rule_features, y):
        """
        Select important rules using Lasso regression
        
        Args:
            rule_features: Binary rule matrix (n_samples, n_rules)
            y: Target values
            
        Returns:
            selected_rules: List of selected rules
            importances: Importance (coefficient) of each selected rule
        """
        # Remove rules that never fire or always fire
        rule_variance = np.var(rule_features, axis=0)
        valid_rules = rule_variance > 0
        
        if valid_rules.sum() == 0:
            print("   ⚠️  No valid rules found!")
            return [], []
        
        rule_features_valid = rule_features[:, valid_rules]
        
        # Fit Lasso with cross-validation to select alpha
        lasso = LassoCV(
            cv=5,
            random_state=self.random_state,
            max_iter=5000,
            n_alphas=100
        )
        lasso.fit(rule_features_valid, y)
        
        self.lasso_model = lasso
        
        # Get non-zero coefficients (selected rules)
        coefs = lasso.coef_
        selected_mask = np.abs(coefs) > 1e-6
        
        if selected_mask.sum() == 0:
            print("   ⚠️  Lasso selected no rules (alpha too high)!")
            # Fallback: select top rules by coefficient from unregularized fit
            selected_mask = np.argsort(np.abs(coefs))[-20:]  # Top 20 rules
        
        valid_rule_indices = np.where(valid_rules)[0]
        selected_rule_indices = valid_rule_indices[selected_mask]
        selected_importances = np.abs(coefs[selected_mask])
        
        return selected_rule_indices.tolist(), selected_importances.tolist()
    
    def _categorize_rules(self, selected_indices, importances, X, y_pred, stage_labels):
        """
        Categorize selected rules by stage and prediction type.
        """
        categorized = []

        for idx, importance in zip(selected_indices, importances):
            if idx >= len(self.all_rules):
                continue

            rule = self.all_rules[idx]
            mask = self._evaluate_rule(X, rule["conditions"])

            if mask.sum() == 0:
                continue

            stages_for_rule = stage_labels[mask]
            majority_stage = (
                pd.Series(stages_for_rule).mode()[0]
                if len(stages_for_rule) > 0
                else "Unknown"
            )

            avg_rul = float(np.mean(y_pred[mask]))
            prediction = self._prediction_from_rul(avg_rul)

            condition_strs = []
            for feat, op, thresh in rule["conditions"]:
                condition_strs.append(f"{feat} {op} {thresh:.6f}")

            categorized_rule = {
                "rule_id": f"RULE_{len(categorized):03d}",
                "stage": majority_stage,
                "conditions": rule["conditions"],
                "condition_str": " AND ".join(condition_strs),
                "prediction": prediction,
                "importance": float(importance),
                "support": int(mask.sum()),
                "avg_rul": avg_rul,
                "coverage": float(mask.sum() / len(mask)),
                "specificity": self._rule_specificity(rule["conditions"]),
            }

            categorized.append(categorized_rule)

        categorized.sort(
            key=lambda x: (x["importance"], x["specificity"], -x["coverage"]),
            reverse=True,
        )
        return categorized
    
    def _evaluate_rule(self, X, conditions):
        """Evaluate which samples satisfy a rule"""
        mask = np.ones(X.shape[0], dtype=bool)
        
        for feature_name, operator, threshold in conditions:
            feature_idx = self.feature_names.index(feature_name)
            
            if operator == '<=':
                mask &= (X[:, feature_idx] <= threshold)
            else:  # '>'
                mask &= (X[:, feature_idx] > threshold)
        
        return mask
    
    def predict(self, X):
        """
        Make predictions using learned rules
        
        Args:
            X: Feature matrix (can be 3D with sequences)
            
        Returns:
            predictions: RUL predictions using rule ensemble
        """
        if X.ndim == 3:
            X = X[:, -1, :]  # Use last timestep
        
        # Recreate rule features for this data
        all_rules = []
        for tree in self.rule_ensemble.estimators_:
            tree_rules = self._extract_rules_from_tree(tree[0].tree_, 0)
            all_rules.extend(tree_rules)
        
        rule_features = self._create_rule_features(X, all_rules)
        
        # Use Lasso model to make predictions
        predictions = self.lasso_model.predict(rule_features)
        
        return predictions
    
    def get_rules_dataframe(self):
        """Return rules as pandas DataFrame"""
        if not self.rules:
            return pd.DataFrame()
        return pd.DataFrame(self.rules)
    
    def print_rules(self, stage=None, min_importance=0.0, top_k=None):
        """
        Pretty print extracted rules
        
        Args:
            stage: Filter by stage ('Early', 'Mid', 'Late') or None for all
            min_importance: Minimum importance threshold
            top_k: Show only top K rules
        """
        filtered = [
            r for r in self.rules
            if (stage is None or r['stage'] == stage) and r['importance'] >= min_importance
        ]
        
        if top_k:
            filtered = filtered[:top_k]
        
        if not filtered:
            print("No rules match the criteria.")
            return
        
        print(f"\n{'='*80}")
        print(f"RULEFIT EXTRACTED RULES (Showing {len(filtered)} rules)")
        print(f"{'='*80}\n")
        
        for rule in filtered:
            print(f"🔹 {rule['rule_id']} (Importance: {rule['importance']:.4f})")
            print(f"   Stage: {rule['stage']} | Prediction: {rule['prediction']}")
            print(f"   IF {rule['condition_str']}")
            print(f"   THEN Expected RUL ≈ {rule['avg_rul']:.0f} cycles")
            print(f"   Support: {rule['support']} samples ({rule['coverage']*100:.1f}% coverage)")
            print(f"{'-'*80}\n")
    
    def save_rules(self, filepath='../outputs/rulefit_rules.csv'):
        """Save rules to CSV"""
        df = self.get_rules_dataframe()
        if not df.empty:
            df.to_csv(filepath, index=False)
            print(f"\n✅ Rules saved to: {filepath}")
        else:
            print("\n⚠️  No rules to save.")


# Convenience function for notebook use
def extract_rules_with_rulefit(
    model,
    X_test,
    y_test,
    stage_labels,
    feature_names,
    n_estimators=100,
    max_depth=3,
    target_scaler=None,
    stage_boundaries=None,
):
    """
    Wrapper function to extract rules using RuleFit.
    """
    rulefit = RuleFit(
        model=model,
        feature_names=feature_names,
        n_estimators=n_estimators,
        max_depth=max_depth,
        memory_par=0.01,
        random_state=42,
        target_scaler=target_scaler,
        stage_boundaries=stage_boundaries,
    )

    rulefit.fit(X_test, y_test, stage_labels)
    return rulefit