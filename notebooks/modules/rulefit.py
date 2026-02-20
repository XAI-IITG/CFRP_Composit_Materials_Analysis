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
    
    def __init__(self, model, feature_names, n_estimators=100, max_depth=3, 
                 tree_size=4, memory_par=0.01, random_state=42):
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
        """
        self.model = model
        self.feature_names = feature_names
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.tree_size = tree_size
        self.memory_par = memory_par
        self.random_state = random_state
        self.device = next(model.parameters()).device
        
        # Will be populated during fit
        self.rules = []
        self.rule_ensemble = None
        self.lasso_model = None
        self.linear_coefs = None
        
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
        print("\n" + "="*80)
        print("RULEFIT RULE EXTRACTION")
        print("="*80)
        
        # Step 1: Get model predictions
        print("\n1️⃣ Getting model predictions...")
        y_pred = self._get_predictions(X)
        
        # Use last timestep features for rule extraction
        X_2d = X[:, -1, :]  # (n_samples, n_features)
        
        print(f"   ✓ Got {len(y_pred)} predictions")
        print(f"   Prediction range: [{y_pred.min():.0f}, {y_pred.max():.0f}] cycles")
        
        # Step 2: Train tree ensemble on predictions
        print("\n2️⃣ Training tree ensemble...")
        self.rule_ensemble = self._train_tree_ensemble(X_2d, y_pred)
        print(f"   ✓ Trained {self.n_estimators} trees")
        
        # Step 3: Extract all rules from trees
        print("\n3️⃣ Extracting rules from trees...")
        all_rules = self._extract_rules_from_ensemble(X_2d)
        print(f"   ✓ Extracted {len(all_rules)} candidate rules")
        
        # Step 4: Create rule feature matrix
        print("\n4️⃣ Creating rule feature matrix...")
        rule_features = self._create_rule_features(X_2d, all_rules)
        print(f"   ✓ Created feature matrix: {rule_features.shape}")
        
        # Step 5: Select important rules using Lasso
        print("\n5️⃣ Selecting important rules with Lasso...")
        selected_rules, rule_importances = self._select_rules_lasso(
            rule_features, y_pred
        )
        print(f"   ✓ Selected {len(selected_rules)} important rules")
        
        # Step 6: Categorize rules by stage and prediction
        print("\n6️⃣ Categorizing rules by stage...")
        self.rules = self._categorize_rules(
            selected_rules, rule_importances, X_2d, y_pred, stage_labels
        )
        print(f"   ✓ Categorized {len(self.rules)} final rules")
        
        print("\n" + "="*80)
        print(f"✅ RULEFIT COMPLETE: {len(self.rules)} rules extracted")
        print("="*80)
        
        return self
    
    def _get_predictions(self, X):
        """Get model predictions using PyTorch"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_pred_normalized = self.model(X_tensor).cpu().numpy().flatten()
        
        # Note: If predictions are normalized, you need to inverse transform
        # For now, assuming they're in actual RUL cycles
        return y_pred_normalized
    
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
        Categorize selected rules by stage and prediction type
        
        Args:
            selected_indices: Indices of selected rules
            importances: Importance scores
            X: Feature matrix
            y_pred: Model predictions
            stage_labels: Stage labels
            
        Returns:
            List of categorized rule dictionaries
        """
        # Get all rules that were extracted
        all_rules = []
        for tree in self.rule_ensemble.estimators_:
            tree_rules = self._extract_rules_from_tree(tree[0].tree_, 0)
            all_rules.extend(tree_rules)
        
        categorized = []
        
        for idx, importance in zip(selected_indices, importances):
            if idx >= len(all_rules):
                continue
                
            rule = all_rules[idx]
            
            # Find samples that satisfy this rule
            mask = self._evaluate_rule(X, rule['conditions'])
            
            if mask.sum() == 0:
                continue
            
            # Determine majority stage
            stages_for_rule = stage_labels[mask]
            majority_stage = pd.Series(stages_for_rule).mode()[0] if len(stages_for_rule) > 0 else 'Unknown'
            
            # Determine prediction category
            avg_rul = np.mean(y_pred[mask])
            if avg_rul < 60000:
                prediction = 'Critical'
            elif avg_rul < 120000:
                prediction = 'Warning'
            else:
                prediction = 'Normal'
            
            # Format conditions as string
            condition_strs = []
            for feat, op, thresh in rule['conditions']:
                condition_strs.append(f"{feat} {op} {thresh:.3f}")
            
            categorized_rule = {
                'rule_id': f"RULE_{len(categorized):03d}",
                'stage': majority_stage,
                'conditions': rule['conditions'],
                'condition_str': ' AND '.join(condition_strs),
                'prediction': prediction,
                'importance': float(importance),
                'support': int(mask.sum()),
                'avg_rul': float(avg_rul),
                'coverage': float(mask.sum() / len(mask))
            }
            
            categorized.append(categorized_rule)
        
        # Sort by importance
        categorized.sort(key=lambda x: x['importance'], reverse=True)
        
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
def extract_rules_with_rulefit(model, X_test, y_test, stage_labels, feature_names, 
                                n_estimators=100, max_depth=3):
    """
    Wrapper function to extract rules using RuleFit
    
    Args:
        model: Trained PyTorch model
        X_test: Test features (n_samples, seq_length, n_features)
        y_test: Test targets (actual RUL cycles)
        stage_labels: Stage labels for each sample
        feature_names: List of feature names
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        
    Returns:
        rulefit: Fitted RuleFit object with extracted rules
    """
    rulefit = RuleFit(
        model=model,
        feature_names=feature_names,
        n_estimators=n_estimators,
        max_depth=max_depth,
        memory_par=0.01,
        random_state=42
    )
    
    rulefit.fit(X_test, y_test, stage_labels)
    
    return rulefit