import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Literal
import pandas as pd
from dataclasses import dataclass
from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


@dataclass
class EnsembleRule:
    """Represents a rule extracted from ensemble"""
    conditions: List[Tuple[str, str, float]]  # (feature, operator, threshold)
    prediction: float  # Predicted RUL value
    stage: str  # Degradation stage
    ensemble_votes: int  # Number of trees supporting this rule
    avg_confidence: float  # Average confidence across trees
    support: int  # Number of samples covered
    diversity: float  # Diversity score (how many different trees)
    
    def __str__(self):
        cond_str = ' AND '.join([f"{feat} {op} {val:.4f}" for feat, op, val in self.conditions])
        return f"IF {cond_str} THEN RUL ≈ {self.prediction:.0f} ({self.stage}) [votes={self.ensemble_votes}, conf={self.avg_confidence:.3f}]"
    
    def __repr__(self):
        return self.__str__()


class REFNE:
    """
    Rule Extraction from Neural Network Ensemble.
    
    Uses ensemble of decision trees to approximate neural network,
    then extracts and consolidates rules from the ensemble.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        feature_names: List[str],
        target_scaler=None,
        ensemble_method: Literal['random_forest', 'gradient_boosting', 'bagging'] = 'random_forest',
        n_estimators: int = 100,
        max_depth: int = 6,
        min_samples_split: int = 50,
        min_samples_leaf: int = 20,
        sample_size: int = 1000,
        synthetic_ratio: float = 0.3,
        min_ensemble_votes: int = 3,
        min_rule_confidence: float = 0.6,
        stage_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
        random_state: int = 42
    ):
        """
        Initialize REFNE rule extractor.
        
        Args:
            model: Trained PyTorch model
            device: torch.device
            feature_names: List of feature names
            target_scaler: Scaler for inverse transforming predictions
            ensemble_method: Type of ensemble ('random_forest', 'gradient_boosting', 'bagging')
            n_estimators: Number of trees in ensemble
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            sample_size: Number of samples for training ensemble
            synthetic_ratio: Ratio of synthetic to real samples
            min_ensemble_votes: Minimum trees that must support a rule
            min_rule_confidence: Minimum confidence threshold
            stage_thresholds: Optional RUL thresholds for stages
            random_state: Random seed
        """
        self.model = model
        self.device = device
        self.feature_names = feature_names
        self.target_scaler = target_scaler
        self.ensemble_method = ensemble_method
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.sample_size = sample_size
        self.synthetic_ratio = synthetic_ratio
        self.min_ensemble_votes = min_ensemble_votes
        self.min_rule_confidence = min_rule_confidence
        self.random_state = random_state
        
        # Define stage thresholds
        if stage_thresholds is None:
            self.stage_thresholds = {
                'critical': (0, 60000),
                'progressive': (60000, 120000),
                'early_damage': (120000, 180000),
                'healthy': (180000, 230000)
            }
        else:
            self.stage_thresholds = stage_thresholds
        
        self.ensemble = None
        self.rules = []
        self.tree_rules = []  # Rules from individual trees
        self.feature_importance_ = None
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    
    def _query_model(self, X: np.ndarray) -> np.ndarray:
        """Query model predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        if self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    
    def _generate_synthetic_samples(
        self, 
        X_real: np.ndarray, 
        n_synthetic: int
    ) -> np.ndarray:
        """Generate synthetic samples by perturbing real samples"""
        n_samples, seq_len, n_features = X_real.shape
        
        # Gaussian noise perturbation (50%)
        n_gaussian = int(n_synthetic * 0.5)
        base_samples = X_real[np.random.choice(n_samples, n_gaussian)]
        noise_std = np.std(X_real, axis=0) * 0.15
        synthetic_gaussian = base_samples + np.random.normal(0, noise_std, base_samples.shape)
        
        # Feature swapping (30%)
        n_swap = int(n_synthetic * 0.3)
        synthetic_swap = []
        for _ in range(n_swap):
            sample = X_real[np.random.choice(n_samples)].copy()
            n_swap_features = max(1, int(n_features * 0.4))
            swap_features = np.random.choice(n_features, n_swap_features, replace=False)
            donor = X_real[np.random.choice(n_samples)]
            sample[:, swap_features] = donor[:, swap_features]
            synthetic_swap.append(sample)
        synthetic_swap = np.array(synthetic_swap)
        
        # Interpolation (20%)
        n_interp = n_synthetic - n_gaussian - n_swap
        synthetic_interp = []
        for _ in range(n_interp):
            idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
            alpha = np.random.uniform(0.3, 0.7)
            interp_sample = alpha * X_real[idx1] + (1 - alpha) * X_real[idx2]
            synthetic_interp.append(interp_sample)
        synthetic_interp = np.array(synthetic_interp) if synthetic_interp else np.array([]).reshape(0, seq_len, n_features)
        
        X_synthetic = np.vstack([synthetic_gaussian, synthetic_swap, synthetic_interp])
        
        return X_synthetic
    
    
    def _assign_stage(self, rul: float) -> str:
        """Assign degradation stage based on RUL value"""
        for stage_name, (min_rul, max_rul) in self.stage_thresholds.items():
            if min_rul <= rul < max_rul:
                return stage_name
        return 'unknown'
    
    
    def _build_ensemble(self, X_flat: np.ndarray, y: np.ndarray):
        """Build ensemble of decision trees"""
        print(f"\n🌳 Building {self.ensemble_method} ensemble...")
        print(f"   Number of trees: {self.n_estimators}")
        print(f"   Max depth: {self.max_depth}")
        
        if self.ensemble_method == 'random_forest':
            self.ensemble = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.ensemble_method == 'gradient_boosting':
            self.ensemble = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        elif self.ensemble_method == 'bagging':
            from sklearn.ensemble import BaggingRegressor
            base_tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            self.ensemble = BaggingRegressor(
                estimator=base_tree,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        self.ensemble.fit(X_flat, y)
        
        # Calculate feature importance
        if hasattr(self.ensemble, 'feature_importances_'):
            self.feature_importance_ = dict(zip(self.feature_names, self.ensemble.feature_importances_))
        
        print(f"   ✅ Ensemble trained")
    
    
    def _extract_rules_from_tree(
        self, 
        tree: DecisionTreeRegressor, 
        tree_id: int
    ) -> List[Dict]:
        """Extract rules from a single decision tree"""
        tree_obj = tree.tree_
        
        rules = []
        
        def recurse(node, conditions):
            """Recursively traverse tree"""
            
            # Leaf node
            if tree_obj.feature[node] == _tree.TREE_UNDEFINED:
                prediction = tree_obj.value[node][0][0]
                n_samples = tree_obj.n_node_samples[node]
                impurity = tree_obj.impurity[node]
                
                rule = {
                    'tree_id': tree_id,
                    'conditions': conditions.copy(),
                    'prediction': prediction,
                    'stage': self._assign_stage(prediction),
                    'n_samples': n_samples,
                    'impurity': impurity,
                    'confidence': 1.0 - impurity / (impurity + 1e-10)  # Normalized confidence
                }
                rules.append(rule)
                return
            
            # Internal node
            feature_idx = tree_obj.feature[node]
            threshold = tree_obj.threshold[node]
            feature_name = self.feature_names[feature_idx]
            
            # Left child (<=)
            left_conditions = conditions + [(feature_name, '<=', threshold)]
            recurse(tree_obj.children_left[node], left_conditions)
            
            # Right child (>)
            right_conditions = conditions + [(feature_name, '>', threshold)]
            recurse(tree_obj.children_right[node], right_conditions)
        
        recurse(0, [])
        
        return rules
    
    
    def _extract_rules_from_ensemble(self):
        """Extract rules from all trees in ensemble"""
        print(f"\n📋 Extracting rules from {self.n_estimators} trees...")
        
        all_tree_rules = []
        
        # Get individual trees from ensemble
        if self.ensemble_method == 'random_forest':
            trees = self.ensemble.estimators_
        elif self.ensemble_method == 'gradient_boosting':
            trees = [est[0] for est in self.ensemble.estimators_]
        elif self.ensemble_method == 'bagging':
            trees = self.ensemble.estimators_
        
        # Extract rules from each tree
        for tree_id, tree in enumerate(trees):
            tree_rules = self._extract_rules_from_tree(tree, tree_id)
            all_tree_rules.extend(tree_rules)
            
            if (tree_id + 1) % 20 == 0:
                print(f"   Processed {tree_id + 1}/{self.n_estimators} trees...")
        
        print(f"   ✅ Extracted {len(all_tree_rules)} total rules from all trees")
        
        self.tree_rules = all_tree_rules
    
    
    def _consolidate_rules(self):
        """Consolidate rules from multiple trees using voting"""
        print(f"\n🔄 Consolidating rules via ensemble voting...")
        
        # Group similar rules (same conditions)
        rule_groups = defaultdict(list)
        
        for rule in self.tree_rules:
            # Create hashable key from conditions
            conditions_tuple = tuple(sorted(rule['conditions']))
            rule_groups[conditions_tuple].append(rule)
        
        print(f"   Found {len(rule_groups)} unique rule patterns")
        
        # Aggregate rules
        consolidated_rules = []
        
        for conditions_tuple, similar_rules in rule_groups.items():
            n_votes = len(similar_rules)
            
            # Filter by minimum votes
            if n_votes < self.min_ensemble_votes:
                continue
            
            # Calculate aggregate statistics
            predictions = [r['prediction'] for r in similar_rules]
            confidences = [r['confidence'] for r in similar_rules]
            supports = [r['n_samples'] for r in similar_rules]
            stages = [r['stage'] for r in similar_rules]
            
            avg_prediction = np.mean(predictions)
            avg_confidence = np.mean(confidences)
            total_support = sum(supports)
            majority_stage = Counter(stages).most_common(1)[0][0]
            
            # Calculate diversity (how many different trees)
            tree_ids = set([r['tree_id'] for r in similar_rules])
            diversity = len(tree_ids) / self.n_estimators
            
            # Create consolidated rule
            if avg_confidence >= self.min_rule_confidence:
                ensemble_rule = EnsembleRule(
                    conditions=list(conditions_tuple),
                    prediction=avg_prediction,
                    stage=majority_stage,
                    ensemble_votes=n_votes,
                    avg_confidence=avg_confidence,
                    support=total_support,
                    diversity=diversity
                )
                consolidated_rules.append(ensemble_rule)
        
        # Sort by votes and confidence
        consolidated_rules.sort(key=lambda r: (r.ensemble_votes, r.avg_confidence), reverse=True)
        
        print(f"   ✅ Consolidated to {len(consolidated_rules)} high-quality rules")
        print(f"   (Min votes: {self.min_ensemble_votes}, Min confidence: {self.min_rule_confidence})")
        
        self.rules = consolidated_rules
    
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Extract rules using REFNE.
        
        Args:
            X: Input samples (n_samples, seq_len, n_features)
            y: Optional ground truth (not used, we query model)
        """
        print(f"\n{'='*80}")
        print(f"REFNE (Rule Extraction from Neural Network Ensemble)")
        print(f"{'='*80}")
        
        n_samples = X.shape[0]
        
        print(f"\n📊 Dataset Configuration:")
        print(f"   Real samples: {n_samples}")
        print(f"   Synthetic ratio: {self.synthetic_ratio}")
        print(f"   Sequence length: {X.shape[1]}")
        print(f"   Features: {X.shape[2]}")
        
        # Generate synthetic samples
        n_synthetic = int(n_samples * self.synthetic_ratio)
        print(f"\n🔄 Generating {n_synthetic} synthetic samples...")
        X_synthetic = self._generate_synthetic_samples(X, n_synthetic)
        
        # Combine real and synthetic
        X_combined = np.vstack([X, X_synthetic])
        
        # Query model
        print(f"\n🔄 Querying neural network for predictions...")
        y_combined = self._query_model(X_combined)
        
        print(f"\n📈 Prediction Statistics:")
        print(f"   Mean RUL: {np.mean(y_combined):.0f} cycles")
        print(f"   Std RUL: {np.std(y_combined):.0f} cycles")
        print(f"   Min RUL: {np.min(y_combined):.0f} cycles")
        print(f"   Max RUL: {np.max(y_combined):.0f} cycles")
        
        # Flatten sequences
        X_flat = X_combined[:, -1, :]
        
        # Build ensemble
        self._build_ensemble(X_flat, y_combined)
        
        # Extract rules from ensemble
        self._extract_rules_from_ensemble()
        
        # Consolidate rules
        self._consolidate_rules()
        
        print(f"\n✅ REFNE extraction complete!")
        print(f"   Ensemble trees: {self.n_estimators}")
        print(f"   Total tree rules: {len(self.tree_rules)}")
        print(f"   Consolidated rules: {len(self.rules)}")
        print(f"   Avg ensemble votes: {np.mean([r.ensemble_votes for r in self.rules]):.1f}")
        print(f"{'='*80}")
    
    
    def print_rules(
        self, 
        top_k: int = 15, 
        stage: Optional[str] = None,
        sort_by: Literal['votes', 'confidence', 'diversity'] = 'votes'
    ):
        """
        Print extracted rules.
        
        Args:
            top_k: Number of rules to print
            stage: Optional stage filter
            sort_by: Sorting criterion
        """
        rules = self.rules
        
        if stage:
            rules = [r for r in rules if r.stage == stage]
            title = f"TOP {top_k} RULES FOR {stage.upper()} STAGE"
        else:
            title = f"TOP {top_k} RULES (ALL STAGES)"
        
        # Sort rules
        if sort_by == 'votes':
            rules = sorted(rules, key=lambda r: r.ensemble_votes, reverse=True)
        elif sort_by == 'confidence':
            rules = sorted(rules, key=lambda r: r.avg_confidence, reverse=True)
        elif sort_by == 'diversity':
            rules = sorted(rules, key=lambda r: r.diversity, reverse=True)
        
        print(f"\n{'='*80}")
        print(title)
        print(f"{'='*80}\n")
        
        for i, rule in enumerate(rules[:top_k], 1):
            cond_str = ' AND '.join([f"{feat} {op} {val:.4f}" for feat, op, val in rule.conditions])
            print(f"Rule {i}:")
            print(f"  IF {cond_str}")
            print(f"  THEN RUL ≈ {rule.prediction:.0f} cycles (Stage: {rule.stage})")
            print(f"  Ensemble Support: {rule.ensemble_votes}/{self.n_estimators} trees ({rule.ensemble_votes/self.n_estimators*100:.0f}%)")
            print(f"  Confidence: {rule.avg_confidence:.3f} | Diversity: {rule.diversity:.3f} | Samples: {rule.support}")
            print()
    
    
    def get_rules_dataframe(self) -> pd.DataFrame:
        """Get rules as pandas DataFrame"""
        if not self.rules:
            return pd.DataFrame()
        
        rows = []
        for i, rule in enumerate(self.rules):
            cond_str = ' AND '.join([f"{feat} {op} {val:.4f}" for feat, op, val in rule.conditions])
            
            rows.append({
                'rule_id': i,
                'conditions': cond_str,
                'n_conditions': len(rule.conditions),
                'prediction': rule.prediction,
                'stage': rule.stage,
                'ensemble_votes': rule.ensemble_votes,
                'vote_percentage': rule.ensemble_votes / self.n_estimators * 100,
                'confidence': rule.avg_confidence,
                'diversity': rule.diversity,
                'support': rule.support
            })
        
        return pd.DataFrame(rows)
    
    
    def get_feature_importance(self, top_k: int = 10) -> Dict[str, float]:
        """Get feature importance from ensemble"""
        if self.feature_importance_ is None:
            return {}
        
        sorted_features = sorted(self.feature_importance_.items(), 
                                key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:top_k])
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble"""
        if self.ensemble is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        X_flat = X[:, -1, :]
        return self.ensemble.predict(X_flat)
    
    
    def evaluate_fidelity(self, X_test: np.ndarray) -> Dict[str, float]:
        """Evaluate how well ensemble approximates neural network"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Neural network predictions
        nn_predictions = self._query_model(X_test)
        
        # Ensemble predictions
        ensemble_predictions = self.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(nn_predictions, ensemble_predictions)
        rmse = np.sqrt(mean_squared_error(nn_predictions, ensemble_predictions))
        r2 = r2_score(nn_predictions, ensemble_predictions)
        
        # Agreement (within 10%)
        agreement = np.mean(np.abs(nn_predictions - ensemble_predictions) / (np.abs(nn_predictions) + 1e-10) < 0.1)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'agreement': agreement
        }
    
    
    def visualize_ensemble_statistics(self, save_path: Optional[str] = None):
        """Visualize ensemble and rule statistics"""
        rules_df = self.get_rules_dataframe()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Ensemble votes distribution
        axes[0, 0].hist(rules_df['ensemble_votes'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(self.min_ensemble_votes, color='red', linestyle='--', linewidth=2, label=f'Min votes: {self.min_ensemble_votes}')
        axes[0, 0].set_xlabel('Ensemble Votes', fontsize=11)
        axes[0, 0].set_ylabel('Number of Rules', fontsize=11)
        axes[0, 0].set_title('Ensemble Vote Distribution', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3, axis='y')
        
        # 2. Confidence distribution
        axes[0, 1].hist(rules_df['confidence'], bins=20, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(self.min_rule_confidence, color='red', linestyle='--', linewidth=2, label=f'Min conf: {self.min_rule_confidence}')
        axes[0, 1].set_xlabel('Confidence', fontsize=11)
        axes[0, 1].set_ylabel('Number of Rules', fontsize=11)
        axes[0, 1].set_title('Confidence Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3, axis='y')
        
        # 3. Rules per stage
        stage_counts = rules_df['stage'].value_counts()
        axes[0, 2].bar(stage_counts.index, stage_counts.values, color='green', alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Stage', fontsize=11)
        axes[0, 2].set_ylabel('Number of Rules', fontsize=11)
        axes[0, 2].set_title('Rules per Stage', fontsize=13, fontweight='bold')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(alpha=0.3, axis='y')
        
        # 4. Votes vs Confidence scatter
        scatter = axes[1, 0].scatter(rules_df['ensemble_votes'], rules_df['confidence'], 
                                     c=rules_df['diversity'], cmap='viridis', s=100, alpha=0.6)
        axes[1, 0].set_xlabel('Ensemble Votes', fontsize=11)
        axes[1, 0].set_ylabel('Confidence', fontsize=11)
        axes[1, 0].set_title('Votes vs Confidence (color=diversity)', fontsize=13, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Diversity')
        
        # 5. Rule complexity
        axes[1, 1].hist(rules_df['n_conditions'], bins=range(1, rules_df['n_conditions'].max() + 2), 
                        color='purple', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Number of Conditions', fontsize=11)
        axes[1, 1].set_ylabel('Number of Rules', fontsize=11)
        axes[1, 1].set_title('Rule Complexity Distribution', fontsize=13, fontweight='bold')
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        # 6. Feature importance (if available)
        if self.feature_importance_:
            top_features = self.get_feature_importance(top_k=10)
            axes[1, 2].barh(list(top_features.keys()), list(top_features.values()), color='teal', alpha=0.7)
            axes[1, 2].set_xlabel('Importance', fontsize=11)
            axes[1, 2].set_title('Top 10 Feature Importance', fontsize=13, fontweight='bold')
            axes[1, 2].grid(alpha=0.3, axis='x')
        else:
            axes[1, 2].text(0.5, 0.5, 'Feature importance\nnot available', 
                           ha='center', va='center', fontsize=12)
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to: {save_path}")
        
        plt.show()