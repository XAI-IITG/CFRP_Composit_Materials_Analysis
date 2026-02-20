import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import pandas as pd


class KnowledgeTrepan:
    """
    Knowledge Trepan algorithm for rule extraction from neural networks.
    
    Builds decision tree by querying trained model on original + synthetic samples.
    Extracts interpretable IF-THEN rules from the tree.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        feature_names: List[str],
        target_scaler=None,
        max_depth: int = 5,
        min_samples_split: int = 50,
        min_samples_leaf: int = 20,
        sample_size: int = 1000,
        synthetic_ratio: float = 0.3,
        random_state: int = 42
    ):
        """
        Initialize Knowledge Trepan extractor.
        
        Args:
            model: Trained PyTorch model
            device: torch.device ('cpu' or 'cuda')
            feature_names: List of feature names
            target_scaler: Scaler for inverse transforming predictions
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            sample_size: Number of samples per node expansion
            synthetic_ratio: Ratio of synthetic to real samples
            random_state: Random seed
        """
        self.model = model
        self.device = device
        self.feature_names = feature_names
        self.target_scaler = target_scaler
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.sample_size = sample_size
        self.synthetic_ratio = synthetic_ratio
        self.random_state = random_state
        
        self.tree = None
        self.rules = []
        self.feature_importance_ = None
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    
    def _generate_synthetic_samples(
        self, 
        X_real: np.ndarray, 
        n_synthetic: int,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Generate synthetic samples by perturbing real samples.
        
        Args:
            X_real: Real samples (n_samples, seq_len, n_features)
            n_synthetic: Number of synthetic samples to generate
            constraints: Optional feature constraints (min/max values)
        
        Returns:
            Synthetic samples with same shape as X_real
        """
        n_samples, seq_len, n_features = X_real.shape
        
        # Strategy 1: Gaussian noise perturbation (70%)
        n_gaussian = int(n_synthetic * 0.7)
        base_samples = X_real[np.random.choice(n_samples, n_gaussian)]
        noise_std = np.std(X_real, axis=0) * 0.1  # 10% of std
        synthetic_gaussian = base_samples + np.random.normal(0, noise_std, base_samples.shape)
        
        # Strategy 2: Feature swapping (20%)
        n_swap = int(n_synthetic * 0.2)
        synthetic_swap = []
        for _ in range(n_swap):
            sample = X_real[np.random.choice(n_samples)].copy()
            # Swap 30% of features with another sample
            n_swap_features = max(1, int(n_features * 0.3))
            swap_features = np.random.choice(n_features, n_swap_features, replace=False)
            donor = X_real[np.random.choice(n_samples)]
            sample[:, swap_features] = donor[:, swap_features]
            synthetic_swap.append(sample)
        synthetic_swap = np.array(synthetic_swap)
        
        # Strategy 3: Interpolation (10%)
        n_interp = n_synthetic - n_gaussian - n_swap
        synthetic_interp = []
        for _ in range(n_interp):
            idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
            alpha = np.random.uniform(0.2, 0.8)
            interp_sample = alpha * X_real[idx1] + (1 - alpha) * X_real[idx2]
            synthetic_interp.append(interp_sample)
        synthetic_interp = np.array(synthetic_interp) if synthetic_interp else np.array([]).reshape(0, seq_len, n_features)
        
        # Combine all synthetic samples
        X_synthetic = np.vstack([synthetic_gaussian, synthetic_swap, synthetic_interp])
        
        # Apply constraints if provided
        if constraints:
            for feat_idx, (min_val, max_val) in constraints.items():
                X_synthetic[:, :, feat_idx] = np.clip(X_synthetic[:, :, feat_idx], min_val, max_val)
        
        return X_synthetic
    
    
    def _query_model(self, X: np.ndarray) -> np.ndarray:
        """
        Query model predictions for samples.
        
        Args:
            X: Input samples (n_samples, seq_len, n_features)
        
        Returns:
            Predictions in actual scale (not normalized)
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform if scaler provided
        if self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    
    def fit(self, X: np.ndarray, y: np.ndarray = None, stage_labels: np.ndarray = None):
        """
        Fit Knowledge Trepan tree by querying the model.
        
        Args:
            X: Real training samples (n_samples, seq_len, n_features)
            y: Optional ground truth labels (not used, we query model)
            stage_labels: Optional degradation stage labels
        """
        print(f"\n{'='*80}")
        print(f"KNOWLEDGE TREPAN RULE EXTRACTION")
        print(f"{'='*80}")
        
        n_samples = X.shape[0]
        n_synthetic = int(n_samples * self.synthetic_ratio)
        
        print(f"\n📊 Dataset Configuration:")
        print(f"   Real samples: {n_samples}")
        print(f"   Synthetic samples: {n_synthetic}")
        print(f"   Total samples: {n_samples + n_synthetic}")
        print(f"   Sequence length: {X.shape[1]}")
        print(f"   Features: {X.shape[2]}")
        
        # Generate synthetic samples
        print(f"\n🔄 Generating synthetic samples...")
        X_synthetic = self._generate_synthetic_samples(X, n_synthetic)
        
        # Combine real and synthetic
        X_combined = np.vstack([X, X_synthetic])
        
        # Query model for predictions
        print(f"🔄 Querying model for predictions...")
        y_combined = self._query_model(X_combined)
        
        print(f"\n📈 Prediction Statistics:")
        print(f"   Mean RUL: {np.mean(y_combined):.0f} cycles")
        print(f"   Std RUL: {np.std(y_combined):.0f} cycles")
        print(f"   Min RUL: {np.min(y_combined):.0f} cycles")
        print(f"   Max RUL: {np.max(y_combined):.0f} cycles")
        
        # Flatten sequences to use last timestep features
        # (Alternatively, could use sequence aggregation)
        X_flat = X_combined[:, -1, :]  # Use last timestep
        
        print(f"\n🌳 Building decision tree...")
        print(f"   Max depth: {self.max_depth}")
        print(f"   Min samples split: {self.min_samples_split}")
        print(f"   Min samples leaf: {self.min_samples_leaf}")
        
        # Build decision tree
        self.tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        
        self.tree.fit(X_flat, y_combined)
        
        # Calculate feature importance
        self.feature_importance_ = dict(zip(self.feature_names, self.tree.feature_importances_))
        
        # Extract rules
        print(f"\n📋 Extracting rules from tree...")
        self.rules = self._extract_rules_from_tree()
        
        print(f"\n✅ Extraction complete!")
        print(f"   Total rules extracted: {len(self.rules)}")
        print(f"   Tree nodes: {self.tree.tree_.node_count}")
        print(f"   Tree leaves: {self.tree.get_n_leaves()}")
        print(f"{'='*80}")
    
    
    def _extract_rules_from_tree(self) -> List[Dict]:
        """
        Extract IF-THEN rules from fitted decision tree.
        
        Returns:
            List of rule dictionaries
        """
        tree = self.tree.tree_
        feature_names = self.feature_names
        
        rules = []
        
        def recurse(node, conditions, path_nodes):
            """Recursively traverse tree and extract rules"""
            
            # Leaf node - create rule
            if tree.feature[node] == _tree.TREE_UNDEFINED:
                rule = {
                    'conditions': conditions.copy(),
                    'prediction': tree.value[node][0][0],
                    'n_samples': tree.n_node_samples[node],
                    'impurity': tree.impurity[node],
                    'path_length': len(conditions)
                }
                rules.append(rule)
                return
            
            # Internal node - split
            feature_idx = tree.feature[node]
            threshold = tree.threshold[node]
            feature_name = feature_names[feature_idx]
            
            # Left child (feature <= threshold)
            left_conditions = conditions + [(feature_name, '<=', threshold)]
            recurse(tree.children_left[node], left_conditions, path_nodes + [node])
            
            # Right child (feature > threshold)
            right_conditions = conditions + [(feature_name, '>', threshold)]
            recurse(tree.children_right[node], right_conditions, path_nodes + [node])
        
        # Start recursion from root
        recurse(0, [], [])
        
        return rules
    
    
    def get_rules_dataframe(self, sort_by: str = 'n_samples') -> pd.DataFrame:
        """
        Get rules as pandas DataFrame.
        
        Args:
            sort_by: Column to sort by ('n_samples', 'prediction', 'impurity')
        
        Returns:
            DataFrame with rule information
        """
        if not self.rules:
            return pd.DataFrame()
        
        rows = []
        for i, rule in enumerate(self.rules):
            # Format conditions as string
            cond_str = ' AND '.join([f"{name} {op} {val:.4f}" for name, op, val in rule['conditions']])
            
            rows.append({
                'rule_id': i,
                'conditions': cond_str,
                'prediction': rule['prediction'],
                'n_samples': rule['n_samples'],
                'coverage': rule['n_samples'] / sum(r['n_samples'] for r in self.rules),
                'impurity': rule['impurity'],
                'complexity': rule['path_length']
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
        
        return df
    
    
    def print_rules(self, top_k: int = 10, min_coverage: float = 0.01):
        """
        Print extracted rules in human-readable format.
        
        Args:
            top_k: Number of top rules to print
            min_coverage: Minimum coverage threshold
        """
        df = self.get_rules_dataframe()
        df_filtered = df[df['coverage'] >= min_coverage].head(top_k)
        
        print(f"\n{'='*80}")
        print(f"TOP {min(top_k, len(df_filtered))} RULES (Coverage >= {min_coverage*100:.1f}%)")
        print(f"{'='*80}\n")
        
        for idx, row in df_filtered.iterrows():
            print(f"Rule {row['rule_id'] + 1}:")
            print(f"  IF {row['conditions']}")
            print(f"  THEN RUL ≈ {row['prediction']:.0f} cycles")
            print(f"  Coverage: {row['coverage']*100:.1f}% ({row['n_samples']} samples)")
            print(f"  Complexity: {row['complexity']} conditions")
            print(f"  Impurity: {row['impurity']:.4f}")
            print()
    
    
    def visualize_tree(self, max_depth: int = 3, save_path: Optional[str] = None):
        """
        Visualize the decision tree.
        
        Args:
            max_depth: Maximum depth to display
            save_path: Optional path to save figure
        """
        from sklearn.tree import plot_tree
        
        fig, ax = plt.subplots(figsize=(20, 10))
        
        plot_tree(
            self.tree,
            feature_names=self.feature_names,
            filled=True,
            rounded=True,
            fontsize=10,
            max_depth=max_depth,
            ax=ax
        )
        
        ax.set_title(f'Knowledge Trepan Decision Tree (max depth={max_depth})', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Tree visualization saved to: {save_path}")
        
        plt.show()
    
    
    def get_feature_importance(self, top_k: int = 10) -> Dict[str, float]:
        """
        Get feature importance from the tree.
        
        Args:
            top_k: Number of top features to return
        
        Returns:
            Dictionary of feature importances
        """
        if self.feature_importance_ is None:
            return {}
        
        sorted_features = sorted(self.feature_importance_.items(), 
                                key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:top_k])
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the extracted tree.
        
        Args:
            X: Input samples (n_samples, seq_len, n_features)
        
        Returns:
            Predictions
        """
        if self.tree is None:
            raise ValueError("Tree not fitted. Call fit() first.")
        
        X_flat = X[:, -1, :]  # Use last timestep
        return self.tree.predict(X_flat)
    
    
    def evaluate_fidelity(self, X_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate how well the tree approximates the neural network.
        
        Args:
            X_test: Test samples
        
        Returns:
            Dictionary of fidelity metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Get neural network predictions
        nn_predictions = self._query_model(X_test)
        
        # Get tree predictions
        tree_predictions = self.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(nn_predictions, tree_predictions)
        rmse = np.sqrt(mean_squared_error(nn_predictions, tree_predictions))
        r2 = r2_score(nn_predictions, tree_predictions)
        
        # Agreement (predictions within 10% of each other)
        agreement = np.mean(np.abs(nn_predictions - tree_predictions) / (np.abs(nn_predictions) + 1e-10) < 0.1)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'agreement': agreement
        }