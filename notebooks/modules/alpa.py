import numpy as np
import torch
from typing import List, Dict, Tuple, Set, Optional
import pandas as pd
from dataclasses import dataclass
from itertools import combinations, product
import re


@dataclass
class LogicRule:
    """Represents a first-order logic rule"""
    head: str  # Consequent (e.g., "high_rul")
    body: List[Tuple[str, str, float]]  # List of (feature, operator, threshold)
    confidence: float  # Rule confidence [0, 1]
    support: int  # Number of samples covered
    precision: float  # Precision on covered samples
    lift: float  # Lift compared to baseline
    
    def __str__(self):
        body_str = ' ∧ '.join([f"{feat} {op} {val:.4f}" for feat, op, val in self.body])
        return f"{self.head} ← {body_str} (conf={self.confidence:.3f}, supp={self.support})"
    
    def __repr__(self):
        return self.__str__()


class ALPA:
    """
    Adaptive Logic Programming Algorithm for rule extraction.
    
    Extracts first-order logic rules from neural network predictions
    using inductive logic programming principles.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        feature_names: List[str],
        target_scaler=None,
        min_confidence: float = 0.7,
        min_support: int = 50,
        min_precision: float = 0.6,
        max_rule_length: int = 5,
        beam_width: int = 10,
        n_quantiles: int = 5,
        stage_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
        random_state: int = 42
    ):
        """
        Initialize ALPA rule extractor.
        
        Args:
            model: Trained PyTorch model
            device: torch.device
            feature_names: List of feature names
            target_scaler: Scaler for inverse transforming predictions
            min_confidence: Minimum rule confidence threshold
            min_support: Minimum number of samples a rule must cover
            min_precision: Minimum precision threshold
            max_rule_length: Maximum number of conditions in a rule
            beam_width: Beam search width for rule generation
            n_quantiles: Number of quantiles for discretization
            stage_thresholds: Optional RUL thresholds for stages
            random_state: Random seed
        """
        self.model = model
        self.device = device
        self.feature_names = feature_names
        self.target_scaler = target_scaler
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.min_precision = min_precision
        self.max_rule_length = max_rule_length
        self.beam_width = beam_width
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        
        # Define stage thresholds (default CFRP stages)
        if stage_thresholds is None:
            self.stage_thresholds = {
                'critical': (0, 60000),
                'progressive': (60000, 120000),
                'early_damage': (120000, 180000),
                'healthy': (180000, 230000)
            }
        else:
            self.stage_thresholds = stage_thresholds
        
        self.rules = []
        self.discretization_thresholds = {}
        
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
    
    
    def _discretize_features(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Discretize continuous features using quantile-based binning.
        
        Args:
            X: Input samples (n_samples, seq_len, n_features)
        
        Returns:
            Discretized features and threshold dictionary
        """
        X_flat = X[:, -1, :]  # Use last timestep
        n_samples, n_features = X_flat.shape
        
        thresholds = {}
        
        for feat_idx in range(n_features):
            feat_values = X_flat[:, feat_idx]
            
            # Calculate quantiles
            quantiles = np.linspace(0, 1, self.n_quantiles + 1)[1:-1]  # Exclude 0 and 1
            thresholds[feat_idx] = np.quantile(feat_values, quantiles)
        
        return thresholds
    
    
    def _assign_stage_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Assign degradation stage labels based on RUL values.
        
        Args:
            y: RUL values (in actual cycles)
        
        Returns:
            Stage labels
        """
        labels = np.empty(len(y), dtype=object)
        
        for stage_name, (min_rul, max_rul) in self.stage_thresholds.items():
            mask = (y >= min_rul) & (y < max_rul)
            labels[mask] = stage_name
        
        return labels
    
    
    def _generate_candidate_conditions(
        self, 
        X: np.ndarray, 
        feature_idx: int
    ) -> List[Tuple[str, str, float]]:
        """
        Generate candidate conditions for a feature.
        
        Args:
            X: Input samples (flattened, last timestep)
            feature_idx: Feature index
        
        Returns:
            List of (feature_name, operator, threshold) tuples
        """
        feature_name = self.feature_names[feature_idx]
        thresholds = self.discretization_thresholds[feature_idx]
        
        conditions = []
        
        # Generate conditions for each threshold
        for threshold in thresholds:
            conditions.append((feature_name, '<=', threshold))
            conditions.append((feature_name, '>', threshold))
        
        return conditions
    
    
    def _evaluate_rule(
        self, 
        conditions: List[Tuple[str, str, float]], 
        X_flat: np.ndarray, 
        y_labels: np.ndarray,
        target_class: str
    ) -> Dict[str, float]:
        """
        Evaluate a candidate rule.
        
        Args:
            conditions: List of conditions
            X_flat: Flattened input samples
            y_labels: Stage labels
            target_class: Target stage to predict
        
        Returns:
            Evaluation metrics
        """
        # Apply conditions to get covered samples
        mask = np.ones(len(X_flat), dtype=bool)
        
        for feat_name, op, threshold in conditions:
            feat_idx = self.feature_names.index(feat_name)
            feat_values = X_flat[:, feat_idx]
            
            if op == '<=':
                mask &= (feat_values <= threshold)
            elif op == '>':
                mask &= (feat_values > threshold)
            elif op == '<':
                mask &= (feat_values < threshold)
            elif op == '>=':
                mask &= (feat_values >= threshold)
        
        support = np.sum(mask)
        
        if support == 0:
            return {'support': 0, 'precision': 0, 'confidence': 0, 'lift': 0}
        
        # Calculate precision (how many covered samples are target class)
        covered_labels = y_labels[mask]
        precision = np.mean(covered_labels == target_class)
        
        # Calculate confidence (precision weighted by support)
        confidence = precision * (support / len(X_flat))
        
        # Calculate lift (how much better than baseline)
        baseline = np.mean(y_labels == target_class)
        lift = precision / baseline if baseline > 0 else 0
        
        return {
            'support': support,
            'precision': precision,
            'confidence': confidence,
            'lift': lift
        }
    
    
    def _beam_search_rules(
        self,
        X_flat: np.ndarray,
        y_labels: np.ndarray,
        target_class: str
    ) -> List[LogicRule]:
        """
        Use beam search to find high-quality rules.
        
        Args:
            X_flat: Flattened input samples
            y_labels: Stage labels
            target_class: Target class to predict
        
        Returns:
            List of extracted rules
        """
        # Initialize beam with single-condition rules
        beam = []
        
        for feat_idx in range(len(self.feature_names)):
            candidate_conditions = self._generate_candidate_conditions(X_flat, feat_idx)
            
            for condition in candidate_conditions:
                metrics = self._evaluate_rule([condition], X_flat, y_labels, target_class)
                
                if metrics['support'] >= self.min_support:
                    beam.append({
                        'conditions': [condition],
                        'metrics': metrics
                    })
        
        # Sort beam by confidence
        beam = sorted(beam, key=lambda x: x['metrics']['confidence'], reverse=True)[:self.beam_width]
        
        # Expand beam iteratively
        final_rules = []
        
        for rule_length in range(2, self.max_rule_length + 1):
            new_beam = []
            
            for rule_candidate in beam:
                current_conditions = rule_candidate['conditions']
                
                # Try adding each possible condition
                for feat_idx in range(len(self.feature_names)):
                    additional_conditions = self._generate_candidate_conditions(X_flat, feat_idx)
                    
                    for new_condition in additional_conditions:
                        # Avoid duplicate features
                        if any(cond[0] == new_condition[0] for cond in current_conditions):
                            continue
                        
                        expanded_conditions = current_conditions + [new_condition]
                        metrics = self._evaluate_rule(expanded_conditions, X_flat, y_labels, target_class)
                        
                        # Only keep if improvement
                        if (metrics['support'] >= self.min_support and 
                            metrics['precision'] >= self.min_precision and
                            metrics['confidence'] >= self.min_confidence):
                            
                            new_beam.append({
                                'conditions': expanded_conditions,
                                'metrics': metrics
                            })
            
            if not new_beam:
                break
            
            # Keep top beam_width rules
            new_beam = sorted(new_beam, key=lambda x: x['metrics']['confidence'], reverse=True)[:self.beam_width]
            beam = new_beam
        
        # Convert beam to LogicRule objects
        for rule_candidate in beam:
            final_rules.append(LogicRule(
                head=target_class,
                body=rule_candidate['conditions'],
                confidence=rule_candidate['metrics']['confidence'],
                support=rule_candidate['metrics']['support'],
                precision=rule_candidate['metrics']['precision'],
                lift=rule_candidate['metrics']['lift']
            ))
        
        return final_rules
    
    
    def _remove_redundant_rules(self, rules: List[LogicRule]) -> List[LogicRule]:
        """
        Remove redundant and subsumed rules.
        
        Args:
            rules: List of rules
        
        Returns:
            Pruned rule list
        """
        # Sort by confidence
        rules = sorted(rules, key=lambda r: r.confidence, reverse=True)
        
        pruned_rules = []
        
        for rule in rules:
            # Check if rule is subsumed by existing rules
            is_redundant = False
            
            for existing_rule in pruned_rules:
                # If existing rule has same head and is more general (fewer conditions)
                if (existing_rule.head == rule.head and
                    len(existing_rule.body) <= len(rule.body) and
                    set(existing_rule.body).issubset(set(rule.body))):
                    is_redundant = True
                    break
            
            if not is_redundant:
                pruned_rules.append(rule)
        
        return pruned_rules
    
    
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Extract logic rules from the model.
        
        Args:
            X: Input samples (n_samples, seq_len, n_features)
            y: Optional ground truth (not used, we query model)
        """
        print(f"\n{'='*80}")
        print(f"ALPA LOGIC RULE EXTRACTION")
        print(f"{'='*80}")
        
        n_samples = X.shape[0]
        
        print(f"\n📊 Dataset Configuration:")
        print(f"   Samples: {n_samples}")
        print(f"   Sequence length: {X.shape[1]}")
        print(f"   Features: {X.shape[2]}")
        
        # Discretize features
        print(f"\n🔄 Discretizing features into {self.n_quantiles} bins...")
        self.discretization_thresholds = self._discretize_features(X)
        
        print(f"   Discretization complete. Thresholds:")
        for feat_idx, thresholds in list(self.discretization_thresholds.items())[:5]:
            print(f"      {self.feature_names[feat_idx]}: {thresholds}")
        if len(self.discretization_thresholds) > 5:
            print(f"      ... ({len(self.discretization_thresholds) - 5} more features)")
        
        # Query model predictions
        print(f"\n🔄 Querying model for predictions...")
        y_pred = self._query_model(X)
        
        # Assign stage labels
        y_labels = self._assign_stage_labels(y_pred)
        
        print(f"\n🎯 Stage Distribution:")
        for stage_name in self.stage_thresholds.keys():
            count = np.sum(y_labels == stage_name)
            pct = count / len(y_labels) * 100
            print(f"   {stage_name:20s}: {count:4d} samples ({pct:5.1f}%)")
        
        # Flatten sequences
        X_flat = X[:, -1, :]
        
        # Extract rules for each stage
        print(f"\n🔍 Extracting rules using beam search...")
        print(f"   Min confidence: {self.min_confidence}")
        print(f"   Min support: {self.min_support}")
        print(f"   Min precision: {self.min_precision}")
        print(f"   Beam width: {self.beam_width}")
        
        all_rules = []
        
        for stage_name in self.stage_thresholds.keys():
            print(f"\n   🎯 Stage: {stage_name}...")
            stage_rules = self._beam_search_rules(X_flat, y_labels, stage_name)
            print(f"      Found {len(stage_rules)} candidate rules")
            all_rules.extend(stage_rules)
        
        # Remove redundant rules
        print(f"\n🔄 Pruning redundant rules...")
        self.rules = self._remove_redundant_rules(all_rules)
        
        print(f"\n✅ Extraction complete!")
        print(f"   Total rules: {len(self.rules)}")
        print(f"   Avg rule length: {np.mean([len(r.body) for r in self.rules]):.1f} conditions")
        print(f"   Avg confidence: {np.mean([r.confidence for r in self.rules]):.3f}")
        print(f"{'='*80}")
    
    
    def get_rules_by_stage(self, stage: str) -> List[LogicRule]:
        """Get rules for a specific degradation stage"""
        return [r for r in self.rules if r.head == stage]
    
    
    def print_rules(
        self, 
        stage: Optional[str] = None, 
        top_k: int = 10,
        sort_by: str = 'confidence'
    ):
        """
        Print extracted rules.
        
        Args:
            stage: Optional stage filter
            top_k: Number of rules to print
            sort_by: Sort criterion ('confidence', 'support', 'precision', 'lift')
        """
        if stage:
            rules = self.get_rules_by_stage(stage)
            title = f"TOP {top_k} RULES FOR {stage.upper()}"
        else:
            rules = self.rules
            title = f"TOP {top_k} RULES (ALL STAGES)"
        
        # Sort rules
        if sort_by == 'confidence':
            rules = sorted(rules, key=lambda r: r.confidence, reverse=True)
        elif sort_by == 'support':
            rules = sorted(rules, key=lambda r: r.support, reverse=True)
        elif sort_by == 'precision':
            rules = sorted(rules, key=lambda r: r.precision, reverse=True)
        elif sort_by == 'lift':
            rules = sorted(rules, key=lambda r: r.lift, reverse=True)
        
        print(f"\n{'='*80}")
        print(title)
        print(f"{'='*80}\n")
        
        for i, rule in enumerate(rules[:top_k], 1):
            print(f"Rule {i}:")
            print(f"  IF {' AND '.join([f'{feat} {op} {val:.4f}' for feat, op, val in rule.body])}")
            print(f"  THEN degradation_stage = {rule.head}")
            print(f"  Confidence: {rule.confidence:.3f} | Support: {rule.support} | Precision: {rule.precision:.3f} | Lift: {rule.lift:.2f}")
            print()
    
    
    def get_rules_dataframe(self) -> pd.DataFrame:
        """Get rules as pandas DataFrame"""
        if not self.rules:
            return pd.DataFrame()
        
        rows = []
        for i, rule in enumerate(self.rules):
            body_str = ' ∧ '.join([f"{feat} {op} {val:.4f}" for feat, op, val in rule.body])
            
            rows.append({
                'rule_id': i,
                'stage': rule.head,
                'conditions': body_str,
                'n_conditions': len(rule.body),
                'confidence': rule.confidence,
                'support': rule.support,
                'precision': rule.precision,
                'lift': rule.lift
            })
        
        return pd.DataFrame(rows)
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict degradation stage using extracted rules.
        
        Args:
            X: Input samples
        
        Returns:
            Predicted stages
        """
        X_flat = X[:, -1, :]
        predictions = []
        
        for sample in X_flat:
            # Find matching rules
            matching_rules = []
            
            for rule in self.rules:
                matches = True
                for feat_name, op, threshold in rule.body:
                    feat_idx = self.feature_names.index(feat_name)
                    feat_value = sample[feat_idx]
                    
                    if op == '<=' and feat_value > threshold:
                        matches = False
                        break
                    elif op == '>' and feat_value <= threshold:
                        matches = False
                        break
                
                if matches:
                    matching_rules.append(rule)
            
            # Use highest confidence rule
            if matching_rules:
                best_rule = max(matching_rules, key=lambda r: r.confidence)
                predictions.append(best_rule.head)
            else:
                predictions.append('unknown')
        
        return np.array(predictions)
    
    
    def evaluate_fidelity(self, X: np.ndarray) -> Dict[str, float]:
        """
        Evaluate how well rules match neural network predictions.
        
        Args:
            X: Test samples
        
        Returns:
            Fidelity metrics
        """
        # Get neural network predictions
        nn_predictions = self._query_model(X)
        nn_labels = self._assign_stage_labels(nn_predictions)
        
        # Get rule predictions
        rule_predictions = self.predict(X)
        
        # Calculate agreement
        agreement = np.mean(nn_labels == rule_predictions)
        
        # Per-stage accuracy
        stage_accuracies = {}
        for stage in self.stage_thresholds.keys():
            mask = nn_labels == stage
            if np.sum(mask) > 0:
                stage_accuracies[stage] = np.mean(rule_predictions[mask] == stage)
        
        return {
            'agreement': agreement,
            'stage_accuracies': stage_accuracies,
            'unknown_rate': np.mean(rule_predictions == 'unknown')
        }