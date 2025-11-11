"""
XAI Explainer using gradient-based attribution and Knowledge Graph
Supports Transformer, LSTM, and DRL models (PPO, DDPG, DQN)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict
from .kgbuilder import CFRPKnowledgeGraph


class XAIExplainer:
    """Explainable AI with Knowledge Graph"""
    
    def __init__(self, kg: CFRPKnowledgeGraph, device):
        """
        Args:
            kg: Knowledge graph instance
            device: torch device (cpu or cuda)
        """
        self.kg = kg
        self.device = device
    
    def explain_prediction(self, model, sample, feature_names, model_type='standard', 
                          top_k=5, target_scaler=None):
        """Generate comprehensive explanation for a prediction
        
        Args:
            model: The trained model (Transformer, LSTM, or DRL agent)
            sample: Input sample (already batched, shape: [1, seq_len, features])
            feature_names: List of feature names
            model_type: 'transformer', 'lstm', 'ddpg', 'ppo', 'dqn'
            top_k: Number of top features to return
            target_scaler: sklearn scaler to inverse transform predictions (REQUIRED!)
            
        Returns:
            Dictionary with explanation results
        """
        
        if target_scaler is None:
            raise ValueError("❌ target_scaler is required to convert normalized predictions to actual RUL!")
        
        # Convert to tensor if needed
        if not isinstance(sample, torch.Tensor):
            sample_tensor = torch.FloatTensor(sample).to(self.device)
        else:
            sample_tensor = sample.to(self.device)
        
        # 1. Get model prediction based on model type
        model.eval()
        with torch.no_grad():
            pred_normalized = model(sample_tensor)
            pred_normalized_value = pred_normalized.cpu().item()
            
            # ⚠️ CRITICAL: Inverse transform to get actual RUL in cycles
            pred_rul = target_scaler.inverse_transform([[pred_normalized_value]])[0, 0]
        
        # 2. Compute gradient-based feature importance
        try:
            sample_tensor_grad = sample_tensor.clone().detach().requires_grad_(True)
            
            output = model(sample_tensor_grad)
            output.backward()
            
            # Get gradient magnitude as feature importance
            # Shape: (batch=1, seq_len, features)
            if sample_tensor_grad.grad is not None:
                # Average over time dimension (dim=1) to get per-feature importance
                gradients = sample_tensor_grad.grad.abs().mean(dim=1).squeeze()  # Shape: [n_features]
                
                # If gradients are all zero or near-zero (vanishing gradient problem)
                if gradients.max() < 1e-10:
                    print("⚠️  Warning: Vanishing gradients detected. Using input saliency instead.")
                    # Use input magnitude weighted by output sensitivity
                    input_magnitude = sample_tensor.abs().mean(dim=1).squeeze()
                    feature_importance_values = input_magnitude.cpu().numpy()
                else:
                    feature_importance_values = gradients.cpu().numpy()
            else:
                raise ValueError("Gradients are None")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not compute gradients: {e}")
            # Fallback: Use mean absolute values as importance (input magnitude)
            feature_importance_values = np.abs(sample_tensor.cpu().numpy()).mean(axis=1).squeeze()
        
        # Normalize feature importance to sum to 1 for better interpretability
        total_importance = np.sum(np.abs(feature_importance_values))
        if total_importance > 1e-10:
            feature_importance_values = feature_importance_values / total_importance
        else:
            print("⚠️  Warning: All feature importances are zero!")
        
        # Ensure we have the right number of features
        if len(feature_importance_values) != len(feature_names):
            print(f"⚠️  Feature mismatch: {len(feature_importance_values)} importance values vs {len(feature_names)} names")
            # Pad or truncate to match
            if len(feature_importance_values) < len(feature_names):
                feature_importance_values = np.pad(
                    feature_importance_values, 
                    (0, len(feature_names) - len(feature_importance_values))
                )
            else:
                feature_importance_values = feature_importance_values[:len(feature_names)]
        
        # 3. Get feature importance dictionary
        feature_importance = dict(zip(feature_names, feature_importance_values))
        
        # 4. Get top-k features sorted by importance
        top_features = sorted(feature_importance.items(), 
                             key=lambda x: abs(x[1]), reverse=True)[:top_k]
        
        # 5. Generate KG-based explanation paths
        paths = []
        for feature_name, importance in top_features:
            # Find paths from this feature through the knowledge graph
            if self.kg.graph.has_node(feature_name):
                # Get all paths from feature to degradation stages
                stage_nodes = [n for n, d in self.kg.graph.nodes(data=True) 
                              if d.get('type') == 'stage']
                
                for stage in stage_nodes:
                    try:
                        import networkx as nx
                        for path in nx.all_simple_paths(self.kg.graph, feature_name, stage, cutoff=3):
                            if len(path) > 1:  # Must have at least feature -> phenomenon
                                path_strength = importance * 0.8  # Weight by feature importance
                                path_info = {
                                    'nodes': [{'type': self.kg.graph.nodes[n].get('type', 'unknown'),
                                             'name': n} for n in path],
                                    'strength': float(path_strength)
                                }
                                paths.append(path_info)
                    except:
                        pass  # No path found
        
        # Sort paths by strength
        paths = sorted(paths, key=lambda x: x['strength'], reverse=True)[:10]
        
        # 6. Create comprehensive explanation
        explanation = {
            'predicted_rul': float(pred_rul),  # In actual cycles (inverse transformed)
            'predicted_rul_normalized': float(pred_normalized_value),  # Keep normalized for reference
            'actual_rul': None,  # Not provided in this API
            'feature_importances': top_features,
            'feature_importance': feature_importance,
            'paths': paths,
            'model_type': model_type
        }
        
        return explanation
    
    def visualize_explanation(self, explanation):
        """Visualize the explanation"""
        
        print("="*60)
        print("🔍 PREDICTION EXPLANATION")
        print("="*60)
        print(f"\n📊 Prediction: {explanation['predicted_rul']:.0f} cycles")
        print(f"   Actual: {explanation['actual_rul']:.0f} cycles")
        print(f"   Error: {explanation['error']:.0f} cycles ({explanation['error_percent']:.1f}%)")
        
        print(f"\n🎯 Top Contributing Features:")
        for feature, importance in explanation['top_features']:
            print(f"   • {feature}: {importance:.4f}")
        
        print(f"\n🧠 Knowledge Graph Explanations:")
        if explanation['kg_explanations']:
            for i, exp in enumerate(explanation['kg_explanations'], 1):
                print(f"   {i}. {exp}")
        else:
            print(f"   (No direct paths found in knowledge graph)")
        
        print("="*60)
