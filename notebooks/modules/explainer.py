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
    
    def __init__(self, model, kg: CFRPKnowledgeGraph, feature_names: List[str], model_type='standard'):
        """
        Args:
            model: The trained model (Transformer, LSTM, or DRL agent)
            kg: Knowledge graph instance
            feature_names: List of feature names
            model_type: 'standard' (Transformer/LSTM), 'ddpg', 'ppo', 'dqn', or 'drl'
        """
        self.model = model
        self.kg = kg
        self.feature_names = feature_names
        self.model_type = model_type
    
    def explain_prediction(self, sample_sequence, actual_rul, device, target_scaler):
        """Generate comprehensive explanation for a prediction"""
        
        # 1. Get model prediction based on model type
        self.model.eval()
        with torch.no_grad():
            if self.model_type == 'ddpg':
                # DDPG uses actor network for prediction
                pred_normalized = self.model(torch.FloatTensor(sample_sequence).unsqueeze(0).to(device))
            elif self.model_type in ['ppo', 'dqn', 'drl']:
                # Other DRL models
                pred_normalized = self.model(torch.FloatTensor(sample_sequence).unsqueeze(0).to(device))
            else:
                # Standard models (Transformer/LSTM)
                pred_normalized = self.model(torch.FloatTensor(sample_sequence).unsqueeze(0).to(device))
            
            pred_rul = target_scaler.inverse_transform(pred_normalized.cpu().numpy().reshape(-1, 1))[0][0]
        
        # 2. Compute gradient-based feature importance
        try:
            sample_tensor = torch.FloatTensor(sample_sequence).unsqueeze(0).to(device)
            sample_tensor.requires_grad = True
            
            if self.model_type == 'ddpg':
                # For DDPG, use actor network
                output = self.model(sample_tensor)
            elif self.model_type in ['ppo', 'dqn', 'drl']:
                output = self.model(sample_tensor)
            else:
                output = self.model(sample_tensor)
            
            output.backward()
            
            # Get gradient magnitude as feature importance
            gradients = sample_tensor.grad.abs().mean(dim=1).squeeze()  # Average over time
            feature_importance_values = gradients.cpu().numpy()
            
        except Exception as e:
            print(f"⚠️  Warning: Could not compute gradients: {e}")
            # Fallback: Use mean absolute values as importance
            feature_importance_values = np.abs(sample_sequence).mean(axis=0)
        
        # 3. Get feature importance dictionary
        feature_importance = dict(zip(self.feature_names, feature_importance_values))
        
        # 4. Generate KG-based explanations
        kg_explanations = self.kg.get_explanation_path(feature_importance, pred_rul)
        
        # 5. Create comprehensive explanation
        explanation = {
            'predicted_rul': pred_rul,
            'actual_rul': actual_rul,
            'error': abs(pred_rul - actual_rul),
            'error_percent': (abs(pred_rul - actual_rul) / max(actual_rul, 1)) * 100,
            'top_features': sorted(feature_importance.items(), 
                                 key=lambda x: abs(x[1]), reverse=True)[:5],
            'kg_explanations': kg_explanations,
            'feature_importance': feature_importance
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
