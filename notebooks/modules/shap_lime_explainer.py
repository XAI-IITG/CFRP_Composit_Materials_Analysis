import numpy as np
import torch
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt

class SHAPLIMEExplainer:
    """
    SHAP and LIME explainer for RUL prediction models
    """
    
    def __init__(self, model, X_train, feature_names, model_type='transformer', device='cpu'):
        """
        Args:
            model: Trained PyTorch model
            X_train: Training data for SHAP background (shape: n_samples, seq_len, features)
            feature_names: List of feature names
            model_type: 'transformer', 'lstm', or 'ddpg'
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.model.eval()
        self.X_train = X_train
        self.feature_names = feature_names
        self.model_type = model_type
        self.device = device
        
    def _predict_wrapper(self, X):
        """
        Wrapper function for model prediction
        Handles different input shapes for SHAP/LIME
        
        Args:
            X: Input data (can be 2D or 3D)
        Returns:
            predictions: NumPy array of predictions
        """
        # Handle different input shapes
        if len(X.shape) == 2:  # (n_samples, features) - LIME format
            # Reshape to (n_samples, seq_len, features)
            seq_len = self.X_train.shape[1]
            n_features = X.shape[1]
            
            if self.model_type == 'ddpg':
                # DDPG expects flattened input (seq_len * features)
                X_tensor = torch.FloatTensor(X).to(self.device)
            else:
                # Reshape: (n_samples, features) -> (n_samples, seq_len, features)
                # Repeat last timestep to fill sequence
                X_reshaped = np.repeat(X[:, np.newaxis, :], seq_len, axis=1)
                X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
        
        elif len(X.shape) == 3:  # (n_samples, seq_len, features) - SHAP format
            if self.model_type == 'ddpg':
                # Flatten for DDPG
                X_flat = X.reshape(X.shape[0], -1)
                X_tensor = torch.FloatTensor(X_flat).to(self.device)
            else:
                X_tensor = torch.FloatTensor(X).to(self.device)
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")
        
        # Get predictions
        with torch.no_grad():
            if self.model_type in ['transformer', 'lstm']:
                preds = self.model(X_tensor).squeeze().cpu().numpy()
            elif self.model_type == 'ddpg':
                preds = self.model(X_tensor).cpu().numpy()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Ensure 1D array
        if len(preds.shape) == 0:
            preds = np.array([preds])
        
        return preds
    
    def explain_with_shap(self, test_sample, target_scaler=None, n_background=100):
        """
        Generate SHAP explanation for a test sample
        
        Args:
            test_sample: Test sequence (shape: seq_len, features) or (1, seq_len, features)
            target_scaler: Scaler for inverse transformation
            n_background: Number of background samples for SHAP
        
        Returns:
            explanation: Dictionary with SHAP values and plots
        """
        print("🔍 Generating SHAP Explanation...")
        
        # Ensure correct shape
        if len(test_sample.shape) == 2:
            test_sample = test_sample[np.newaxis, :]  # Add batch dimension
        
        # Select background data (random subset of training data)
        n_train = self.X_train.shape[0]
        bg_indices = np.random.choice(n_train, min(n_background, n_train), replace=False)
        background = self.X_train[bg_indices]
        
        # For SHAP, we need to flatten sequences to 2D
        # Shape: (n_samples, seq_len * features)
        if self.model_type != 'ddpg':
            background_flat = background.reshape(background.shape[0], -1)
            test_flat = test_sample.reshape(test_sample.shape[0], -1)
        else:
            background_flat = background.reshape(background.shape[0], -1)
            test_flat = test_sample.reshape(test_sample.shape[0], -1)
        
        # Create explainer
        explainer = shap.KernelExplainer(
            model=lambda x: self._predict_wrapper(x.reshape(-1, *test_sample.shape[1:])),
            data=background_flat
        )
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(test_flat)
        
        # Get prediction
        pred_normalized = self._predict_wrapper(test_sample)[0]
        pred_actual = target_scaler.inverse_transform([[pred_normalized]])[0, 0] if target_scaler else pred_normalized
        
        # Aggregate SHAP values across sequence (average over timesteps)
        seq_len = self.X_train.shape[1]
        n_features = self.X_train.shape[2]
        shap_values_3d = shap_values.reshape(1, seq_len, n_features)
        shap_values_avg = np.mean(np.abs(shap_values_3d), axis=1)[0]  # Shape: (n_features,)
        
        # Create feature importance dictionary
        feature_importance = {
            feat: float(shap_val) 
            for feat, shap_val in zip(self.feature_names, shap_values_avg)
        }
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"✓ SHAP Explanation Generated!")
        print(f"  Predicted RUL: {pred_actual:.0f} cycles")
        print(f"  Top 5 Features:")
        for i, (feat, importance) in enumerate(sorted_features[:5], 1):
            print(f"    {i}. {feat}: {importance:.6f}")
        
        return {
            'shap_values': shap_values,
            'shap_values_avg': shap_values_avg,
            'feature_importance': feature_importance,
            'sorted_features': sorted_features,
            'predicted_rul': pred_actual,
            'predicted_rul_normalized': pred_normalized,
            'explainer': explainer,
            'test_sample': test_sample
        }
    
    def explain_with_lime(self, test_sample, target_scaler=None, num_features=10):
        """
        Generate LIME explanation for a test sample
        
        Args:
            test_sample: Test sequence (shape: seq_len, features) or (1, seq_len, features)
            target_scaler: Scaler for inverse transformation
            num_features: Number of top features to show
        
        Returns:
            explanation: Dictionary with LIME explanation
        """
        print("🔍 Generating LIME Explanation...")
        
        # Ensure correct shape
        if len(test_sample.shape) == 2:
            test_sample = test_sample[np.newaxis, :]  # Add batch dimension
        
        # Flatten training data for LIME
        # LIME expects 2D input: (n_samples, n_features)
        # We'll use the LAST TIMESTEP as the sample representation
        X_train_last = self.X_train[:, -1, :]  # Shape: (n_samples, n_features)
        test_last = test_sample[0, -1, :]      # Shape: (n_features,)
        
        # Create LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train_last,
            feature_names=self.feature_names,
            mode='regression',
            verbose=False
        )
        
        # Predict function for LIME (expects 2D input)
        def predict_fn(X):
            return self._predict_wrapper(X)
        
        # Generate explanation
        exp = explainer.explain_instance(
            data_row=test_last,
            predict_fn=predict_fn,
            num_features=num_features
        )
        
        # Get prediction
        pred_normalized = self._predict_wrapper(test_sample)[0]
        pred_actual = target_scaler.inverse_transform([[pred_normalized]])[0, 0] if target_scaler else pred_normalized
        
        # Extract feature importance
        feature_importance = dict(exp.as_list())
        
        print(f"✓ LIME Explanation Generated!")
        print(f"  Predicted RUL: {pred_actual:.0f} cycles")
        print(f"  Top {num_features} Features:")
        for i, (feat, importance) in enumerate(exp.as_list(), 1):
            print(f"    {i}. {feat}: {importance:.6f}")
        
        return {
            'lime_explanation': exp,
            'feature_importance': feature_importance,
            'predicted_rul': pred_actual,
            'predicted_rul_normalized': pred_normalized,
            'explainer': explainer
        }
    
    def visualize_shap(self, shap_explanation, save_path=None):
        """
        Visualize SHAP values
        
        Args:
            shap_explanation: Output from explain_with_shap()
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Waterfall plot
        ax = axes[0]
        sorted_features = shap_explanation['sorted_features']
        features = [f[0] for f in sorted_features[:10]]
        values = [f[1] for f in sorted_features[:10]]
        
        colors = ['red' if v > 0 else 'blue' for v in values]
        ax.barh(features, values, color=colors, alpha=0.7)
        ax.set_xlabel('SHAP Value (Impact on RUL)', fontsize=12)
        ax.set_title(f'SHAP Feature Importance\nPredicted RUL: {shap_explanation["predicted_rul"]:.0f} cycles', 
                    fontsize=13, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(alpha=0.3, axis='x')
        
        # Plot 2: Force plot (simplified)
        ax = axes[1]
        ax.bar(range(len(values)), values, color=colors, alpha=0.7)
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_ylabel('SHAP Value', fontsize=12)
        ax.set_title('SHAP Values for Top Features', fontsize=13, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✓ SHAP visualization generated")
    
    def visualize_lime(self, lime_explanation, save_path=None):
        """
        Visualize LIME explanation
        
        Args:
            lime_explanation: Output from explain_with_lime()
            save_path: Path to save figure (optional)
        """
        exp = lime_explanation['lime_explanation']
        
        # Get feature importance
        features_weights = exp.as_list()
        features = [f[0] for f in features_weights]
        weights = [f[1] for f in features_weights]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if w > 0 else 'red' for w in weights]
        ax.barh(features, weights, color=colors, alpha=0.7)
        ax.set_xlabel('LIME Weight (Impact on RUL)', fontsize=12)
        ax.set_title(f'LIME Feature Importance\nPredicted RUL: {lime_explanation["predicted_rul"]:.0f} cycles', 
                    fontsize=13, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✓ LIME visualization generated")