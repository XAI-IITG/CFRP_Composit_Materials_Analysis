import numpy as np
import torch
import torch.nn as nn
import shap


class ModelWrapper(nn.Module):
    """
    Wrap base sequence model so output is always [batch, 1].
    This avoids shape instability across different PyTorch models.
    """

    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)

        if isinstance(y, (list, tuple)):
            y = y[0]

        if y.dim() == 0:
            y = y.view(1, 1)
        elif y.dim() == 1:
            y = y.unsqueeze(-1)

        return y


class SHAPBenchmarkExplainer:
    """
    SHAP explainer for benchmarking against rule-based explanations.
    Assumes sequence input shape [batch, seq_len, n_features].
    """

    def __init__(
        self,
        model,
        background_data,
        feature_names=None,
        device=None,
        max_background=100,
        prefer_deep=True,
    ):
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = ModelWrapper(model).to(self.device)
        self.model.eval()

        self.background_data = self._to_tensor(background_data)
        if self.background_data.dim() != 3:
            raise ValueError(
                "background_data must have shape [batch, seq_len, n_features]."
            )

        if self.background_data.shape[0] > max_background:
            idx = np.linspace(
                0, self.background_data.shape[0] - 1, max_background, dtype=int
            )
            self.background_data = self.background_data[idx]

        self.n_features = int(self.background_data.shape[-1])

        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
        else:
            if len(feature_names) != self.n_features:
                raise ValueError(
                    f"feature_names length {len(feature_names)} does not match n_features {self.n_features}."
                )
            self.feature_names = list(feature_names)

        self.explainer = None
        self.explainer_name = None
        self._init_explainer(prefer_deep=prefer_deep)

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.float().to(self.device)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _init_explainer(self, prefer_deep=True):
        errors = []

        if prefer_deep:
            try:
                self.explainer = shap.DeepExplainer(self.model, self.background_data)
                self.explainer_name = "DeepExplainer"
                return
            except Exception as e:
                errors.append(f"DeepExplainer failed: {e}")

        try:
            self.explainer = shap.GradientExplainer(self.model, self.background_data)
            self.explainer_name = "GradientExplainer"
            return
        except Exception as e:
            errors.append(f"GradientExplainer failed: {e}")

        raise RuntimeError("Unable to initialize SHAP explainer. " + " | ".join(errors))

    def _normalize_shap_output(self, shap_values):
        """
        Convert SHAP output to [seq_len, n_features].
        """

        arr = shap_values[0] if isinstance(shap_values, list) else shap_values
        arr = np.asarray(arr)
        arr = np.squeeze(arr)

        if arr.ndim == 2:
            return arr

        if arr.ndim == 3:
            return np.mean(np.abs(arr), axis=0)

        raise ValueError(f"Unexpected SHAP output shape after squeeze: {arr.shape}")

    def get_shap_matrix(self, target_sequence):
        x = self._to_tensor(target_sequence)

        if x.dim() == 2:
            x = x.unsqueeze(0)

        if x.dim() != 3:
            raise ValueError(
                "target_sequence must have shape [seq_len, n_features] or [batch, seq_len, n_features]."
            )

        with torch.enable_grad():
            shap_values = self.explainer.shap_values(x)

        return self._normalize_shap_output(shap_values)

    def get_feature_importances(self, target_sequence, top_k=5):
        """
        Returns top_k list of tuples: (feature_name, importance_score).
        Importance is mean absolute SHAP value across sequence length.
        """
        shap_matrix = self.get_shap_matrix(target_sequence)
        importances = np.mean(np.abs(shap_matrix), axis=0)

        top_indices = np.argsort(importances)[::-1][:top_k]
        return [
            (self.feature_names[i], float(importances[i]))
            for i in top_indices
        ]

    def get_full_importance_vector(self, target_sequence):
        """
        Returns dict: feature_name -> mean absolute SHAP importance.
        """
        shap_matrix = self.get_shap_matrix(target_sequence)
        importances = np.mean(np.abs(shap_matrix), axis=0)
        return {name: float(score) for name, score in zip(self.feature_names, importances)}