# Quick Reference: Add This Cell to Your Notebook

## 📦 Save All Models & Metrics - Add to End of Notebook

```python
# ============================================================================
# SAVE ALL TRAINED MODELS AND METRICS
# ============================================================================

import sys
sys.path.append('../scripts')
from save_models_and_metrics import save_all_models_and_metrics

# Save everything
saved_files = save_all_models_and_metrics(
    all_results=all_results,
    trained_models=trained_models,
    scalers=scalers,
    datasets=datasets,
    interpolation_info=info_uniform,
    sequence_datasets=sequence_datasets
)

print("✅ All models and metrics saved!")
```

## 📊 What Gets Saved

| Category | Location | Content |
|----------|----------|---------|
| **Models** | `outputs/saved_models/` | All Transformer, LSTM, PPO, DDPG models + scalers |
| **Metrics** | `outputs/reports/` | JSON files with MSE, RMSE, MAE, R², MAPE |
| **Plots** | `outputs/visualizations/saved_plots/` | Training curves, predictions plots |

## 🔄 Quick Model Loading

```python
import torch
import pickle
import json

# Load PyTorch model
checkpoint = torch.load('outputs/saved_models/transformer_Medium-term_TIMESTAMP.pth')
model = TransformerRULPredictor(input_dim=checkpoint['input_dim'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load scaler
with open('outputs/saved_models/scaler_Medium-term_TIMESTAMP.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load metrics
with open('outputs/reports/all_model_metrics_TIMESTAMP.json', 'r') as f:
    metrics = json.load(f)
```

## 📈 Access Specific Metrics

```python
# Get RMSE for Transformer on Medium-term dataset
rmse = metrics['Medium-term']['transformer']['metrics']['RMSE']
print(f"Transformer RMSE: {rmse:.2f}")

# Get all metrics for a model
tf_metrics = metrics['Medium-term']['transformer']['metrics']
print(f"Transformer Performance:")
print(f"  RMSE: {tf_metrics['RMSE']:.2f}")
print(f"  MAE:  {tf_metrics['MAE']:.2f}")
print(f"  R²:   {tf_metrics['R2']:.4f}")
```

## 🎯 Models Available for XAI Analysis

After saving, you'll have:

1. **Transformer Models** (3 configurations: Short, Medium, Long-term)
2. **LSTM Models** (3 configurations: Short, Medium, Long-term)
3. **PPO Agent** (Deep Reinforcement Learning)
4. **DDPG Agent** (Deep Reinforcement Learning)

All with complete:
- Model weights
- Training histories
- Performance metrics
- Prediction results
- Scalers for data preprocessing

## 📁 File Organization

```
outputs/
├── saved_models/          # All .pth and .pkl model files
├── reports/              # All .json metric files
└── visualizations/
    └── saved_plots/      # All .png visualization files
```

---

**Important**: Files are timestamped (YYYYMMDD_HHMMSS) to prevent overwrites!
