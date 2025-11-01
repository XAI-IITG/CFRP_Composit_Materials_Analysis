# Model Saving & Metrics Export Guide

## Overview

This guide explains how to save all trained models, metrics, and results from the CFRP Composite Materials Analysis project.

## What Gets Saved

### 1. Models (→ `outputs/saved_models/`)

All trained models are saved with complete state:

- **Transformer Models** (Short-term, Medium-term, Long-term)
  - Model state dict
  - Architecture information
  - Input dimensions and sequence length
  
- **LSTM Models** (Short-term, Medium-term, Long-term)
  - Model state dict
  - Architecture information
  - Input dimensions and sequence length

- **DRL Agents**
  - PPO Agent (Actor-Critic network)
  - DDPG Agent (Actor and Critic networks separately)

- **Scalers** (for data normalization)
  - Feature scalers (StandardScaler)
  - Target scalers (MinMaxScaler)

### 2. Metrics (→ `outputs/reports/`)

Comprehensive performance metrics saved as JSON:

- **Per-Model Metrics**:
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² (R-squared Score)
  - MAPE (Mean Absolute Percentage Error)

- **Training History**:
  - Training losses per epoch
  - Validation losses per epoch
  - Number of epochs

- **Experiment Summary**:
  - Best models for each metric
  - Complete results for all configurations
  - Dataset information

### 3. Visualizations (→ `outputs/visualizations/saved_plots/`)

High-quality plots for analysis:

- **Training History Plots**
  - Training vs Validation loss curves
  - Saved as PNG (300 DPI)

- **Prediction Plots**
  - Predictions vs Actual values scatter plots
  - Perfect prediction line reference
  - Metrics displayed in title

- **Comparison Plots**
  - Model performance comparison across metrics
  - Bar charts for RMSE, MAE, R², MAPE
  - Grouped by dataset configuration

## File Naming Convention

All files are timestamped for version control:

```
{model_name}_{dataset}_{YYYYMMDD_HHMMSS}.{extension}
```

Examples:
- `transformer_Medium-term_20250101_143022.pth`
- `all_model_metrics_20250101_143022.json`
- `lstm_Short-term_training_history_20250101_143022.png`

## Folder Structure

```
outputs/
├── saved_models/
│   ├── transformer_Short-term_YYYYMMDD_HHMMSS.pth
│   ├── transformer_Medium-term_YYYYMMDD_HHMMSS.pth
│   ├── transformer_Long-term_YYYYMMDD_HHMMSS.pth
│   ├── lstm_Short-term_YYYYMMDD_HHMMSS.pth
│   ├── lstm_Medium-term_YYYYMMDD_HHMMSS.pth
│   ├── lstm_Long-term_YYYYMMDD_HHMMSS.pth
│   ├── ppo_agent_YYYYMMDD_HHMMSS.pth
│   ├── ddpg_actor_YYYYMMDD_HHMMSS.pth
│   ├── ddpg_critic_YYYYMMDD_HHMMSS.pth
│   ├── scaler_Short-term_YYYYMMDD_HHMMSS.pkl
│   ├── scaler_Medium-term_YYYYMMDD_HHMMSS.pkl
│   └── scaler_Long-term_YYYYMMDD_HHMMSS.pkl
│
├── reports/
│   ├── all_model_metrics_YYYYMMDD_HHMMSS.json
│   ├── model_comparison_YYYYMMDD_HHMMSS.json
│   ├── experiment_summary_YYYYMMDD_HHMMSS.json
│   ├── experiment_summary_YYYYMMDD_HHMMSS.txt
│   ├── transformer_Short-term_training_history_YYYYMMDD_HHMMSS.json
│   ├── transformer_Medium-term_training_history_YYYYMMDD_HHMMSS.json
│   ├── transformer_Long-term_training_history_YYYYMMDD_HHMMSS.json
│   ├── lstm_Short-term_training_history_YYYYMMDD_HHMMSS.json
│   ├── lstm_Medium-term_training_history_YYYYMMDD_HHMMSS.json
│   └── lstm_Long-term_training_history_YYYYMMDD_HHMMSS.json
│
└── visualizations/
    └── saved_plots/
        ├── transformer_Short-term_training_history_YYYYMMDD_HHMMSS.png
        ├── transformer_Medium-term_training_history_YYYYMMDD_HHMMSS.png
        ├── transformer_Long-term_training_history_YYYYMMDD_HHMMSS.png
        ├── transformer_Short-term_predictions_YYYYMMDD_HHMMSS.png
        ├── transformer_Medium-term_predictions_YYYYMMDD_HHMMSS.png
        ├── transformer_Long-term_predictions_YYYYMMDD_HHMMSS.png
        ├── lstm_Short-term_training_history_YYYYMMDD_HHMMSS.png
        ├── lstm_Medium-term_training_history_YYYYMMDD_HHMMSS.png
        ├── lstm_Long-term_training_history_YYYYMMDD_HHMMSS.png
        ├── lstm_Short-term_predictions_YYYYMMDD_HHMMSS.png
        ├── lstm_Medium-term_predictions_YYYYMMDD_HHMMSS.png
        ├── lstm_Long-term_predictions_YYYYMMDD_HHMMSS.png
        ├── model_comparison_RMSE_YYYYMMDD_HHMMSS.png
        ├── model_comparison_MAE_YYYYMMDD_HHMMSS.png
        ├── model_comparison_R2_YYYYMMDD_HHMMSS.png
        └── model_comparison_MAPE_YYYYMMDD_HHMMSS.png
```

## How to Use

### In Notebook (Recommended)

At the end of your training in `layup1.ipynb`, add a new cell:

```python
# Import the saving utility
import sys
sys.path.append('../scripts')
from save_models_and_metrics import save_all_models_and_metrics

# Execute the saving function
saved_files = save_all_models_and_metrics(
    all_results=all_results,           # Dictionary with all model results
    trained_models=trained_models,      # Dictionary with trained model instances
    scalers=scalers,                   # Dictionary with data scalers
    datasets=datasets,                 # Dictionary with dataset information
    interpolation_info=info_uniform,   # Information about data interpolation
    sequence_datasets=sequence_datasets  # Dictionary with sequence datasets
)

print("\n✅ All models and metrics have been saved!")
```

### Loading Saved Models

#### PyTorch Models

```python
import torch

# Load a saved model
checkpoint = torch.load('outputs/saved_models/transformer_Medium-term_20250101_143022.pth')

# Create model instance
model = TransformerRULPredictor(input_dim=checkpoint['input_dim'])

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded from epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Architecture: {checkpoint['model_architecture']}")
```

#### Scalers

```python
import pickle

# Load scaler
with open('outputs/saved_models/scaler_Medium-term_20250101_143022.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Use for prediction
scaled_data = scaler['feature_scaler'].transform(new_data)
```

#### Metrics

```python
import json

# Load metrics
with open('outputs/reports/all_model_metrics_20250101_143022.json', 'r') as f:
    metrics = json.load(f)

# Access specific model metrics
transformer_rmse = metrics['Medium-term']['transformer']['metrics']['RMSE']
print(f"Transformer RMSE: {transformer_rmse}")
```

## JSON File Structures

### 1. All Model Metrics (`all_model_metrics_*.json`)

```json
{
  "Short-term": {
    "transformer": {
      "metrics": {
        "MSE": 1234.56,
        "RMSE": 35.14,
        "MAE": 28.92,
        "R2": 0.9234,
        "MAPE": 12.45
      },
      "train_losses": [0.123, 0.098, ...],
      "val_losses": [0.145, 0.112, ...],
      "config": {...},
      "input_dim": 15,
      "seq_len": 5
    },
    "lstm": {...}
  },
  "Medium-term": {...},
  "Long-term": {...}
}
```

### 2. Model Comparison (`model_comparison_*.json`)

```json
{
  "Short-term": {
    "transformer": {
      "MSE": 1234.56,
      "RMSE": 35.14,
      "MAE": 28.92,
      "R2": 0.9234,
      "MAPE": 12.45
    },
    "lstm": {...}
  },
  "Medium-term": {...},
  "Long-term": {...}
}
```

### 3. Experiment Summary (`experiment_summary_*.json`)

```json
{
  "timestamp": "20250101_143022",
  "best_models": {
    "RMSE": {
      "model": "Transformer",
      "dataset": "Medium-term",
      "value": 28.45
    },
    "MAE": {...},
    "R2": {...},
    "MAPE": {...}
  },
  "experiment_summary": {...},
  "data_info": {
    "interpolation": {...}
  }
}
```

### 4. Training History (`*_training_history_*.json`)

```json
{
  "train_losses": [0.234, 0.189, 0.145, ...],
  "val_losses": [0.267, 0.201, 0.178, ...],
  "epochs": 50
}
```

## Model State Dict Structure

PyTorch model files (`.pth`) contain:

```python
{
    'model_state_dict': OrderedDict(...),  # Model weights
    'model_architecture': str,              # String representation of model
    'timestamp': str,                       # Save timestamp
    'optimizer_state_dict': OrderedDict(...),  # Optional optimizer state
    'epoch': int,                           # Optional epoch number
    'dataset': str,                         # Dataset configuration used
    'model_type': str,                      # transformer/lstm/ppo/ddpg
    'input_dim': int,                       # Number of input features
    'seq_len': int                          # Sequence length
}
```

## Best Practices

1. **Always save after training** - Don't risk losing trained models
2. **Check file sizes** - Models should be 1-50MB typically
3. **Version control** - Timestamps prevent accidental overwrites
4. **Backup critical models** - Copy best-performing models to separate folder
5. **Document experiments** - Review `experiment_summary_*.txt` for readability

## Using for Explainability (XAI)

For your XAI work, you'll need:

1. **Load trained models**:
   ```python
   model = load_model('outputs/saved_models/transformer_Medium-term_*.pth')
   ```

2. **Use metrics for analysis**:
   - Compare model behaviors
   - Identify which features are most important
   - Understand failure modes

3. **Prediction data**:
   - Use saved predictions vs actual for error analysis
   - Identify where models fail (high RUL vs low RUL)

4. **Training dynamics**:
   - Analyze loss curves for overfitting
   - Understand convergence behavior

## Troubleshooting

### Issue: "Module not found"
**Solution**: Make sure you're in the notebook directory:
```python
import sys
sys.path.append('../scripts')
```

### Issue: "File already exists"
**Solution**: Timestamps prevent this. Each run creates new files.

### Issue: "Out of disk space"
**Solution**: Models are ~5-20MB each. Clean old experiments:
```bash
# Keep only recent files
find outputs/saved_models/ -mtime +30 -delete
```

### Issue: "Cannot load model"
**Solution**: Ensure model architecture matches:
```python
# Check saved architecture
checkpoint = torch.load('model.pth')
print(checkpoint['model_architecture'])
```

## Advanced: Selective Saving

If you only want to save specific models:

```python
from save_models_and_metrics import ModelSaver

saver = ModelSaver()

# Save only best model
best_model_path = saver.save_pytorch_model(
    best_model,
    "best_transformer",
    additional_info={'test_rmse': 28.45}
)

# Save only metrics
metrics_path = saver.save_metrics(all_results)
```

## Summary

✅ All models, metrics, and visualizations are automatically saved  
✅ Organized folder structure for easy navigation  
✅ Timestamped files prevent overwrites  
✅ Complete state preservation for reproducibility  
✅ Ready for XAI analysis and model deployment  

For questions or issues, refer to the main README or contact the project team.
