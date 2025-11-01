# Model Saving System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        layup1.ipynb (Notebook)                      │
│                                                                     │
│  • Data Loading & Preprocessing                                    │
│  • Feature Engineering                                             │
│  • Model Training (Transformer, LSTM, PPO, DDPG)                   │
│  • Evaluation & Metrics Collection                                 │
│                                                                     │
│  Variables Available:                                              │
│    - all_results: Dict with all model results                      │
│    - trained_models: Dict with PyTorch models                      │
│    - scalers: Dict with preprocessing scalers                      │
│    - datasets: Dict with dataset configurations                    │
│                                                                     │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  ADD THIS CELL AT THE END:                             │        │
│  │                                                          │        │
│  │  from save_models_and_metrics import (...)             │        │
│  │  saved_files = save_all_models_and_metrics(...)        │        │
│  └────────────────────────────────────────────────────────┘        │
└──────────────────────────┬──────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│              scripts/save_models_and_metrics.py                     │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐         │
│  │              ModelSaver Class                        │         │
│  │                                                       │         │
│  │  • save_pytorch_model()                              │         │
│  │  • save_sklearn_model()                              │         │
│  │  • save_metrics()                                    │         │
│  │  • save_training_history()                           │         │
│  │  • save_predictions_plot()                           │         │
│  │  • save_comparison_metrics()                         │         │
│  │  • create_summary_report()                           │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                     │
│  Main Function: save_all_models_and_metrics()                      │
│    ↓                                                                │
│    Orchestrates all saving operations                              │
└──────────────────────────┬──────────────────────────────────────────┘
                          │
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│   MODELS     │  │   METRICS    │  │  VISUALIZATIONS  │
│              │  │              │  │                  │
│ outputs/     │  │ outputs/     │  │ outputs/         │
│ saved_models/│  │ reports/     │  │ visualizations/  │
│              │  │              │  │ saved_plots/     │
└──────────────┘  └──────────────┘  └──────────────────┘
```

## Data Flow

```
Input Data (all_results, trained_models, etc.)
        │
        ▼
┌─────────────────────────────────────┐
│  ModelSaver Initialization          │
│  • Create directory structure       │
│  • Generate timestamp               │
└────────────┬────────────────────────┘
            │
            ├─────────────────────────────┐
            │                             │
            ▼                             ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Save Models        │      │  Save Metrics       │
│  (.pth, .pkl)       │      │  (.json)            │
│                     │      │                     │
│  • Transformers     │      │  • All metrics      │
│  • LSTMs            │      │  • Comparisons      │
│  • DRL Agents       │      │  • Summaries        │
│  • Scalers          │      │  • Training history │
└─────────────────────┘      └─────────────────────┘
            │                             │
            └─────────────┬───────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │  Generate Visualizations│
            │  (.png)                 │
            │                         │
            │  • Training curves      │
            │  • Prediction plots     │
            │  • Comparison charts    │
            └─────────────────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │  Create Summary Reports │
            │  (.json, .txt)          │
            │                         │
            │  • Best models          │
            │  • Experiment summary   │
            │  • Readable text report │
            └─────────────────────────┘
                         │
                         ▼
                ┌────────────────┐
                │  Return Paths  │
                │  Dictionary    │
                └────────────────┘
```

## File Organization

```
CFRP_Composit_Materials_Analysis/
│
├── notebooks/
│   └── layup1.ipynb                    ← Add saving cell here
│
├── scripts/
│   ├── save_models_and_metrics.py      ← Main implementation
│   └── execute_save.py                 ← Example usage
│
├── docs/
│   ├── README.md                       ← Documentation index
│   ├── IMPLEMENTATION_SUMMARY.md       ← System overview
│   ├── MODEL_SAVING_GUIDE.md           ← Complete guide
│   ├── QUICK_REFERENCE.md              ← Code snippets
│   └── ARCHITECTURE.md                 ← This file
│
└── outputs/
    ├── saved_models/                   ← Model files (.pth, .pkl)
    │   ├── transformer_Short-term_TIMESTAMP.pth
    │   ├── transformer_Medium-term_TIMESTAMP.pth
    │   ├── transformer_Long-term_TIMESTAMP.pth
    │   ├── lstm_Short-term_TIMESTAMP.pth
    │   ├── lstm_Medium-term_TIMESTAMP.pth
    │   ├── lstm_Long-term_TIMESTAMP.pth
    │   ├── ppo_agent_TIMESTAMP.pth
    │   ├── ddpg_actor_TIMESTAMP.pth
    │   ├── ddpg_critic_TIMESTAMP.pth
    │   ├── scaler_Short-term_TIMESTAMP.pkl
    │   ├── scaler_Medium-term_TIMESTAMP.pkl
    │   └── scaler_Long-term_TIMESTAMP.pkl
    │
    ├── reports/                        ← Metric files (.json, .txt)
    │   ├── all_model_metrics_TIMESTAMP.json
    │   ├── model_comparison_TIMESTAMP.json
    │   ├── experiment_summary_TIMESTAMP.json
    │   ├── experiment_summary_TIMESTAMP.txt
    │   └── *_training_history_TIMESTAMP.json (x6)
    │
    └── visualizations/
        └── saved_plots/                ← Plot files (.png)
            ├── *_training_history_TIMESTAMP.png (x6)
            ├── *_predictions_TIMESTAMP.png (x6)
            └── model_comparison_*_TIMESTAMP.png (x4)
```

## Component Responsibilities

### 1. ModelSaver Class

```
┌────────────────────────────────────────────────────────┐
│                  ModelSaver                            │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Responsibilities:                                     │
│  • Directory management                                │
│  • File naming with timestamps                         │
│  • Model serialization (PyTorch, Pickle)               │
│  • Metrics serialization (JSON)                        │
│  • Visualization generation (Matplotlib)               │
│  • Report generation (JSON, TXT)                       │
│                                                        │
│  Methods:                                              │
│  ├─ save_pytorch_model()      → .pth files            │
│  ├─ save_sklearn_model()      → .pkl files            │
│  ├─ save_metrics()            → .json files           │
│  ├─ save_training_history()   → .json + .png         │
│  ├─ save_predictions_plot()   → .png files            │
│  ├─ save_comparison_metrics() → .json + .png         │
│  └─ create_summary_report()   → .json + .txt         │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 2. Main Function

```
save_all_models_and_metrics()
    │
    ├─► Validate inputs
    │
    ├─► Initialize ModelSaver
    │
    ├─► For each dataset configuration:
    │   ├─► Save Transformer model
    │   ├─► Save LSTM model
    │   ├─► Save scaler
    │   ├─► Save training history
    │   └─► Save prediction plots
    │
    ├─► Save DRL agents (PPO, DDPG)
    │
    ├─► Save all metrics (comprehensive JSON)
    │
    ├─► Create comparison visualizations
    │
    ├─► Generate summary reports
    │
    └─► Return dictionary of saved file paths
```

## Model State Structure

### PyTorch Model (.pth)

```
{
    'model_state_dict': OrderedDict(
        [layer_name, layer_weights],
        ...
    ),
    'model_architecture': str,
    'timestamp': 'YYYYMMDD_HHMMSS',
    'dataset': 'Short-term' | 'Medium-term' | 'Long-term',
    'model_type': 'transformer' | 'lstm' | 'ppo' | 'ddpg',
    'input_dim': int,
    'seq_len': int,
    'optimizer_state_dict': OrderedDict(...),  # Optional
    'epoch': int  # Optional
}
```

### Scaler (.pkl)

```
{
    'feature_scaler': StandardScaler(
        mean_: array([...]),
        var_: array([...]),
        scale_: array([...])
    ),
    'target_scaler': MinMaxScaler(
        min_: float,
        scale_: float,
        data_min_: float,
        data_max_: float
    )
}
```

### Metrics (.json)

```
{
    'dataset_name': {
        'model_type': {
            'metrics': {
                'MSE': float,
                'RMSE': float,
                'MAE': float,
                'R2': float,
                'MAPE': float
            },
            'train_losses': [float, ...],
            'val_losses': [float, ...],
            'config': {...},
            'predictions': [float, ...],
            'targets': [float, ...]
        }
    }
}
```

## Usage Flow

```
┌──────────────────┐
│  1. Train Models │
│     in Notebook  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│  2. Import Saver     │
│     from scripts/    │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────────────┐
│  3. Call Saving Function     │
│     with all_results, etc.   │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  4. Automatic Saving         │
│     • Models                 │
│     • Metrics                │
│     • Plots                  │
│     • Reports                │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  5. Check outputs/ folder    │
│     • Verify files created   │
│     • Review summary report  │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  6. Use for XAI Analysis     │
│     • Load models            │
│     • Access metrics         │
│     • Analyze results        │
└──────────────────────────────┘
```

## Error Handling

```
save_all_models_and_metrics()
    │
    ├─ Try: Save each model
    │   └─ Catch: Print warning, continue
    │
    ├─ Try: Save each metric file
    │   └─ Catch: Print warning, continue
    │
    ├─ Try: Generate each plot
    │   └─ Catch: Print warning, continue
    │
    └─ Finally: Return saved_files dict
                (contains successful saves)
```

## Benefits of This Architecture

### 1. Modularity
- Each saving operation is independent
- Failures don't stop the entire process
- Easy to extend with new model types

### 2. Maintainability
- Clear separation of concerns
- Well-documented functions
- Type hints for clarity

### 3. Reproducibility
- Complete state preservation
- Timestamped versions
- Configuration tracking

### 4. Usability
- Single function call
- Automatic organization
- Comprehensive reports

### 5. Scalability
- Handles multiple models
- Supports various formats
- Extensible for new metrics

## Integration Points

### For Explainability (XAI)

```
Saved Models
     │
     ├─► SHAP Analysis
     │   └─ Load model → Compute SHAP values → Visualize
     │
     ├─► LIME Analysis
     │   └─ Load model → Generate explanations → Interpret
     │
     ├─► Attention Visualization
     │   └─ Load Transformer → Extract attention → Plot
     │
     └─► Feature Importance
         └─ Load model + metrics → Analyze → Report
```

### For Deployment

```
Saved Models + Scalers
     │
     ├─► API Endpoint
     │   └─ Load → Serve predictions
     │
     ├─► Batch Processing
     │   └─ Load → Process data → Export
     │
     └─► Real-time Monitoring
         └─ Load → Predict → Alert
```

## Summary

This architecture provides:

✅ **Complete Automation** - One function saves everything  
✅ **Professional Organization** - Structured folder hierarchy  
✅ **Version Control** - Timestamped files  
✅ **Comprehensive Coverage** - Models, metrics, plots, reports  
✅ **Easy Integration** - Simple API, clear documentation  
✅ **XAI Ready** - Models preserved for explanation  
✅ **Deployment Ready** - Complete state for production  

---

**For implementation details, see:**
- `scripts/save_models_and_metrics.py` - Source code
- `docs/MODEL_SAVING_GUIDE.md` - Usage guide
- `docs/QUICK_REFERENCE.md` - Code snippets
