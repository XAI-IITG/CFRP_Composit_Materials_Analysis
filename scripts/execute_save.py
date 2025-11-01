# ============================================================================
# SAVE ALL MODELS AND METRICS - EXECUTION CELL
# ============================================================================

"""
This cell should be executed at the end of all training to save:
1. All trained models (Transformer, LSTM, PPO, DDPG)
2. All metrics (MSE, RMSE, MAE, R2, MAPE)
3. Training histories
4. Prediction plots
5. Comparison reports
"""

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

print("\n✅ All models and metrics have been saved successfully!")
print(f"\n📁 Files are organized in the following structure:")
print(f"   - Models: outputs/saved_models/")
print(f"   - Metrics: outputs/reports/")
print(f"   - Plots: outputs/visualizations/saved_plots/")
