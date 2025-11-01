"""
Script to save all trained models and their metrics to appropriate directories.
This script should be executed at the end of the notebook training process.
"""

import torch
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class ModelSaver:
    """
    Utility class to save models, metrics, and training histories
    """
    
    def __init__(self, base_dir="outputs"):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "saved_models"
        self.reports_dir = self.base_dir / "reports"
        self.plots_dir = self.base_dir / "visualizations" / "saved_plots"
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_pytorch_model(self, model, model_name, optimizer=None, epoch=None, additional_info=None):
        """
        Save PyTorch model with complete state
        
        Args:
            model: PyTorch model
            model_name: Name for the saved model
            optimizer: Optional optimizer state
            epoch: Optional epoch number
            additional_info: Dictionary with additional information
        """
        save_path = self.models_dir / f"{model_name}_{self.timestamp}.pth"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_architecture': str(model),
            'timestamp': self.timestamp,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if additional_info is not None:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, save_path)
        print(f"✅ Saved PyTorch model: {save_path}")
        
        return str(save_path)
    
    def save_sklearn_model(self, model, model_name):
        """
        Save scikit-learn compatible models using pickle
        
        Args:
            model: Sklearn model
            model_name: Name for the saved model
        """
        save_path = self.models_dir / f"{model_name}_{self.timestamp}.pkl"
        
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✅ Saved sklearn model: {save_path}")
        
        return str(save_path)
    
    def save_metrics(self, metrics_dict, filename="all_metrics"):
        """
        Save metrics dictionary to JSON file
        
        Args:
            metrics_dict: Dictionary containing all metrics
            filename: Name for the JSON file
        """
        save_path = self.reports_dir / f"{filename}_{self.timestamp}.json"
        
        # Create a clean copy without non-serializable objects (like PyTorch models)
        clean_metrics = self._clean_metrics_dict(metrics_dict)
        
        with open(save_path, 'w') as f:
            json.dump(clean_metrics, f, indent=4, cls=NumpyEncoder)
        
        print(f"✅ Saved metrics: {save_path}")
        
        return str(save_path)
    
    def _clean_metrics_dict(self, metrics_dict):
        """
        Extract ONLY numerical metrics from dictionary, excluding model objects
        
        The all_results dictionary has this structure:
        {
            'dataset_name': {
                'transformer': {
                    'model': <TransformerRULPredictor object>,  # SKIP THIS
                    'MAE': 0.038,                                # KEEP THIS
                    'MSE': 0.003,                                # KEEP THIS
                    'RMSE': 0.056,                               # KEEP THIS
                    ...
                },
                'lstm': {...},
                'ppo': {...}
            }
        }
        
        Args:
            metrics_dict: Dictionary potentially containing model objects and metrics
            
        Returns:
            Clean dictionary with only JSON-serializable numerical metrics
        """
        import torch.nn as nn
        
        def _extract_metrics_only(d):
            """Recursively extract only numerical metrics"""
            if not isinstance(d, dict):
                # Return basic types as-is
                if isinstance(d, (int, float, str, bool, type(None))):
                    return d
                elif isinstance(d, (list, tuple)):
                    return [_extract_metrics_only(item) for item in d if not isinstance(item, nn.Module)]
                else:
                    # Skip non-serializable objects
                    return None
            
            cleaned = {}
            for key, value in d.items():
                # Skip known model-related keys
                if key in ['model', 'scaler', 'optimizer', 'scheduler']:
                    continue
                
                # Skip PyTorch model instances directly
                if isinstance(value, nn.Module):
                    continue
                
                # Skip complex objects that aren't basic types
                if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None), np.ndarray, np.integer, np.floating)):
                    continue
                
                # Recursively process nested dictionaries
                if isinstance(value, dict):
                    nested_result = _extract_metrics_only(value)
                    if nested_result:  # Only add if not empty
                        cleaned[key] = nested_result
                
                # Process lists/tuples
                elif isinstance(value, (list, tuple)):
                    list_result = [_extract_metrics_only(item) for item in value]
                    # Filter out None values
                    list_result = [item for item in list_result if item is not None]
                    if list_result:
                        cleaned[key] = list_result
                
                # Keep basic serializable types
                elif isinstance(value, (int, float, str, bool, type(None))):
                    cleaned[key] = value
                
                # Handle numpy types (NumpyEncoder will handle these)
                elif isinstance(value, (np.integer, np.floating)):
                    cleaned[key] = value
                elif isinstance(value, np.ndarray):
                    cleaned[key] = value
                
            return cleaned if cleaned else None
        
        result = _extract_metrics_only(metrics_dict)
        return result if result else {}
    
    def save_training_history(self, train_losses, val_losses, model_name):
        """
        Save training history as JSON and plot
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            model_name: Name of the model
        """
        # Save as JSON
        history_dict = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs': len(train_losses)
        }
        
        json_path = self.reports_dir / f"{model_name}_training_history_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(history_dict, f, indent=4, cls=NumpyEncoder)
        
        # Save as plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} - Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.plots_dir / f"{model_name}_training_history_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved training history: {json_path}")
        print(f"✅ Saved training plot: {plot_path}")
        
        return str(json_path), str(plot_path)
    
    def save_predictions_plot(self, predictions, targets, model_name, metrics=None):
        """
        Save predictions vs actual values plot
        
        Args:
            predictions: Predicted values
            targets: Actual target values
            model_name: Name of the model
            metrics: Optional metrics dictionary
        """
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(targets, predictions, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual RUL (cycles)', fontsize=12)
        plt.ylabel('Predicted RUL (cycles)', fontsize=12)
        
        # Add metrics to title if provided
        if metrics:
            title = f'{model_name} - Predictions vs Actual\n'
            title += f'R² = {metrics.get("R2", 0):.4f}, RMSE = {metrics.get("RMSE", 0):.2f}, MAE = {metrics.get("MAE", 0):.2f}'
            plt.title(title, fontsize=14)
        else:
            plt.title(f'{model_name} - Predictions vs Actual', fontsize=14)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = self.plots_dir / f"{model_name}_predictions_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved predictions plot: {plot_path}")
        
        return str(plot_path)
    
    def save_comparison_metrics(self, all_results, filename="model_comparison"):
        """
        Save comprehensive comparison of all models
        
        Args:
            all_results: Dictionary with results from all models
            filename: Name for the comparison file
        """
        comparison = {}
        
        for dataset_name, results in all_results.items():
            comparison[dataset_name] = {}
            
            for model_type, model_data in results.items():
                # Skip non-dictionary entries and model objects
                if not isinstance(model_data, dict):
                    continue
                    
                if model_type in ['transformer', 'lstm', 'drl', 'ensemble', 'ppo', 'ddpg']:
                    metrics = model_data.get('metrics', {})
                    
                    # Only save if metrics exist
                    if metrics:
                        comparison[dataset_name][model_type] = {
                            'MSE': float(metrics.get('MSE', 0)),
                            'RMSE': float(metrics.get('RMSE', 0)),
                            'MAE': float(metrics.get('MAE', 0)),
                            'R2': float(metrics.get('R2', 0)),
                            'MAPE': float(metrics.get('MAPE', 0)),
                        }
        
        # Save as JSON
        json_path = self.reports_dir / f"{filename}_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(comparison, f, indent=4, cls=NumpyEncoder)
        
        print(f"✅ Saved model comparison: {json_path}")
        
        # Create comparison visualization
        self._plot_comparison(comparison, filename)
        
        return str(json_path)
    
    def _plot_comparison(self, comparison, filename):
        """Create comparison plots for all models"""
        
        metrics_to_plot = ['RMSE', 'MAE', 'R2', 'MAPE']
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 6))
            
            datasets = list(comparison.keys())
            x = np.arange(len(datasets))
            width = 0.2
            
            # Get all model types
            all_model_types = set()
            for dataset_data in comparison.values():
                all_model_types.update(dataset_data.keys())
            
            model_types = sorted(list(all_model_types))
            
            for i, model_type in enumerate(model_types):
                values = []
                for dataset in datasets:
                    value = comparison[dataset].get(model_type, {}).get(metric, 0)
                    values.append(value)
                
                plt.bar(x + i * width, values, width, label=model_type.capitalize())
            
            plt.xlabel('Dataset Configuration')
            plt.ylabel(metric)
            plt.title(f'Model Comparison - {metric}')
            plt.xticks(x + width * (len(model_types) - 1) / 2, datasets, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            plot_path = self.plots_dir / f"{filename}_{metric}_{self.timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved comparison plot ({metric}): {plot_path}")
    
    def create_summary_report(self, all_results, interpolation_info=None, training_config=None):
        """
        Create a comprehensive summary report
        
        Args:
            all_results: Dictionary with all model results
            interpolation_info: Information about data interpolation
            training_config: Training configuration details
        """
        report = {
            'timestamp': self.timestamp,
            'experiment_summary': {},
            'best_models': {},
            'data_info': {},
            'training_config': training_config or {}
        }
        
        # Add interpolation info
        if interpolation_info:
            report['data_info']['interpolation'] = interpolation_info
        
        # Find best models for each metric
        for metric in ['RMSE', 'MAE', 'R2', 'MAPE']:
            best_value = None
            best_model = None
            best_dataset = None
            
            for dataset_name, results in all_results.items():
                for model_type, model_data in results.items():
                    # Skip non-dictionary entries and model objects
                    if not isinstance(model_data, dict):
                        continue
                        
                    if model_type in ['transformer', 'lstm', 'drl', 'ensemble', 'ppo', 'ddpg']:
                        metrics = model_data.get('metrics', {})
                        value = metrics.get(metric, None)
                        
                        if value is not None:
                            # For R2, higher is better; for others, lower is better
                            if metric == 'R2':
                                if best_value is None or value > best_value:
                                    best_value = value
                                    best_model = f"{model_type.capitalize()}"
                                    best_dataset = dataset_name
                            else:
                                if best_value is None or value < best_value:
                                    best_value = value
                                    best_model = f"{model_type.capitalize()}"
                                    best_dataset = dataset_name
            
            report['best_models'][metric] = {
                'model': best_model,
                'dataset': best_dataset,
                'value': float(best_value) if best_value is not None else None
            }
        
        # Add experiment summary
        for dataset_name, results in all_results.items():
            report['experiment_summary'][dataset_name] = {}
            
            for model_type, model_data in results.items():
                # Skip non-dictionary entries and model objects
                if not isinstance(model_data, dict):
                    continue
                    
                if model_type in ['transformer', 'lstm', 'drl', 'ensemble', 'ppo', 'ddpg']:
                    metrics = model_data.get('metrics', {})
                    config = model_data.get('config', {})
                    
                    # Only add if metrics exist
                    if metrics:
                        report['experiment_summary'][dataset_name][model_type] = {
                            'metrics': {
                                'MSE': float(metrics.get('MSE', 0)),
                                'RMSE': float(metrics.get('RMSE', 0)),
                                'MAE': float(metrics.get('MAE', 0)),
                                'R2': float(metrics.get('R2', 0)),
                                'MAPE': float(metrics.get('MAPE', 0)),
                            },
                            'config': config
                        }
        
        # Save report
        report_path = self.reports_dir / f"experiment_summary_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, cls=NumpyEncoder)
        
        print(f"✅ Saved experiment summary: {report_path}")
        
        # Also create a readable text version
        txt_path = self.reports_dir / f"experiment_summary_{self.timestamp}.txt"
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EXPERIMENT SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            
            f.write("BEST MODELS BY METRIC:\n")
            f.write("-"*80 + "\n")
            for metric, info in report['best_models'].items():
                model_name = info.get('model', 'N/A')
                dataset_name = info.get('dataset', 'N/A')
                value = info.get('value')
                if value is not None:
                    f.write(f"{metric:10s}: {model_name:15s} ({dataset_name}) = {value:.4f}\n")
                else:
                    f.write(f"{metric:10s}: No data available\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED RESULTS BY DATASET:\n")
            f.write("="*80 + "\n\n")
            
            for dataset_name, dataset_results in report['experiment_summary'].items():
                f.write(f"\n{dataset_name}:\n")
                f.write("-"*80 + "\n")
                
                for model_type, model_info in dataset_results.items():
                    f.write(f"\n  {model_type.capitalize()}:\n")
                    metrics = model_info['metrics']
                    f.write(f"    RMSE: {metrics['RMSE']:8.2f}\n")
                    f.write(f"    MAE:  {metrics['MAE']:8.2f}\n")
                    f.write(f"    R²:   {metrics['R2']:8.4f}\n")
                    f.write(f"    MAPE: {metrics['MAPE']:8.2f}%\n")
        
        print(f"✅ Saved text summary: {txt_path}")
        
        return str(report_path), str(txt_path)


def save_all_models_and_metrics(all_results, trained_models, scalers, datasets, 
                                interpolation_info=None, sequence_datasets=None):
    """
    Main function to save all models and metrics
    
    Args:
        all_results: Dictionary with all model results
        trained_models: Dictionary with trained model instances
        scalers: Dictionary with data scalers
        datasets: Dictionary with dataset information
        interpolation_info: Information about data interpolation
        sequence_datasets: Dictionary with sequence dataset information
    
    Returns:
        Dictionary with paths to all saved files
    """
    
    print("\n" + "="*80)
    print("SAVING ALL MODELS AND METRICS")
    print("="*80 + "\n")
    
    saver = ModelSaver()
    saved_files = {
        'models': {},
        'metrics': {},
        'plots': {},
        'reports': {}
    }
    
    # 1. Save all PyTorch models
    print("\n📦 Saving Models...")
    print("-"*80)
    
    for dataset_name, models in trained_models.items():
        saved_files['models'][dataset_name] = {}
        
        # Save Transformer
        if 'transformer' in models:
            model_path = saver.save_pytorch_model(
                models['transformer'],
                f"transformer_{dataset_name}",
                additional_info={
                    'dataset': dataset_name,
                    'model_type': 'transformer',
                    'input_dim': all_results[dataset_name]['input_dim'],
                    'seq_len': all_results[dataset_name]['seq_len']
                }
            )
            saved_files['models'][dataset_name]['transformer'] = model_path
        
        # Save LSTM
        if 'lstm' in models:
            model_path = saver.save_pytorch_model(
                models['lstm'],
                f"lstm_{dataset_name}",
                additional_info={
                    'dataset': dataset_name,
                    'model_type': 'lstm',
                    'input_dim': all_results[dataset_name]['input_dim'],
                    'seq_len': all_results[dataset_name]['seq_len']
                }
            )
            saved_files['models'][dataset_name]['lstm'] = model_path
    
    # 2. Save DRL agents (PPO, DDPG) if available
    if 'ppo_agent' in globals():
        model_path = saver.save_pytorch_model(
            ppo_agent.actor_critic,
            "ppo_agent",
            additional_info={'model_type': 'ppo_drl'}
        )
        saved_files['models']['ppo'] = model_path
    
    if 'ddpg_agent' in globals():
        # Save both actor and critic
        actor_path = saver.save_pytorch_model(
            ddpg_agent.actor,
            "ddpg_actor",
            additional_info={'model_type': 'ddpg_drl_actor'}
        )
        critic_path = saver.save_pytorch_model(
            ddpg_agent.critic,
            "ddpg_critic",
            additional_info={'model_type': 'ddpg_drl_critic'}
        )
        saved_files['models']['ddpg'] = {'actor': actor_path, 'critic': critic_path}
    
    # 3. Save scalers
    print("\n📊 Saving Scalers...")
    print("-"*80)
    
    for dataset_name, scaler_dict in scalers.items():
        scaler_path = saver.save_sklearn_model(scaler_dict, f"scaler_{dataset_name}")
        saved_files['scalers'] = saved_files.get('scalers', {})
        saved_files['scalers'][dataset_name] = scaler_path
    
    # 4. Save all metrics
    print("\n📈 Saving Metrics...")
    print("-"*80)
    
    metrics_path = saver.save_metrics(all_results, "all_model_metrics")
    saved_files['metrics']['all'] = metrics_path
    
    # 5. Save training histories
    print("\n📉 Saving Training Histories...")
    print("-"*80)
    
    for dataset_name, results in all_results.items():
        # Transformer history
        if 'transformer' in results:
            json_path, plot_path = saver.save_training_history(
                results['transformer']['train_losses'],
                results['transformer']['val_losses'],
                f"transformer_{dataset_name}"
            )
            saved_files['plots'][f"transformer_{dataset_name}_history"] = plot_path
        
        # LSTM history
        if 'lstm' in results:
            json_path, plot_path = saver.save_training_history(
                results['lstm']['train_losses'],
                results['lstm']['val_losses'],
                f"lstm_{dataset_name}"
            )
            saved_files['plots'][f"lstm_{dataset_name}_history"] = plot_path
    
    # 6. Save prediction plots
    print("\n🎨 Saving Prediction Plots...")
    print("-"*80)
    
    for dataset_name, results in all_results.items():
        # Transformer predictions
        if 'transformer' in results:
            plot_path = saver.save_predictions_plot(
                results['transformer']['metrics']['predictions'],
                results['transformer']['metrics']['targets'],
                f"transformer_{dataset_name}",
                results['transformer']['metrics']
            )
            saved_files['plots'][f"transformer_{dataset_name}_predictions"] = plot_path
        
        # LSTM predictions
        if 'lstm' in results:
            plot_path = saver.save_predictions_plot(
                results['lstm']['metrics']['predictions'],
                results['lstm']['metrics']['targets'],
                f"lstm_{dataset_name}",
                results['lstm']['metrics']
            )
            saved_files['plots'][f"lstm_{dataset_name}_predictions"] = plot_path
    
    # 7. Save model comparison
    print("\n🏆 Saving Model Comparisons...")
    print("-"*80)
    
    comparison_path = saver.save_comparison_metrics(all_results)
    saved_files['reports']['comparison'] = comparison_path
    
    # 8. Create summary report
    print("\n📄 Creating Summary Report...")
    print("-"*80)
    
    report_path, txt_path = saver.create_summary_report(
        all_results,
        interpolation_info=interpolation_info
    )
    saved_files['reports']['summary_json'] = report_path
    saved_files['reports']['summary_txt'] = txt_path
    
    print("\n" + "="*80)
    print("✅ ALL MODELS AND METRICS SAVED SUCCESSFULLY!")
    print("="*80)
    
    # Print summary of saved files
    print("\n📁 Summary of Saved Files:")
    print("-"*80)
    print(f"Models saved:          {sum(len(v) if isinstance(v, dict) else 1 for v in saved_files['models'].values())}")
    print(f"Scalers saved:         {len(saved_files.get('scalers', {}))}")
    print(f"Metric files saved:    {len(saved_files['metrics'])}")
    print(f"Plots saved:           {len(saved_files['plots'])}")
    print(f"Reports saved:         {len(saved_files['reports'])}")
    
    return saved_files


if __name__ == "__main__":
    print("This script should be imported and called from the notebook.")
    print("Use: save_all_models_and_metrics(all_results, trained_models, scalers, datasets)")
