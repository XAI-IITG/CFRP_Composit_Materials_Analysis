"""
KG-Enriched Transformer Training Script
=========================================
Trains the KGEnrichedTransformerRULPredictor model using:
- Preprocessed time-series data from the existing pipeline
- Node2Vec KG embeddings as additional input context

Usage:
    cd <project_root>
    python scripts/kg_enriched_training.py
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'notebooks'))

from modules.model_architecture import (
    TransformerRULPredictor,
    KGEnrichedTransformerRULPredictor
)
from modules.kg_embeddings import KGEmbeddingGenerator


# ============================================================================
# Custom Dataset with KG embeddings
# ============================================================================

class KGEnrichedDataset(Dataset):
    """Dataset that returns (time_series_seq, kg_embedding, target_rul)."""
    
    def __init__(self, X, y, specimen_ids, kg_gen):
        """
        Args:
            X: Time-series sequences (n_samples, seq_len, n_features) numpy array
            y: Target RUL values (n_samples,) numpy array
            specimen_ids: Specimen IDs per sample (n_samples,) numpy array
            kg_gen: Fitted KGEmbeddingGenerator instance
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        # Precompute KG embeddings for each sample
        kg_embeddings = []
        for sid in specimen_ids:
            emb = kg_gen.get_coupon_embedding(sid)
            kg_embeddings.append(emb)
        self.kg_embed = torch.FloatTensor(np.array(kg_embeddings))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.kg_embed[idx], self.y[idx]


# ============================================================================
# Training / Evaluation functions
# ============================================================================

def train_epoch_kg(model, loader, criterion, optimizer, device):
    """Train for one epoch with KG embeddings."""
    model.train()
    total_loss = 0
    for batch_X, batch_kg, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_kg = batch_kg.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X, batch_kg)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate_kg(model, loader, criterion, device):
    """Evaluate model with KG embeddings."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_X, batch_kg, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_kg = batch_kg.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X, batch_kg)
            loss = criterion(outputs.squeeze(), batch_y)
            
            total_loss += loss.item()
            predictions.extend(outputs.squeeze().cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    
    return avg_loss, mae, rmse, predictions, targets


def train_model_kg(model, train_loader, val_loader, criterion, optimizer,
                   scheduler, num_epochs, patience, device, model_name):
    """Train model with early stopping."""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_mae': [], 'val_rmse': []
    }
    
    for epoch in range(num_epochs):
        train_loss = train_epoch_kg(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae, val_rmse, _, _ = evaluate_kg(model, val_loader, criterion, device)
        
        if scheduler is not None:
            scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"MAE: {val_mae:.4f} | "
                  f"RMSE: {val_rmse:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    print(f"✓ Training complete! Best val loss: {best_val_loss:.4f}")
    
    return model, history


# ============================================================================
# Main training pipeline
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Paths
    models_dir = project_root / 'outputs' / 'saved_models'
    kg_ttl_path = project_root / 'data' / 'ontology' / 'cfrp_ontology_populated.ttl'
    
    # ========================================================================
    # Step 1: Load preprocessed data
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: Loading preprocessed data")
    print("="*70)
    
    with open(models_dir / 'preprocessed_data_combined.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    spec_train = data['specimen_ids_train']
    spec_val = data['specimen_ids_val']
    spec_test = data['specimen_ids_test']
    n_features = data['n_features']
    seq_length = data['sequence_length']

    # ========================================================================


    with open(models_dir / 'target_scaler_combined.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    
    print(f"  Train: {X_train.shape[0]} sequences")
    print(f"  Val:   {X_val.shape[0]} sequences")
    print(f"  Test:  {X_test.shape[0]} sequences")
    print(f"  Features: {n_features}, Seq length: {seq_length}")
    print(f"  Specimens: {np.unique(spec_train)}")
    
    # ========================================================================
    # Step 2: Generate KG embeddings
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: Generating KG embeddings")
    print("="*70)
    
    kg_gen = KGEmbeddingGenerator(
        ttl_path=kg_ttl_path,
        embed_dim=16,
        walk_length=20,
        num_walks=50,
        seed=42
    )
    kg_gen.fit()
    
    kg_embed_dim = kg_gen.kg_embed_dim
    print(f"  KG embedding dim: {kg_embed_dim}")
    
    # Show what embeddings look like for each specimen
    print("\n  Per-specimen KG embedding norms:")
    for sid in np.unique(spec_train):
        emb = kg_gen.get_coupon_embedding(sid)
        print(f"    {sid}: norm={np.linalg.norm(emb):.4f}")
    
    # ========================================================================
    # Step 3: Create datasets with KG embeddings
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: Creating KG-enriched datasets")
    print("="*70)
    
    train_dataset = KGEnrichedDataset(X_train, y_train, spec_train, kg_gen)
    val_dataset = KGEnrichedDataset(X_val, y_val, spec_val, kg_gen)
    test_dataset = KGEnrichedDataset(X_test, y_test, spec_test, kg_gen)
    
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Verify shapes
    sample = next(iter(train_loader))
    print(f"  Sample batch shapes: X={sample[0].shape}, KG={sample[1].shape}, y={sample[2].shape}")
    
    # ========================================================================
    # Step 4: Initialize and train KG-enriched model
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: Training KG-Enriched Transformer")
    print("="*70)
    
    config = {
        'input_dim': n_features,         # 16
        'kg_embed_dim': kg_embed_dim,     # 80
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 128,
        'dropout': 0.1,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 100,
        'patience': 15
    }
    
    print("\nModel Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    model = KGEnrichedTransformerRULPredictor(
        input_dim=config['input_dim'],
        kg_embed_dim=config['kg_embed_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    model, history = train_model_kg(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=config['epochs'],
        patience=config['patience'],
        device=device,
        model_name="KG-Enriched Transformer"
    )
    
    # Save model
    save_path = models_dir / 'transformer_kg_enriched.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\n✓ Model saved to: {save_path}")
    
    # ========================================================================
    # Step 5: Evaluate on test set
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: Test Set Evaluation")
    print("="*70)
    
    test_loss, test_mae_scaled, test_rmse_scaled, preds_scaled, targets_scaled = \
        evaluate_kg(model, test_loader, criterion, device)
    
    # Convert to actual cycles
    preds_actual = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    targets_actual = target_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()
    
    mae_actual = np.mean(np.abs(preds_actual - targets_actual))
    rmse_actual = np.sqrt(np.mean((preds_actual - targets_actual)**2))
    
    print(f"\nKG-Enriched Transformer Results:")
    print(f"  Test Loss (MSE):          {test_loss:.4f}")
    print(f"  Test MAE (scaled):        {test_mae_scaled:.4f}")
    print(f"  Test RMSE (scaled):       {test_rmse_scaled:.4f}")
    print(f"  Test MAE (actual cycles): {mae_actual:,.0f}")
    print(f"  Test RMSE (actual cycles):{rmse_actual:,.0f}")
    
    # ========================================================================
    # Step 6: Run baseline comparison
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: Baseline Comparison")
    print("="*70)
    
    # Load and evaluate baseline Transformer
    baseline_model = TransformerRULPredictor(
        input_dim=n_features,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1
    ).to(device)
    
    baseline_path = models_dir / 'transformer_rul_combined.pth'
    if baseline_path.exists():
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))
        
        # Evaluate baseline on test set (uses original DataLoader without KG)
        from torch.utils.data import TensorDataset
        baseline_test = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        baseline_loader = DataLoader(baseline_test, batch_size=BATCH_SIZE, shuffle=False)
        
        baseline_model.eval()
        baseline_preds = []
        baseline_targets_list = []
        with torch.no_grad():
            for batch_X, batch_y in baseline_loader:
                batch_X = batch_X.to(device)
                outputs = baseline_model(batch_X)
                baseline_preds.extend(outputs.squeeze().cpu().numpy())
                baseline_targets_list.extend(batch_y.numpy())
        
        baseline_preds = np.array(baseline_preds)
        baseline_preds_actual = target_scaler.inverse_transform(
            baseline_preds.reshape(-1, 1)).flatten()
        
        baseline_mae = np.mean(np.abs(baseline_preds_actual - targets_actual))
        baseline_rmse = np.sqrt(np.mean((baseline_preds_actual - targets_actual)**2))
        
        print(f"\n  Baseline Transformer MAE: {baseline_mae:,.0f} cycles")
        print(f"  Baseline Transformer RMSE: {baseline_rmse:,.0f} cycles")
    else:
        baseline_mae = None
        print(f"\n  ⚠ Baseline model not found at {baseline_path}")
    
    # ========================================================================
    # Step 7: Summary
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    print(f"\n  {'Model':<30} {'MAE (cycles)':>15} {'RMSE (cycles)':>15}")
    print(f"  {'-'*60}")
    
    if baseline_mae is not None:
        print(f"  {'Baseline Transformer':<30} {baseline_mae:>15,.0f} {baseline_rmse:>15,.0f}")
    
    print(f"  {'KG-Enriched Transformer':<30} {mae_actual:>15,.0f} {rmse_actual:>15,.0f}")
    
    if baseline_mae is not None:
        improvement = baseline_mae - mae_actual
        improvement_pct = (improvement / baseline_mae) * 100
        
        if improvement > 0:
            print(f"\n  ✓ KG enrichment improved MAE by {improvement:,.0f} cycles ({improvement_pct:.1f}%)")
        else:
            print(f"\n  ⚠ KG enrichment did not improve MAE (diff: {improvement:,.0f} cycles)")
    
    print(f"\n  Prediction range: [{preds_actual.min():,.0f}, {preds_actual.max():,.0f}]")
    print(f"  Actual range:     [{targets_actual.min():,.0f}, {targets_actual.max():,.0f}]")
    print(f"  Prediction std:   {np.std(preds_actual):,.0f}")
    
    # Save results
    results = {
        'kg_enriched_mae': float(mae_actual),
        'kg_enriched_rmse': float(rmse_actual),
        'baseline_mae': float(baseline_mae) if baseline_mae is not None else None,
        'baseline_rmse': float(baseline_rmse) if baseline_mae is not None else None,
        'config': config,
        'history': history,
        'kg_graph_stats': kg_gen.get_graph_stats(),
    }
    
    with open(models_dir / 'kg_enriched_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Results saved to: {models_dir / 'kg_enriched_results.pkl'}")
    
    print("\n" + "="*70)
    print("KG-ENRICHED TRAINING COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
