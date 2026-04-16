"""
TS-RMF Random Split Experiment
==================================================

Runs a random split experiment (K-Fold cross validation without grouping)
on the 4-specimen dataset, predicting fractional RUL but converting it
back to absolute cycles for final metrics comparison.

This is to replicate the data leakage evaluation style that yielded ~5000 cycles 
error previously, and compare the Baseline Transformer with TS-RMF under 
these conditions.

Usage:
    python scripts/run_random_split_eval.py --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from time_sliced_graph_builder import TimeSlicedCFRPKG, collate_graph_batches
from grouped_split_experiment import (
    PlainTransformerBaseline,
    TimeSlicedRelationalKGRUL,
    CFRPWindowDataset,
    SequenceStandardizer,
    auto_device,
    make_stage_labels
)


@dataclass
class FoldData:
    X: np.ndarray
    y_rul: np.ndarray
    y_stage: np.ndarray
    specimen_ids: np.ndarray
    current_cycles: np.ndarray
    max_lives: np.ndarray


class RandomSplitDataset(Dataset):
    def __init__(self, fd: FoldData):
        self.X = fd.X.astype(np.float32)
        self.y_rul = fd.y_rul.astype(np.float32)
        self.y_stage = fd.y_stage.astype(np.int64)
        self.specimen_ids = fd.specimen_ids.astype(str)
        self.current_cycles = fd.current_cycles.astype(np.int64)
        self.max_lives = fd.max_lives.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return {
            "x": self.X[idx],
            "y_rul": self.y_rul[idx],
            "y_stage": self.y_stage[idx],
            "specimen_id": self.specimen_ids[idx],
            "current_cycle": self.current_cycles[idx],
            "max_life": self.max_lives[idx]
        }


def collate_fn(batch):
    x = torch.tensor(np.stack([b["x"] for b in batch]), dtype=torch.float32)
    y_rul = torch.tensor([b["y_rul"] for b in batch], dtype=torch.float32)
    y_stage = torch.tensor([b["y_stage"] for b in batch], dtype=torch.long)
    specimen_ids = [b["specimen_id"] for b in batch]
    current_cycles = [b["current_cycle"] for b in batch]
    max_lives = np.array([b["max_life"] for b in batch], dtype=np.float32)
    return {
        "x": x,
        "y_rul": y_rul,
        "y_stage": y_stage,
        "specimen_ids": specimen_ids,
        "current_cycles": current_cycles,
        "max_lives": max_lives
    }


def make_loader(fd: FoldData, batch_size: int, shuffle: bool) -> DataLoader:
    ds = RandomSplitDataset(fd)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=False)


@torch.no_grad()
def evaluate_in_cycles(model, loader, kg_builder, device):
    model.eval()
    y_true_cycles, y_pred_cycles = [], []
    
    for batch in loader:
        x = batch["x"].to(device)
        graphs = collate_graph_batches(kg_builder, batch["specimen_ids"], batch["current_cycles"])
        outputs = model(x, graphs)
        
        y_pred_frac = outputs["rul"].detach().cpu().numpy()
        y_true_frac = batch["y_rul"].detach().cpu().numpy()
        max_lives = batch["max_lives"]
        
        # Convert fraction back to absolute cycles using max_lives
        y_t_cyc = y_true_frac * max_lives
        y_p_cyc = y_pred_frac * max_lives
        
        y_true_cycles.extend(y_t_cyc)
        y_pred_cycles.extend(y_p_cyc)
        
    y_true_arr = np.array(y_true_cycles)
    y_pred_arr = np.array(y_pred_cycles)
    
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr)**2)))
    
    return mae, rmse


def train_one_fold(model_name: str, train_fd: FoldData, val_fd: FoldData, test_fd: FoldData, kg_builder: TimeSlicedCFRPKG, epochs: int, batch_size: int, lr: float, lambda_stage: float, lambda_kg: float, device: torch.device):
    train_loader = make_loader(train_fd, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_fd, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(test_fd, batch_size=batch_size, shuffle=False)

    num_stages = int(max(train_fd.y_stage.max(), val_fd.y_stage.max(), test_fd.y_stage.max()) + 1)
    
    if model_name == "plain_transformer":
        model = PlainTransformerBaseline(input_dim=train_fd.X.shape[-1], num_stages=num_stages, d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    else:
        model = TimeSlicedRelationalKGRUL(
            input_dim=train_fd.X.shape[-1],
            node_feat_dim=kg_builder.num_feat_dim,
            num_node_types=len(kg_builder.node_type2id),
            num_relations=max(1, len(kg_builder.rel2id) * 2 + 8),
            d_model=64, nhead=4, num_layers=2, dim_feedforward=128, num_stages=num_stages,
        )
    
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    patience = 10
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y_rul = batch["y_rul"].to(device)
            y_stage = batch["y_stage"].to(device)
            graphs = collate_graph_batches(kg_builder, batch["specimen_ids"], batch["current_cycles"])

            outputs = model(x, graphs)
            rul_loss = nn.functional.smooth_l1_loss(outputs["rul"], y_rul)
            stage_loss = nn.functional.cross_entropy(outputs["stage_logits"], y_stage)
            loss = rul_loss + lambda_stage * stage_loss
            if model_name == "ts_rmf":
                loss += lambda_kg * model.transe_loss(graphs)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y_rul = batch["y_rul"].to(device)
                y_stage = batch["y_stage"].to(device)
                graphs = collate_graph_batches(kg_builder, batch["specimen_ids"], batch["current_cycles"])
                
                outputs = model(x, graphs)
                vrul = nn.functional.smooth_l1_loss(outputs["rul"], y_rul)
                vstage = nn.functional.cross_entropy(outputs["stage_logits"], y_stage)
                vloss = vrul + lambda_stage * vstage
                if model_name == "ts_rmf":
                    vloss += lambda_kg * model.transe_loss(graphs)
                val_losses.append(float(vloss.item()))
                
        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")

        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate in absolute cycles for easy comparison
    mae, rmse = evaluate_in_cycles(model, test_loader, kg_builder, device)
    return mae, rmse


def main():
    parser = argparse.ArgumentParser(description="Random-split experiment for TS-RMF vs Baseline in Absolute Cycles")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-stage", type=float, default=0.5)
    parser.add_argument("--lambda-kg", type=float, default=0.05)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device if args.device else auto_device()
    project_root = Path(__file__).resolve().parent.parent
    ttl_path = str(project_root / "data" / "ontology" / "cfrp_ontology_populated.ttl")
    npz_path = str(project_root / "data" / "processed" / "cfrp_windows_4spec.npz")
    
    # Load dataset
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y_rul = data["y_rul"].astype(np.float32)
    specimen_ids = data["specimen_ids"].astype(str)
    current_cycles = data["current_cycles"].astype(np.int64)
    max_lives = data["max_lives"].astype(np.int64)
    y_stage = make_stage_labels(y_rul)
    
    kg_builder = TimeSlicedCFRPKG(ttl_path)
    device_obj = torch.device(device)

    print("=" * 70)
    print("RANDOM SPLIT LEAKAGE SCENARIO (Reported in Absolute Cycles)")
    print("=" * 70)
    print(f"  Dataset: {npz_path}")
    print(f"  Total sequences: {len(X)}")
    print(f"  Evaluated with {args.n_splits}-Fold CV (Random Split, no grouped leakage prevention)")
    
    outer = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    
    models = ["plain_transformer", "ts_rmf"]
    results = {m: {"mae": [], "rmse": []} for m in models}

    for fold_id, (trainval_idx, test_idx) in enumerate(outer.split(X), start=1):
        print(f"\n  --- [ Fold {fold_id} / {args.n_splits} ] ---")
        train_idx, val_idx = train_test_split(trainval_idx, test_size=0.2, random_state=42 + fold_id)

        # Quick standardizer for X per fold
        x_scaler = SequenceStandardizer().fit(X[train_idx])
        
        def subset_fd(idx):
            return FoldData(
                X=x_scaler.transform(X[idx]),
                y_rul=y_rul[idx],
                y_stage=y_stage[idx],
                specimen_ids=specimen_ids[idx],
                current_cycles=current_cycles[idx],
                max_lives=max_lives[idx]
            )
            
        train_fd = subset_fd(train_idx)
        val_fd = subset_fd(val_idx)
        test_fd = subset_fd(test_idx)

        for model_name in models:
            mae, rmse = train_one_fold(
                model_name=model_name,
                train_fd=train_fd,
                val_fd=val_fd,
                test_fd=test_fd,
                kg_builder=kg_builder,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lambda_stage=args.lambda_stage,
                lambda_kg=args.lambda_kg,
                device=device_obj,
            )
            results[model_name]["mae"].append(mae)
            results[model_name]["rmse"].append(rmse)
            print(f"    > {model_name:20s}: MAE = {mae:,.0f} cycles | RMSE = {rmse:,.0f} cycles")

    print("\n" + "=" * 70)
    print("FINAL COMPARISON (Data Leakage Random Split Scenario)")
    print("=" * 70)
    print(f"\n  {'Model':<30} {'Mean MAE (cycles)':>20} {'Mean RMSE (cycles)':>20}")
    print(f"  {'-' * 74}")

    for m in models:
        mean_mae = np.mean(results[m]["mae"])
        std_mae = np.std(results[m]["mae"])
        mean_rmse = np.mean(results[m]["rmse"])
        std_rmse = np.std(results[m]["rmse"])
        print(f"  {m:<30} {mean_mae:>10,.0f} +/- {std_mae:<7,.0f} {mean_rmse:>10,.0f} +/- {std_rmse:<7,.0f}")

    diff = np.mean(results["plain_transformer"]["mae"]) - np.mean(results["ts_rmf"]["mae"])
    if diff > 0:
        print(f"\n  [OK] TS-RMF improved MAE by {diff:,.0f} cycles over Baseline under Random Split.")
    else:
        print(f"\n  [INFO] Baseline is still better. TS-RMF MAE was {-diff:,.0f} cycles higher under Random Split.")


if __name__ == "__main__":
    from dataclasses import dataclass
    main()
