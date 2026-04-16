from __future__ import annotations

"""
Grouped-split experiment runner for time-sliced relational KG CFRP RUL models.

Expected NPZ keys:
    X              : float array [N, T, D]
    y_rul          : float array [N]
    specimen_ids   : string array [N]
    current_cycles : int array [N]
Optional keys:
    y_stage        : int array [N]

Alternatively, use --pickle to load from the existing preprocessed pickle format.
If neither --dataset nor --pickle is provided, the script runs a synthetic smoke test.
"""

import argparse
import json
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# Filter unhelpful sklearn unpickling warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", category=UserWarning)

# Handle imports whether run from scripts/ dir or project root
try:
    from time_sliced_graph_builder import TimeSlicedCFRPKG, collate_graph_batches
    from time_sliced_relational_kg_rul import TimeSlicedRelationalKGRUL
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from time_sliced_graph_builder import TimeSlicedCFRPKG, collate_graph_batches
    from time_sliced_relational_kg_rul import TimeSlicedRelationalKGRUL


class PlainTransformerBaseline(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, num_stages: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.rul_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.stage_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, num_stages))

    def forward(self, x_seq: torch.Tensor, _graphs=None) -> Dict[str, torch.Tensor]:
        z = self.input_proj(x_seq)
        z = self.encoder(z)
        pooled = z.mean(dim=1)
        rul = self.rul_head(pooled).squeeze(-1)
        stage_logits = self.stage_head(pooled)
        return {"rul": rul, "stage_logits": stage_logits}


@dataclass
class FoldData:
    X: np.ndarray
    y_rul: np.ndarray
    y_stage: np.ndarray
    specimen_ids: np.ndarray
    current_cycles: np.ndarray


class CFRPWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y_rul: np.ndarray, y_stage: np.ndarray, specimen_ids: np.ndarray, current_cycles: np.ndarray):
        self.X = X.astype(np.float32)
        self.y_rul = y_rul.astype(np.float32)
        self.y_stage = y_stage.astype(np.int64)
        self.specimen_ids = specimen_ids.astype(str)
        self.current_cycles = current_cycles.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return {
            "x": self.X[idx],
            "y_rul": self.y_rul[idx],
            "y_stage": self.y_stage[idx],
            "specimen_id": self.specimen_ids[idx],
            "current_cycle": self.current_cycles[idx],
        }


class SequenceStandardizer:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SequenceStandardizer":
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fit before transform")
        return ((X - self.mean_) / self.std_).astype(np.float32)


class TargetStandardizer:
    def __init__(self):
        self.mean_: float = 0.0
        self.std_: float = 1.0

    def fit(self, y: np.ndarray) -> "TargetStandardizer":
        self.mean_ = float(y.mean())
        self.std_ = float(y.std())
        if self.std_ < 1e-8:
            self.std_ = 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return (y * self.std_ + self.mean_).astype(np.float32)


def collate_fn(batch):
    x = torch.tensor(np.stack([b["x"] for b in batch]), dtype=torch.float32)
    y_rul = torch.tensor([b["y_rul"] for b in batch], dtype=torch.float32)
    y_stage = torch.tensor([b["y_stage"] for b in batch], dtype=torch.long)
    specimen_ids = [b["specimen_id"] for b in batch]
    current_cycles = [int(b["current_cycle"]) for b in batch]
    return {
        "x": x,
        "y_rul": y_rul,
        "y_stage": y_stage,
        "specimen_ids": specimen_ids,
        "current_cycles": current_cycles,
    }


def make_stage_labels(y_rul: np.ndarray, n_stages: int = 4) -> np.ndarray:
    qs = np.quantile(y_rul, np.linspace(0, 1, n_stages + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    return np.digitize(y_rul, qs[1:-1], right=True).astype(np.int64)


def _generate_synthetic() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a small synthetic dataset for smoke testing."""
    rng = np.random.default_rng(42)
    n, T, d = 96, 10, 16
    specimens = np.array([f"S{sid}" for sid in np.repeat([11, 12, 17, 18, 19, 20], n // 6)])
    current_cycles = np.tile(np.linspace(1000, 30000, n // 6, dtype=np.int64), 6)
    X = rng.normal(size=(n, T, d)).astype(np.float32)
    degradation = (current_cycles / current_cycles.max()).astype(np.float32)
    X[:, :, 0] += degradation[:, None] * 2.0
    X[:, :, 1] += degradation[:, None] * 1.2
    y_rul = (1.0 - degradation) * 50000.0 + rng.normal(scale=2500.0, size=n)
    y_stage = make_stage_labels(y_rul)
    return X, y_rul.astype(np.float32), y_stage, specimens.astype(str), current_cycles.astype(np.int64)


def _load_from_pickle(
    pickle_path: str,
    scaler_path: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from the existing CFRP preprocessed pickle format.

    Recovers raw cycle counts from the standardized 'cycles' column (idx 14)
    using the feature scaler. Recombines train/val/test into a single array
    so grouped splitting can re-partition properly.
    """
    CYCLES_IDX = 14

    pkl_path = Path(pickle_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    X = np.concatenate([data["X_train"], data["X_val"], data["X_test"]], axis=0).astype(np.float32)
    y_rul = np.concatenate([data["y_train"], data["y_val"], data["y_test"]], axis=0).astype(np.float32)
    specimen_ids = np.concatenate(
        [data["specimen_ids_train"], data["specimen_ids_val"], data["specimen_ids_test"]], axis=0
    ).astype(str)

    # Recover raw cycle counts
    if scaler_path is not None:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        cycles_mean = scaler.mean_[CYCLES_IDX]
        cycles_scale = scaler.scale_[CYCLES_IDX]
        cycles_scaled = X[:, -1, CYCLES_IDX]
        current_cycles = np.maximum((cycles_scaled * cycles_scale + cycles_mean), 0).astype(np.int64)
    else:
        # Fallback: use the raw column value (may be scaled)
        current_cycles = np.maximum(X[:, -1, CYCLES_IDX], 0).astype(np.int64)
        print("  WARNING: No feature scaler provided -- cycle counts may be inaccurate")

    y_stage = make_stage_labels(y_rul)

    print(f"  Loaded from pickle: {X.shape[0]} samples, {X.shape[2]} features")
    print(f"  Specimens: {np.unique(specimen_ids)}")
    print(f"  Cycle range: [{current_cycles.min():,} -- {current_cycles.max():,}]")

    return X, y_rul, y_stage, specimen_ids, current_cycles


def load_dataset(
    npz_path: str | None = None,
    pickle_path: str | None = None,
    scaler_path: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data from NPZ, pickle, or generate synthetic."""
    if pickle_path is not None:
        return _load_from_pickle(pickle_path, scaler_path)

    if npz_path is None:
        return _generate_synthetic()

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y_rul = data["y_rul"].astype(np.float32)
    specimen_ids = data["specimen_ids"].astype(str)
    current_cycles = data["current_cycles"].astype(np.int64)
    if "y_stage" in data:
        y_stage = data["y_stage"].astype(np.int64)
    else:
        y_stage = make_stage_labels(y_rul)
    return X, y_rul, y_stage, specimen_ids, current_cycles


def nested_group_split(train_idx: np.ndarray, groups: np.ndarray, val_fraction: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    train_rel, val_rel = next(splitter.split(train_idx, groups=groups[train_idx]))
    return train_idx[train_rel], train_idx[val_rel]


def subset_fold(X, y_rul, y_stage, specimen_ids, current_cycles, idx: np.ndarray) -> FoldData:
    return FoldData(X[idx], y_rul[idx], y_stage[idx], specimen_ids[idx], current_cycles[idx])


def prepare_fold_data(train_fold: FoldData, val_fold: FoldData, test_fold: FoldData):
    x_scaler = SequenceStandardizer().fit(train_fold.X)
    y_scaler = TargetStandardizer().fit(train_fold.y_rul)

    def tx(fd: FoldData) -> FoldData:
        return FoldData(
            X=x_scaler.transform(fd.X),
            y_rul=y_scaler.transform(fd.y_rul),
            y_stage=fd.y_stage,
            specimen_ids=fd.specimen_ids,
            current_cycles=fd.current_cycles,
        )

    return tx(train_fold), tx(val_fold), tx(test_fold), x_scaler, y_scaler


def make_loader(fd: FoldData, batch_size: int, shuffle: bool) -> DataLoader:
    ds = CFRPWindowDataset(fd.X, fd.y_rul, fd.y_stage, fd.specimen_ids, fd.current_cycles)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


def per_specimen_mae(y_true: np.ndarray, y_pred: np.ndarray, specimen_ids: np.ndarray) -> Dict[str, float]:
    out = {}
    for sid in sorted(set(specimen_ids.tolist())):
        mask = specimen_ids == sid
        out[str(sid)] = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
    return out


def load_target_scaler(path: str | Path) -> Optional[object]:
    """Load the original MinMaxScaler used to scale y_rul to [0, 1]."""
    path = Path(path)
    if not path.exists():
        print(f"  WARNING: target scaler not found at {path} -- metrics will be in [0,1] space")
        return None
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def to_real_cycles(y: np.ndarray, target_scaler) -> np.ndarray:
    """Convert [0, 1] normalized RUL values back to real cycle counts."""
    if target_scaler is None:
        return y
    return target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()


@torch.no_grad()
def evaluate(model, loader, kg_builder, device, y_scaler, target_scaler=None):
    model.eval()
    y_true_all, y_pred_all, sid_all = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        graphs = collate_graph_batches(kg_builder, batch["specimen_ids"], batch["current_cycles"])
        graphs = [g for g in graphs]
        outputs = model(x, graphs)
        y_pred = outputs["rul"].detach().cpu().numpy()
        y_true = batch["y_rul"].detach().cpu().numpy()
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        sid_all.extend(batch["specimen_ids"])

    # Undo the per-fold TargetStandardizer -> back to [0, 1]
    y_true = y_scaler.inverse_transform(np.concatenate(y_true_all))
    y_pred = y_scaler.inverse_transform(np.concatenate(y_pred_all))

    # Undo the original MinMaxScaler -> back to real cycle counts
    y_true = to_real_cycles(y_true, target_scaler)
    y_pred = to_real_cycles(y_pred, target_scaler)

    specimen_ids = np.array(sid_all, dtype=str)

    metrics = mae_rmse(y_true, y_pred)
    metrics["per_specimen_mae"] = per_specimen_mae(y_true, y_pred, specimen_ids)
    return metrics, y_true, y_pred, specimen_ids


def build_model(model_name: str, input_dim: int, kg_builder: TimeSlicedCFRPKG, num_stages: int, device: torch.device):
    if model_name == "plain_transformer":
        model = PlainTransformerBaseline(input_dim=input_dim, num_stages=num_stages, d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    elif model_name == "ts_rmf":
        model = TimeSlicedRelationalKGRUL(
            input_dim=input_dim,
            node_feat_dim=kg_builder.num_feat_dim,
            num_node_types=len(kg_builder.node_type2id),
            num_relations=max(1, len(kg_builder.rel2id) * 2 + 8),
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            num_stages=num_stages,
        )
    else:
        raise ValueError(f"Unknown model_name={model_name}")
    return model.to(device)


def train_one_fold(model_name: str, train_fd: FoldData, val_fd: FoldData, test_fd: FoldData, kg_builder: TimeSlicedCFRPKG, epochs: int, batch_size: int, lr: float, lambda_stage: float, lambda_kg: float, device: torch.device):
    train_loader = make_loader(train_fd, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_fd, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(test_fd, batch_size=batch_size, shuffle=False)

    num_stages = int(max(train_fd.y_stage.max(), val_fd.y_stage.max(), test_fd.y_stage.max()) + 1)
    model = build_model(model_name, train_fd.X.shape[-1], kg_builder, num_stages, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    patience = 10
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y_rul = batch["y_rul"].to(device)
            y_stage = batch["y_stage"].to(device)
            graphs = collate_graph_batches(kg_builder, batch["specimen_ids"], batch["current_cycles"])

            outputs = model(x, graphs)
            rul_loss = nn.functional.smooth_l1_loss(outputs["rul"], y_rul)
            stage_loss = nn.functional.cross_entropy(outputs["stage_logits"], y_stage)
            total_loss = rul_loss + lambda_stage * stage_loss
            if model_name == "ts_rmf":
                kg_loss = model.transe_loss(graphs)
                total_loss = total_loss + lambda_kg * kg_loss
            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            running += float(total_loss.item())

        # Compute validation on scaled target space to track early stopping without data leakage.
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
                    vloss = vloss + lambda_kg * model.transe_loss(graphs)
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

    return model, test_loader


def run_grouped_experiment(
    dataset_path: str | None,
    ttl_path: str,
    model_name: str,
    n_splits: int,
    val_fraction: float,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_stage: float,
    lambda_kg: float,
    random_state: int,
    device: str,
    pickle_path: str | None = None,
    scaler_path: str | None = None,
    target_scaler_path: str | None = None,
    checkpoint_dir: str | None = None,
):
    X, y_rul, y_stage, specimen_ids, current_cycles = load_dataset(
        npz_path=dataset_path,
        pickle_path=pickle_path,
        scaler_path=scaler_path,
    )
    kg_builder = TimeSlicedCFRPKG(ttl_path)
    device_obj = torch.device(device)

    # Load the original target scaler (MinMaxScaler) to convert back to cycles
    target_scaler = None
    if target_scaler_path is not None:
        target_scaler = load_target_scaler(target_scaler_path)
        if target_scaler is not None:
            print(f"  [OK] Loaded target scaler: range=[{target_scaler.data_min_[0]:,.0f}, {target_scaler.data_max_[0]:,.0f}] cycles")

    units = "cycles" if target_scaler is not None else "normalized"

    outer = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold_id, (trainval_idx, test_idx) in enumerate(outer.split(X, y_rul, groups=specimen_ids), start=1):
        print(f"\n  --- [ Fold {fold_id} / {n_splits} ] ---")
        train_idx, val_idx = nested_group_split(trainval_idx, specimen_ids, val_fraction=val_fraction, random_state=random_state + fold_id)

        train_raw = subset_fold(X, y_rul, y_stage, specimen_ids, current_cycles, train_idx)
        val_raw = subset_fold(X, y_rul, y_stage, specimen_ids, current_cycles, val_idx)
        test_raw = subset_fold(X, y_rul, y_stage, specimen_ids, current_cycles, test_idx)

        # Log graph diagnostic for this fold's test specimens
        test_sids = sorted(set(specimen_ids[test_idx]))
        for sid in test_sids:
            mask = specimen_ids[test_idx] == sid
            cycs = current_cycles[test_idx][mask]
            try:
                g_lo = kg_builder.build_sample_graph(sid, int(cycs.min()))
                g_hi = kg_builder.build_sample_graph(sid, int(cycs.max()))
                print(f"    > Test Specimen {sid} | Cycles: {cycs.min():,} -> {cycs.max():,} | Nodes: {g_lo.node_count} -> {g_hi.node_count}")
            except Exception as e:
                print(f"    > Test Specimen {sid} | Graph diagnostic ERROR: {e}")

        train_fd, val_fd, test_fd, x_scaler, y_scaler = prepare_fold_data(train_raw, val_raw, test_raw)

        model, test_loader = train_one_fold(
            model_name=model_name,
            train_fd=train_fd,
            val_fd=val_fd,
            test_fd=test_fd,
            kg_builder=kg_builder,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lambda_stage=lambda_stage,
            lambda_kg=lambda_kg,
            device=device_obj,
        )

        test_metrics, y_true, y_pred, sid_test = evaluate(
            model, test_loader, kg_builder, device_obj, y_scaler,
            target_scaler=target_scaler,
        )

        # Save per-fold checkpoint
        if checkpoint_dir is not None:
            ckpt_dir = Path(checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"{model_name}_fold{fold_id}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"    > Checkpoint: {ckpt_path.name}")

        fold_results.append({
            "fold": fold_id,
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "per_specimen_mae": test_metrics["per_specimen_mae"],
            "test_specimens": sorted(set(sid_test.tolist())),
        })
        print(f"    > Metrics: MAE = {test_metrics['mae']:,.0f} {units} | RMSE = {test_metrics['rmse']:,.0f} {units}")

    summary = {
        "model_name": model_name,
        "n_splits": n_splits,
        "units": units,
        "mean_test_mae": float(np.mean([fr["test_mae"] for fr in fold_results])),
        "std_test_mae": float(np.std([fr["test_mae"] for fr in fold_results])),
        "mean_test_rmse": float(np.mean([fr["test_rmse"] for fr in fold_results])),
        "std_test_rmse": float(np.std([fr["test_rmse"] for fr in fold_results])),
        "folds": fold_results,
    }
    return summary


def auto_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Grouped-split experiment for TS-RMF and baseline models"
    )
    # Data sources (can use either --dataset or --pickle)
    parser.add_argument("--dataset", type=str, default=None, help="Path to NPZ dataset")
    parser.add_argument("--pickle", type=str, default=None, help="Path to preprocessed pickle (alternative to --dataset)")
    parser.add_argument("--feature-scaler", type=str, default=None, help="Path to feature scaler pickle (needed with --pickle for cycle recovery)")
    parser.add_argument("--target-scaler", type=str, default=None, help="Path to target_scaler_combined.pkl to report metrics in real cycles")

    parser.add_argument("--ttl", type=str, required=True, help="Path to populated ontology TTL")
    parser.add_argument("--model", type=str, default="ts_rmf", choices=["ts_rmf", "plain_transformer"])
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-stage", type=float, default=0.5)
    parser.add_argument("--lambda-kg", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto-detect)")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save per-fold model checkpoints")
    args = parser.parse_args()

    device = args.device if args.device else auto_device()
    print(f"Using device: {device}")

    summary = run_grouped_experiment(
        dataset_path=args.dataset,
        ttl_path=args.ttl,
        model_name=args.model,
        n_splits=args.n_splits,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_stage=args.lambda_stage,
        lambda_kg=args.lambda_kg,
        random_state=args.random_state,
        device=device,
        pickle_path=args.pickle,
        scaler_path=args.feature_scaler,
        target_scaler_path=args.target_scaler,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[OK] Results saved to: {args.out}")


if __name__ == "__main__":
    main()
