#!/usr/bin/env python
"""Compare constant-current and tiny-CNN KL OccWorld baselines."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tools.data_converter.kl_occworld_dataset import (
    IGNORE_INDEX,
    KLOccWorldSequenceDataset,
)
from tools.tutorials.occworld_toy.step01_current_to_future_baseline import (
    PersistenceResidualOccWorld,
    TinyOccWorld,
    change_aware_world_cross_entropy,
    masked_world_cross_entropy,
)


CLASS_NAMES = ('free', 'static_occupied', 'instance_occupied')


def _update_confusion(confusion: np.ndarray,
                      prediction: torch.Tensor,
                      target: torch.Tensor,
                      extra_valid: Optional[torch.Tensor] = None):
    valid = target != IGNORE_INDEX
    if extra_valid is not None:
        valid &= extra_valid
    if not torch.any(valid):
        return
    target_valid = target[valid].to(torch.long)
    prediction_valid = prediction[valid].to(torch.long)
    encoded = target_valid * len(CLASS_NAMES) + prediction_valid
    counts = torch.bincount(
        encoded.cpu(), minlength=len(CLASS_NAMES) ** 2)
    confusion += counts.reshape(len(CLASS_NAMES), len(CLASS_NAMES)).numpy()


def _summarize_confusion(confusion: np.ndarray) -> Dict[str, object]:
    true_positive = np.diag(confusion).astype(np.float64)
    target_count = confusion.sum(axis=1).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    union = target_count + predicted_count - true_positive
    iou = np.divide(
        true_positive, union,
        out=np.full_like(true_positive, np.nan), where=union > 0)
    total = float(confusion.sum())
    return {
        'voxel_count': int(total),
        'accuracy': float(true_positive.sum() / total) if total else 0.0,
        'iou': {
            name: None if not np.isfinite(value) else float(value)
            for name, value in zip(CLASS_NAMES, iou)
        },
        'mean_iou': float(np.nanmean(iou)) if np.any(np.isfinite(iou)) else 0.0,
        'confusion_target_rows_prediction_columns': confusion.tolist(),
    }


def _summarize_horizons(confusions: np.ndarray) -> Dict[str, object]:
    return {
        'overall': _summarize_confusion(confusions.sum(axis=0)),
        'by_horizon': [
            _summarize_confusion(confusion)
            for confusion in confusions
        ],
    }


def _constant_current_prediction(batch: Dict[str, object]) -> torch.Tensor:
    current = batch['input_state_3d']
    target = batch['target_class_3d']
    prediction = torch.zeros_like(current, dtype=torch.long)
    known = current != 0
    prediction[known] = current[known] - 1
    return prediction[:, None].expand(-1, target.shape[1], -1, -1, -1)


def evaluate_constant_current(loader: DataLoader) -> Dict[str, object]:
    horizon_count = loader.dataset[0]['target_class_3d'].shape[0]
    all_confusion = np.zeros(
        (horizon_count, len(CLASS_NAMES), len(CLASS_NAMES)),
        dtype=np.int64)
    visible_confusion = np.zeros_like(all_confusion)
    unknown_confusion = np.zeros_like(all_confusion)
    change_confusion = np.zeros_like(all_confusion)
    target_counts = np.zeros(horizon_count, dtype=np.int64)
    visible_counts = np.zeros(horizon_count, dtype=np.int64)
    change_counts = np.zeros(horizon_count, dtype=np.int64)
    for batch in loader:
        target = batch['target_class_3d']
        prediction = _constant_current_prediction(batch)
        current_visible = batch['input_valid_3d'][:, None].expand_as(target)
        for horizon in range(horizon_count):
            _update_confusion(
                all_confusion[horizon], prediction[:, horizon],
                target[:, horizon])
            _update_confusion(
                visible_confusion[horizon], prediction[:, horizon],
                target[:, horizon], current_visible[:, horizon])
            _update_confusion(
                unknown_confusion[horizon], prediction[:, horizon],
                target[:, horizon], ~current_visible[:, horizon])
            changed = target[:, horizon] != prediction[:, horizon]
            _update_confusion(
                change_confusion[horizon], prediction[:, horizon],
                target[:, horizon], changed)
            target_valid = target[:, horizon] != IGNORE_INDEX
            target_counts[horizon] += int(target_valid.sum())
            visible_counts[horizon] += int(
                (target_valid & current_visible[:, horizon]).sum())
            change_counts[horizon] += int(
                (target_valid & changed).sum())
    result = _summarize_horizons(all_confusion)
    result['current_visible_subset'] = _summarize_horizons(
        visible_confusion)
    result['current_unknown_subset'] = _summarize_horizons(
        unknown_confusion)
    result['state_change_subset'] = _summarize_horizons(
        change_confusion)
    result['current_visible_coverage_by_horizon'] = [
        float(visible / total) if total else 0.0
        for visible, total in zip(visible_counts, target_counts)
    ]
    result['state_change_ratio_by_horizon'] = [
        float(changed / total) if total else 0.0
        for changed, total in zip(change_counts, target_counts)
    ]
    return result


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader,
                   device: torch.device,
                   input_key: str = 'input_one_hot_3d') -> Dict[str, object]:
    model.eval()
    horizon_count = loader.dataset[0]['target_class_3d'].shape[0]
    confusions = np.zeros(
        (horizon_count, len(CLASS_NAMES), len(CLASS_NAMES)),
        dtype=np.int64)
    unknown_confusions = np.zeros_like(confusions)
    change_confusions = np.zeros_like(confusions)
    for batch in loader:
        inputs = batch[input_key].to(device)
        target = batch['target_class_3d']
        prediction = model(inputs).argmax(dim=2).cpu()
        persistence = _constant_current_prediction(batch)
        current_unknown = ~batch['input_valid_3d'][:, None].expand_as(target)
        for horizon in range(horizon_count):
            _update_confusion(
                confusions[horizon], prediction[:, horizon],
                target[:, horizon])
            _update_confusion(
                unknown_confusions[horizon], prediction[:, horizon],
                target[:, horizon], current_unknown[:, horizon])
            changed = target[:, horizon] != persistence[:, horizon]
            _update_confusion(
                change_confusions[horizon], prediction[:, horizon],
                target[:, horizon], changed)
    result = _summarize_horizons(confusions)
    result['current_unknown_subset'] = _summarize_horizons(
        unknown_confusions)
    result['state_change_subset'] = _summarize_horizons(
        change_confusions)
    return result


def train_model(model: torch.nn.Module, loader: DataLoader,
                epochs: int, class_weights: torch.Tensor,
                device: torch.device, learning_rate: float,
                change_voxel_weight: float = 1.0,
                input_key: str = 'input_one_hot_3d') -> List[float]:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for batch in loader:
            inputs = batch[input_key].to(device)
            target = batch['target_class_3d'].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if change_voxel_weight > 1.0:
                loss = change_aware_world_cross_entropy(
                    logits, target,
                    batch['input_state_3d'].to(device),
                    class_weights=class_weights,
                    change_voxel_weight=change_voxel_weight)
            else:
                loss = masked_world_cross_entropy(
                    logits, target, class_weights=class_weights)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            batch_count += 1
        mean_loss = loss_sum / max(batch_count, 1)
        epoch_losses.append(mean_loss)
        print(f'epoch={epoch + 1}/{epochs} mean_loss={mean_loss:.6f}')
    return epoch_losses


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-root', required=True)
    parser.add_argument('--train-count', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--hidden-channels', type=int, default=4)
    parser.add_argument('--persistence-logit-scale', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=3e-3)
    parser.add_argument('--change-voxel-weight', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--class-weights', type=float, nargs=3,
                        default=[0.384, 1.443, 1.172])
    parser.add_argument('--out-file', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.epochs < 0 or args.batch_size < 1 or args.torch_threads < 1 or
            args.learning_rate <= 0 or args.change_voxel_weight < 1):
        raise ValueError('Invalid training parameters')
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    dataset = KLOccWorldSequenceDataset(args.sequence_root)
    if args.train_count < 1 or args.train_count >= len(dataset):
        raise ValueError('train-count must leave at least one test sample')
    train_indices = list(range(args.train_count))
    test_indices = list(range(args.train_count, len(dataset)))
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0)
    first_sample = dataset[0]
    model = TinyOccWorld(
        in_channels=first_sample['input_one_hot_3d'].shape[0],
        hidden_channels=args.hidden_channels,
        horizon_count=first_sample['target_class_3d'].shape[0])
    device = torch.device(args.device)
    model.to(device)
    class_weights = torch.tensor(
        args.class_weights, dtype=torch.float32, device=device)

    constant_metrics = evaluate_constant_current(test_loader)
    untrained_metrics = evaluate_model(model, test_loader, device)
    epoch_losses = train_model(
        model, train_loader, args.epochs, class_weights, device,
        learning_rate=args.learning_rate)
    trained_metrics = evaluate_model(model, test_loader, device)

    torch.manual_seed(args.seed)
    residual_model = PersistenceResidualOccWorld(
        in_channels=first_sample['input_one_hot_3d'].shape[0],
        hidden_channels=args.hidden_channels,
        horizon_count=first_sample['target_class_3d'].shape[0],
        persistence_logit_scale=args.persistence_logit_scale)
    residual_model.to(device)
    residual_untrained_metrics = evaluate_model(
        residual_model, test_loader, device)
    residual_epoch_losses = train_model(
        residual_model, train_loader, args.epochs, class_weights, device,
        learning_rate=args.learning_rate,
        change_voxel_weight=args.change_voxel_weight)
    residual_trained_metrics = evaluate_model(
        residual_model, test_loader, device)
    train_references = [
        int(dataset[index]['reference_index']) for index in train_indices]
    test_references = [
        int(dataset[index]['reference_index']) for index in test_indices]
    summary = {
        'sample_count': len(dataset),
        'train_count': len(train_indices),
        'test_count': len(test_indices),
        'train_reference_indices': train_references,
        'test_reference_indices': test_references,
        'epochs': args.epochs,
        'hidden_channels': args.hidden_channels,
        'persistence_logit_scale': args.persistence_logit_scale,
        'learning_rate': args.learning_rate,
        'change_voxel_weight': args.change_voxel_weight,
        'class_weights': args.class_weights,
        'constant_current': constant_metrics,
        'tiny_cnn_untrained': untrained_metrics,
        'training_epoch_losses': epoch_losses,
        'tiny_cnn_trained': trained_metrics,
        'persistence_residual_untrained': residual_untrained_metrics,
        'persistence_residual_training_epoch_losses': (
            residual_epoch_losses),
        'persistence_residual_trained': residual_trained_metrics,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'tiny_model_state_dict': model.state_dict(),
            'persistence_residual_state_dict': (
                residual_model.state_dict()),
            'summary': summary,
        }, args.checkpoint)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
