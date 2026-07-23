#!/usr/bin/env python
"""Train and evaluate an explicit temporal persistence change gate."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tools.analysis_tools.evaluate_kl_occworld_baselines import (
    CLASS_NAMES,
    _summarize_horizons,
    _update_confusion,
    evaluate_constant_current,
)
from tools.data_converter.kl_occworld_dataset import (
    IGNORE_INDEX,
    KLOccWorldSequenceDataset,
)
from tools.tutorials.occworld_toy.step03_change_gate_baseline import (
    GatedTemporalOccWorld,
    gated_occworld_loss,
)


def _summarize_change_counts(counts: np.ndarray) -> Dict[str, object]:
    true_positive, false_positive, false_negative, true_negative = (
        counts.astype(np.float64))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator
        if precision_denominator else 0.0)
    recall = (
        true_positive / recall_denominator
        if recall_denominator else 0.0)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0)
    false_positive_rate = (
        false_positive / (false_positive + true_negative)
        if false_positive + true_negative else 0.0)
    return {
        'true_positive': int(true_positive),
        'false_positive': int(false_positive),
        'false_negative': int(false_negative),
        'true_negative': int(true_negative),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'false_positive_rate': float(false_positive_rate),
    }


def _summarize_change_horizons(counts: np.ndarray) -> Dict[str, object]:
    return {
        'overall': _summarize_change_counts(counts.sum(axis=0)),
        'by_horizon': [
            _summarize_change_counts(horizon_counts)
            for horizon_counts in counts
        ],
    }


@torch.no_grad()
def evaluate_gate(model: GatedTemporalOccWorld,
                  loader: DataLoader,
                  threshold: float,
                  device: torch.device) -> Dict[str, object]:
    model.eval()
    horizon_count = loader.dataset[0]['target_class_3d'].shape[0]
    confusions = np.zeros(
        (horizon_count, len(CLASS_NAMES), len(CLASS_NAMES)),
        dtype=np.int64)
    unknown_confusions = np.zeros_like(confusions)
    change_confusions = np.zeros_like(confusions)
    change_counts = np.zeros((horizon_count, 4), dtype=np.int64)
    for batch in loader:
        target = batch['target_class_3d']
        outputs = model.predict(
            batch['history_one_hot_3d'].to(device), threshold)
        prediction = outputs['prediction_class'].cpu()
        persistence = outputs['persistence_class'].cpu()
        change_probability = outputs['change_probability'].cpu()
        current_unknown = ~batch['input_valid_3d'][:, None].expand_as(target)
        for horizon in range(horizon_count):
            horizon_target = target[:, horizon]
            horizon_prediction = prediction[:, horizon]
            valid = horizon_target != IGNORE_INDEX
            true_change = valid & (
                horizon_target != persistence[:, horizon])
            predicted_change = valid & (
                change_probability[:, horizon] >= threshold)
            _update_confusion(
                confusions[horizon], horizon_prediction,
                horizon_target)
            _update_confusion(
                unknown_confusions[horizon], horizon_prediction,
                horizon_target, current_unknown[:, horizon])
            _update_confusion(
                change_confusions[horizon], horizon_prediction,
                horizon_target, true_change)
            change_counts[horizon] += np.asarray([
                int((predicted_change & true_change).sum()),
                int((predicted_change & ~true_change & valid).sum()),
                int((~predicted_change & true_change).sum()),
                int((~predicted_change & ~true_change & valid).sum()),
            ], dtype=np.int64)
    result = _summarize_horizons(confusions)
    result['current_unknown_subset'] = _summarize_horizons(
        unknown_confusions)
    result['state_change_subset'] = _summarize_horizons(
        change_confusions)
    result['change_detection'] = _summarize_change_horizons(
        change_counts)
    result['threshold'] = float(threshold)
    return result


def train_gate(model: GatedTemporalOccWorld,
               loader: DataLoader,
               epochs: int,
               learning_rate: float,
               change_positive_weight: float,
               change_focal_gamma: float,
               changed_class_loss_weight: float,
               device: torch.device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rows = []
    for epoch in range(epochs):
        model.train()
        sums = {'loss': 0.0, 'change_loss': 0.0,
                'changed_class_loss': 0.0}
        batch_count = 0
        for batch in loader:
            history = batch['history_one_hot_3d'].to(device)
            target = batch['target_class_3d'].to(device)
            current_state = batch['input_state_3d'].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(history)
            losses = gated_occworld_loss(
                outputs, target, current_state,
                change_positive_weight=change_positive_weight,
                change_focal_gamma=change_focal_gamma,
                changed_class_loss_weight=(
                    changed_class_loss_weight))
            losses['loss'].backward()
            optimizer.step()
            for key in sums:
                sums[key] += float(losses[key].detach())
            batch_count += 1
        row = {
            key: value / max(batch_count, 1)
            for key, value in sums.items()
        }
        row['epoch'] = epoch + 1
        rows.append(row)
        print(
            f'epoch={epoch + 1}/{epochs} '
            f'loss={row["loss"]:.6f} '
            f'change={row["change_loss"]:.6f} '
            f'class={row["changed_class_loss"]:.6f}')
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-root', required=True)
    parser.add_argument('--history-root', required=True)
    parser.add_argument('--train-count', type=int, default=10)
    parser.add_argument('--validation-count', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--hidden-channels', type=int, default=4)
    parser.add_argument('--change-prior', type=float, default=0.05)
    parser.add_argument('--change-positive-weight', type=float, default=10.0)
    parser.add_argument('--change-focal-gamma', type=float, default=0.0)
    parser.add_argument('--changed-class-loss-weight', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=3e-3)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
        0.45, 0.50, 0.60, 0.70, 0.80, 0.90])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out-file', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.epochs < 0 or args.batch_size < 1 or args.torch_threads < 1 or
            args.learning_rate <= 0 or args.change_positive_weight <= 0 or
            args.change_focal_gamma < 0):
        raise ValueError('Invalid change-gate training parameters')
    if any(not 0.0 <= threshold <= 1.0 for threshold in args.thresholds):
        raise ValueError('Thresholds must be between 0 and 1')
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    dataset = KLOccWorldSequenceDataset(
        args.sequence_root,
        expected_shape=(10, 120, 160),
        history_root=args.history_root,
        drop_missing_history=True)
    test_start = args.train_count + args.validation_count
    if (args.train_count < 1 or args.validation_count < 1 or
            test_start >= len(dataset)):
        raise ValueError('Split must leave train, validation and test samples')
    train_indices = list(range(args.train_count))
    validation_indices = list(range(args.train_count, test_start))
    test_indices = list(range(test_start, len(dataset)))
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=args.batch_size,
        shuffle=True, num_workers=0)
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=0)
    test_loader = DataLoader(
        Subset(dataset, test_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=0)
    sample = dataset[0]
    model = GatedTemporalOccWorld(
        history_count=sample['history_one_hot_3d'].shape[0],
        state_channels=sample['history_one_hot_3d'].shape[1],
        hidden_channels=args.hidden_channels,
        horizon_count=sample['target_class_3d'].shape[0],
        change_prior=args.change_prior,
        include_frame_differences=True)
    device = torch.device(args.device)
    model.to(device)
    training_rows = train_gate(
        model, train_loader, args.epochs, args.learning_rate,
        args.change_positive_weight, args.change_focal_gamma,
        args.changed_class_loss_weight, device)

    validation_sweep = []
    for threshold in sorted(set(args.thresholds)):
        metrics = evaluate_gate(
            model, validation_loader, threshold, device)
        change = metrics['change_detection']['overall']
        validation_sweep.append({
            'threshold': float(threshold),
            'precision': change['precision'],
            'recall': change['recall'],
            'f1': change['f1'],
            'overall_accuracy': metrics['overall']['accuracy'],
            'overall_mean_iou': metrics['overall']['mean_iou'],
            'state_change_mean_iou': (
                metrics['state_change_subset']['overall']['mean_iou']),
        })
    selected = max(
        validation_sweep,
        key=lambda row: (row['f1'], row['precision'], row['threshold']))
    selected_threshold = selected['threshold']
    constant_test = evaluate_constant_current(test_loader)
    gated_test = evaluate_gate(
        model, test_loader, selected_threshold, device)
    gated_test_fixed_05 = evaluate_gate(
        model, test_loader, 0.5, device)

    summary = {
        'sample_count': len(dataset),
        'train_reference_indices': [
            int(dataset[index]['reference_index'])
            for index in train_indices],
        'validation_reference_indices': [
            int(dataset[index]['reference_index'])
            for index in validation_indices],
        'test_reference_indices': [
            int(dataset[index]['reference_index'])
            for index in test_indices],
        'epochs': args.epochs,
        'hidden_channels': args.hidden_channels,
        'parameter_count': sum(
            parameter.numel() for parameter in model.parameters()),
        'change_prior': args.change_prior,
        'change_positive_weight': args.change_positive_weight,
        'change_focal_gamma': args.change_focal_gamma,
        'changed_class_loss_weight': args.changed_class_loss_weight,
        'learning_rate': args.learning_rate,
        'training': training_rows,
        'validation_threshold_sweep': validation_sweep,
        'selected_threshold': selected_threshold,
        'selected_validation_metrics': selected,
        'constant_current_test': constant_test,
        'gated_test': gated_test,
        'gated_test_fixed_threshold_0_5': gated_test_fixed_05,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'summary': summary,
        }, args.checkpoint)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
