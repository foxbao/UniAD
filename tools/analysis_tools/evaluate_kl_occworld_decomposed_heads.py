#!/usr/bin/env python
"""Train and evaluate decomposed reveal and visible-transition heads."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence


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
from tools.analysis_tools.evaluate_kl_occworld_change_gate import (
    _summarize_change_counts,
    _summarize_change_horizons,
)
from tools.data_converter.kl_occworld_dataset import (
    IGNORE_INDEX,
    KLOccWorldSequenceDataset,
)
from tools.tutorials.occworld_toy.step03_change_gate_baseline import (
    persistence_class_from_state,
)
from tools.tutorials.occworld_toy.step04_decomposed_change_baseline import (
    DecomposedTemporalOccWorld,
    decomposed_occworld_loss,
)


def _balanced_class_weights(counts: np.ndarray,
                            power: float) -> np.ndarray:
    counts = counts.astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError(f'Every class needs training samples, got {counts}')
    if not 0.0 <= power <= 1.0:
        raise ValueError('Class weight power must be between 0 and 1')
    weights = counts ** (-power)
    return weights / weights.mean()


def measure_training_balance(dataset, indices: Sequence[int],
                             class_weight_power: float) -> Dict[str, object]:
    reveal_counts = np.zeros(2, dtype=np.int64)
    transition_counts = np.zeros(2, dtype=np.int64)
    reveal_classes = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    transition_classes = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for index in indices:
        sample = dataset[index]
        target = sample['target_class_3d']
        known = target != IGNORE_INDEX
        current_state = sample['input_state_3d']
        current_unknown = (current_state == 0)[None].expand_as(target)
        persistence = persistence_class_from_state(
            current_state[None], target.shape[0])[0]
        reveal_positive = current_unknown & known
        reveal_counts += np.asarray([
            int(reveal_positive.sum()),
            int((current_unknown & ~known).sum()),
        ], dtype=np.int64)
        visible_valid = ~current_unknown & known
        transition_positive = visible_valid & (target != persistence)
        transition_counts += np.asarray([
            int(transition_positive.sum()),
            int((visible_valid & ~transition_positive).sum()),
        ], dtype=np.int64)
        for class_id in range(len(CLASS_NAMES)):
            reveal_classes[class_id] += int(
                (reveal_positive & (target == class_id)).sum())
            transition_classes[class_id] += int(
                (transition_positive & (target == class_id)).sum())
    return {
        'reveal_positive_negative': reveal_counts.tolist(),
        'transition_positive_negative': transition_counts.tolist(),
        'reveal_class_counts': reveal_classes.tolist(),
        'transition_class_counts': transition_classes.tolist(),
        'balanced_reveal_positive_weight': float(
            reveal_counts[1] / reveal_counts[0]),
        'balanced_transition_positive_weight': float(
            transition_counts[1] / transition_counts[0]),
        'balanced_reveal_class_weights': _balanced_class_weights(
            reveal_classes, class_weight_power).tolist(),
        'balanced_transition_class_weights': _balanced_class_weights(
            transition_classes, class_weight_power).tolist(),
    }


def _binary_counts(predicted: torch.Tensor,
                   truth: torch.Tensor,
                   scope: torch.Tensor) -> np.ndarray:
    return np.asarray([
        int((predicted & truth & scope).sum()),
        int((predicted & ~truth & scope).sum()),
        int((~predicted & truth & scope).sum()),
        int((~predicted & ~truth & scope).sum()),
    ], dtype=np.int64)


@torch.no_grad()
def sweep_gate_thresholds(model: DecomposedTemporalOccWorld,
                          loader: DataLoader,
                          thresholds: Sequence[float],
                          device: torch.device) -> Dict[str, object]:
    model.eval()
    horizon_count = loader.dataset[0]['target_class_3d'].shape[0]
    reveal_counts = np.zeros(
        (len(thresholds), horizon_count, 4), dtype=np.int64)
    transition_counts = np.zeros_like(reveal_counts)
    for batch in loader:
        target = batch['target_class_3d']
        known = target != IGNORE_INDEX
        current_state = batch['input_state_3d']
        current_unknown = (
            current_state == 0)[:, None].expand_as(target)
        persistence = persistence_class_from_state(
            current_state, target.shape[1])
        visible_valid = ~current_unknown & known
        true_transition = visible_valid & (target != persistence)
        outputs = model(batch['history_one_hot_3d'].to(device))
        reveal_probability = outputs['reveal_logits'].sigmoid().cpu()
        transition_probability = outputs['transition_logits'].sigmoid().cpu()
        for threshold_index, threshold in enumerate(thresholds):
            predicted_reveal = reveal_probability >= threshold
            predicted_transition = transition_probability >= threshold
            for horizon in range(horizon_count):
                reveal_counts[threshold_index, horizon] += _binary_counts(
                    predicted_reveal[:, horizon], known[:, horizon],
                    current_unknown[:, horizon])
                transition_counts[
                    threshold_index, horizon] += _binary_counts(
                        predicted_transition[:, horizon],
                        true_transition[:, horizon],
                        visible_valid[:, horizon])

    def rows_for(counts):
        rows = []
        for threshold, threshold_counts in zip(thresholds, counts):
            metrics = _summarize_change_counts(
                threshold_counts.sum(axis=0))
            rows.append({'threshold': float(threshold), **metrics})
        return rows

    reveal_rows = rows_for(reveal_counts)
    transition_rows = rows_for(transition_counts)
    return {
        'reveal': reveal_rows,
        'visible_transition': transition_rows,
        'selected_reveal': max(
            reveal_rows,
            key=lambda row: (row['f1'], row['precision'], row['threshold'])),
        'selected_visible_transition': max(
            transition_rows,
            key=lambda row: (row['f1'], row['precision'], row['threshold'])),
    }


@torch.no_grad()
def evaluate_decomposed(model: DecomposedTemporalOccWorld,
                        loader: DataLoader,
                        reveal_threshold: float,
                        transition_threshold: float,
                        decision_mode: str,
                        device: torch.device) -> Dict[str, object]:
    model.eval()
    horizon_count = loader.dataset[0]['target_class_3d'].shape[0]
    shape = (horizon_count, len(CLASS_NAMES), len(CLASS_NAMES))
    confusions = np.zeros(shape, dtype=np.int64)
    unknown_confusions = np.zeros(shape, dtype=np.int64)
    change_confusions = np.zeros(shape, dtype=np.int64)
    reveal_confusions = np.zeros(shape, dtype=np.int64)
    transition_confusions = np.zeros(shape, dtype=np.int64)
    reveal_counts = np.zeros((horizon_count, 4), dtype=np.int64)
    transition_counts = np.zeros_like(reveal_counts)
    for batch in loader:
        target = batch['target_class_3d']
        known = target != IGNORE_INDEX
        current_state = batch['input_state_3d']
        current_unknown = (
            current_state == 0)[:, None].expand_as(target)
        outputs = model.predict(
            batch['history_one_hot_3d'].to(device),
            reveal_threshold=reveal_threshold,
            transition_threshold=transition_threshold,
            decision_mode=decision_mode)
        prediction = outputs['prediction_class'].cpu()
        persistence = outputs['persistence_class'].cpu()
        reveal_probability = outputs['reveal_probability'].cpu()
        transition_probability = outputs['transition_probability'].cpu()
        state_change = known & (target != persistence)
        reveal_subset = current_unknown & known
        visible_valid = ~current_unknown & known
        visible_transition = visible_valid & (target != persistence)
        predicted_reveal = reveal_probability >= reveal_threshold
        predicted_transition = (
            transition_probability >= transition_threshold)
        for horizon in range(horizon_count):
            _update_confusion(
                confusions[horizon], prediction[:, horizon],
                target[:, horizon])
            _update_confusion(
                unknown_confusions[horizon], prediction[:, horizon],
                target[:, horizon], reveal_subset[:, horizon])
            _update_confusion(
                change_confusions[horizon], prediction[:, horizon],
                target[:, horizon], state_change[:, horizon])
            _update_confusion(
                reveal_confusions[horizon], prediction[:, horizon],
                target[:, horizon], reveal_subset[:, horizon])
            _update_confusion(
                transition_confusions[horizon], prediction[:, horizon],
                target[:, horizon], visible_transition[:, horizon])
            reveal_counts[horizon] += _binary_counts(
                predicted_reveal[:, horizon], known[:, horizon],
                current_unknown[:, horizon])
            transition_counts[horizon] += _binary_counts(
                predicted_transition[:, horizon],
                visible_transition[:, horizon], visible_valid[:, horizon])
    return {
        'overall': _summarize_horizons(confusions)['overall'],
        'by_horizon': _summarize_horizons(confusions)['by_horizon'],
        'current_unknown_subset': _summarize_horizons(unknown_confusions),
        'state_change_subset': _summarize_horizons(change_confusions),
        'reveal_completion_subset': _summarize_horizons(reveal_confusions),
        'visible_transition_subset': _summarize_horizons(
            transition_confusions),
        'reveal_detection': _summarize_change_horizons(reveal_counts),
        'visible_transition_detection': _summarize_change_horizons(
            transition_counts),
        'reveal_threshold': float(reveal_threshold),
        'transition_threshold': float(transition_threshold),
        'decision_mode': decision_mode,
    }


def train_model(model, loader, epochs, learning_rate,
                reveal_positive_weight, transition_positive_weight,
                reveal_class_weights, transition_class_weights,
                device):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rows = []
    loss_names = (
        'loss', 'reveal_gate_loss', 'reveal_class_loss',
        'transition_gate_loss', 'transition_class_loss')
    for epoch in range(epochs):
        model.train()
        sums = {name: 0.0 for name in loss_names}
        batch_count = 0
        for batch in loader:
            history = batch['history_one_hot_3d'].to(device)
            target = batch['target_class_3d'].to(device)
            current_state = batch['input_state_3d'].to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = decomposed_occworld_loss(
                model(history), target, current_state,
                reveal_positive_weight=reveal_positive_weight,
                transition_positive_weight=transition_positive_weight,
                reveal_class_weights=reveal_class_weights,
                transition_class_weights=transition_class_weights)
            losses['loss'].backward()
            optimizer.step()
            for name in loss_names:
                sums[name] += float(losses[name].detach())
            batch_count += 1
        row = {
            name: value / max(batch_count, 1)
            for name, value in sums.items()
        }
        row['epoch'] = epoch + 1
        rows.append(row)
        print(
            f'epoch={epoch + 1}/{epochs} loss={row["loss"]:.6f} '
            f'reveal={row["reveal_gate_loss"]:.6f}/'
            f'{row["reveal_class_loss"]:.6f} transition='
            f'{row["transition_gate_loss"]:.6f}/'
            f'{row["transition_class_loss"]:.6f}')
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
    parser.add_argument('--reveal-prior', type=float, default=0.05)
    parser.add_argument('--transition-prior', type=float, default=0.01)
    parser.add_argument('--reveal-positive-weight', type=float)
    parser.add_argument('--transition-positive-weight', type=float)
    parser.add_argument('--class-weight-power', type=float, default=1.0)
    parser.add_argument('--decision-mode',
                        choices=('hard', 'probabilistic', 'reveal_only'),
                        default='probabilistic')
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
            args.learning_rate <= 0 or
            not 0.0 <= args.class_weight_power <= 1.0):
        raise ValueError('Invalid training parameters')
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

    balance = measure_training_balance(
        dataset, train_indices, args.class_weight_power)
    reveal_positive_weight = (
        balance['balanced_reveal_positive_weight']
        if args.reveal_positive_weight is None
        else args.reveal_positive_weight)
    transition_positive_weight = (
        balance['balanced_transition_positive_weight']
        if args.transition_positive_weight is None
        else args.transition_positive_weight)
    if reveal_positive_weight <= 0 or transition_positive_weight <= 0:
        raise ValueError('Positive weights must be greater than zero')
    device = torch.device(args.device)
    reveal_class_weights = torch.tensor(
        balance['balanced_reveal_class_weights'],
        dtype=torch.float32, device=device)
    transition_class_weights = torch.tensor(
        balance['balanced_transition_class_weights'],
        dtype=torch.float32, device=device)
    sample = dataset[0]
    model = DecomposedTemporalOccWorld(
        history_count=sample['history_one_hot_3d'].shape[0],
        state_channels=sample['history_one_hot_3d'].shape[1],
        hidden_channels=args.hidden_channels,
        horizon_count=sample['target_class_3d'].shape[0],
        reveal_prior=args.reveal_prior,
        transition_prior=args.transition_prior,
        include_frame_differences=True).to(device)
    training = train_model(
        model, train_loader, args.epochs, args.learning_rate,
        reveal_positive_weight, transition_positive_weight,
        reveal_class_weights, transition_class_weights, device)
    threshold_sweep = sweep_gate_thresholds(
        model, validation_loader, sorted(set(args.thresholds)), device)
    reveal_threshold = threshold_sweep['selected_reveal']['threshold']
    transition_threshold = threshold_sweep[
        'selected_visible_transition']['threshold']
    test_metrics = evaluate_decomposed(
        model, test_loader, reveal_threshold, transition_threshold,
        args.decision_mode, device)
    summary = {
        'sample_count': len(dataset),
        'train_reference_indices': [
            int(dataset[index]['reference_index']) for index in train_indices],
        'validation_reference_indices': [
            int(dataset[index]['reference_index'])
            for index in validation_indices],
        'test_reference_indices': [
            int(dataset[index]['reference_index']) for index in test_indices],
        'epochs': args.epochs,
        'hidden_channels': args.hidden_channels,
        'parameter_count': sum(
            parameter.numel() for parameter in model.parameters()),
        'reveal_prior': args.reveal_prior,
        'transition_prior': args.transition_prior,
        'learning_rate': args.learning_rate,
        'class_weight_power': args.class_weight_power,
        'decision_mode': args.decision_mode,
        'training_balance': balance,
        'used_reveal_positive_weight': reveal_positive_weight,
        'used_transition_positive_weight': transition_positive_weight,
        'training': training,
        'validation_threshold_sweep': threshold_sweep,
        'constant_current_test': evaluate_constant_current(test_loader),
        'decomposed_test': test_metrics,
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
