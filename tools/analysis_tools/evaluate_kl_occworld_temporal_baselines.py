#!/usr/bin/env python
"""Compare single-frame and temporal-fusion KL OccWorld baselines."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from tools.analysis_tools.evaluate_kl_occworld_baselines import (
    evaluate_constant_current,
    evaluate_model,
    train_model,
)
from tools.data_converter.kl_occworld_dataset import (
    KLOccWorldSequenceDataset,
)
from tools.tutorials.occworld_toy.step01_current_to_future_baseline import (
    PersistenceResidualOccWorld,
)
from tools.tutorials.occworld_toy.step02_temporal_fusion_baseline import (
    TemporalFusionOccWorld,
)


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-root', required=True)
    parser.add_argument('--history-root', required=True)
    parser.add_argument('--train-count', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--hidden-channels', type=int, default=4)
    parser.add_argument('--persistence-logit-scale', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=3e-3)
    parser.add_argument('--change-voxel-weight', type=float, default=10.0)
    parser.add_argument('--class-weights', type=float, nargs=3,
                        default=[0.380, 1.451, 1.169])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out-file', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.epochs < 0 or args.batch_size < 1 or args.torch_threads < 1 or
            args.learning_rate <= 0 or args.change_voxel_weight < 1):
        raise ValueError('Invalid temporal training parameters')
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    dataset = KLOccWorldSequenceDataset(
        args.sequence_root,
        expected_shape=(10, 120, 160),
        history_root=args.history_root,
        drop_missing_history=True)
    if args.train_count < 1 or args.train_count >= len(dataset):
        raise ValueError('train-count must leave at least one test sample')
    train_indices = list(range(args.train_count))
    test_indices = list(range(args.train_count, len(dataset)))
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0)
    sample = dataset[0]
    history_count = sample['history_one_hot_3d'].shape[0]
    state_channels = sample['history_one_hot_3d'].shape[1]
    horizon_count = sample['target_class_3d'].shape[0]
    device = torch.device(args.device)
    class_weights = torch.tensor(
        args.class_weights, dtype=torch.float32, device=device)

    torch.manual_seed(args.seed)
    single_model = PersistenceResidualOccWorld(
        in_channels=state_channels,
        hidden_channels=args.hidden_channels,
        horizon_count=horizon_count,
        persistence_logit_scale=args.persistence_logit_scale).to(device)
    constant_metrics = evaluate_constant_current(test_loader)
    single_untrained = evaluate_model(
        single_model, test_loader, device)
    single_losses = train_model(
        single_model, train_loader, args.epochs, class_weights, device,
        learning_rate=args.learning_rate,
        change_voxel_weight=args.change_voxel_weight)
    single_trained = evaluate_model(
        single_model, test_loader, device)

    torch.manual_seed(args.seed)
    temporal_model = TemporalFusionOccWorld(
        history_count=history_count,
        state_channels=state_channels,
        hidden_channels=args.hidden_channels,
        horizon_count=horizon_count,
        persistence_logit_scale=args.persistence_logit_scale,
        include_frame_differences=True).to(device)
    temporal_untrained = evaluate_model(
        temporal_model, test_loader, device,
        input_key='history_one_hot_3d')
    temporal_losses = train_model(
        temporal_model, train_loader, args.epochs, class_weights, device,
        learning_rate=args.learning_rate,
        change_voxel_weight=args.change_voxel_weight,
        input_key='history_one_hot_3d')
    temporal_trained = evaluate_model(
        temporal_model, test_loader, device,
        input_key='history_one_hot_3d')

    summary = {
        'sample_count': len(dataset),
        'train_count': len(train_indices),
        'test_count': len(test_indices),
        'train_reference_indices': [
            int(dataset[index]['reference_index'])
            for index in train_indices],
        'test_reference_indices': [
            int(dataset[index]['reference_index'])
            for index in test_indices],
        'epochs': args.epochs,
        'hidden_channels': args.hidden_channels,
        'history_count': history_count,
        'persistence_logit_scale': args.persistence_logit_scale,
        'learning_rate': args.learning_rate,
        'change_voxel_weight': args.change_voxel_weight,
        'class_weights': args.class_weights,
        'constant_current': constant_metrics,
        'single_frame_parameter_count': _parameter_count(single_model),
        'single_frame_untrained': single_untrained,
        'single_frame_training_losses': single_losses,
        'single_frame_trained': single_trained,
        'temporal_parameter_count': _parameter_count(temporal_model),
        'temporal_untrained': temporal_untrained,
        'temporal_training_losses': temporal_losses,
        'temporal_trained': temporal_trained,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open('w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'single_frame_state_dict': single_model.state_dict(),
            'temporal_state_dict': temporal_model.state_dict(),
            'summary': summary,
        }, args.checkpoint)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
