from types import SimpleNamespace

import torch

from projects.mmdet3d_plugin.uniad.dense_heads.motion_head_lidar import (
    MotionHeadLidar,
)


class _DummyBoxes:

    def __init__(self, tensor):
        self.tensor = tensor

    @property
    def gravity_center(self):
        center = self.tensor[:, :3].clone()
        center[:, 2] += self.tensor[:, 5] * 0.5
        return center

    @property
    def yaw(self):
        return self.tensor[:, 6]


def test_planning_actor_context_keeps_masked_boxes_aligned_with_top_mode():
    boxes_tensor = torch.tensor([
        [10.0, 20.0, -1.0, 4.0, 2.0, 2.0, 0.1],
        [30.0, 40.0, -0.5, 5.0, 2.5, 3.0, 0.2],
        [-5.0, 6.0, -1.5, 3.0, 1.5, 1.0, -0.3],
    ])
    track_scores = torch.tensor([0.8, 0.5, 0.6])
    mode_scores = torch.tensor([[[0.0, -1.0], [-2.0, -1.0],
                                 [-3.0, -0.1]]])
    mode_preds = torch.zeros((1, 3, 2, 4, 5))
    mode_preds[0, 0, 0, :, :2] = torch.tensor([1.0, 2.0])
    mode_preds[0, 2, 1, :, :2] = torch.tensor([3.0, 4.0])
    outs_motion = dict(
        all_traj_scores=[mode_scores],
        all_traj_preds=[mode_preds])
    track_boxes = [[
        _DummyBoxes(boxes_tensor), track_scores,
        torch.tensor([0, 1, 0]), torch.arange(3), None,
    ]]

    MotionHeadLidar._attach_planning_actor_context(
        SimpleNamespace(), outs_motion, track_boxes,
        torch.tensor([True, False, True]), with_sdc=False)

    assert torch.equal(
        outs_motion['planning_actor_boxes_3d'][0], boxes_tensor[[0, 2]])
    assert torch.allclose(
        outs_motion['planning_actor_future'][0, 0, 0],
        torch.tensor([11.0, 22.0]))
    assert torch.allclose(
        outs_motion['planning_actor_future'][0, 1, 0],
        torch.tensor([-2.0, 10.0]))
    assert torch.allclose(
        outs_motion['planning_actor_scores'][0],
        torch.tensor([0.8, 0.6 * torch.exp(torch.tensor(-0.1))]))
    assert outs_motion['planning_actor_valid'].tolist() == [[True, True]]


def test_empty_planning_actor_context_preserves_batch_and_schema():
    owner = SimpleNamespace(predict_steps=12)
    outs_motion = {}

    MotionHeadLidar._attach_empty_planning_actor_context(
        owner, outs_motion, torch.zeros((3, 2, 0, 8)))

    assert outs_motion['planning_actor_future'].shape == (2, 0, 12, 2)
    assert outs_motion['planning_actor_boxes_3d'].shape == (2, 0, 7)
    assert outs_motion['planning_actor_scores'].shape == (2, 0)
    assert outs_motion['planning_actor_valid'].dtype == torch.bool
