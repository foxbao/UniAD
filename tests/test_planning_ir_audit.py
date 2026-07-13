import unittest

import torch

from projects.mmdet3d_plugin.uniad.dense_heads.map_multimodal_planner import (
    MapMultimodalPlanner,
)
from tools.analysis_tools.planning_ir_schema import (
    PlanningIRValidationError,
    fallback_planning_ir,
    validate_planning_ir,
)


class PlanningIRSchemaTest(unittest.TestCase):

    def setUp(self):
        self.candidates = [
            dict(candidate_id=7, source='map', valid=True, path_index=1,
                 speed_profile_index=2, lateral_offset_index=3),
            dict(candidate_id=8, source='fallback', valid=True,
                 path_index=None, speed_profile_index=None,
                 lateral_offset_index=None),
        ]

    def test_valid_map_candidate(self):
        payload = dict(
            schema_version='planning-ir/v1',
            selected_candidate_id=7,
            maneuver='YIELD',
            lane_path_index=1,
            speed_profile_index=2,
            lateral_offset_index=3,
            yield_actor_id='42',
            risk_flags=['FRONT_CONFLICT'],
            rule_ids=['rule-1'],
            confidence=0.8,
            ttl_frames=2,
        )
        normalized = validate_planning_ir(
            payload, self.candidates, actor_ids=['42'], rule_ids=['rule-1'])
        self.assertEqual(normalized['selected_candidate_id'], 7)

    def test_fallback_and_factor_mismatch(self):
        fallback = fallback_planning_ir(8)
        self.assertEqual(
            validate_planning_ir(fallback, self.candidates), fallback)
        invalid = dict(fallback, selected_candidate_id=7,
                       speed_profile_index=9)
        with self.assertRaises(PlanningIRValidationError):
            validate_planning_ir(invalid, self.candidates)


class MapMultimodalPlannerAuditTest(unittest.TestCase):

    def test_eval_topk_plus_fallback_and_train_default(self):
        planner = MapMultimodalPlanner(
            embed_dims=8, planning_steps=6, num_heads=2, dropout=0.0,
            use_candidate_cost=True, audit_topk=2)
        plan_query = torch.zeros(1, 1, 8)
        fallback = torch.zeros(1, 6, 2)
        outs_map = dict(
            planning_candidates=torch.randn(1, 3, 6, 2),
            planning_candidate_valid=torch.tensor([[True, True, False]]),
        )
        planner.eval()
        output = planner(plan_query, fallback, outs_map)
        self.assertEqual(
            tuple(output['multimodal_audit_indices'].shape), (1, 3))
        self.assertEqual(
            output['multimodal_audit_indices'][0, -1].item(), 3)
        self.assertEqual(
            tuple(output['multimodal_audit_refined_candidates'].shape),
            (1, 3, 6, 2))
        self.assertTrue(torch.equal(
            output['multimodal_fallback_traj'], fallback))

        planner.train()
        output = planner(plan_query, fallback, outs_map)
        self.assertNotIn('multimodal_audit_indices', output)


class MapMultimodalPlannerSetRerankerTest(unittest.TestCase):

    def test_topk_fallback_and_loss_keys(self):
        planner = MapMultimodalPlanner(
            embed_dims=8, planning_steps=6, num_heads=2, dropout=0.0,
            use_candidate_cost=True,
            use_set_reranker=True,
            set_topk=2,
            set_num_layers=1,
            set_num_heads=2,
            set_ffn_dims=16,
            set_dropout=0.0,
            set_use_raw_candidates=True)
        plan_query = torch.zeros(1, 1, 8)
        fallback = torch.zeros(1, 6, 2)
        outs_map = dict(
            planning_candidates=torch.randn(1, 3, 6, 2),
            planning_candidate_valid=torch.tensor([[True, True, False]]),
        )
        actor_future = torch.zeros(1, 1, 6, 2)
        actor_future[..., 0] = 10.0
        outs_motion = dict(
            planning_actor_future=actor_future,
            planning_actor_sizes=torch.tensor([[[4.0, 2.0]]]),
            planning_actor_yaws=torch.zeros(1, 1),
            planning_actor_scores=torch.ones(1, 1),
            planning_actor_valid=torch.ones(1, 1, dtype=torch.bool),
        )

        planner.train()
        output = planner(
            plan_query, fallback, outs_map, outs_motion=outs_motion)
        self.assertEqual(tuple(output['multimodal_set_indices'].shape), (1, 3))
        self.assertEqual(output['multimodal_set_indices'][0, -1].item(), 3)
        self.assertEqual(output['multimodal_selected_index'].item(), 3)
        self.assertTrue(torch.equal(
            output['multimodal_selected_traj'], fallback))
        self.assertTrue(torch.isfinite(
            output['multimodal_set_safety_features']).all())

        gt = torch.zeros(1, 6, 3)
        valid = torch.ones(1, 6, dtype=torch.bool)
        losses = planner.loss(output, gt, valid, future_gt_bbox=None)
        for key in (
                'loss_multimodal_set_cost',
                'loss_multimodal_set_ranking',
                'loss_multimodal_set_collision'):
            self.assertIn(key, losses)
            self.assertTrue(torch.isfinite(losses[key]))


if __name__ == '__main__':
    unittest.main()
