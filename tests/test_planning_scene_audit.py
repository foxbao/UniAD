import csv
import os
import tempfile
import unittest

from tools.analysis_tools.build_planning_manifest_splits import load_manifest
from tools.analysis_tools.build_planning_scene_audit import (
    auto_classify,
    infer_control_mode_proxy,
)


class PlanningSceneAuditTest(unittest.TestCase):

    def test_control_mode_proxy_is_conservative_around_stops(self):
        stopped = dict(path_length_m=1.0, stop_ratio=0.9,
                       speed_p50_mps=0.0, speed_p90_mps=0.1,
                       slow_ratio=1.0)
        manual = dict(path_length_m=20.0, stop_ratio=0.1,
                      speed_p50_mps=0.4, speed_p90_mps=1.0,
                      slow_ratio=0.8)
        automatic = dict(path_length_m=50.0, stop_ratio=0.0,
                         speed_p50_mps=2.0, speed_p90_mps=3.0,
                         slow_ratio=0.1)

        self.assertEqual(
            infer_control_mode_proxy(stopped, 0.8)[0], 'StoppedOrUnknown')
        self.assertEqual(
            infer_control_mode_proxy(manual, 0.8)[0], 'LikelyManual')
        self.assertEqual(
            infer_control_mode_proxy(automatic, 0.8)[0], 'LikelyAuto')

    def test_uncertain_confidence_is_low_when_scores_are_close(self):
        row = dict(
            path_length_m=5.0,
            progress_ratio=0.5,
            stop_ratio=0.5,
            pose_revisit_ratio=0.4,
            focus_target_orbit_deg=0.0,
            static_target_orbit_deg=0.0,
            control_mode_proxy='MixedOrUnknown',
            duration_s=10.0,
            net_displacement_m=2.5,
            frame_count=20,
            turn_deg_per_m=0.0,
            planning_repeat_ratio=0.0,
            static_slow_ratio=0.5,
            median_objects_per_frame=1.0,
            moving_ratio=0.5,
        )
        label, confidence, probe_score, natural_score, _ = auto_classify(row)
        self.assertEqual(label, 'Uncertain')
        self.assertAlmostEqual(confidence, abs(probe_score - natural_score))
        self.assertLess(confidence, 0.5)


class PlanningManifestSplitTest(unittest.TestCase):

    def write_manifest(self, rows):
        directory = tempfile.TemporaryDirectory()
        path = os.path.join(directory.name, 'manifest.csv')
        fieldnames = ('split', 'scene_token', 'auto_label', 'human_label',
                      'control_mode_proxy', 'human_control_mode',
                      'planning_usable')
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return directory, path

    def test_human_manifest_requires_explicit_usability(self):
        directory, path = self.write_manifest([dict(
            split='val', scene_token='scene-1', auto_label='NaturalRun',
            human_label='NaturalRun', control_mode_proxy='LikelyAuto',
            human_control_mode='Auto', planning_usable='')])
        self.addCleanup(directory.cleanup)
        with self.assertRaisesRegex(ValueError, 'planning_usable'):
            load_manifest(path, 'human', allow_unreviewed=False)

    def test_human_manifest_requires_explicit_control_mode(self):
        directory, path = self.write_manifest([dict(
            split='val', scene_token='scene-1', auto_label='NaturalRun',
            human_label='NaturalRun', control_mode_proxy='LikelyAuto',
            human_control_mode='', planning_usable='1')])
        self.addCleanup(directory.cleanup)
        with self.assertRaisesRegex(ValueError, 'human_control_mode'):
            load_manifest(path, 'human', allow_unreviewed=False)

    def test_auto_manifest_derives_usability_from_label(self):
        directory, path = self.write_manifest([
            dict(split='val', scene_token='natural',
                 auto_label='NaturalRun', human_label='',
                 control_mode_proxy='LikelyAuto', human_control_mode='',
                 planning_usable=''),
            dict(split='val', scene_token='probe',
                 auto_label='DetectionProbe', human_label='',
                 control_mode_proxy='LikelyManual', human_control_mode='',
                 planning_usable=''),
        ])
        self.addCleanup(directory.cleanup)
        decisions = load_manifest(path, 'auto', allow_unreviewed=False)
        self.assertTrue(decisions[('val', 'natural')]['usable'])
        self.assertEqual(
            decisions[('val', 'natural')]['control_mode'], 'Auto')
        self.assertFalse(decisions[('val', 'probe')]['usable'])
        self.assertEqual(
            decisions[('val', 'probe')]['control_mode'], 'Manual')


if __name__ == '__main__':
    unittest.main()
