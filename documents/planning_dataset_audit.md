# Planning Dataset Audit and Clean-Split Protocol

## 1. Why this gate is required

The current train and validation annotations originate from a dataset collected
primarily for 3D detection. Some records are therefore not natural planning
demonstrations: the ego IGV can remain near a target, circle a WheelCrane or
Crane, or repeatedly reposition to improve sensing coverage.

The ego vehicle also has two control modes:

- manual joystick operation, normally slower and commonly used for deliberate
  data collection;
- automatic operation, normally faster and closer to the deployed planning
  distribution.

Neither train nor validation can consequently be treated as a planning ground
truth set without a scene-level audit. This affects both supervision and model
selection. A lower L2 on the original validation set can reward imitation of
collection behavior rather than better autonomous planning.

The current info files do not contain a control-mode field. Speed can only
provide a review hint, not control-mode ground truth. Normal automatic driving
can stop, and deliberate target collection can still be fast.

## 2. Implemented audit

Run from the `UniAD` directory:

```bash
python tools/analysis_tools/build_planning_scene_audit.py \
  --infos \
    train=data/kl_8/kl_infos_train.pkl \
    val=data/kl_8/kl_infos_val.pkl \
  --out-dir \
    projects/work_dirs/stage2_e2e_lidar/planning_scene_audit \
  --plot-mode review \
  --max-train-plots 180
```

The audit computes scene-level route progress, speed and stop distributions,
pose revisits, cumulative turning, repeated planning labels, command balance,
object density, and ego orbit around tracked static targets. It produces:

- `scene_features.csv`: complete automatic features;
- `scene_manifest.csv`: persistent human decisions;
- `review_queue.csv`: review order;
- `review_index.html`: trajectory and speed plots;
- `plots/`: all validation plots and prioritized training plots;
- `summary.json` and `README.md`: run summary.

Rerunning the tool preserves all human columns in `scene_manifest.csv`.

Start the local click-through reviewer with:

```bash
python tools/analysis_tools/serve_planning_scene_review.py \
  --audit-dir \
    projects/work_dirs/stage2_e2e_lidar/planning_scene_audit \
  --port 8765
```

Open `http://127.0.0.1:8765`. Each save atomically updates
`scene_manifest.csv`; the default queue is the unreviewed validation split.

## 3. Current automatic triage

The first full audit covers 649 scenes and 49,172 frames:

| split | scenes | frames | NaturalRun | DetectionProbe | Uncertain |
|---|---:|---:|---:|---:|---:|
| train | 584 | 43,981 | 374 / 27,629 | 29 / 2,608 | 181 / 13,744 |
| val | 65 | 5,191 | 35 / 2,405 | 4 / 427 | 26 / 2,359 |

The speed-based control-mode proxy counts 327 `LikelyAuto`, 106
`LikelyManual`, 175 `MixedOrUnknown`, and 41 `StoppedOrUnknown` scenes. These
are suggestions only. In particular, the 2,405-frame automatic validation
clean split is not yet a publishable or training-ready result because 2,359
validation frames remain uncertain.

## 4. Human labels

Every validation scene must be reviewed. Start with validation, then all
`DetectionProbe` candidates, then uncertain training scenes.

`human_label`:

- `NaturalRun`: normal route execution representative of desired behavior;
- `OperationalStop`: legitimate queueing, yielding, loading wait, or safety
  stop rather than deliberate sensing collection;
- `DetectionProbe`: target orbit, repeated inspection, artificial parking, or
  other collection behavior not intended as a planning demonstration;
- `Uncertain`: intent cannot be established from available evidence.

`human_control_mode` is one of `Auto`, `Manual`, `Mixed`, or `Unknown`.

`planning_usable` is an independent `1` or `0`. Use `1` only when the scene is
a behavior the deployed planner should imitate. A scene can be semantically
normal yet remain unusable due to manual-control dynamics, bad localization,
trajectory discontinuity, or invalid future labels.

The missing information that needs human or source-system input is the true
control mode and collection intent. If the original vehicle log contains a
manual/automatic state, joining it by timestamp should replace the speed proxy
before final split generation.

## 5. Generate reviewed planning splits

After filling all human fields:

```bash
python tools/analysis_tools/build_planning_manifest_splits.py \
  --infos \
    train=data/kl_8/kl_infos_train.pkl \
    val=data/kl_8/kl_infos_val.pkl \
  --manifest \
    projects/work_dirs/stage2_e2e_lidar/planning_scene_audit/scene_manifest.csv \
  --out-dir data/kl_8/planning_reviewed \
  --label-source human
```

The command refuses incomplete human labels, control modes, or missing
`planning_usable` values. It preserves the source pickle metadata and emits
clean, natural, operational, detection-probe, uncertain, rejected, and four
control-mode diagnostic subsets.

`--label-source auto` is supported only for pipeline checks and exploratory
statistics. It must not be used to claim final model quality.

## 6. Experiment sequence

1. Review all 65 validation scenes and create `val_planning_clean` plus
   diagnostic natural, operational, manual, and collection subsets.
2. Re-evaluate the same frozen D2, D3-A, and D3-A.1 checkpoints. Report both
   frame-micro and scene-macro L2/collision, with separate control-mode and
   behavior buckets.
3. Review training scenes and train one equal-schedule clean-data control from
   the same initialization. Do not mix a data change with a model change.
4. Only then resume D3-A.2. Compare four cells: raw-data baseline, raw-data
   D3-A.2, clean-data baseline, and clean-data D3-A.2.
5. Keep the original validation set as a collection-distribution diagnostic,
   not as the sole planning benchmark.

The immediate decision gate is therefore data validity, not another planner
architecture tweak.
