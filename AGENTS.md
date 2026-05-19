# Repository Guidelines

## Project Structure & Module Organization

This repository is a UniAD/OpenMMLab-style autonomous driving project. Core model code lives under `projects/mmdet3d_plugin/`, with UniAD detectors, heads, hooks, and APIs in `projects/mmdet3d_plugin/uniad/`. Experiment configs are in `projects/configs/`, split into `stage1_track_map/`, `stage2_e2e/`, and shared `_base_/` files. Entrypoint scripts are in `tools/`, including training, evaluation, data conversion, and visualization helpers. Setup and workflow documentation is in `docs/`; visual assets used by the README are in `sources/`; container setup is in `docker/`.

Expected local-only runtime directories include `data/` for nuScenes files and generated infos, and `ckpts/` for pretrained checkpoints.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install UniAD Python requirements after the OpenMMLab stack from `docs/INSTALL.md`.
- `./tools/uniad_create_data.sh`: generate nuScenes temporal info files under `data/infos/`.
- `./tools/uniad_dist_train.sh ./projects/configs/stage1_track_map/base_track_map.py 8`: launch distributed stage-1 training.
- `./tools/uniad_dist_eval.sh ./projects/configs/stage1_track_map/base_track_map.py ./ckpts/uniad_base_track_map.pth 8`: evaluate a checkpoint.
- `python ./tools/analysis_tools/visualize/run.py --predroot RESULTS.pkl --out_folder OUTPUT --demo_video test_demo.avi --project_to_cam True`: render prediction visualizations.

Use the matching `uniad_slurm_train.sh` and `uniad_slurm_eval.sh` wrappers on Slurm clusters.

## Coding Style & Naming Conventions

Follow the existing Python style: 4-space indentation, snake_case for functions, variables, and config keys, PascalCase for classes, and OpenMMLab registries such as `@DETECTORS.register_module()`. Keep config filenames descriptive and lowercase, for example `base_track_map.py` or `base_e2e.py`. Prefer extending existing modules in `projects/mmdet3d_plugin/uniad/` over adding parallel abstractions.

## Testing Guidelines

There is no standalone unit-test suite in this checkout. Validate changes with the smallest relevant training or evaluation path. For model, dataset, or config changes, run `tools/test.py` or `uniad_dist_eval.sh` against a known checkpoint and config. For data pipeline changes, regenerate infos with `uniad_create_data.sh` and verify the expected files under `data/infos/`. Record GPU count, config path, checkpoint path, and key metrics in your PR.

## Commit & Pull Request Guidelines

Recent commits are short and lowercase, with occasional scoped prefixes such as `bugfix: planning_evaluation_strategy`. Keep commits focused and use imperative summaries when possible. Pull requests should include a concise description, affected configs/modules, reproduction or evaluation commands, metric changes, and any dataset/checkpoint assumptions. Add screenshots or video paths for visualization changes.

## Security & Configuration Tips

Do not commit datasets, checkpoints, generated results, or machine-specific paths. Keep large artifacts in `data/`, `ckpts/`, or external storage, and document required downloads in the PR.
