import argparse
import os
import subprocess

import matplotlib

matplotlib.use('Agg')

from tools.analysis_tools.visualize.run import Visualizer


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('1', 'true', 'yes', 'y'):
        return True
    if value in ('0', 'false', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def write_video(frame_dir, out_path, fps=4, downsample=2):
    ext = os.path.splitext(out_path)[1].lower()
    if ext != '.webm':
        raise ValueError('run_e2e_subset.py writes .webm videos only.')

    scale = f'scale=trunc(iw/{downsample}/2)*2:trunc(ih/{downsample}/2)*2'
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate',
        str(fps),
        '-i',
        os.path.join(frame_dir, '%03d.jpg'),
        '-vf',
        scale,
        '-c:v',
        'libvpx-vp9',
        '-crf',
        '32',
        '-b:v',
        '0',
        '-pix_fmt',
        'yuv420p',
        out_path,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predroot', required=True, help='Path to UniAD results pkl')
    parser.add_argument('--out-folder', required=True, help='Folder for rendered frames')
    parser.add_argument('--demo-video', required=True, help='Output webm path')
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--max-frames', type=int, default=80)
    parser.add_argument('--project-to-cam', type=parse_bool, default=True)
    parser.add_argument('--with-map', type=parse_bool, default=True)
    args = parser.parse_args()

    render_cfg = dict(
        with_occ_map=False,
        with_map=args.with_map,
        with_planning=True,
        with_pred_box=True,
        with_pred_traj=True,
        show_gt_boxes=False,
        show_lidar=False,
        show_command=True,
        show_hd_map=False,
        show_sdc_car=True,
        show_legend=True,
        show_sdc_traj=False,
    )

    viser = Visualizer(
        version=args.version,
        predroot=args.predroot,
        dataroot=args.dataroot,
        **render_cfg,
    )

    os.makedirs(args.out_folder, exist_ok=True)
    rendered = 0
    for sample in viser.nusc.sample:
        sample_token = sample['token']
        if sample_token not in viser.token_set:
            continue
        out_prefix = os.path.join(args.out_folder, str(rendered).zfill(3))
        viser.visualize_bev(sample_token, out_prefix)
        if args.project_to_cam:
            viser.visualize_cam(sample_token, out_prefix)
            viser.combine(out_prefix)
        rendered += 1
        if rendered >= args.max_frames:
            break

    if rendered == 0:
        raise RuntimeError('No sample tokens from predroot matched the nuScenes samples.')

    os.makedirs(os.path.dirname(args.demo_video), exist_ok=True)
    write_video(args.out_folder, args.demo_video, fps=4, downsample=2)
    print(f'Rendered {rendered} frames to {args.out_folder}')
    print(f'Video saved to {args.demo_video}')


if __name__ == '__main__':
    main()
