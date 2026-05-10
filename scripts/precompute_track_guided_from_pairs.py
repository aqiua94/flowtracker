import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"] if isinstance(data, dict) else data


def resolve_repo_path(path, repo_root):
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def resolve_pair_path(path, pair_root):
    path = Path(path)
    if path.is_absolute():
        return path
    return pair_root / path


def run_command(cmd, dry_run=False):
    print(" ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def should_skip(path, overwrite):
    return path.exists() and not overwrite


def precompute_sample(sample, args, pair_root, repo_root):
    image1 = resolve_repo_path(sample["image1"], repo_root)
    image2 = resolve_repo_path(sample["image2"], repo_root)
    flowseek_dir = resolve_pair_path(sample["flowseek_output_dir"], pair_root)
    sampler_dir = resolve_pair_path(sample["sampler_output_dir"], pair_root)
    cotracker_dir = resolve_pair_path(sample["cotracker_output_dir"], pair_root)
    alignment_dir = resolve_pair_path(sample["alignment_output_dir"], pair_root)
    rasterizer_dir = resolve_pair_path(sample["rasterizer_output_dir"], pair_root)

    for path in [flowseek_dir, sampler_dir, cotracker_dir, alignment_dir, rasterizer_dir]:
        path.mkdir(parents=True, exist_ok=True)

    flow_path = flowseek_dir / "flow.npy"
    queries_path = sampler_dir / "queries.csv"
    tracks_path = cotracker_dir / "tracks.npy"
    prior_path = rasterizer_dir / "g_track.npy"
    enhanced_dir = alignment_dir / "trajectory_attention"
    enhanced_flow_path = enhanced_dir / "enhanced_track_flow.npy"

    if not should_skip(flow_path, args.overwrite):
        run_command(
            [
                sys.executable,
                "demo_pair.py",
                "--cfg",
                args.cfg,
                "--model",
                args.model,
                "--image1",
                str(image1),
                "--image2",
                str(image2),
                "--output_dir",
                str(flowseek_dir),
                "--max_size",
                str(args.flowseek_max_size),
            ],
            args.dry_run,
        )
    else:
        print(f"skip FlowSeek: {flow_path}", flush=True)

    if not should_skip(queries_path, args.overwrite):
        run_command(
            [
                sys.executable,
                "demo_adaptive_sampler.py",
                "--flow",
                str(flow_path),
                "--image",
                str(image1),
                "--output_dir",
                str(sampler_dir),
                "--num_points",
                str(args.num_points),
                "--cell_size",
                str(args.cell_size),
                "--min_points_per_cell",
                str(args.min_points_per_cell),
                "--max_points_per_cell",
                str(args.max_points_per_cell),
                "--min_distance",
                str(args.min_distance),
                "--flow_gradient_weight",
                str(args.flow_gradient_weight),
                "--flow_magnitude_weight",
                str(args.flow_magnitude_weight),
                "--image_edge_weight",
                str(args.image_edge_weight),
            ],
            args.dry_run,
        )
    else:
        print(f"skip sampler: {queries_path}", flush=True)

    if not should_skip(tracks_path, args.overwrite):
        run_command(
            [
                sys.executable,
                "demo_cotracker_pair.py",
                "--frames",
                str(image1),
                str(image2),
                "--queries",
                str(queries_path),
                "--output_dir",
                str(cotracker_dir),
                "--max_size",
                str(args.cotracker_max_size),
                "--device",
                args.device,
            ],
            args.dry_run,
        )
    else:
        print(f"skip CoTracker: {tracks_path}", flush=True)

    if not should_skip(alignment_dir / "track_flow.npy", args.overwrite):
        run_command(
            [
                sys.executable,
                "demo_track_flow_alignment.py",
                "--flow",
                str(flow_path),
                "--tracks",
                str(tracks_path),
                "--visibility",
                str(cotracker_dir / "visibility.npy"),
                "--confidence",
                str(cotracker_dir / "confidence.npy"),
                "--image",
                str(image1),
                "--output_dir",
                str(alignment_dir),
                "--pair_index",
                "0",
                "--min_confidence",
                str(args.min_confidence),
            ],
            args.dry_run,
        )
    else:
        print(f"skip alignment: {alignment_dir / 'track_flow.npy'}", flush=True)

    raster_track_flow = alignment_dir / "track_flow.npy"
    raster_confidence = alignment_dir / "confidence.npy"
    raster_valid_mask = alignment_dir / "valid_mask.npy"
    if args.use_trajectory_attention:
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        if not should_skip(enhanced_flow_path, args.overwrite):
            run_command(
                [
                    sys.executable,
                    "demo_trajectory_attention.py",
                    "--points",
                    str(alignment_dir / "points.npy"),
                    "--track_flow",
                    str(alignment_dir / "track_flow.npy"),
                    "--valid_mask",
                    str(alignment_dir / "valid_mask.npy"),
                    "--confidence",
                    str(alignment_dir / "confidence.npy"),
                    "--endpoint_error",
                    str(alignment_dir / "endpoint_error.npy"),
                    "--flow",
                    str(flow_path),
                    "--output_dir",
                    str(enhanced_dir),
                    "--spatial_sigma",
                    str(args.attention_spatial_sigma),
                    "--motion_sigma",
                    str(args.attention_motion_sigma),
                    "--endpoint_error_scale",
                    str(args.attention_endpoint_error_scale),
                    "--self_weight",
                    str(args.attention_self_weight),
                ]
                + (["--promote_invalid"] if args.attention_promote_invalid else []),
                args.dry_run,
            )
        else:
            print(f"skip trajectory attention: {enhanced_flow_path}", flush=True)
        raster_track_flow = enhanced_flow_path
        raster_confidence = enhanced_dir / "enhanced_confidence.npy"
        raster_valid_mask = enhanced_dir / "enhanced_valid_mask.npy"

    if not should_skip(prior_path, args.overwrite):
        run_command(
            [
                sys.executable,
                "demo_track_prior_rasterizer.py",
                "--points",
                str(alignment_dir / "points.npy"),
                "--track_flow",
                str(raster_track_flow),
                "--valid_mask",
                str(raster_valid_mask),
                "--confidence",
                str(raster_confidence),
                "--endpoint_error",
                str(alignment_dir / "endpoint_error.npy"),
                "--flow",
                str(flow_path),
                "--output_dir",
                str(rasterizer_dir),
            ]
            + (
                ["--max_endpoint_error", str(args.max_alignment_epe)]
                if args.max_alignment_epe is not None
                else []
            ),
            args.dry_run,
        )
    else:
        print(f"skip rasterizer: {prior_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="Path to *_pairs.json from prepare_sintel_track_guided_manifest.py")
    parser.add_argument("--cfg", default="config/eval/flowseek-T.json")
    parser.add_argument("--model", default="weights/flowseek_T_CT.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--flowseek_max_size", type=int, default=384)
    parser.add_argument("--cotracker_max_size", type=int, default=0)
    parser.add_argument("--num_points", type=int, default=128)
    parser.add_argument("--cell_size", type=int, default=48)
    parser.add_argument("--min_points_per_cell", type=int, default=1)
    parser.add_argument("--max_points_per_cell", type=int, default=6)
    parser.add_argument("--min_distance", type=float, default=6.0)
    parser.add_argument("--flow_gradient_weight", type=float, default=0.7)
    parser.add_argument("--flow_magnitude_weight", type=float, default=0.3)
    parser.add_argument("--image_edge_weight", type=float, default=0.0)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--max_alignment_epe", type=float, default=None, help="Drop rasterized track points with alignment EPE above this value")
    parser.add_argument("--use_trajectory_attention", action="store_true", help="Enhance sparse track flow before rasterization")
    parser.add_argument("--attention_spatial_sigma", type=float, default=96.0)
    parser.add_argument("--attention_motion_sigma", type=float, default=8.0)
    parser.add_argument("--attention_endpoint_error_scale", type=float, default=3.0)
    parser.add_argument("--attention_self_weight", type=float, default=1.0)
    parser.add_argument("--attention_promote_invalid", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    pair_path = Path(args.pairs).resolve()
    pair_root = pair_path.parent
    samples = load_pairs(pair_path)
    end = len(samples) if args.max_samples is None else min(len(samples), args.start + args.max_samples)

    for index in range(args.start, end):
        sample = samples[index]
        print(f"\n[{index + 1}/{len(samples)}] {sample.get('id', 'sample')}", flush=True)
        precompute_sample(sample, args, pair_root, repo_root)


if __name__ == "__main__":
    main()
