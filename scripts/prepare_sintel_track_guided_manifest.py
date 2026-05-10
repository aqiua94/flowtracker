import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "core"))

from utils.frame_utils import readFlow


def list_sintel_samples(root, split, dstype, scenes=None, max_pairs=None, start_index=0):
    root = Path(root)
    image_root = root / split / dstype
    flow_root = root / split / "flow"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing Sintel image directory: {image_root}")
    if not flow_root.is_dir():
        raise FileNotFoundError(f"Missing Sintel flow directory: {flow_root}")

    scene_filter = set(scenes) if scenes else None
    samples = []
    seen = 0
    for scene_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        scene = scene_dir.name
        if scene_filter is not None and scene not in scene_filter:
            continue

        frames = sorted(scene_dir.glob("*.png"))
        flows = sorted((flow_root / scene).glob("*.flo"))
        num_pairs = min(len(frames) - 1, len(flows))
        for index in range(num_pairs):
            if seen < start_index:
                seen += 1
                continue
            samples.append(
                {
                    "scene": scene,
                    "frame_index": index,
                    "image1": str(frames[index]),
                    "image2": str(frames[index + 1]),
                    "gt_flow_flo": str(flows[index]),
                }
            )
            seen += 1
            if max_pairs is not None and len(samples) >= max_pairs:
                return samples
    return samples


def sample_id(sample):
    frame_name = Path(sample["gt_flow_flo"]).stem
    return f"{sample['scene']}_{frame_name}"


def maybe_relative(path, base):
    path = Path(path).resolve()
    base = Path(base).resolve()
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def convert_gt_flow(sample, sample_dir):
    flow = readFlow(sample["gt_flow_flo"])
    if flow is None:
        raise ValueError(f"Could not read GT flow: {sample['gt_flow_flo']}")
    flow = flow.astype(np.float32)
    valid = np.isfinite(flow).all(axis=2) & (np.abs(flow).max(axis=2) < 1e9)

    gt_flow_path = sample_dir / "gt_flow.npy"
    valid_path = sample_dir / "valid.npy"
    np.save(gt_flow_path, flow)
    np.save(valid_path, valid.astype(np.float32))
    return gt_flow_path, valid_path


def build_manifest(samples, output_root, manifest_path, convert_gt, make_dirs):
    manifest_path = Path(manifest_path)
    output_root = Path(output_root)
    manifest_samples = []
    pair_records = []

    for sample in samples:
        sid = sample_id(sample)
        sample_dir = output_root / sample["scene"] / sid
        flowseek_dir = sample_dir / "flowseek"
        sampler_dir = sample_dir / "sampler"
        cotracker_dir = sample_dir / "cotracker"
        alignment_dir = sample_dir / "alignment"
        rasterizer_dir = sample_dir / "rasterizer"

        if make_dirs:
            for path in [flowseek_dir, sampler_dir, cotracker_dir, alignment_dir, rasterizer_dir]:
                path.mkdir(parents=True, exist_ok=True)

        if convert_gt:
            gt_flow_path, valid_path = convert_gt_flow(sample, sample_dir)
        else:
            gt_flow_path = sample_dir / "gt_flow.npy"
            valid_path = sample_dir / "valid.npy"

        manifest_samples.append(
            {
                "initial_flow": maybe_relative(flowseek_dir / "flow.npy", manifest_path.parent),
                "track_prior": maybe_relative(rasterizer_dir / "g_track.npy", manifest_path.parent),
                "gt_flow": maybe_relative(gt_flow_path, manifest_path.parent),
                "valid": maybe_relative(valid_path, manifest_path.parent),
            }
        )
        pair_records.append(
            {
                **sample,
                "id": sid,
                "sample_dir": maybe_relative(sample_dir, manifest_path.parent),
                "flowseek_output_dir": maybe_relative(flowseek_dir, manifest_path.parent),
                "sampler_output_dir": maybe_relative(sampler_dir, manifest_path.parent),
                "cotracker_output_dir": maybe_relative(cotracker_dir, manifest_path.parent),
                "alignment_output_dir": maybe_relative(alignment_dir, manifest_path.parent),
                "rasterizer_output_dir": maybe_relative(rasterizer_dir, manifest_path.parent),
            }
        )

    write_json(manifest_path, {"samples": manifest_samples})
    pair_list_path = manifest_path.with_name(manifest_path.stem + "_pairs.json")
    write_json(pair_list_path, {"samples": pair_records})
    return pair_list_path, len(manifest_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sintel_root", default="data/MPI-Sintel")
    parser.add_argument("--split", default="training")
    parser.add_argument("--dstype", default="clean", choices=["clean", "final"])
    parser.add_argument("--output_root", default="precomputed/track_guided_sintel_clean")
    parser.add_argument("--manifest", default="precomputed/track_guided_sintel_clean/manifest.json")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--no_convert_gt", action="store_true", help="Only plan paths; do not convert Sintel .flo files")
    parser.add_argument("--no_make_dirs", action="store_true", help="Do not create per-sample output directories")
    args = parser.parse_args()

    samples = list_sintel_samples(
        root=args.sintel_root,
        split=args.split,
        dstype=args.dstype,
        scenes=args.scenes,
        max_pairs=args.max_pairs,
        start_index=args.start_index,
    )
    if not samples:
        raise ValueError("No Sintel frame pairs found.")

    pair_list_path, count = build_manifest(
        samples=samples,
        output_root=args.output_root,
        manifest_path=args.manifest,
        convert_gt=not args.no_convert_gt,
        make_dirs=not args.no_make_dirs,
    )

    print(f"Prepared {count} Sintel {args.dstype} samples.")
    print(f"Manifest: {args.manifest}")
    print(f"Pair list: {pair_list_path}")
    first = samples[0]
    print(f"First pair: {first['image1']} -> {first['image2']}")


if __name__ == "__main__":
    main()
