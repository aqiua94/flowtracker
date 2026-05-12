import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.utils.frame_utils import readPFM


def rel(path, root):
    return str(Path(path).resolve().relative_to(root.resolve()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler_root", default="data/Sampler/FlyingThings3D")
    parser.add_argument("--output_root", default="precomputed/track_guided_sampler_flyingthings_smoke_attn")
    parser.add_argument("--manifest", default="precomputed/track_guided_sampler_flyingthings_smoke_attn/manifest.json")
    parser.add_argument("--max_pairs", type=int, default=2)
    args = parser.parse_args()

    sampler_root = Path(args.sampler_root)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest)
    pair_path = manifest_path.with_name(manifest_path.stem + "_pairs.json")
    output_root.mkdir(parents=True, exist_ok=True)

    image_dir = sampler_root / "RGB_cleanpass" / "left"
    flow_dir = sampler_root / "optical_flow" / "forward"
    images = sorted(image_dir.glob("*.png"))
    if len(images) < 2:
        raise ValueError(f"Need at least two images in {image_dir}")

    manifest_samples = []
    pair_samples = []
    for index, image1 in enumerate(images[:-1]):
        if len(pair_samples) >= args.max_pairs:
            break
        image2 = images[index + 1]
        frame_id = image1.stem
        flow_pfm = flow_dir / f"{frame_id}.pfm"
        if not flow_pfm.exists():
            raise FileNotFoundError(flow_pfm)

        sample_id = f"FlyingThings3D_frame_{frame_id}"
        sample_dir = output_root / "FlyingThings3D" / sample_id
        flowseek_dir = sample_dir / "flowseek"
        sampler_dir = sample_dir / "sampler"
        cotracker_dir = sample_dir / "cotracker"
        alignment_dir = sample_dir / "alignment"
        rasterizer_dir = sample_dir / "rasterizer"
        sample_dir.mkdir(parents=True, exist_ok=True)

        gt_flow = readPFM(flow_pfm).astype(np.float32)
        if gt_flow.ndim == 3 and gt_flow.shape[2] >= 2:
            gt_flow = gt_flow[..., :2]
        else:
            raise ValueError(f"Expected two-channel flow PFM, got {gt_flow.shape}: {flow_pfm}")
        valid = np.isfinite(gt_flow).all(axis=2).astype(np.float32)
        gt_flow = np.nan_to_num(gt_flow, nan=0.0, posinf=0.0, neginf=0.0)
        np.save(sample_dir / "gt_flow.npy", gt_flow)
        np.save(sample_dir / "valid.npy", valid)

        manifest_samples.append(
            {
                "initial_flow": rel(flowseek_dir / "flow.npy", manifest_path.parent),
                "track_prior": rel(rasterizer_dir / "g_track.npy", manifest_path.parent),
                "gt_flow": rel(sample_dir / "gt_flow.npy", manifest_path.parent),
                "valid": rel(sample_dir / "valid.npy", manifest_path.parent),
                "dataset": "Sampler/FlyingThings3D",
                "sample_id": sample_id,
            }
        )
        pair_samples.append(
            {
                "dataset": "Sampler/FlyingThings3D",
                "id": sample_id,
                "image1": rel(image1, Path.cwd()),
                "image2": rel(image2, Path.cwd()),
                "gt_flow_pfm": rel(flow_pfm, Path.cwd()),
                "sample_dir": rel(sample_dir, output_root),
                "flowseek_output_dir": rel(flowseek_dir, output_root),
                "sampler_output_dir": rel(sampler_dir, output_root),
                "cotracker_output_dir": rel(cotracker_dir, output_root),
                "alignment_output_dir": rel(alignment_dir, output_root),
                "rasterizer_output_dir": rel(rasterizer_dir, output_root),
            }
        )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"samples": manifest_samples}, f, indent=2)
    with open(pair_path, "w", encoding="utf-8") as f:
        json.dump({"samples": pair_samples}, f, indent=2)
    print(f"Wrote {len(manifest_samples)} samples to {manifest_path}")
    print(f"Wrote {len(pair_samples)} pairs to {pair_path}")


if __name__ == "__main__":
    main()
