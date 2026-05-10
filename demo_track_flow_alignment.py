import argparse
import os

import cv2
import numpy as np

from core.track_guidance.cotracker_wrapper import (
    compare_track_flow_to_dense,
    draw_alignment_overlay,
    load_optional_array,
    save_alignment_outputs,
)


def read_rgb(path):
    if path is None:
        return None
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", required=True, help="FlowSeek flow.npy with shape (H, W, 2)")
    parser.add_argument("--tracks", required=True, help="CoTracker tracks.npy with shape (T, N, 2)")
    parser.add_argument("--visibility", default=None, help="Optional CoTracker visibility.npy with shape (T, N)")
    parser.add_argument("--confidence", default=None, help="Optional CoTracker confidence.npy with shape (T, N)")
    parser.add_argument("--image", default=None, help="Optional frame image for visualization")
    parser.add_argument("--output_dir", default="demo_alignment_outputs/stage4_smoke")
    parser.add_argument("--pair_index", type=int, default=0)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--max_vis_points", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    flow = np.load(args.flow).astype(np.float32)
    tracks = np.load(args.tracks).astype(np.float32)
    visibility = load_optional_array(args.visibility)
    confidence = load_optional_array(args.confidence)
    image = read_rgb(args.image)

    if image is not None and image.shape[:2] != flow.shape[:2]:
        raise ValueError(f"Image size {image.shape[:2]} does not match flow size {flow.shape[:2]}")

    alignment = compare_track_flow_to_dense(
        flow,
        tracks,
        visibility=visibility,
        confidence=confidence,
        pair_index=args.pair_index,
        min_confidence=args.min_confidence,
    )

    save_alignment_outputs(args.output_dir, alignment)
    draw_alignment_overlay(
        image,
        alignment,
        os.path.join(args.output_dir, "alignment_overlay.png"),
        max_points=args.max_vis_points,
    )

    stats = alignment["stats"]
    print(f"Saved sparse track-flow alignment outputs to {args.output_dir}")
    print(f"Valid tracks: {stats.num_valid}/{stats.num_tracks} ({stats.valid_ratio:.3f})")
    print(f"Mean CoTracker displacement: [{stats.mean_track_dx:.4f}, {stats.mean_track_dy:.4f}]")
    print(f"Mean FlowSeek sampled flow: [{stats.mean_flowseek_dx:.4f}, {stats.mean_flowseek_dy:.4f}]")
    print(
        "Endpoint error: "
        f"mean={stats.mean_endpoint_error:.4f}, "
        f"median={stats.median_endpoint_error:.4f}, "
        f"max={stats.max_endpoint_error:.4f}"
    )


if __name__ == "__main__":
    main()
