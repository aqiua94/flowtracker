import argparse
import os

import cv2
import numpy as np

from core.track_guidance.rasterizer import (
    rasterize_track_prior,
    save_raster_outputs,
    save_raster_visualizations,
)


def infer_shape(args):
    if args.height is not None and args.width is not None:
        return args.height, args.width
    if args.flow is not None:
        flow = np.load(args.flow)
        if flow.ndim != 3 or flow.shape[2] != 2:
            raise ValueError(f"Expected flow with shape (H, W, 2), got {flow.shape}")
        return flow.shape[:2]
    if args.image is not None:
        image = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        return image.shape[:2]
    raise ValueError("Provide --height/--width, --flow, or --image to infer the output size.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True, help="Stage-4 points.npy with shape (N, 2)")
    parser.add_argument("--track_flow", required=True, help="Stage-4 track_flow.npy with shape (N, 2)")
    parser.add_argument("--valid_mask", default=None, help="Optional valid_mask.npy with shape (N,)")
    parser.add_argument("--confidence", default=None, help="Optional confidence.npy with shape (N,)")
    parser.add_argument("--endpoint_error", default=None, help="Optional alignment endpoint_error.npy with shape (N,)")
    parser.add_argument("--max_endpoint_error", type=float, default=None, help="Drop track points with alignment EPE above this value")
    parser.add_argument("--flow", default=None, help="Optional FlowSeek flow.npy for output shape inference")
    parser.add_argument("--image", default=None, help="Optional image for output shape inference")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--output_dir", default="demo_rasterizer_outputs/stage5_smoke")
    parser.add_argument("--channels_first", action="store_true", help="Save G_track as (5, H, W) instead of (H, W, 5)")
    parser.add_argument("--normalize_distance", action="store_true", help="Normalize distance channel by image diagonal")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    points = np.load(args.points).astype(np.float32)
    track_flow = np.load(args.track_flow).astype(np.float32)
    valid_mask = np.load(args.valid_mask) if args.valid_mask is not None else None
    confidence = np.load(args.confidence).astype(np.float32) if args.confidence is not None else None
    if args.endpoint_error is not None and args.max_endpoint_error is not None:
        endpoint_error = np.load(args.endpoint_error).reshape(-1).astype(np.float32)
        epe_mask = np.isfinite(endpoint_error) & (endpoint_error <= args.max_endpoint_error)
        valid_mask = epe_mask if valid_mask is None else (np.asarray(valid_mask).reshape(-1).astype(bool) & epe_mask)
    image_shape = infer_shape(args)

    prior, stats = rasterize_track_prior(
        points,
        track_flow,
        image_shape=image_shape,
        valid_mask=valid_mask,
        confidence=confidence,
        normalize_distance=args.normalize_distance,
        channels_first=args.channels_first,
    )

    prior_hwc = np.transpose(prior, (1, 2, 0)) if args.channels_first else prior
    save_raster_outputs(args.output_dir, prior, stats, prefix="g_track")
    save_raster_visualizations(args.output_dir, prior_hwc, prefix="g_track")

    print(f"Saved track prior to {os.path.join(args.output_dir, 'g_track.npy')}")
    print(f"G_track shape: {prior.shape}")
    print(f"Valid input points: {stats.num_valid_points}/{stats.num_input_points}")
    print(f"Rasterized pixels: {stats.num_rasterized_pixels}")
    print(f"Distance range: [0.0000, {stats.max_distance:.4f}], mean={stats.mean_distance:.4f}")


if __name__ == "__main__":
    main()
