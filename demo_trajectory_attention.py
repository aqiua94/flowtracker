import argparse
import json
import os

import cv2
import numpy as np

from core.track_guidance.trajectory_attention import trajectory_attention_enhance


def save_attention_visualization(path, attention):
    attention = np.asarray(attention, dtype=np.float32)
    if attention.size == 0:
        return
    vmax = float(np.percentile(attention, 99)) if np.any(attention > 0) else 1.0
    image = np.clip(attention / max(vmax, 1e-6), 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    cv2.imwrite(path, image)


def save_flow_delta_visualization(path, points, flow_delta, image_shape=None):
    points = np.asarray(points, dtype=np.float32)
    flow_delta = np.asarray(flow_delta, dtype=np.float32).reshape(-1)
    if image_shape is None:
        width = int(max(np.ceil(points[:, 0].max() + 8), 1))
        height = int(max(np.ceil(points[:, 1].max() + 8), 1))
    else:
        height, width = int(image_shape[0]), int(image_shape[1])
    canvas = np.zeros((height, width), dtype=np.float32)
    if len(points):
        xy = np.round(points).astype(np.int32)
        x = np.clip(xy[:, 0], 0, width - 1)
        y = np.clip(xy[:, 1], 0, height - 1)
        for px, py, value in zip(x, y, flow_delta):
            canvas[py, px] = max(canvas[py, px], float(value))
    if float(canvas.max()) > 0:
        canvas = canvas / float(canvas.max())
    image = cv2.applyColorMap((canvas * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(path, image)


def infer_shape(args):
    if args.height is not None and args.width is not None:
        return args.height, args.width
    if args.flow is not None:
        flow = np.load(args.flow)
        if flow.ndim != 3:
            raise ValueError(f"Expected flow array with shape (H, W, C), got {flow.shape}")
        return flow.shape[:2]
    if args.image is not None:
        image = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        return image.shape[:2]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True, help="Stage-4 points.npy with shape (N, 2)")
    parser.add_argument("--track_flow", required=True, help="Stage-4 track_flow.npy with shape (N, 2)")
    parser.add_argument("--valid_mask", default=None, help="Optional valid_mask.npy with shape (N,)")
    parser.add_argument("--confidence", default=None, help="Optional confidence.npy with shape (N,)")
    parser.add_argument("--endpoint_error", default=None, help="Optional alignment endpoint_error.npy with shape (N,)")
    parser.add_argument("--flow", default=None, help="Optional flow.npy for visualization shape inference")
    parser.add_argument("--image", default=None, help="Optional image for visualization shape inference")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--spatial_sigma", type=float, default=96.0)
    parser.add_argument("--motion_sigma", type=float, default=8.0)
    parser.add_argument("--endpoint_error_scale", type=float, default=3.0)
    parser.add_argument("--self_weight", type=float, default=1.0)
    parser.add_argument("--promote_invalid", action="store_true")
    parser.add_argument("--min_output_confidence", type=float, default=0.05)
    parser.add_argument("--output_dir", default="demo_trajectory_attention_outputs/stage7_smoke")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    points = np.load(args.points).astype(np.float32)
    track_flow = np.load(args.track_flow).astype(np.float32)
    valid_mask = np.load(args.valid_mask) if args.valid_mask is not None else None
    confidence = np.load(args.confidence).astype(np.float32) if args.confidence is not None else None
    endpoint_error = np.load(args.endpoint_error).astype(np.float32) if args.endpoint_error is not None else None

    enhanced_flow, enhanced_confidence, enhanced_valid, attention, stats = trajectory_attention_enhance(
        points,
        track_flow,
        valid_mask=valid_mask,
        confidence=confidence,
        endpoint_error=endpoint_error,
        spatial_sigma=args.spatial_sigma,
        motion_sigma=args.motion_sigma,
        endpoint_error_scale=args.endpoint_error_scale,
        self_weight=args.self_weight,
        promote_invalid=args.promote_invalid,
        min_output_confidence=args.min_output_confidence,
    )

    np.save(os.path.join(args.output_dir, "enhanced_track_flow.npy"), enhanced_flow)
    np.save(os.path.join(args.output_dir, "enhanced_confidence.npy"), enhanced_confidence)
    np.save(os.path.join(args.output_dir, "enhanced_valid_mask.npy"), enhanced_valid)
    np.save(os.path.join(args.output_dir, "attention.npy"), attention)
    with open(os.path.join(args.output_dir, "trajectory_attention_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2, allow_nan=True)

    image_shape = infer_shape(args)
    flow_delta = np.linalg.norm(enhanced_flow - track_flow, axis=1)
    save_attention_visualization(os.path.join(args.output_dir, "attention_matrix.png"), attention)
    save_flow_delta_visualization(os.path.join(args.output_dir, "flow_delta_points.png"), points, flow_delta, image_shape=image_shape)

    print(f"Saved enhanced tracks to {args.output_dir}")
    print(f"Input valid points: {stats.num_valid_input}/{stats.num_points}")
    print(f"Output valid points: {stats.num_valid_output}/{stats.num_points}")
    print(f"Mean reliability: {stats.mean_reliability:.6f}")
    print(f"Mean flow delta: {stats.mean_flow_delta:.6f}")
    print(f"Max flow delta: {stats.max_flow_delta:.6f}")


if __name__ == "__main__":
    main()
