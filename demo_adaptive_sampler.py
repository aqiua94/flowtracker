import argparse
import os

import cv2
import numpy as np

from core.track_guidance.sampler import SamplingConfig, adaptive_sample_points, points_to_cotracker_queries


def read_rgb(path):
    if path is None:
        return None
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_weight_map(path, weight_map):
    normalized = weight_map.astype(np.float32)
    if float(normalized.max()) > float(normalized.min()):
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    else:
        normalized = np.zeros_like(normalized, dtype=np.float32)
    heat = (normalized * 255.0).clip(0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    cv2.imwrite(path, heat)


def save_points_overlay(path, image, points, weights):
    if image is None:
        height = int(points[:, 1].max() + 16) if len(points) else 256
        width = int(points[:, 0].max() + 16) if len(points) else 256
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        canvas = image.copy()

    if len(points):
        w = weights.astype(np.float32)
        if float(w.max()) > float(w.min()):
            w = (w - w.min()) / (w.max() - w.min())
        else:
            w = np.ones_like(w, dtype=np.float32) * 0.5

        for (x, y), score in zip(points, w):
            color = (int(255 * score), int(255 * (1.0 - abs(score - 0.5) * 2.0)), int(255 * (1.0 - score)))
            cv2.circle(canvas, (int(round(x)), int(round(y))), 2, color, -1, cv2.LINE_AA)

    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", required=True, help="Path to FlowSeek flow.npy with shape (H, W, 2)")
    parser.add_argument("--image", default=None, help="Optional RGB/BGR image for edge weighting and visualization")
    parser.add_argument("--output_dir", default="demo_sampler_outputs/stage3_smoke")
    parser.add_argument("--num_points", type=int, default=256)
    parser.add_argument("--cell_size", type=int, default=32)
    parser.add_argument("--min_points_per_cell", type=int, default=1)
    parser.add_argument("--max_points_per_cell", type=int, default=8)
    parser.add_argument("--min_distance", type=float, default=4.0)
    parser.add_argument("--flow_gradient_weight", type=float, default=0.7)
    parser.add_argument("--flow_magnitude_weight", type=float, default=0.3)
    parser.add_argument("--image_edge_weight", type=float, default=0.0)
    parser.add_argument("--query_frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    flow = np.load(args.flow)
    image = read_rgb(args.image)
    if image is not None and image.shape[:2] != flow.shape[:2]:
        raise ValueError(f"Image size {image.shape[:2]} does not match flow size {flow.shape[:2]}")

    config = SamplingConfig(
        num_points=args.num_points,
        cell_size=args.cell_size,
        min_points_per_cell=args.min_points_per_cell,
        max_points_per_cell=args.max_points_per_cell,
        min_distance=args.min_distance,
        flow_gradient_weight=args.flow_gradient_weight,
        flow_magnitude_weight=args.flow_magnitude_weight,
        image_edge_weight=args.image_edge_weight,
        seed=args.seed,
    )

    points, weights, weight_map = adaptive_sample_points(flow, image=image, config=config)
    queries = points_to_cotracker_queries(points, query_frame=args.query_frame)

    np.save(os.path.join(args.output_dir, "points.npy"), points)
    np.save(os.path.join(args.output_dir, "point_weights.npy"), weights)
    np.save(os.path.join(args.output_dir, "sampling_weight.npy"), weight_map)
    np.savetxt(os.path.join(args.output_dir, "queries.csv"), queries, delimiter=",", header="t,x,y", comments="")
    save_weight_map(os.path.join(args.output_dir, "sampling_weight.png"), weight_map)
    save_points_overlay(os.path.join(args.output_dir, "sampled_points.png"), image, points, weights)

    print(f"Saved {len(points)} sampled points to {os.path.join(args.output_dir, 'points.npy')}")
    print(f"Saved CoTracker queries to {os.path.join(args.output_dir, 'queries.csv')}")
    print(f"Saved visualizations to {args.output_dir}")
    print(f"Point bounds x=[{points[:, 0].min():.1f}, {points[:, 0].max():.1f}], y=[{points[:, 1].min():.1f}, {points[:, 1].max():.1f}]")
    print(f"Weight range [{weights.min():.4f}, {weights.max():.4f}], mean={weights.mean():.4f}")


if __name__ == "__main__":
    main()
