# This file provides a small CoTracker3 smoke test on a short image sequence.

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch


def read_frames(paths, max_size):
    frames = []
    resize_scale = 1.0
    original_size = None

    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read frame: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if original_size is None:
            original_size = image.shape[:2]
        elif image.shape[:2] != original_size:
            raise ValueError(f"Frame sizes differ: {image.shape[:2]} vs {original_size}")
        frames.append(image)

    if max_size is not None and max_size > 0:
        height, width = original_size
        resize_scale = min(1.0, float(max_size) / float(max(height, width)))
        if resize_scale != 1.0:
            resized = []
            new_width = int(round(width * resize_scale))
            new_height = int(round(height * resize_scale))
            for image in frames:
                resized.append(cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA))
            frames = resized

    video_np = np.stack(frames, axis=0)
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()
    return video, video_np, resize_scale


def default_queries(height, width):
    points = [
        (0, width * 0.25, height * 0.25),
        (0, width * 0.50, height * 0.50),
        (0, width * 0.75, height * 0.50),
        (0, width * 0.50, height * 0.75),
    ]
    return np.asarray(points, dtype=np.float32)


def load_queries(path, height, width):
    if path is None:
        return default_queries(height, width)

    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        queries = np.asarray(data, dtype=np.float32)
    else:
        queries = np.genfromtxt(path, delimiter=",", dtype=np.float32, names=None, skip_header=0)
        if queries.ndim > 0 and np.isnan(queries).any():
            queries = np.genfromtxt(path, delimiter=",", dtype=np.float32, names=None, skip_header=1)

    if queries.ndim == 1:
        queries = queries[None]

    if queries.shape[1] == 2:
        frame_ids = np.zeros((queries.shape[0], 1), dtype=np.float32)
        queries = np.concatenate([frame_ids, queries], axis=1)

    if queries.shape[1] != 3:
        raise ValueError("Queries must have columns [t, x, y] or [x, y].")

    if np.any(queries[:, 0] < 0):
        raise ValueError("Query frame indices must be non-negative.")
    if np.any(queries[:, 1] < 0) or np.any(queries[:, 1] > width - 1):
        raise ValueError("Query x coordinates are outside the image.")
    if np.any(queries[:, 2] < 0) or np.any(queries[:, 2] > height - 1):
        raise ValueError("Query y coordinates are outside the image.")

    return queries.astype(np.float32)


def draw_tracks(video_np, tracks, visibility, queries, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 160, 255),
        (255, 220, 80),
        (220, 80, 255),
        (80, 255, 220),
    ]

    for t, frame in enumerate(video_np):
        canvas = frame.copy()
        for i in range(tracks.shape[1]):
            color = colors[i % len(colors)]
            track_xy = tracks[: t + 1, i]
            vis_i = visibility[: t + 1, i]

            for j in range(1, len(track_xy)):
                if not (vis_i[j - 1] and vis_i[j]):
                    continue
                p0 = tuple(np.round(track_xy[j - 1]).astype(int))
                p1 = tuple(np.round(track_xy[j]).astype(int))
                cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)

            if visibility[t, i]:
                center = tuple(np.round(tracks[t, i]).astype(int))
                cv2.circle(canvas, center, 4, color, -1, cv2.LINE_AA)
                cv2.putText(canvas, str(i), (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        query_frame = int(queries[0, 0])
        label = f"frame {t} | query frame {query_frame}"
        cv2.putText(canvas, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"track_{t:04d}.png"), canvas_bgr)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", nargs="+", required=True, help="Input frames in temporal order")
    parser.add_argument("--queries", default=None, help="Optional CSV/TXT/JSON queries with [t,x,y] or [x,y]")
    parser.add_argument("--output_dir", default="demo_cotracker_outputs/stage2_smoke")
    parser.add_argument("--max_size", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid_size", type=int, default=0, help="Use CoTracker grid mode instead of manual queries when > 0")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    os.makedirs(args.output_dir, exist_ok=True)

    video, video_np, resize_scale = read_frames(args.frames, args.max_size)
    _, _, _, height, width = video.shape
    queries_np = load_queries(args.queries, height, width)
    queries = torch.from_numpy(queries_np)[None].float()

    device = torch.device(args.device)
    video = video.to(device)
    queries = queries.to(device)

    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", trust_repo=True).to(device)
    model.eval()

    if args.grid_size > 0:
        pred_tracks, pred_visibility = model(video, grid_size=args.grid_size)
        queries_np = np.empty((0, 3), dtype=np.float32)
    else:
        pred_tracks, pred_visibility = model(video, queries=queries)

    tracks = pred_tracks[0].detach().cpu().numpy()
    visibility = pred_visibility[0].detach().cpu().numpy().astype(bool)
    confidence = visibility.astype(np.float32)

    if resize_scale != 1.0:
        tracks = tracks / resize_scale
        queries_np[:, 1:] = queries_np[:, 1:] / resize_scale

    np.save(os.path.join(args.output_dir, "tracks.npy"), tracks)
    np.save(os.path.join(args.output_dir, "visibility.npy"), visibility)
    np.save(os.path.join(args.output_dir, "confidence.npy"), confidence)
    np.savetxt(os.path.join(args.output_dir, "queries.csv"), queries_np, delimiter=",", header="t,x,y", comments="")

    draw_video_np = video_np
    if resize_scale != 1.0:
        # For visualization, keep the resized frames and resized tracks so the overlays align.
        tracks_for_vis = pred_tracks[0].detach().cpu().numpy()
        queries_for_vis = load_queries(args.queries, height, width)
    else:
        tracks_for_vis = tracks
        queries_for_vis = queries_np
    draw_tracks(draw_video_np, tracks_for_vis, visibility, queries_for_vis, args.output_dir)

    print(f"Saved tracks to {os.path.join(args.output_dir, 'tracks.npy')}")
    print(f"Saved visibility to {os.path.join(args.output_dir, 'visibility.npy')}")
    print(f"Saved confidence placeholder to {os.path.join(args.output_dir, 'confidence.npy')}")
    print(f"Tracks shape: {tracks.shape}")
    print(f"Visibility shape: {visibility.shape}")
    if tracks.shape[0] >= 2 and tracks.shape[1] > 0:
        displacement = tracks[1] - tracks[0]
        print(f"Mean displacement frame0->frame1: {displacement.mean(axis=0).tolist()}")


if __name__ == "__main__":
    main()
