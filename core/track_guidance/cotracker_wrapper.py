import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class AlignmentStats:
    num_tracks: int
    num_valid: int
    valid_ratio: float
    mean_track_dx: float
    mean_track_dy: float
    mean_flowseek_dx: float
    mean_flowseek_dy: float
    mean_endpoint_error: float
    median_endpoint_error: float
    max_endpoint_error: float

    def to_dict(self):
        return asdict(self)


def load_optional_array(path, fallback=None):
    if path is None:
        return fallback
    return np.load(path)


def bilinear_sample_flow(flow, points):
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected flow with shape (H, W, 2), got {flow.shape}")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {points.shape}")

    height, width = flow.shape[:2]
    x = points[:, 0].astype(np.float32)
    y = points[:, 1].astype(np.float32)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)

    wa = ((x1 - x) * (y1 - y))[:, None]
    wb = ((x1 - x) * (y - y0))[:, None]
    wc = ((x - x0) * (y1 - y))[:, None]
    wd = ((x - x0) * (y - y0))[:, None]

    return (
        wa * flow[y0c, x0c]
        + wb * flow[y1c, x0c]
        + wc * flow[y0c, x1c]
        + wd * flow[y1c, x1c]
    ).astype(np.float32)


def compute_sparse_track_flow(tracks, visibility=None, confidence=None, pair_index=0, min_confidence=0.0, image_shape=None):
    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError(f"Expected tracks with shape (T, N, 2), got {tracks.shape}")
    if pair_index < 0 or pair_index + 1 >= tracks.shape[0]:
        raise ValueError(f"pair_index={pair_index} is outside tracks with T={tracks.shape[0]}")

    start = tracks[pair_index].astype(np.float32)
    end = tracks[pair_index + 1].astype(np.float32)
    track_flow = (end - start).astype(np.float32)

    valid = np.isfinite(start).all(axis=1) & np.isfinite(end).all(axis=1) & np.isfinite(track_flow).all(axis=1)
    if visibility is not None:
        if visibility.shape[:2] != tracks.shape[:2]:
            raise ValueError(f"Visibility shape {visibility.shape} does not match tracks shape {tracks.shape[:2]}")
        valid &= visibility[pair_index].astype(bool) & visibility[pair_index + 1].astype(bool)

    if confidence is None:
        pair_confidence = valid.astype(np.float32)
    else:
        if confidence.shape[:2] != tracks.shape[:2]:
            raise ValueError(f"Confidence shape {confidence.shape} does not match tracks shape {tracks.shape[:2]}")
        pair_confidence = np.minimum(confidence[pair_index], confidence[pair_index + 1]).astype(np.float32)
        valid &= pair_confidence >= min_confidence

    if image_shape is not None:
        height, width = image_shape[:2]
        valid &= start[:, 0] >= 0
        valid &= start[:, 0] <= width - 1
        valid &= start[:, 1] >= 0
        valid &= start[:, 1] <= height - 1
        valid &= end[:, 0] >= 0
        valid &= end[:, 0] <= width - 1
        valid &= end[:, 1] >= 0
        valid &= end[:, 1] <= height - 1

    return {
        "points": start,
        "next_points": end,
        "track_flow": track_flow,
        "valid_mask": valid.astype(bool),
        "confidence": pair_confidence,
    }


def compare_track_flow_to_dense(flow, tracks, visibility=None, confidence=None, pair_index=0, min_confidence=0.0):
    sparse = compute_sparse_track_flow(
        tracks,
        visibility=visibility,
        confidence=confidence,
        pair_index=pair_index,
        min_confidence=min_confidence,
        image_shape=flow.shape[:2],
    )
    dense_at_points = bilinear_sample_flow(flow, sparse["points"])
    error = np.linalg.norm(dense_at_points - sparse["track_flow"], axis=1).astype(np.float32)

    valid = sparse["valid_mask"]
    if np.any(valid):
        stats = AlignmentStats(
            num_tracks=int(tracks.shape[1]),
            num_valid=int(valid.sum()),
            valid_ratio=float(valid.mean()),
            mean_track_dx=float(sparse["track_flow"][valid, 0].mean()),
            mean_track_dy=float(sparse["track_flow"][valid, 1].mean()),
            mean_flowseek_dx=float(dense_at_points[valid, 0].mean()),
            mean_flowseek_dy=float(dense_at_points[valid, 1].mean()),
            mean_endpoint_error=float(error[valid].mean()),
            median_endpoint_error=float(np.median(error[valid])),
            max_endpoint_error=float(error[valid].max()),
        )
    else:
        stats = AlignmentStats(
            num_tracks=int(tracks.shape[1]),
            num_valid=0,
            valid_ratio=0.0,
            mean_track_dx=float("nan"),
            mean_track_dy=float("nan"),
            mean_flowseek_dx=float("nan"),
            mean_flowseek_dy=float("nan"),
            mean_endpoint_error=float("nan"),
            median_endpoint_error=float("nan"),
            max_endpoint_error=float("nan"),
        )

    sparse["flowseek_at_points"] = dense_at_points
    sparse["endpoint_error"] = error
    sparse["stats"] = stats
    return sparse


def save_alignment_outputs(output_dir, alignment):
    np.save(f"{output_dir}/points.npy", alignment["points"])
    np.save(f"{output_dir}/next_points.npy", alignment["next_points"])
    np.save(f"{output_dir}/track_flow.npy", alignment["track_flow"])
    np.save(f"{output_dir}/flowseek_at_points.npy", alignment["flowseek_at_points"])
    np.save(f"{output_dir}/valid_mask.npy", alignment["valid_mask"])
    np.save(f"{output_dir}/confidence.npy", alignment["confidence"])
    np.save(f"{output_dir}/endpoint_error.npy", alignment["endpoint_error"])
    with open(f"{output_dir}/alignment_stats.json", "w", encoding="utf-8") as f:
        json.dump(alignment["stats"].to_dict(), f, indent=2, allow_nan=True)


def draw_alignment_overlay(image, alignment, output_path, max_points=256):
    if image is None:
        points = alignment["points"]
        height = int(max(256, np.nanmax(points[:, 1]) + 16)) if len(points) else 256
        width = int(max(256, np.nanmax(points[:, 0]) + 16)) if len(points) else 256
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        canvas = image.copy()

    valid_indices = np.flatnonzero(alignment["valid_mask"])
    if len(valid_indices) > max_points:
        order = np.argsort(-alignment["endpoint_error"][valid_indices])
        valid_indices = valid_indices[order[:max_points]]

    for idx in valid_indices:
        p0 = alignment["points"][idx]
        p_track = p0 + alignment["track_flow"][idx]
        p_dense = p0 + alignment["flowseek_at_points"][idx]
        p0_i = tuple(np.round(p0).astype(int))
        p_track_i = tuple(np.round(p_track).astype(int))
        p_dense_i = tuple(np.round(p_dense).astype(int))
        cv2.circle(canvas, p0_i, 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(canvas, p0_i, p_track_i, (80, 255, 80), 1, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(canvas, p0_i, p_dense_i, (255, 80, 80), 1, cv2.LINE_AA, tipLength=0.25)

    cv2.putText(canvas, "green: CoTracker | red: FlowSeek", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
