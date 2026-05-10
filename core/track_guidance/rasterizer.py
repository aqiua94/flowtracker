import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class RasterizeStats:
    height: int
    width: int
    num_input_points: int
    num_valid_points: int
    num_rasterized_pixels: int
    mean_confidence: float
    max_distance: float
    mean_distance: float

    def to_dict(self):
        return asdict(self)


def _validate_inputs(points, track_flow, valid_mask=None, confidence=None):
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {points.shape}")
    if track_flow.ndim != 2 or track_flow.shape[1] != 2:
        raise ValueError(f"Expected track_flow with shape (N, 2), got {track_flow.shape}")
    if points.shape[0] != track_flow.shape[0]:
        raise ValueError(f"points and track_flow must have the same N, got {points.shape[0]} and {track_flow.shape[0]}")

    num_points = points.shape[0]
    if valid_mask is None:
        valid_mask = np.ones(num_points, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask).reshape(-1).astype(bool)
        if valid_mask.shape[0] != num_points:
            raise ValueError(f"valid_mask must have N={num_points}, got {valid_mask.shape[0]}")

    if confidence is None:
        confidence = valid_mask.astype(np.float32)
    else:
        confidence = np.asarray(confidence).reshape(-1).astype(np.float32)
        if confidence.shape[0] != num_points:
            raise ValueError(f"confidence must have N={num_points}, got {confidence.shape[0]}")

    finite = np.isfinite(points).all(axis=1) & np.isfinite(track_flow).all(axis=1) & np.isfinite(confidence)
    return valid_mask & finite, confidence


def _nearest_integer_pixels(points, height, width):
    xy = np.round(points).astype(np.int32)
    x = np.clip(xy[:, 0], 0, width - 1)
    y = np.clip(xy[:, 1], 0, height - 1)
    return x, y


def rasterize_track_prior(
    points,
    track_flow,
    image_shape,
    valid_mask=None,
    confidence=None,
    normalize_distance=False,
    channels_first=False,
):
    """Rasterize sparse track flow into [dx, dy, confidence, visibility, distance]."""
    if len(image_shape) < 2:
        raise ValueError(f"Expected image_shape with at least two values, got {image_shape}")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: {(height, width)}")

    points = np.asarray(points, dtype=np.float32)
    track_flow = np.asarray(track_flow, dtype=np.float32)
    valid_mask, confidence = _validate_inputs(points, track_flow, valid_mask=valid_mask, confidence=confidence)

    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] <= width - 1)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= height - 1)
    )
    valid_mask = valid_mask & inside

    prior = np.zeros((height, width, 5), dtype=np.float32)
    best_confidence = np.full((height, width), -np.inf, dtype=np.float32)
    track_pixel_mask = np.zeros((height, width), dtype=bool)

    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices):
        x, y = _nearest_integer_pixels(points[valid_indices], height, width)
        for src_idx, px, py in zip(valid_indices, x, y):
            conf = float(confidence[src_idx])
            if conf < best_confidence[py, px]:
                continue
            best_confidence[py, px] = conf
            prior[py, px, 0:2] = track_flow[src_idx]
            prior[py, px, 2] = conf
            prior[py, px, 3] = 1.0
            track_pixel_mask[py, px] = True

    distance_input = np.ones((height, width), dtype=np.uint8)
    distance_input[track_pixel_mask] = 0
    if track_pixel_mask.any():
        distance = cv2.distanceTransform(distance_input, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(np.float32)
    else:
        distance = np.full((height, width), np.sqrt(height * height + width * width), dtype=np.float32)

    if normalize_distance:
        denom = float(np.sqrt(height * height + width * width))
        if denom > 0:
            distance = distance / denom

    prior[..., 4] = distance

    finite_distance = distance[np.isfinite(distance)]
    stats = RasterizeStats(
        height=height,
        width=width,
        num_input_points=int(points.shape[0]),
        num_valid_points=int(valid_mask.sum()),
        num_rasterized_pixels=int(track_pixel_mask.sum()),
        mean_confidence=float(prior[..., 2][track_pixel_mask].mean()) if track_pixel_mask.any() else float("nan"),
        max_distance=float(finite_distance.max()) if finite_distance.size else float("nan"),
        mean_distance=float(finite_distance.mean()) if finite_distance.size else float("nan"),
    )

    if channels_first:
        prior = np.transpose(prior, (2, 0, 1))
    return prior, stats


def save_raster_outputs(output_dir, prior, stats, prefix="track_prior"):
    np.save(f"{output_dir}/{prefix}.npy", prior)
    with open(f"{output_dir}/{prefix}_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2, allow_nan=True)


def save_raster_visualizations(output_dir, prior_hwc, prefix="track_prior"):
    if prior_hwc.ndim != 3 or prior_hwc.shape[2] != 5:
        raise ValueError(f"Expected HWC prior with 5 channels, got {prior_hwc.shape}")

    flow = prior_hwc[..., 0:2]
    visibility = prior_hwc[..., 3] > 0
    magnitude = np.linalg.norm(flow, axis=-1)
    mag_vis = np.zeros_like(magnitude, dtype=np.uint8)
    if visibility.any() and float(magnitude[visibility].max()) > 0:
        mag_vis[visibility] = np.clip(magnitude[visibility] / magnitude[visibility].max() * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{output_dir}/{prefix}_magnitude.png", cv2.applyColorMap(mag_vis, cv2.COLORMAP_TURBO))

    confidence = np.clip(prior_hwc[..., 2], 0.0, 1.0)
    cv2.imwrite(f"{output_dir}/{prefix}_confidence.png", (confidence * 255.0).astype(np.uint8))

    distance = prior_hwc[..., 4].astype(np.float32)
    if float(distance.max()) > float(distance.min()):
        distance_vis = (distance - distance.min()) / (distance.max() - distance.min())
    else:
        distance_vis = np.zeros_like(distance, dtype=np.float32)
    cv2.imwrite(f"{output_dir}/{prefix}_distance.png", (distance_vis * 255.0).astype(np.uint8))
