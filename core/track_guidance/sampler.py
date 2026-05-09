import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SamplingConfig:
    num_points: int = 256
    cell_size: int = 32
    min_points_per_cell: int = 1
    max_points_per_cell: int = 8
    min_distance: float = 4.0
    flow_gradient_weight: float = 0.7
    flow_magnitude_weight: float = 0.3
    image_edge_weight: float = 0.0
    seed: int = 0


def robust_normalize(values, eps=1e-6):
    values = values.astype(np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)

    valid = values[finite]
    lo = np.percentile(valid, 1.0)
    hi = np.percentile(valid, 99.0)
    if hi - lo < eps:
        return np.zeros_like(values, dtype=np.float32)

    normalized = (values - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def compute_flow_gradient_weight(flow):
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected flow with shape (H, W, 2), got {flow.shape}")

    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    grad = np.sqrt(du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2)
    return robust_normalize(grad)


def compute_flow_magnitude_weight(flow):
    mag = np.linalg.norm(flow.astype(np.float32), axis=-1)
    return robust_normalize(mag)


def compute_image_edge_weight(image):
    if image is None:
        return None

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx**2 + gy**2)
    return robust_normalize(edge)


def compute_sampling_weight(flow, image=None, config=None):
    config = config or SamplingConfig()
    terms = []

    grad = compute_flow_gradient_weight(flow)
    terms.append((config.flow_gradient_weight, grad))

    mag = compute_flow_magnitude_weight(flow)
    terms.append((config.flow_magnitude_weight, mag))

    if config.image_edge_weight > 0:
        edge = compute_image_edge_weight(image)
        if edge is not None:
            terms.append((config.image_edge_weight, edge))

    total_weight = sum(max(0.0, weight) for weight, _ in terms)
    if total_weight <= 0:
        return np.ones(flow.shape[:2], dtype=np.float32)

    sampling = np.zeros(flow.shape[:2], dtype=np.float32)
    for weight, term in terms:
        if weight > 0:
            sampling += (weight / total_weight) * term

    if float(sampling.max()) <= 1e-6:
        sampling = np.ones_like(sampling, dtype=np.float32)
    return sampling.astype(np.float32)


def _cell_slices(height, width, cell_size):
    rows = int(math.ceil(height / cell_size))
    cols = int(math.ceil(width / cell_size))
    for row in range(rows):
        y0 = row * cell_size
        y1 = min(height, (row + 1) * cell_size)
        for col in range(cols):
            x0 = col * cell_size
            x1 = min(width, (col + 1) * cell_size)
            yield row, col, y0, y1, x0, x1


def _allocate_points(weight_map, config):
    height, width = weight_map.shape
    cells = list(_cell_slices(height, width, config.cell_size))
    cell_scores = []
    for _, _, y0, y1, x0, x1 in cells:
        cell_scores.append(float(weight_map[y0:y1, x0:x1].sum()))
    cell_scores = np.asarray(cell_scores, dtype=np.float64)

    num_cells = len(cells)
    base_total = min(config.num_points, config.min_points_per_cell * num_cells)
    counts = np.zeros(num_cells, dtype=np.int32)

    if config.min_points_per_cell > 0 and base_total > 0:
        order = np.arange(num_cells)
        counts[order[:base_total]] = 1
        if config.min_points_per_cell > 1:
            counts[:] = min(config.min_points_per_cell, config.num_points // max(1, num_cells))

    remaining = max(0, config.num_points - int(counts.sum()))
    if remaining == 0:
        return cells, counts

    if float(cell_scores.sum()) <= 1e-8:
        probs = np.ones(num_cells, dtype=np.float64) / num_cells
    else:
        probs = cell_scores / cell_scores.sum()

    raw_extra = probs * remaining
    extra = np.floor(raw_extra).astype(np.int32)
    counts += extra

    overflow = np.maximum(0, counts - config.max_points_per_cell)
    if overflow.any():
        counts -= overflow

    leftover = config.num_points - int(counts.sum())
    if leftover > 0:
        priority = np.argsort(-(raw_extra - np.floor(raw_extra)))
        for idx in priority:
            if leftover <= 0:
                break
            if counts[idx] >= config.max_points_per_cell:
                continue
            counts[idx] += 1
            leftover -= 1

    return cells, counts


def _is_far_enough(point, selected, min_distance):
    if not selected:
        return True
    selected_xy = np.asarray(selected, dtype=np.float32)
    dist = np.linalg.norm(selected_xy - np.asarray(point, dtype=np.float32), axis=1)
    return bool(np.all(dist >= min_distance))


def _select_from_cell(weight_map, y0, y1, x0, x1, count, selected, config, rng):
    if count <= 0:
        return []

    local = weight_map[y0:y1, x0:x1]
    ys, xs = np.indices(local.shape)
    xs = xs.reshape(-1) + x0
    ys = ys.reshape(-1) + y0
    scores = local.reshape(-1).astype(np.float64)

    jitter = rng.random(scores.shape) * 1e-6
    order = np.argsort(-(scores + jitter))

    picked = []
    for idx in order:
        point = (float(xs[idx]), float(ys[idx]))
        if _is_far_enough(point, selected + picked, config.min_distance):
            picked.append(point)
            if len(picked) >= count:
                break

    return picked


def adaptive_sample_points(flow, image=None, config=None):
    config = config or SamplingConfig()
    if config.num_points <= 0:
        raise ValueError("num_points must be positive.")
    if config.cell_size <= 0:
        raise ValueError("cell_size must be positive.")

    weight_map = compute_sampling_weight(flow, image=image, config=config)
    cells, counts = _allocate_points(weight_map, config)
    rng = np.random.default_rng(config.seed)

    selected = []
    point_weights = []
    for cell_index, (_, _, y0, y1, x0, x1) in enumerate(cells):
        picked = _select_from_cell(weight_map, y0, y1, x0, x1, int(counts[cell_index]), selected, config, rng)
        for x, y in picked:
            selected.append((x, y))
            point_weights.append(float(weight_map[int(round(y)), int(round(x))]))

    if len(selected) < config.num_points:
        height, width = weight_map.shape
        ys, xs = np.indices((height, width))
        scores = weight_map.reshape(-1).astype(np.float64)
        order = np.argsort(-(scores + rng.random(scores.shape) * 1e-6))
        for idx in order:
            if len(selected) >= config.num_points:
                break
            y = int(ys.reshape(-1)[idx])
            x = int(xs.reshape(-1)[idx])
            point = (float(x), float(y))
            if _is_far_enough(point, selected, config.min_distance):
                selected.append(point)
                point_weights.append(float(weight_map[y, x]))

    points = np.asarray(selected[: config.num_points], dtype=np.float32)
    weights = np.asarray(point_weights[: config.num_points], dtype=np.float32)
    return points, weights, weight_map


def points_to_cotracker_queries(points, query_frame=0):
    frame_ids = np.full((points.shape[0], 1), float(query_frame), dtype=np.float32)
    return np.concatenate([frame_ids, points.astype(np.float32)], axis=1)
