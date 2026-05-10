from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class TrajectoryAttentionStats:
    num_points: int
    num_valid_input: int
    num_valid_output: int
    mean_input_confidence: float
    mean_output_confidence: float
    mean_reliability: float
    mean_flow_delta: float
    max_flow_delta: float

    def to_dict(self):
        return asdict(self)


def _as_bool_mask(mask, num_points):
    if mask is None:
        return np.ones(num_points, dtype=bool)
    mask = np.asarray(mask).reshape(-1).astype(bool)
    if mask.shape[0] != num_points:
        raise ValueError(f"valid_mask must have N={num_points}, got {mask.shape[0]}")
    return mask


def _as_confidence(confidence, valid_mask):
    if confidence is None:
        return valid_mask.astype(np.float32)
    confidence = np.asarray(confidence).reshape(-1).astype(np.float32)
    if confidence.shape[0] != valid_mask.shape[0]:
        raise ValueError(f"confidence must have N={valid_mask.shape[0]}, got {confidence.shape[0]}")
    return np.clip(confidence, 0.0, 1.0)


def _safe_softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    denom = weights.sum(axis=1, keepdims=True)
    return weights / np.maximum(denom, 1e-12)


def compute_track_reliability(valid_mask, confidence, endpoint_error=None, endpoint_error_scale=3.0):
    reliability = valid_mask.astype(np.float32) * np.clip(confidence, 0.0, 1.0)
    if endpoint_error is not None:
        endpoint_error = np.asarray(endpoint_error).reshape(-1).astype(np.float32)
        if endpoint_error.shape[0] != reliability.shape[0]:
            raise ValueError(f"endpoint_error must have N={reliability.shape[0]}, got {endpoint_error.shape[0]}")
        finite = np.isfinite(endpoint_error)
        epe_weight = np.zeros_like(reliability, dtype=np.float32)
        epe_weight[finite] = np.exp(-endpoint_error[finite] / max(float(endpoint_error_scale), 1e-6))
        reliability *= epe_weight
    return np.clip(reliability, 0.0, 1.0)


def trajectory_attention_enhance(
    points,
    track_flow,
    valid_mask=None,
    confidence=None,
    endpoint_error=None,
    spatial_sigma=96.0,
    motion_sigma=8.0,
    endpoint_error_scale=3.0,
    self_weight=1.0,
    promote_invalid=False,
    min_output_confidence=0.05,
):
    """Enhance sparse track flow with reliability-aware cross-track attention.

    The module is intentionally conservative: reliable visible tracks are kept
    close to their original displacement, while low-reliability tracks borrow a
    weighted consensus from nearby reliable tracks with similar motion.
    """
    points = np.asarray(points, dtype=np.float32)
    track_flow = np.asarray(track_flow, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {points.shape}")
    if track_flow.ndim != 2 or track_flow.shape[1] != 2:
        raise ValueError(f"Expected track_flow with shape (N, 2), got {track_flow.shape}")
    if points.shape[0] != track_flow.shape[0]:
        raise ValueError(f"points and track_flow must have same N, got {points.shape[0]} and {track_flow.shape[0]}")

    num_points = points.shape[0]
    valid_mask = _as_bool_mask(valid_mask, num_points)
    confidence = _as_confidence(confidence, valid_mask)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(track_flow).all(axis=1) & np.isfinite(confidence)
    valid_mask = valid_mask & finite

    reliability = compute_track_reliability(
        valid_mask,
        confidence,
        endpoint_error=endpoint_error,
        endpoint_error_scale=endpoint_error_scale,
    )
    source_weight = reliability.copy()
    if source_weight.max() <= 0:
        attention = np.eye(num_points, dtype=np.float32)
        stats = TrajectoryAttentionStats(
            num_points=num_points,
            num_valid_input=int(valid_mask.sum()),
            num_valid_output=int(valid_mask.sum()),
            mean_input_confidence=float(confidence[valid_mask].mean()) if valid_mask.any() else float("nan"),
            mean_output_confidence=float(confidence[valid_mask].mean()) if valid_mask.any() else float("nan"),
            mean_reliability=float(reliability.mean()),
            mean_flow_delta=0.0,
            max_flow_delta=0.0,
        )
        return track_flow.copy(), confidence.copy(), valid_mask.copy(), attention, stats

    diff_xy = points[:, None, :] - points[None, :, :]
    spatial_dist2 = np.sum(diff_xy * diff_xy, axis=-1)
    diff_flow = track_flow[:, None, :] - track_flow[None, :, :]
    motion_dist2 = np.sum(diff_flow * diff_flow, axis=-1)

    logits = (
        -spatial_dist2 / (2.0 * max(float(spatial_sigma), 1e-6) ** 2)
        - motion_dist2 / (2.0 * max(float(motion_sigma), 1e-6) ** 2)
        + np.log(np.maximum(source_weight[None, :], 1e-12))
    )
    logits[:, source_weight <= 0] = -1.0e9
    if self_weight > 0:
        logits[np.arange(num_points), np.arange(num_points)] += float(self_weight)

    attention = _safe_softmax(logits).astype(np.float32)
    consensus_flow = attention @ track_flow
    consensus_confidence = attention @ reliability

    keep_ratio = reliability[:, None]
    enhanced_flow = keep_ratio * track_flow + (1.0 - keep_ratio) * consensus_flow
    enhanced_confidence = np.maximum(confidence * reliability, consensus_confidence).astype(np.float32)
    enhanced_valid = valid_mask.copy()
    if promote_invalid:
        enhanced_valid = enhanced_valid | (enhanced_confidence >= float(min_output_confidence))

    flow_delta = np.linalg.norm(enhanced_flow - track_flow, axis=1)
    output_valid = enhanced_valid.astype(bool)
    stats = TrajectoryAttentionStats(
        num_points=num_points,
        num_valid_input=int(valid_mask.sum()),
        num_valid_output=int(output_valid.sum()),
        mean_input_confidence=float(confidence[valid_mask].mean()) if valid_mask.any() else float("nan"),
        mean_output_confidence=float(enhanced_confidence[output_valid].mean()) if output_valid.any() else float("nan"),
        mean_reliability=float(reliability[valid_mask].mean()) if valid_mask.any() else float("nan"),
        mean_flow_delta=float(flow_delta[valid_mask].mean()) if valid_mask.any() else 0.0,
        max_flow_delta=float(flow_delta[valid_mask].max()) if valid_mask.any() else 0.0,
    )
    return enhanced_flow.astype(np.float32), enhanced_confidence, enhanced_valid, attention, stats
