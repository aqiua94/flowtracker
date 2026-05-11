import torch
import torch.nn.functional as F


def masked_l1(pred, target, valid=None, eps=1e-6):
    loss = (pred - target).abs()
    if valid is None:
        return loss.mean()

    if valid.ndim == pred.ndim - 1:
        valid = valid[:, None]
    valid = valid.float()
    while valid.ndim < loss.ndim:
        valid = valid.unsqueeze(1)
    valid = valid.expand_as(loss)
    denom = valid.sum().clamp_min(eps)
    return (loss * valid).sum() / denom


def endpoint_error(pred, target, valid=None, eps=1e-6):
    epe = torch.linalg.vector_norm(pred - target, dim=1)
    if valid is None:
        return epe.mean()
    valid = valid.float()
    denom = valid.sum().clamp_min(eps)
    return (epe * valid).sum() / denom


def track_prior_loss(refined_flow, track_prior, eps=1e-6):
    track_flow = track_prior[:, 0:2]
    confidence = track_prior[:, 2:3].clamp(0.0, 1.0)
    visibility = track_prior[:, 3:4].clamp(0.0, 1.0)
    weight = confidence * visibility
    loss = (refined_flow - track_flow).abs() * weight
    denom = weight.sum().clamp_min(eps) * refined_flow.shape[1]
    return loss.sum() / denom


def smoothness_loss(flow):
    dx = (flow[..., :, 1:] - flow[..., :, :-1]).abs().mean()
    dy = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
    return dx + dy


def _as_valid_mask(valid, pred):
    if valid is None:
        return None
    valid = valid.float()
    while valid.ndim < pred.ndim - 1:
        valid = valid.unsqueeze(0)
    return valid


def _masked_mean(values, valid=None, eps=1e-6):
    if valid is None:
        return values.mean()
    valid = valid.float()
    while valid.ndim < values.ndim:
        valid = valid.unsqueeze(1)
    valid = valid.expand_as(values)
    return (values * valid).sum() / valid.sum().clamp_min(eps)


def prior_coverage(track_prior):
    confidence = track_prior[:, 2:3].clamp(0.0, 1.0)
    visibility = track_prior[:, 3:4].clamp(0.0, 1.0)
    support = (confidence * visibility > 0).float()
    return support.flatten(1).mean(dim=1)


def unsafe_prior_weight(track_prior, min_safe_prior_coverage=0.000225, gate_distance_scale=48.0):
    confidence = track_prior[:, 2:3].clamp(0.0, 1.0)
    visibility = track_prior[:, 3:4].clamp(0.0, 1.0)
    reliability = confidence * visibility

    if track_prior.shape[1] >= 5 and gate_distance_scale > 0:
        distance = track_prior[:, 4:5].clamp_min(0.0)
        local_support = torch.exp(-distance / float(gate_distance_scale))
    else:
        local_support = torch.ones_like(reliability)

    safe_local = (reliability * local_support).clamp(0.0, 1.0)
    unsafe = 1.0 - safe_local

    if min_safe_prior_coverage > 0:
        coverage = prior_coverage(track_prior).view(-1, 1, 1, 1)
        low_coverage = (1.0 - coverage / float(min_safe_prior_coverage)).clamp(0.0, 1.0)
        unsafe = torch.maximum(unsafe, low_coverage.expand_as(unsafe))

    return unsafe.clamp(0.0, 1.0)


def no_harm_loss(refined_flow, initial_flow, gt_flow, valid=None, margin=0.0):
    refined_epe = torch.linalg.vector_norm(refined_flow - gt_flow, dim=1, keepdim=True)
    initial_epe = torch.linalg.vector_norm(initial_flow - gt_flow, dim=1, keepdim=True).detach()
    penalty = F.relu(refined_epe - initial_epe + float(margin))
    return _masked_mean(penalty, valid=valid)


def gate_safety_loss(gate, unsafe_weight, valid=None):
    return _masked_mean(gate * unsafe_weight, valid=valid)


def update_safety_loss(refined_flow, initial_flow, unsafe_weight, valid=None):
    update = (refined_flow - initial_flow).abs().mean(dim=1, keepdim=True)
    return _masked_mean(update * unsafe_weight, valid=valid)


def fusion_loss(
    refined_flow,
    initial_flow,
    track_prior,
    gate=None,
    gt_flow=None,
    valid=None,
    lambda_flow=1.0,
    lambda_track=0.2,
    lambda_smooth=0.01,
    lambda_no_harm=0.0,
    lambda_gate_safety=0.0,
    lambda_update_safety=0.0,
    no_harm_margin=0.0,
    min_safe_prior_coverage=0.000225,
    gate_distance_scale=48.0,
):
    losses = {}
    if gt_flow is not None:
        losses["flow_l1"] = masked_l1(refined_flow, gt_flow, valid=valid)
        losses["epe"] = endpoint_error(refined_flow, gt_flow, valid=valid).detach()
        losses["initial_epe"] = endpoint_error(initial_flow, gt_flow, valid=valid).detach()
        losses["no_harm"] = no_harm_loss(
            refined_flow,
            initial_flow,
            gt_flow,
            valid=valid,
            margin=no_harm_margin,
        )
    else:
        losses["flow_l1"] = refined_flow.new_tensor(0.0)
        losses["epe"] = refined_flow.new_tensor(float("nan"))
        losses["initial_epe"] = refined_flow.new_tensor(float("nan"))
        losses["no_harm"] = refined_flow.new_tensor(0.0)

    losses["track_l1"] = track_prior_loss(refined_flow, track_prior)
    losses["smoothness"] = smoothness_loss(refined_flow)
    unsafe_weight = unsafe_prior_weight(
        track_prior,
        min_safe_prior_coverage=min_safe_prior_coverage,
        gate_distance_scale=gate_distance_scale,
    )
    if gate is not None:
        losses["gate_safety"] = gate_safety_loss(gate, unsafe_weight, valid=valid)
    else:
        losses["gate_safety"] = refined_flow.new_tensor(0.0)
    losses["update_safety"] = update_safety_loss(refined_flow, initial_flow, unsafe_weight, valid=valid)
    losses["prior_coverage"] = prior_coverage(track_prior).mean().detach()
    losses["total"] = (
        lambda_flow * losses["flow_l1"]
        + lambda_track * losses["track_l1"]
        + lambda_smooth * losses["smoothness"]
        + lambda_no_harm * losses["no_harm"]
        + lambda_gate_safety * losses["gate_safety"]
        + lambda_update_safety * losses["update_safety"]
    )
    return losses
