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


def fusion_loss(
    refined_flow,
    initial_flow,
    track_prior,
    gt_flow=None,
    valid=None,
    lambda_flow=1.0,
    lambda_track=0.2,
    lambda_smooth=0.01,
):
    losses = {}
    if gt_flow is not None:
        losses["flow_l1"] = masked_l1(refined_flow, gt_flow, valid=valid)
        losses["epe"] = endpoint_error(refined_flow, gt_flow, valid=valid).detach()
        losses["initial_epe"] = endpoint_error(initial_flow, gt_flow, valid=valid).detach()
    else:
        losses["flow_l1"] = refined_flow.new_tensor(0.0)
        losses["epe"] = refined_flow.new_tensor(float("nan"))
        losses["initial_epe"] = refined_flow.new_tensor(float("nan"))

    losses["track_l1"] = track_prior_loss(refined_flow, track_prior)
    losses["smoothness"] = smoothness_loss(refined_flow)
    losses["total"] = (
        lambda_flow * losses["flow_l1"]
        + lambda_track * losses["track_l1"]
        + lambda_smooth * losses["smoothness"]
    )
    return losses
