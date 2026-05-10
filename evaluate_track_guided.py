import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from core.track_guidance.fusion_net import TrackGuidedFusionNet
from core.utils.flow_viz import flow_to_image
from train_track_guided import PrecomputedTrackGuidedDataset


def flow_epe(flow, gt_flow, valid):
    epe = np.linalg.norm(flow - gt_flow, axis=2)
    valid = valid.astype(bool)
    if valid.sum() == 0:
        return float("nan")
    return float(epe[valid].mean())


def flow_l1(flow, gt_flow, valid):
    l1 = np.abs(flow - gt_flow).mean(axis=2)
    valid = valid.astype(bool)
    if valid.sum() == 0:
        return float("nan")
    return float(l1[valid].mean())


def save_error_map(path, error, valid=None):
    error = error.astype(np.float32)
    if valid is not None:
        shown = error.copy()
        shown[~valid.astype(bool)] = 0.0
    else:
        shown = error
    vmax = float(np.percentile(shown[shown > 0], 95)) if np.any(shown > 0) else 1.0
    normalized = np.clip(shown / max(vmax, 1e-6), 0.0, 1.0)
    image = (normalized * 255.0).astype(np.uint8)
    image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), image)


def save_flow_vis(path, flow):
    image = flow_to_image(flow.astype(np.float32), convert_to_bgr=True)
    cv2.imwrite(str(path), image)


def load_model(checkpoint_path, device, hidden_dim=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    if hidden_dim is None:
        hidden_dim = int(ckpt_args.get("hidden_dim", 32))
    model = TrackGuidedFusionNet(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, hidden_dim


def tensor_to_flow_hwc(tensor):
    return tensor.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)


def prior_coverage(track_prior):
    confidence = track_prior[2].detach().cpu().numpy()
    return float((confidence > 0).mean())


def should_use_safe_fallback(record, args):
    if not args.safe_refinement:
        return False, ""
    if record["prior_coverage"] < args.min_prior_coverage:
        return True, "low_prior_coverage"
    if record["gate_mean"] < args.min_gate_mean:
        return True, "low_gate_mean"
    if record["delta_mean_abs"] > args.max_delta_mean_abs:
        return True, "large_delta"
    return False, ""


def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, hidden_dim = load_model(args.checkpoint, device=device, hidden_dim=args.hidden_dim)
    dataset = PrecomputedTrackGuidedDataset(args.manifest)

    records = []
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            initial_flow = item["initial_flow"][None].to(device)
            track_prior = item["track_prior"][None].to(device)
            gt_flow = item.get("gt_flow")
            valid = item.get("valid")
            if gt_flow is None:
                raise ValueError("Evaluation requires gt_flow in the manifest.")
            if valid is None:
                valid = torch.ones(gt_flow.shape[-2:], dtype=torch.float32)

            output = model(initial_flow, track_prior)
            refined_flow_t = output["refined_flow"][0]
            delta_flow_t = output["delta_flow"][0]
            gate_t = output["gate"][0, 0]

            initial_np = tensor_to_flow_hwc(item["initial_flow"])
            refined_np = tensor_to_flow_hwc(refined_flow_t)
            gt_np = tensor_to_flow_hwc(gt_flow)
            valid_np = valid.detach().cpu().numpy().astype(bool)
            delta_np = tensor_to_flow_hwc(delta_flow_t)
            gate_np = gate_t.detach().cpu().numpy().astype(np.float32)

            record = {
                "index": index,
                "prior_coverage": prior_coverage(item["track_prior"]),
                "delta_mean_abs": float(np.abs(delta_np).mean()),
                "gate_mean": float(gate_np.mean()),
                "gate_max": float(gate_np.max()),
            }
            fallback, fallback_reason = should_use_safe_fallback(record, args)
            if fallback:
                refined_np = initial_np.copy()
                delta_np = np.zeros_like(delta_np)

            initial_error = np.linalg.norm(initial_np - gt_np, axis=2)
            refined_error = np.linalg.norm(refined_np - gt_np, axis=2)
            improvement = initial_error - refined_error

            record.update(
                {
                    "initial_epe": flow_epe(initial_np, gt_np, valid_np),
                    "refined_epe": flow_epe(refined_np, gt_np, valid_np),
                    "initial_l1": flow_l1(initial_np, gt_np, valid_np),
                    "refined_l1": flow_l1(refined_np, gt_np, valid_np),
                    "used_fallback": fallback,
                    "fallback_reason": fallback_reason,
                }
            )
            record["epe_delta"] = record["refined_epe"] - record["initial_epe"]
            records.append(record)

            if index < args.num_visualizations:
                sample_dir = output_dir / f"sample_{index:04d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                np.save(sample_dir / "refined_flow.npy", refined_np)
                np.save(sample_dir / "delta_flow.npy", delta_np)
                np.save(sample_dir / "gate.npy", gate_np)
                save_flow_vis(sample_dir / "initial_flow_vis.png", initial_np)
                save_flow_vis(sample_dir / "refined_flow_vis.png", refined_np)
                save_flow_vis(sample_dir / "gt_flow_vis.png", gt_np)
                save_error_map(sample_dir / "initial_error.png", initial_error, valid_np)
                save_error_map(sample_dir / "refined_error.png", refined_error, valid_np)
                save_error_map(sample_dir / "improvement.png", np.maximum(improvement, 0.0), valid_np)

    initial_epes = np.asarray([r["initial_epe"] for r in records], dtype=np.float32)
    refined_epes = np.asarray([r["refined_epe"] for r in records], dtype=np.float32)
    summary = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "hidden_dim": hidden_dim,
        "safe_refinement": args.safe_refinement,
        "min_prior_coverage": args.min_prior_coverage,
        "min_gate_mean": args.min_gate_mean,
        "max_delta_mean_abs": args.max_delta_mean_abs,
        "num_samples": len(records),
        "mean_initial_epe": float(initial_epes.mean()),
        "mean_refined_epe": float(refined_epes.mean()),
        "mean_epe_delta": float((refined_epes - initial_epes).mean()),
        "num_improved": int((refined_epes < initial_epes).sum()),
        "num_worse": int((refined_epes > initial_epes).sum()),
        "num_fallback": int(sum(r["used_fallback"] for r in records)),
        "samples": records,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluated {summary['num_samples']} samples.")
    print(f"Mean initial EPE: {summary['mean_initial_epe']:.6f}")
    print(f"Mean refined EPE: {summary['mean_refined_epe']:.6f}")
    print(f"Mean EPE delta:   {summary['mean_epe_delta']:.6f}")
    print(f"Improved/worse:   {summary['num_improved']}/{summary['num_worse']}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="demo_fusion_outputs/eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_visualizations", type=int, default=4)
    parser.add_argument("--safe_refinement", action="store_true", help="Fall back to initial FlowSeek flow when unsupervised safety checks fail")
    parser.add_argument("--min_prior_coverage", type=float, default=0.0, help="Minimum nonzero track-prior confidence pixel fraction")
    parser.add_argument("--min_gate_mean", type=float, default=0.0, help="Minimum mean refinement gate required to keep the refined flow")
    parser.add_argument("--max_delta_mean_abs", type=float, default=float("inf"), help="Maximum mean absolute delta flow allowed before fallback")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
