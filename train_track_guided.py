import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.track_guidance.fusion_net import TrackGuidedFusionNet
from core.track_guidance.losses import fusion_loss


def masked_mean_epe(pred, target, valid=None, eps=1e-6):
    epe = torch.linalg.vector_norm(pred - target, dim=1)
    if valid is None:
        return epe.mean()
    valid = valid.float()
    return (epe * valid).sum() / valid.sum().clamp_min(eps)


class PrecomputedTrackGuidedDataset(Dataset):
    def __init__(self, manifest):
        self.root = Path(manifest).resolve().parent
        with open(manifest, "r", encoding="utf-8") as f:
            if manifest.endswith(".jsonl"):
                self.samples = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                self.samples = data["samples"] if isinstance(data, dict) else data
        if not self.samples:
            raise ValueError(f"No samples found in manifest: {manifest}")

    def _resolve(self, path):
        path = Path(path)
        if path.is_absolute():
            return path
        return self.root / path

    def _load_flow_chw(self, sample, key):
        array = np.load(self._resolve(sample[key])).astype(np.float32)
        if array.ndim != 3 or array.shape[2] != 2:
            raise ValueError(f"{key} must have shape (H, W, 2), got {array.shape}")
        return torch.from_numpy(array).permute(2, 0, 1)

    def _load_prior_chw(self, sample):
        prior = np.load(self._resolve(sample["track_prior"])).astype(np.float32)
        if prior.ndim == 3 and prior.shape[2] == 5:
            prior = np.transpose(prior, (2, 0, 1))
        elif prior.ndim == 3 and prior.shape[0] == 5:
            pass
        else:
            raise ValueError(f"track_prior must have shape (H, W, 5) or (5, H, W), got {prior.shape}")
        return torch.from_numpy(prior)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        item = {
            "initial_flow": self._load_flow_chw(sample, "initial_flow"),
            "track_prior": self._load_prior_chw(sample),
        }
        if "gt_flow" in sample and sample["gt_flow"]:
            item["gt_flow"] = self._load_flow_chw(sample, "gt_flow")
        if "valid" in sample and sample["valid"]:
            valid = np.load(self._resolve(sample["valid"])).astype(np.float32)
            item["valid"] = torch.from_numpy(valid)
        elif "gt_flow" in item:
            item["valid"] = torch.ones(item["gt_flow"].shape[-2:], dtype=torch.float32)
        return item


def make_smoke_manifest(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "stage6_smoke_manifest.json"
    sample = {
        "initial_flow": str(Path(args.initial_flow).resolve()),
        "track_prior": str(Path(args.track_prior).resolve()),
    }
    if args.gt_flow:
        sample["gt_flow"] = str(Path(args.gt_flow).resolve())
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump([sample], f, indent=2)
    return str(manifest_path)


def evaluate_model(model, dataset, device, batch_size=1):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    refined_epes = []
    initial_epes = []
    with torch.no_grad():
        for batch in loader:
            initial_flow = batch["initial_flow"].to(device)
            track_prior = batch["track_prior"].to(device)
            gt_flow = batch.get("gt_flow")
            valid = batch.get("valid")
            if gt_flow is None:
                continue
            gt_flow = gt_flow.to(device)
            if valid is not None:
                valid = valid.to(device)
            output = model(initial_flow, track_prior)
            refined_epes.append(masked_mean_epe(output["refined_flow"], gt_flow, valid=valid).detach())
            initial_epes.append(masked_mean_epe(initial_flow, gt_flow, valid=valid).detach())
    model.train()
    if not refined_epes:
        return None
    refined_epe = torch.stack(refined_epes).mean().item()
    initial_epe = torch.stack(initial_epes).mean().item()
    return {
        "val_initial_epe": initial_epe,
        "val_refined_epe": refined_epe,
        "val_epe_delta": refined_epe - initial_epe,
    }


def save_checkpoint(path, model, args, history, step, best_val=None):
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "history": history,
            "step": step,
            "best_val": best_val,
        },
        path,
    )


def learning_rate_for_step(args, step):
    if args.lr_schedule == "constant":
        return args.lr

    if args.lr_warmup_steps > 0 and step <= args.lr_warmup_steps:
        warmup_ratio = step / float(args.lr_warmup_steps)
        return args.lr * warmup_ratio

    decay_start = min(args.lr_warmup_steps, args.steps)
    decay_steps = max(1, args.steps - decay_start)
    progress = min(1.0, max(0.0, (step - decay_start) / float(decay_steps)))

    if args.lr_schedule == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr + (args.lr - args.min_lr) * cosine

    if args.lr_schedule == "step":
        drops = int((step - decay_start) // max(1, args.lr_step_size))
        return max(args.min_lr, args.lr * (args.lr_gamma ** drops))

    raise ValueError(f"Unknown lr_schedule: {args.lr_schedule}")


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def train(args):
    manifest = args.manifest or make_smoke_manifest(args)
    dataset = PrecomputedTrackGuidedDataset(manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dataset = PrecomputedTrackGuidedDataset(args.val_manifest) if args.val_manifest else None

    device = torch.device(args.device)
    model = TrackGuidedFusionNet(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    model.train()
    step = 0
    best_val = None
    while step < args.steps:
        for batch in loader:
            step += 1
            lr = learning_rate_for_step(args, step)
            set_optimizer_lr(optimizer, lr)
            initial_flow = batch["initial_flow"].to(device)
            track_prior = batch["track_prior"].to(device)
            gt_flow = batch.get("gt_flow")
            valid = batch.get("valid")
            if gt_flow is not None:
                gt_flow = gt_flow.to(device)
            if valid is not None:
                valid = valid.to(device)

            output = model(initial_flow, track_prior)
            losses = fusion_loss(
                output["refined_flow"],
                initial_flow,
                track_prior,
                gate=output["gate"],
                gt_flow=gt_flow,
                valid=valid,
                lambda_flow=args.lambda_flow,
                lambda_track=args.lambda_track,
                lambda_smooth=args.lambda_smooth,
                lambda_no_harm=args.lambda_no_harm,
                lambda_gate_safety=args.lambda_gate_safety,
                lambda_update_safety=args.lambda_update_safety,
                no_harm_margin=args.no_harm_margin,
                min_safe_prior_coverage=args.min_safe_prior_coverage,
                gate_distance_scale=args.gate_distance_scale,
            )

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            record = {name: float(value.detach().cpu()) for name, value in losses.items()}
            record["step"] = step
            record["lr"] = lr
            if val_dataset is not None and (step == 1 or step % args.val_interval == 0 or step == args.steps):
                val_record = evaluate_model(model, val_dataset, device=device, batch_size=args.val_batch_size)
                if val_record is not None:
                    record.update(val_record)
                    if best_val is None or val_record["val_refined_epe"] < best_val["val_refined_epe"]:
                        best_val = {"step": step, **val_record}
                        os.makedirs(args.output_dir, exist_ok=True)
                        save_checkpoint(
                            os.path.join(args.output_dir, "fusion_net_best_val.pth"),
                            model,
                            args,
                            history,
                            step,
                            best_val=best_val,
                        )
            history.append(record)
            if step == 1 or step % args.log_interval == 0 or step == args.steps:
                message = (
                    f"step {step:04d} "
                    f"lr={record['lr']:.3e} "
                    f"total={record['total']:.6f} "
                    f"flow_l1={record['flow_l1']:.6f} "
                    f"track_l1={record['track_l1']:.6f} "
                    f"smooth={record['smoothness']:.6f} "
                    f"no_harm={record['no_harm']:.6f} "
                    f"gate_safe={record['gate_safety']:.6f} "
                    f"update_safe={record['update_safety']:.6f} "
                    f"coverage={record['prior_coverage']:.7f}"
                )
                if "val_refined_epe" in record:
                    message += (
                        f" val_refined={record['val_refined_epe']:.6f}"
                        f" val_delta={record['val_epe_delta']:.6f}"
                    )
                print(message)
            if step >= args.steps:
                break

    os.makedirs(args.output_dir, exist_ok=True)
    save_checkpoint(os.path.join(args.output_dir, "fusion_net_smoke.pth"), model, args, history, step, best_val=best_val)
    with open(os.path.join(args.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, allow_nan=True)

    print(f"Saved checkpoint to {os.path.join(args.output_dir, 'fusion_net_smoke.pth')}")
    if best_val is not None:
        print(
            f"Best val checkpoint: {os.path.join(args.output_dir, 'fusion_net_best_val.pth')} "
            f"step={best_val['step']} val_refined={best_val['val_refined_epe']:.6f} "
            f"val_delta={best_val['val_epe_delta']:.6f}"
        )
    print(f"Initial total loss: {history[0]['total']:.6f}")
    print(f"Final total loss: {history[-1]['total']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None, help="JSON/JSONL manifest with initial_flow, track_prior, optional gt_flow")
    parser.add_argument("--val_manifest", default=None, help="Optional validation manifest used to save fusion_net_best_val.pth")
    parser.add_argument("--initial_flow", default=None, help="Convenience input used when --manifest is omitted")
    parser.add_argument("--track_prior", default=None, help="Convenience input used when --manifest is omitted")
    parser.add_argument("--gt_flow", default=None, help="Optional GT flow for supervised smoke training")
    parser.add_argument("--output_dir", default="demo_fusion_outputs/stage6_smoke")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--lr_schedule", choices=["constant", "cosine", "step"], default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_step_size", type=int, default=200)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_flow", type=float, default=1.0)
    parser.add_argument("--lambda_track", type=float, default=0.2)
    parser.add_argument("--lambda_smooth", type=float, default=0.01)
    parser.add_argument("--lambda_no_harm", type=float, default=0.0, help="Penalize pixels where refinement is worse than initial flow")
    parser.add_argument("--lambda_gate_safety", type=float, default=0.0, help="Suppress gate in low-reliability prior regions")
    parser.add_argument("--lambda_update_safety", type=float, default=0.0, help="Suppress flow updates in low-reliability prior regions")
    parser.add_argument("--no_harm_margin", type=float, default=0.0, help="Allowed EPE increase before no-harm penalty activates")
    parser.add_argument("--min_safe_prior_coverage", type=float, default=0.000225, help="Coverage below this value is treated as globally unsafe")
    parser.add_argument("--gate_distance_scale", type=float, default=48.0, help="Distance-transform scale for local prior reliability")
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--val_interval", type=int, default=50)
    args = parser.parse_args()

    if args.manifest is None and (args.initial_flow is None or args.track_prior is None):
        raise ValueError("Provide --manifest, or both --initial_flow and --track_prior.")

    train(args)


if __name__ == "__main__":
    main()
