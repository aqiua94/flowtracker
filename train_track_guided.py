import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.track_guidance.fusion_net import TrackGuidedFusionNet
from core.track_guidance.losses import fusion_loss


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


def train(args):
    manifest = args.manifest or make_smoke_manifest(args)
    dataset = PrecomputedTrackGuidedDataset(manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device(args.device)
    model = TrackGuidedFusionNet(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    model.train()
    step = 0
    while step < args.steps:
        for batch in loader:
            step += 1
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
                gt_flow=gt_flow,
                valid=valid,
                lambda_flow=args.lambda_flow,
                lambda_track=args.lambda_track,
                lambda_smooth=args.lambda_smooth,
            )

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            record = {name: float(value.detach().cpu()) for name, value in losses.items()}
            record["step"] = step
            history.append(record)
            if step == 1 or step % args.log_interval == 0 or step == args.steps:
                print(
                    f"step {step:04d} "
                    f"total={record['total']:.6f} "
                    f"flow_l1={record['flow_l1']:.6f} "
                    f"track_l1={record['track_l1']:.6f} "
                    f"smooth={record['smoothness']:.6f}"
                )
            if step >= args.steps:
                break

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "history": history,
        },
        os.path.join(args.output_dir, "fusion_net_smoke.pth"),
    )
    with open(os.path.join(args.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, allow_nan=True)

    print(f"Saved checkpoint to {os.path.join(args.output_dir, 'fusion_net_smoke.pth')}")
    print(f"Initial total loss: {history[0]['total']:.6f}")
    print(f"Final total loss: {history[-1]['total']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None, help="JSON/JSONL manifest with initial_flow, track_prior, optional gt_flow")
    parser.add_argument("--initial_flow", default=None, help="Convenience input used when --manifest is omitted")
    parser.add_argument("--track_prior", default=None, help="Convenience input used when --manifest is omitted")
    parser.add_argument("--gt_flow", default=None, help="Optional GT flow for supervised smoke training")
    parser.add_argument("--output_dir", default="demo_fusion_outputs/stage6_smoke")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_flow", type=float, default=1.0)
    parser.add_argument("--lambda_track", type=float, default=0.2)
    parser.add_argument("--lambda_smooth", type=float, default=0.01)
    parser.add_argument("--log_interval", type=int, default=5)
    args = parser.parse_args()

    if args.manifest is None and (args.initial_flow is None or args.track_prior is None):
        raise ValueError("Provide --manifest, or both --initial_flow and --track_prior.")

    train(args)


if __name__ == "__main__":
    main()
