import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


PATH_KEYS = (
    "initial_flow",
    "track_prior",
    "gt_flow",
    "valid",
)


def load_samples(path):
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"] if isinstance(data, dict) else data
    return path.parent, samples


def resolve_path(root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def sample_id(sample):
    flow_path = Path(sample["initial_flow"])
    sample_dir = flow_path.parents[1]
    return str(sample_dir)


def scene_name(sample):
    return sample_id(sample).split("/", 1)[0]


def rebase_sample(root, sample, dst_root, dataset):
    fixed = dict(sample)
    fixed["dataset"] = dataset
    fixed["sample_id"] = sample_id(sample)
    fixed["scene"] = scene_name(sample)
    for key in PATH_KEYS:
        value = sample.get(key)
        if value:
            fixed[key] = os.path.relpath(resolve_path(root, value), dst_root)
    return fixed


def select_evenly(samples, count):
    if count <= 0 or len(samples) <= count:
        return list(samples)
    if count == 1:
        return [samples[len(samples) // 2]]
    last = len(samples) - 1
    indices = sorted({round(i * last / (count - 1)) for i in range(count)})
    return [samples[i] for i in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_manifest", default="precomputed/track_guided_sintel_clean_full_attn/manifest.json")
    parser.add_argument("--final_manifest", default="precomputed/track_guided_sintel_final_full_attn/manifest.json")
    parser.add_argument("--train_manifest", default="precomputed/track_guided_sintel_clean_final_train_excl_val230_attn/manifest.json")
    parser.add_argument("--val_manifest", default="precomputed/track_guided_sintel_clean_final_val230_attn/manifest.json")
    parser.add_argument("--val_per_scene_per_dataset", type=int, default=5)
    args = parser.parse_args()

    train_path = Path(args.train_manifest).resolve()
    val_path = Path(args.val_manifest).resolve()
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    for dataset, manifest in (("clean", args.clean_manifest), ("final", args.final_manifest)):
        root, samples = load_samples(manifest)
        by_scene = defaultdict(list)
        for sample in samples:
            by_scene[scene_name(sample)].append(sample)
        for scene in sorted(by_scene):
            scene_samples = by_scene[scene]
            val_samples = select_evenly(scene_samples, args.val_per_scene_per_dataset)
            val_ids = {sample_id(sample) for sample in val_samples}
            for sample in scene_samples:
                split = "val" if sample_id(sample) in val_ids else "train"
                all_records.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "scene": scene,
                        "sample": sample,
                        "root": root,
                    }
                )

    train_samples = [
        rebase_sample(record["root"], record["sample"], train_path.parent, record["dataset"])
        for record in all_records
        if record["split"] == "train"
    ]
    val_samples = [
        rebase_sample(record["root"], record["sample"], val_path.parent, record["dataset"])
        for record in all_records
        if record["split"] == "val"
    ]

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump({"samples": train_samples}, f, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump({"samples": val_samples}, f, indent=2)

    print(f"Wrote {len(train_samples)} train samples to {train_path}")
    print(f"Wrote {len(val_samples)} val samples to {val_path}")


if __name__ == "__main__":
    main()
