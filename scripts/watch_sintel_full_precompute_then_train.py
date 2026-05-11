import datetime as dt
import subprocess
import time
from pathlib import Path


ROOT = Path("precomputed/track_guided_sintel_clean_full_attn")
TOTAL = 1041
WATCH_LOG = Path("logs/sintel_clean_full_pipeline_watch.log")
TRAIN_LOG = Path("logs/sintel_clean_full_train_attn_safe_loss_cosine.log")


def log(message):
    WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    with WATCH_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")
        f.flush()


def count_precomputed():
    return sum(1 for _ in ROOT.glob("**/rasterizer/g_track.npy"))


def main():
    log("offline watcher started")
    while True:
        count = count_precomputed()
        log(f"precomputed {count}/{TOTAL}")
        if count >= TOTAL:
            break
        time.sleep(120)

    log("precompute complete, starting training")
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/root/miniconda3/envs/flowseek/bin/python",
        "train_track_guided.py",
        "--manifest",
        "precomputed/track_guided_sintel_clean_full_attn/manifest.json",
        "--val_manifest",
        "precomputed/track_guided_sintel_clean_multiscene_val20_attn/manifest.json",
        "--output_dir",
        "demo_fusion_outputs/sintel_clean_full_attn_safe_loss_cosine",
        "--steps",
        "5000",
        "--batch_size",
        "1",
        "--val_batch_size",
        "1",
        "--hidden_dim",
        "32",
        "--lr",
        "3e-4",
        "--min_lr",
        "1e-5",
        "--lr_schedule",
        "cosine",
        "--lr_warmup_steps",
        "100",
        "--lambda_flow",
        "1.0",
        "--lambda_track",
        "0.2",
        "--lambda_smooth",
        "0.01",
        "--lambda_no_harm",
        "0.5",
        "--lambda_gate_safety",
        "0.05",
        "--lambda_update_safety",
        "0.1",
        "--min_safe_prior_coverage",
        "0.00018",
        "--gate_distance_scale",
        "48",
        "--log_interval",
        "25",
        "--val_interval",
        "250",
        "--device",
        "cuda",
    ]
    with TRAIN_LOG.open("w", encoding="utf-8") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    log(f"training finished returncode={result.returncode}")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
