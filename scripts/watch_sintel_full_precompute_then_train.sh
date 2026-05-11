#!/usr/bin/env bash
set -euo pipefail

TOTAL=${TOTAL:-1041}
ROOT=${ROOT:-precomputed/track_guided_sintel_clean_full_attn}
WATCH_LOG=${WATCH_LOG:-logs/sintel_clean_full_pipeline_watch.log}
TRAIN_LOG=${TRAIN_LOG:-logs/sintel_clean_full_train_attn_safe_loss_cosine.log}

mkdir -p "$(dirname "$WATCH_LOG")" "$(dirname "$TRAIN_LOG")"

while true; do
  count=$(find "$ROOT" -path "*/rasterizer/g_track.npy" | wc -l)
  echo "$(date -Iseconds) precomputed ${count}/${TOTAL}" >> "$WATCH_LOG"
  if [ "$count" -ge "$TOTAL" ]; then
    break
  fi
  sleep 120
done

echo "$(date -Iseconds) precompute complete, starting training" >> "$WATCH_LOG"
/root/miniconda3/envs/flowseek/bin/python train_track_guided.py \
  --manifest precomputed/track_guided_sintel_clean_full_attn/manifest.json \
  --val_manifest precomputed/track_guided_sintel_clean_multiscene_val20_attn/manifest.json \
  --output_dir demo_fusion_outputs/sintel_clean_full_attn_safe_loss_cosine \
  --steps 5000 \
  --batch_size 1 \
  --val_batch_size 1 \
  --hidden_dim 32 \
  --lr 3e-4 \
  --min_lr 1e-5 \
  --lr_schedule cosine \
  --lr_warmup_steps 100 \
  --lambda_flow 1.0 \
  --lambda_track 0.2 \
  --lambda_smooth 0.01 \
  --lambda_no_harm 0.5 \
  --lambda_gate_safety 0.05 \
  --lambda_update_safety 0.1 \
  --min_safe_prior_coverage 0.00018 \
  --gate_distance_scale 48 \
  --log_interval 25 \
  --val_interval 250 \
  --device cuda > "$TRAIN_LOG" 2>&1
echo "$(date -Iseconds) training finished" >> "$WATCH_LOG"
