#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/flowseek

LOG=${LOG:-logs/sintel_clean_full_precompute_attn.log}
mkdir -p "$(dirname "$LOG")"

{
  echo "==== $(date -Iseconds) offline precompute resume ===="
  /root/miniconda3/envs/flowseek/bin/python scripts/precompute_track_guided_from_pairs.py \
    --pairs precomputed/track_guided_sintel_clean_full_attn/manifest_pairs.json \
    --cfg config/eval/flowseek-T.json \
    --model weights/flowseek_T_CT.pth \
    --device cuda \
    --flowseek_max_size 384 \
    --cotracker_max_size 0 \
    --num_points 512 \
    --cell_size 32 \
    --min_points_per_cell 1 \
    --max_points_per_cell 8 \
    --min_distance 4.0 \
    --flow_gradient_weight 0.7 \
    --flow_magnitude_weight 0.3 \
    --image_edge_weight 0.0 \
    --min_confidence 0.0 \
    --use_trajectory_attention \
    --attention_spatial_sigma 96 \
    --attention_motion_sigma 8 \
    --attention_endpoint_error_scale 3 \
    --attention_self_weight 1
  echo "==== $(date -Iseconds) offline precompute finished ===="
} >> "$LOG" 2>&1
