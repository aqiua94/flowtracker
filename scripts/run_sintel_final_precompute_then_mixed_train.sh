#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/flowseek

PY=/root/miniconda3/envs/flowseek/bin/python

FINAL_ROOT=precomputed/track_guided_sintel_final_full_attn
FINAL_MANIFEST=${FINAL_ROOT}/manifest.json
FINAL_PAIRS=${FINAL_ROOT}/manifest_pairs.json

CLEAN_MANIFEST=precomputed/track_guided_sintel_clean_full_attn/manifest.json
MIXED_ROOT=precomputed/track_guided_sintel_clean_final_full_attn
MIXED_MANIFEST=${MIXED_ROOT}/manifest.json

PIPELINE_LOG=logs/sintel_final_full_pipeline.log
PRECOMPUTE_LOG=logs/sintel_final_full_precompute_attn.log
TRAIN_LOG=logs/sintel_clean_final_mixed_tempered_batch4_update002.log
TRAIN_OUT=demo_fusion_outputs/sintel_clean_final_mixed_tempered_batch4_update002

mkdir -p logs "${FINAL_ROOT}" "${MIXED_ROOT}"

echo "==== $(date -Iseconds) prepare Sintel final manifest ====" >> "${PIPELINE_LOG}"
"${PY}" scripts/prepare_sintel_track_guided_manifest.py \
  --sintel_root data/MPI-Sintel \
  --split training \
  --dstype final \
  --output_root "${FINAL_ROOT}" \
  --manifest "${FINAL_MANIFEST}" >> "${PIPELINE_LOG}" 2>&1

echo "==== $(date -Iseconds) start Sintel final offline precompute ====" >> "${PIPELINE_LOG}"
{
  echo "==== $(date -Iseconds) offline precompute final resume ===="
  "${PY}" scripts/precompute_track_guided_from_pairs.py \
    --pairs "${FINAL_PAIRS}" \
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
  echo "==== $(date -Iseconds) offline precompute final finished ===="
} >> "${PRECOMPUTE_LOG}" 2>&1

FINAL_COUNT=$(find "${FINAL_ROOT}" -path '*/rasterizer/g_track.npy' | wc -l)
echo "==== $(date -Iseconds) final precomputed ${FINAL_COUNT}/1041 ====" >> "${PIPELINE_LOG}"
if [[ "${FINAL_COUNT}" -lt 1041 ]]; then
  echo "Expected 1041 final priors, got ${FINAL_COUNT}; abort mixed training." >> "${PIPELINE_LOG}"
  exit 1
fi

echo "==== $(date -Iseconds) build mixed clean+final manifest ====" >> "${PIPELINE_LOG}"
"${PY}" - "${CLEAN_MANIFEST}" "${FINAL_MANIFEST}" "${MIXED_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

clean_path, final_path, mixed_path = map(Path, sys.argv[1:])

def load_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"] if isinstance(data, dict) else data

samples = load_samples(clean_path) + load_samples(final_path)
mixed_path.parent.mkdir(parents=True, exist_ok=True)
with open(mixed_path, "w", encoding="utf-8") as f:
    json.dump({"samples": samples}, f, indent=2)
print(f"Wrote {len(samples)} mixed samples to {mixed_path}")
PY

echo "==== $(date -Iseconds) start mixed clean+final training ====" >> "${PIPELINE_LOG}"
"${PY}" train_track_guided.py \
  --manifest "${MIXED_MANIFEST}" \
  --val_manifest precomputed/track_guided_sintel_clean_multiscene_val20_attn/manifest.json \
  --output_dir "${TRAIN_OUT}" \
  --steps 1200 \
  --batch_size 4 \
  --val_batch_size 1 \
  --hidden_dim 32 \
  --lr 3e-4 \
  --min_lr 1e-5 \
  --lr_schedule cosine \
  --lr_warmup_steps 100 \
  --lambda_flow 1.0 \
  --lambda_track 0.3 \
  --lambda_smooth 0.01 \
  --lambda_no_harm 0.0 \
  --lambda_gate_safety 0.0 \
  --lambda_update_safety 0.02 \
  --min_safe_prior_coverage 0.00018 \
  --gate_distance_scale 48 \
  --log_interval 25 \
  --val_interval 50 \
  --device cuda > "${TRAIN_LOG}" 2>&1

echo "==== $(date -Iseconds) mixed training finished ====" >> "${PIPELINE_LOG}"
