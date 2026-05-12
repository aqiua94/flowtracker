# Track-Guided Fusion Training Strategy

Date: 2026-05-12

This document records the currently confirmed training strategy for the
track-guided FusionNet experiment after the clean+final mixed precompute run.
The recommended reusable path is the leakage-free val230 split with the relaxed
safety loss weights below.

## Data

The original fixed mixed manifest is:

```text
precomputed/track_guided_sintel_clean_final_full_attn/manifest.json
```

It contains 2082 samples:

- 1041 Sintel clean full-attention precomputed samples
- 1041 Sintel final full-attention precomputed samples

The manifest paths must point back to the original precompute roots, for
example:

```text
../track_guided_sintel_clean_full_attn/alley_1/alley_1_frame_0001/flowseek/flow.npy
../track_guided_sintel_final_full_attn/alley_1/alley_1_frame_0001/flowseek/flow.npy
```

The old small validation set was:

```text
precomputed/track_guided_sintel_clean_multiscene_val20_attn/manifest.json
```

It contains only 20 clean samples from `ambush_2` and `temple_2`. Use it only
for quick smoke checks.

The expanded validation split is:

```text
precomputed/track_guided_sintel_clean_final_val230_attn/manifest.json
```

It contains 230 samples: 23 scenes x 2 datasets x 5 evenly spaced samples.
For leakage-free Sintel-only training with this validation set, use this train
manifest:

```text
precomputed/track_guided_sintel_clean_final_train_excl_val230_attn/manifest.json
```

This train split contains 1852 samples after excluding the val230 samples. This
is the current recommended training manifest.

## Confirmed Hyperparameters

Recommended command shape:

```bash
/root/miniconda3/envs/flowseek/bin/python train_track_guided.py \
  --manifest precomputed/track_guided_sintel_clean_final_train_excl_val230_attn/manifest.json \
  --val_manifest precomputed/track_guided_sintel_clean_final_val230_attn/manifest.json \
  --output_dir demo_fusion_outputs/<run_name> \
  --steps 100 \
  --batch_size 4 \
  --val_batch_size 1 \
  --hidden_dim 32 \
  --lr 3e-4 \
  --min_lr 1e-5 \
  --lr_schedule cosine \
  --lr_warmup_steps 100 \
  --lambda_flow 1.0 \
  --lambda_track 0.2 \
  --lambda_smooth 0.01 \
  --lambda_no_harm 0.2 \
  --lambda_gate_safety 0.02 \
  --lambda_update_safety 0.05 \
  --min_safe_prior_coverage 0.00018 \
  --gate_distance_scale 48 \
  --log_interval 25 \
  --val_interval 25 \
  --device cuda
```

Use `fusion_net_best_val.pth` as the model artifact. Do not select the final
checkpoint by default, because the final checkpoint has repeatedly degraded
after the early best validation step.

For the latest confirmed run we used `--steps 200`. The best checkpoint still
landed at step 50, so `100` is a reasonable default for reuse. Use `200` when
you want one extra validation point past step 100.

## Rationale

The current confirmed strategy uses the relaxed safety setup:

```text
lambda_track=0.2
lambda_no_harm=0.2
lambda_gate_safety=0.02
lambda_update_safety=0.05
```

The older full-mixed run produced the best small val20 result so far:

```text
checkpoint: demo_fusion_outputs/sintel_clean_final_mixed_relaxed_batch4/fusion_net_best_val.pth
best step: 50
mean initial EPE: 2.209927
mean refined EPE: 2.209288
mean EPE delta: -0.000639
improved/worse: 17/3
```

On the expanded val230 split, the same checkpoint is essentially neutral:

```text
checkpoint: demo_fusion_outputs/sintel_clean_final_mixed_relaxed_batch4/fusion_net_best_val.pth
mean initial EPE: 3.873132
mean refined EPE: 3.873244
mean EPE delta: +0.000113
improved/worse: 94/136
```

This means val20 was too small and optimistic. Future model selection should
use val230, or a larger multi-dataset validation set once additional datasets
are added.

## Current Confirmed Run

Use this run as the current reference:

```text
output_dir: demo_fusion_outputs/sintel_clean_final_val230_relaxed_batch4
train_manifest: precomputed/track_guided_sintel_clean_final_train_excl_val230_attn/manifest.json
val_manifest: precomputed/track_guided_sintel_clean_final_val230_attn/manifest.json
steps: 200
batch_size: 4
best checkpoint: fusion_net_best_val.pth
best step: 50
```

Independent val230 evaluation of the best checkpoint:

```text
checkpoint: demo_fusion_outputs/sintel_clean_final_val230_relaxed_batch4/fusion_net_best_val.pth
mean initial EPE: 3.873132
mean refined EPE: 3.872814
mean EPE delta: -0.000318
improved/worse: 159/71
metrics: demo_fusion_outputs/sintel_clean_final_val230_relaxed_batch4_eval_best/metrics.json
```

Independent val230 evaluation of the final checkpoint:

```text
checkpoint: demo_fusion_outputs/sintel_clean_final_val230_relaxed_batch4/fusion_net_smoke.pth
mean initial EPE: 3.873132
mean refined EPE: 3.873235
mean EPE delta: +0.000103
improved/worse: 93/137
metrics: demo_fusion_outputs/sintel_clean_final_val230_relaxed_batch4_eval_final/metrics.json
```

Conclusion: this is the first clean+final run with a leakage-free expanded
validation split that shows a small positive val230 gain. Reuse this data
split, loss setup, batch size, and best-checkpoint selection rule for the next
round of experiments.

The final step 500 checkpoint from the older full-mixed relaxed run degraded:

```text
mean refined EPE: 2.214797
mean EPE delta: +0.004870
improved/worse: 4/16
```

Because the best point appears early, the recommended default is `--steps 100`
with `--val_interval 25`. If training on substantially more data, keep the same
loss weights first and rely on `fusion_net_best_val.pth`; only increase steps if
the validation curve shows that the best point moves later.

## Comparison Runs

Earlier clean+final mixed run with weak safety:

```text
lambda_track=0.3
lambda_no_harm=0.0
lambda_gate_safety=0.0
lambda_update_safety=0.02
best step: 100
mean refined EPE: 2.209421
mean EPE delta: -0.000506
final step 1200 mean EPE delta: +0.088478
```

Conservative safety run:

```text
lambda_track=0.1
lambda_no_harm=0.5
lambda_gate_safety=0.05
lambda_update_safety=0.1
best step: 25
mean refined EPE: 2.209894
mean EPE delta: -0.000033
```

The weak-safety run can improve, but it degrades severely with longer training.
The conservative run avoids degradation but almost suppresses useful updates.
The relaxed setup is the current best compromise.

## Next Data Scaling Direction

Keep the confirmed hyperparameters unchanged while adding data. Prefer changing
the data mix before changing the training strategy.

Suggested data experiments:

- If optimizing for clean validation, try a clean-heavy mix such as clean:final
  = 2:1.
- If optimizing for broader robustness, keep clean:final = 1:1.
- Add a larger and more representative validation manifest before relying on
  small EPE differences.

When comparing runs, always evaluate both:

```text
fusion_net_best_val.pth
fusion_net_smoke.pth
```

The expected artifact for deployment or downstream experiments is normally:

```text
fusion_net_best_val.pth
```

## FlowSeek Zero-Shot Alignment Plan

The current confirmed run is Sintel-supervised post-refinement because the
FusionNet training split is derived from Sintel clean/final. It should not be
reported as zero-shot Sintel generalization.

To align with the FlowSeek paper's Sintel(train) zero-shot protocol, train the
track-guided FusionNet on FlowSeek's `C -> T` data and evaluate on Sintel
training clean/final without using Sintel GT for training.

### FlowSeek Baseline Weights

For the `C -> T` protocol, use:

```text
T / S: weights/flowseek_T_CT.pth
M / L: weights/flowseek_M_CT.pth
```

`TartanCT` weights are only needed for the paper's `Tartan + C -> T` setting.
They are not needed for the basic `C -> T` alignment.

The checked FlowSeek-T baseline command is:

```bash
/root/miniconda3/envs/flowseek/bin/python evaluate.py \
  --cfg config/eval/flowseek-T.json \
  --model weights/flowseek_T_CT.pth \
  --dataset sintel
```

Observed local result:

```text
Validation clean EPE: 1.14
Validation final EPE: 2.52
```

This is close to the FlowSeek paper's FlowSeek(T) `C -> T` Sintel(train)
numbers: clean `1.12`, final `2.53`.

For M/L evaluation, `weights/depth_anything_v2_vitb.pth` is required in addition
to `weights/flowseek_M_CT.pth`.

### Required Training Data

For a true `C -> T` FusionNet training run, use:

```text
FlyingChairs
FlyingThings3D
```

Default repository paths from `config/datapaths.json`:

```text
data/FlyingChairs/data
data/FlyingThings3D/
```

FlyingChairs should contain flat `.ppm` and `.flo` files:

```text
data/FlyingChairs/data/*.ppm
data/FlyingChairs/data/*.flo
```

FlyingThings3D full data should follow the FlowSeek loader layout:

```text
data/FlyingThings3D/frames_cleanpass/TRAIN/...
data/FlyingThings3D/frames_finalpass/TRAIN/...
data/FlyingThings3D/optical_flow/TRAIN/...
```

The small official sampler pack has a different layout and is only for smoke
testing:

```text
data/Sampler/FlyingThings3D/RGB_cleanpass/left/*.png
data/Sampler/FlyingThings3D/optical_flow/forward/*.pfm
```

We added this adapter for the sampler pack:

```text
scripts/prepare_sampler_track_guided_manifest.py
```

Smoke output:

```text
precomputed/track_guided_sampler_flyingthings_smoke_attn/manifest.json
demo_fusion_outputs/sampler_flyingthings_smoke_train/fusion_net_best_val.pth
```

The sampler smoke verified the full preprocessing and training path, but it is
not a reliable benchmark because it contains only two FlyingThings3D pairs.

### Zero-Shot Training Pipeline

Once full FlyingChairs and FlyingThings3D are available:

1. Build train/val manifests from C/T without using Sintel samples.
2. Precompute for each sample:

```text
FlowSeek initial flow -> adaptive sampling -> CoTracker -> alignment
-> trajectory attention -> rasterized track prior
```

3. Train FusionNet using the current relaxed safety loss setup:

```text
batch_size=4
lambda_flow=1.0
lambda_track=0.2
lambda_smooth=0.01
lambda_no_harm=0.2
lambda_gate_safety=0.02
lambda_update_safety=0.05
val_interval=25
```

4. Select `fusion_net_best_val.pth`, not the final checkpoint.
5. Evaluate on Sintel train clean/final and compare:

```text
FlowSeek initial EPE
Ours refined EPE
Delta = refined - initial
```

The correct claim, if this improves, is:

```text
Starting from the same FlowSeek C->T zero-shot model, track-guided
post-refinement improves Sintel(train) EPE.
```

Do not use the Sintel-supervised `val230` checkpoint for this claim.
