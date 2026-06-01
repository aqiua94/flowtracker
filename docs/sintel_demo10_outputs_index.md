# Sintel Demo10 分步可视化产物说明

本文档说明刚刚用 Sintel clean `alley_1` 前 10 对相邻帧跑出的分步结果。输出根目录为：

```text
precomputed/track_guided_sintel_clean_demo10_attn/
```

这批数据包含 10 个样本：

```text
alley_1_frame_0001
alley_1_frame_0002
...
alley_1_frame_0010
```

每个样本对应一对相邻帧，例如 `alley_1_frame_0001` 表示：

```text
data/MPI-Sintel/training/clean/alley_1/frame_0001.png
data/MPI-Sintel/training/clean/alley_1/frame_0002.png
```


## 0. 运行命令

这批 demo10 结果由两步生成。

第一步：从 Sintel clean `alley_1` 取前 10 对相邻帧，生成 manifest 和每个样本的基础目录：

```bash
/root/miniconda3/envs/flowseek/bin/python scripts/prepare_sintel_track_guided_manifest.py   --sintel_root data/MPI-Sintel   --split training   --dstype clean   --scenes alley_1   --max_pairs 10   --output_root precomputed/track_guided_sintel_clean_demo10_attn   --manifest precomputed/track_guided_sintel_clean_demo10_attn/manifest.json
```

第二步：对这 10 对帧逐样本运行完整分步流水线，包括 FlowSeek 初始光流、自适应采样、CoTracker 点跟踪、稀疏轨迹与光流对齐、trajectory attention 增强，以及最终轨迹先验栅格化：

```bash
/root/miniconda3/envs/flowseek/bin/python scripts/precompute_track_guided_from_pairs.py   --pairs precomputed/track_guided_sintel_clean_demo10_attn/manifest_pairs.json   --cfg config/eval/flowseek-T.json   --model weights/flowseek_T_CT.pth   --device cuda   --max_samples 10   --overwrite   --flowseek_max_size 384   --cotracker_max_size 0   --num_points 512   --cell_size 32   --min_points_per_cell 1   --max_points_per_cell 8   --min_distance 4.0   --flow_gradient_weight 0.7   --flow_magnitude_weight 0.3   --image_edge_weight 0.0   --use_trajectory_attention   --attention_spatial_sigma 96   --attention_motion_sigma 8   --attention_endpoint_error_scale 3   --attention_self_weight 1
```

额外的 `gallery/` 总览拼图是在上述流水线完成后，根据每个样本目录里的图片产物整理生成的，便于快速浏览。

## 0.1 使用的模型

这批 demo10 分步预计算实际用到的是预训练模型和点跟踪模型，没有使用你训练出来的 FusionNet 做 refined flow 评估。

### FlowSeek 预训练模型

```text
weights/flowseek_T_CT.pth
```

它在第二步命令中通过下面参数传入：

```text
--cfg config/eval/flowseek-T.json
--model weights/flowseek_T_CT.pth
```

含义：使用 FlowSeek-T 配置和 `CT` 预训练权重，生成每个样本的初始 dense optical flow。对应输出主要在每个样本的：

```text
flowseek/flow.npy
flowseek/flow_vis.png
```

### Depth Anything V2 预训练模型

```text
weights/depth_anything_v2_vits.pth
```

`config/eval/flowseek-T.json` 中配置了：

```json
"da_size": "vits"
```

因此 `FlowSeek` 初始化时会自动加载 `weights/depth_anything_v2_vits.pth`。这个模型不是单独在命令行里传入的，而是由 FlowSeek 模型内部读取，用来提供深度基础模型特征。

### CoTracker 预训练模型

```text
facebookresearch/co-tracker: cotracker3_offline
```

它由 `demo_cotracker_pair.py` 内部通过 `torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")` 加载。本机已有缓存：

```text
/root/.cache/torch/hub/facebookresearch_co-tracker_main
```

它负责把 `sampler/queries.csv` 里的采样点从第一帧跟踪到第二帧。对应输出主要在每个样本的：

```text
cotracker/tracks.npy
cotracker/visibility.npy
cotracker/track_0000.png
cotracker/track_0001.png
```

### 你训练的 FusionNet 模型

当前仓库里最新确认最好的你训练的 FusionNet 是：

```text
demo_fusion_outputs/sintel_clean_final_val230_relaxed_batch4/fusion_net_best_val.pth
```

对应的最终 checkpoint 是：

```text
demo_fusion_outputs/sintel_clean_final_val230_relaxed_batch4/fusion_net_smoke.pth
```

但注意：这次 demo10 只跑了预计算链路，生成 FlowSeek 初始光流、采样点、CoTracker 轨迹、alignment、trajectory attention 和 `G_track` 轨迹先验；没有运行 `evaluate_track_guided.py`，所以没有把你训练的 FusionNet checkpoint 用到 demo10 上，也没有生成 demo10 的 `refined_flow_vis.png`、`initial_error.png`、`refined_error.png` 这类 refined 评测图。

如果后续要用你训练的最好模型评估这 10 个 demo 样本，应使用：

```text
checkpoint: demo_fusion_outputs/sintel_clean_final_val230_relaxed_batch4/fusion_net_best_val.pth
manifest: precomputed/track_guided_sintel_clean_demo10_attn/manifest.json
```


## 1. 总览图目录

```text
precomputed/track_guided_sintel_clean_demo10_attn/gallery/
```

该目录是为了快速查看每个样本的关键阶段而额外生成的拼图目录。每个样本有一张总览图：

```text
alley_1_frame_0001_overview.png
...
alley_1_frame_0010_overview.png
```

每张总览图从左到右、从上到下包含：

| 面板 | 含义 |
| --- | --- |
| `INPUT` | 当前样本的第一帧原图。 |
| `FLOW` | FlowSeek 预测的初始光流可视化。颜色表示运动方向和幅值。 |
| `SAMPLE WEIGHT` | 自适应采样权重热力图。颜色越亮，表示该区域越容易被选为跟踪点。 |
| `SAMPLED POINTS` | 在原图上叠加最终选出的 CoTracker 查询点。 |
| `COTRACKER VECTORS` | CoTracker 在第二帧上的跟踪结果，点和短线表示从第一帧到第二帧的运动轨迹。 |
| `ALIGNMENT` | CoTracker 稀疏位移与 FlowSeek 采样光流的对齐对比图。 |
| `ATTN DELTA` | trajectory attention 增强前后，稀疏轨迹位移变化较大的位置。 |
| `G_TRACK MAG` | 最终栅格化轨迹先验 `G_track` 的位移幅值图。 |

建议先看 `gallery/`，确认整体流程是否正常，再进入单个样本目录看细节。

## 2. 单个样本目录结构

以 `alley_1_frame_0001` 为例：

```text
precomputed/track_guided_sintel_clean_demo10_attn/alley_1/alley_1_frame_0001/
```

目录内主要分为以下阶段：

```text
flowseek/
sampler/
cotracker/
alignment/
alignment/trajectory_attention/
rasterizer/
```

此外还有：

```text
gt_flow.npy
valid.npy
```

它们来自 Sintel 官方真值光流和有效像素 mask，主要用于数值评估，不是直接给人看的图片。

## 3. `flowseek/`：初始光流结果

```text
flowseek/flow.npy
flowseek/flow_vis.png
```

### `flow.npy`

FlowSeek 对当前相邻帧预测出的初始 dense optical flow，形状为：

```text
H x W x 2
```

最后一维两个通道分别是水平位移 `dx` 和垂直位移 `dy`。

### `flow_vis.png`

`flow.npy` 的颜色可视化结果。

- 颜色主要表示运动方向。
- 颜色亮度或饱和度通常和运动幅值有关。
- 这张图用于快速判断 FlowSeek 预测的整体运动趋势是否合理。

## 4. `sampler/`：自适应采样点

```text
sampler/points.npy
sampler/point_weights.npy
sampler/sampling_weight.npy
sampler/queries.csv
sampler/sampling_weight.png
sampler/sampled_points.png
```

### `sampling_weight.png`

采样权重热力图。

它由 FlowSeek 光流的梯度、光流幅值等信息计算得到。颜色越强，表示这个位置越值得选为 CoTracker 查询点。

直观含义：

- 运动边界附近通常权重更高。
- 光流变化明显的区域通常更容易被采样。
- 纹理或运动信息较少的平坦区域权重可能较低。

### `sampled_points.png`

在第一帧图像上画出最终选中的采样点。

这张图可以用来检查：

- 点是否覆盖了主要运动区域。
- 点是否过度集中在少数位置。
- 图像边缘、遮挡边界、运动剧烈区域是否被采到。

### `queries.csv`

给 CoTracker 使用的查询点文件，格式为：

```text
t,x,y
```

其中 `t=0` 表示这些点都从第一帧开始跟踪。

## 5. `cotracker/`：点跟踪结果

```text
cotracker/tracks.npy
cotracker/visibility.npy
cotracker/confidence.npy
cotracker/queries.csv
cotracker/track_0000.png
cotracker/track_0001.png
```

### `track_0000.png`

第一帧上的查询点可视化。

这张图显示 CoTracker 的起点位置，也就是从哪里开始跟踪。

### `track_0001.png`

第二帧上的跟踪结果可视化。

图中会画出点编号和轨迹线。对于两帧输入来说，轨迹线就是从第一帧点位置到第二帧预测位置的短向量。

这张图用于检查：

- 点是否跟随了真实物体运动。
- 是否有明显跟飞、漂移、跳点。
- 大位移区域的跟踪是否稳定。

### `tracks.npy`

CoTracker 输出的轨迹坐标，形状通常为：

```text
2 x N x 2
```

这里 `2` 是两帧，`N=512` 是采样点数，最后一维是坐标 `(x, y)`。

### `visibility.npy`

每个点在每一帧是否可见。

如果某个点在第二帧不可见或跟踪不可靠，后续 alignment 和 rasterizer 阶段会降低或忽略它的贡献。

## 6. `alignment/`：稀疏轨迹与 FlowSeek 光流对齐

```text
alignment/points.npy
alignment/track_flow.npy
alignment/flowseek_flow.npy
alignment/endpoint_error.npy
alignment/valid_mask.npy
alignment/confidence.npy
alignment/alignment_stats.json
alignment/alignment_overlay.png
```

### `alignment_overlay.png`

这是最重要的诊断图之一。

它把两种位移画在同一张输入图上：

- CoTracker 根据点跟踪得到的稀疏位移。
- FlowSeek 在相同点位置采样得到的光流位移。

用途：

- 检查 CoTracker 和 FlowSeek 的方向是否一致。
- 检查尺度是否一致。
- 找出两者差异很大的区域。

如果两类箭头大体方向一致，说明稀疏轨迹和初始 dense flow 在坐标系上是对齐的。如果某些区域差异很大，可能是遮挡、跟踪失败、FlowSeek 预测错误，或真实运动复杂。

### `endpoint_error.npy`

每个采样点上，CoTracker 位移和 FlowSeek 采样光流之间的 EPE 差异。

数值越大，说明该点的稀疏跟踪结果和 FlowSeek 初始光流越不一致。

### `alignment_stats.json`

当前样本的对齐统计信息，例如：

- 有效轨迹数。
- 总轨迹数。
- 平均 CoTracker 位移。
- 平均 FlowSeek 采样光流。
- 平均、最大 endpoint error。

## 7. `alignment/trajectory_attention/`：轨迹注意力增强

```text
alignment/trajectory_attention/enhanced_track_flow.npy
alignment/trajectory_attention/enhanced_confidence.npy
alignment/trajectory_attention/enhanced_valid_mask.npy
alignment/trajectory_attention/attention.npy
alignment/trajectory_attention/trajectory_attention_stats.json
alignment/trajectory_attention/attention_matrix.png
alignment/trajectory_attention/flow_delta_points.png
```

### `attention_matrix.png`

轨迹点之间的 attention 权重矩阵可视化。

它表示不同轨迹点之间如何互相参考、传播和修正信息。亮色区域表示两个点之间的关联更强。

### `flow_delta_points.png`

显示 trajectory attention 增强前后，哪些点的轨迹位移变化较大。

直观含义：

- 亮点少且分散：attention 只是做了温和修正。
- 亮点很多或集中：该样本中有较多轨迹被明显调整。
- 如果亮点集中在遮挡、运动边界或低纹理区域，说明 attention 可能正在处理不可靠轨迹。

### `enhanced_track_flow.npy`

增强后的稀疏轨迹位移。

后续 `rasterizer/` 不再直接使用原始 `alignment/track_flow.npy`，而是使用这个增强后的版本来生成最终轨迹先验。

### `trajectory_attention_stats.json`

记录 attention 增强的统计信息，例如：

- 输入有效点数量。
- 输出有效点数量。
- 平均可靠性。
- 平均位移改变量。
- 最大位移改变量。

## 8. `rasterizer/`：最终轨迹先验图

```text
rasterizer/g_track.npy
rasterizer/g_track_stats.json
rasterizer/g_track_magnitude.png
rasterizer/g_track_confidence.png
rasterizer/g_track_distance.png
```

### `g_track.npy`

最终给 FusionNet 使用的轨迹先验图，形状为：

```text
H x W x 5
```

5 个通道通常包含：

1. 轨迹位移 `dx`
2. 轨迹位移 `dy`
3. 轨迹置信度
4. 到最近轨迹点的距离
5. 有效性或辅助 mask 信息

这是这条预计算链路的核心产物之一。

### `g_track_magnitude.png`

轨迹位移幅值图。

亮的地方表示该位置附近有较强的轨迹运动先验。由于当前实现主要在有效轨迹点附近栅格化，因此图上通常会看到稀疏点状响应。

### `g_track_confidence.png`

轨迹先验置信度图。

越亮表示该位置附近的轨迹先验越可靠。它综合了 CoTracker 可见性、confidence 以及 trajectory attention 后的可靠性信息。

### `g_track_distance.png`

每个像素到最近有效轨迹点的距离图。

用途：

- 判断轨迹点覆盖是否稀疏。
- 判断哪些区域离有效轨迹点很远，FusionNet 在这些区域不应该过度相信轨迹先验。

### `g_track_stats.json`

记录最终栅格化统计信息，例如：

- 输入点数量。
- 有效点数量。
- 被栅格化的像素数量。
- 距离范围和平均距离。

## 9. 根目录 manifest 文件

```text
manifest.json
manifest_pairs.json
```

### `manifest.json`

训练或评估 `train_track_guided.py` / `evaluate_track_guided.py` 时读取的样本索引。

每个样本记录：

```text
initial_flow
track_prior
gt_flow
valid
```

也就是：

- FlowSeek 初始光流。
- 最终轨迹先验 `G_track`。
- Sintel 真值光流。
- 有效像素 mask。

### `manifest_pairs.json`

预计算流水线使用的帧对索引。

它记录：

- 原始 `image1` / `image2` 路径。
- 每个阶段的输出目录。
- 当前样本 id。

如果以后想重跑某个阶段，可以从这个文件定位到原始输入和各阶段输出目录。

## 10. 快速阅读建议

如果只是想快速确认结果，按这个顺序看：

1. `gallery/*_overview.png`
2. 单个样本的 `sampler/sampled_points.png`
3. 单个样本的 `cotracker/track_0001.png`
4. 单个样本的 `alignment/alignment_overlay.png`
5. 单个样本的 `rasterizer/g_track_magnitude.png`

如果要判断某个样本为什么好或坏，重点看：

1. `alignment/alignment_overlay.png`
2. `alignment/alignment_stats.json`
3. `alignment/trajectory_attention/flow_delta_points.png`
4. `alignment/trajectory_attention/trajectory_attention_stats.json`
5. `rasterizer/g_track_confidence.png`

## 11. FusionNet 最终光流评估与对比图

在完成 demo10 预计算链路后，又使用当前最好 FusionNet checkpoint 对这 10 个样本做了一次 refined flow 评估。

### 运行命令

```bash
/root/miniconda3/envs/flowseek/bin/python evaluate_track_guided.py \
  --manifest precomputed/track_guided_sintel_clean_demo10_attn/manifest.json \
  --checkpoint demo_fusion_outputs/sintel_clean_final_val230_relaxed_batch4/fusion_net_best_val.pth \
  --output_dir demo_fusion_outputs/sintel_demo10_val230_best_eval \
  --num_visualizations 10
```

其中：

```text
checkpoint: demo_fusion_outputs/sintel_clean_final_val230_relaxed_batch4/fusion_net_best_val.pth
manifest: precomputed/track_guided_sintel_clean_demo10_attn/manifest.json
output_dir: demo_fusion_outputs/sintel_demo10_val230_best_eval
```

这一步才真正使用了你训练出来的 FusionNet 模型。它读取预计算阶段生成的：

```text
flowseek/flow.npy
rasterizer/g_track.npy
gt_flow.npy
valid.npy
```

然后输出 refined flow 和误差对比图。

### FusionNet 评估输出目录

```text
demo_fusion_outputs/sintel_demo10_val230_best_eval/
```

关键文件：

```text
metrics.json
sample_0000/
sample_0001/
...
sample_0009/
comparison_gallery/
```

### `metrics.json`

保存这 10 个样本的整体指标和逐样本指标。当前结果为：

```text
Mean initial EPE: 0.432440
Mean refined EPE: 0.432572
Mean EPE delta: +0.000132
Improved/worse: 4/6
```

解释：

- `initial EPE` 是 FlowSeek 初始光流相对 Sintel GT 的误差。
- `refined EPE` 是经过 FusionNet 修正后的最终光流误差。
- `delta = refined - initial`，负数表示 FusionNet 改善，正数表示变差。
- 这 10 个 `alley_1` 样本上，FusionNet 输出和初始 FlowSeek 非常接近，平均略差 `0.000132`，4 个样本改善、6 个样本变差。

### 单个样本评估图

每个 `sample_xxxx/` 目录包含：

```text
initial_flow_vis.png
refined_flow_vis.png
gt_flow_vis.png
initial_error.png
refined_error.png
improvement.png
refined_flow.npy
delta_flow.npy
gate.npy
```

含义如下：

| 文件 | 含义 |
| --- | --- |
| `initial_flow_vis.png` | FlowSeek 初始光流可视化。 |
| `refined_flow_vis.png` | FusionNet 修正后的最终光流可视化。 |
| `gt_flow_vis.png` | Sintel 真值光流可视化。 |
| `initial_error.png` | FlowSeek 初始光流相对 GT 的误差图。 |
| `refined_error.png` | FusionNet 最终光流相对 GT 的误差图。 |
| `improvement.png` | 只显示 FusionNet 改善的区域，越亮表示误差下降越明显。 |
| `refined_flow.npy` | FusionNet 输出的最终光流数组。 |
| `delta_flow.npy` | FusionNet 预测的光流修正量，也就是 refined flow 与 initial flow 的差。 |
| `gate.npy` | FusionNet 内部 gate 图，表示模型对修正量的使用强度。 |

### 最终光流对比拼图

为了更方便肉眼比较，又额外生成了总览拼图目录：

```text
demo_fusion_outputs/sintel_demo10_val230_best_eval/comparison_gallery/
```

每个样本有一张：

```text
sample_0000_flow_compare.png
sample_0001_flow_compare.png
...
sample_0009_flow_compare.png
```

每张拼图包含 6 个面板：

| 面板 | 含义 |
| --- | --- |
| `INITIAL FLOW` | 一开始 FlowSeek 的光流结果。 |
| `REFINED FLOW` | 经过 FusionNet 后的最终光流结果。 |
| `GT FLOW` | Sintel 真值光流。 |
| `INITIAL ERROR` | 初始光流误差。 |
| `REFINED ERROR` | 最终光流误差。 |
| `IMPROVEMENT` | FusionNet 带来的改善区域。 |

这些拼图最适合用来回答“最终光流和一开始的光流相比有什么变化”。

### 和前面预计算图的区别

预计算阶段的图片主要解释轨迹先验是怎么来的：

```text
采样点 -> CoTracker 轨迹 -> alignment 向量 -> trajectory attention -> G_track
```

FusionNet 评估阶段的图片主要解释最终模型有没有改好光流：

```text
initial_flow -> refined_flow -> error/improvement
```

因此 FusionNet 阶段不会再生成采样点或 CoTracker 向量图，它关注的是最终 dense flow 和 GT 的误差变化。

