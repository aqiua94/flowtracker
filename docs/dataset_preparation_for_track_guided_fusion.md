# Track-Guided Fusion 数据集准备指南

## 1. 当前阶段建议

阶段六开始需要训练后处理式 `FusionNet`。FlowSeek 和 CoTracker3 可以继续使用预训练权重，新的训练对象是：

```text
core/track_guidance/fusion_net.py
```

为了先把真实监督训练链路跑通，推荐优先使用 **MPI-Sintel training**，原因是：

- FlowSeek 仓库已经内置 `MpiSintel` 数据读取类。
- Sintel training 提供真实光流 GT。
- 数据规模比 FlyingThings3D 小，更适合第一版最小实验。
- GT 是稠密光流，比 KITTI 稀疏 GT 更适合先验证 FusionNet。

## 2. FlowSeek 官方涉及的数据集

README 中提到评测或训练 FlowSeek 需要手动下载以下数据集：

```text
FlyingChairs
FlyingThings3D
Sintel
KITTI
HD1K
TartanAir
Spring
```

代码中还包含：

```text
LayeredFlow
MegaDepth
Infinigen
Middlebury
vKITTI
```

当前仓库没有提供训练集自动下载脚本。`scripts/get_weights.sh` 只用于下载预训练模型权重，不会下载数据集。

## 3. 默认数据路径

FlowSeek 默认从 `config/datapaths.json` 读取数据路径：

```json
{
  "paths": {
    "tartanair": "data/Tartanair/",
    "chairs": "data/FlyingChairs/data",
    "things": "data/FlyingThings3D/",
    "sintel": "data/MPI-Sintel/",
    "kitti": "data/KITTI/2015/",
    "hd1k": "data/HD1K/",
    "spring": "data/spring/",
    "layeredflow": "data/public_layeredflow_benchmark"
  }
}
```

建议先不要改代码，直接把数据放到默认路径下。

## 4. Sintel 下载内容

请从 Sintel 官网下载 training 数据：

```text
http://sintel.is.tue.mpg.de/
```

阶段六最小实验至少需要：

```text
Training clean images
Training final images
Training flow ground truth
```

如果只想先跑最小版本，可以先只使用 `clean` 图像和 `flow` GT；后续再加入 `final`。

## 5. Sintel 目录结构要求

下载并解压后，整理为以下结构：

```text
data/MPI-Sintel/
  training/
    clean/
      scene_name/
        frame_0001.png
        frame_0002.png
        ...
    final/
      scene_name/
        frame_0001.png
        frame_0002.png
        ...
    flow/
      scene_name/
        frame_0001.flo
        frame_0002.flo
        ...
```

其中：

- `clean` 和 `final` 是两种渲染风格。
- `flow` 中的 `frame_0001.flo` 表示 `frame_0001 -> frame_0002` 的真实光流。
- 每个 scene 中，图片数量通常比 flow 文件数量多 1。

## 6. 最小子集建议

第一版不建议直接全量预计算。建议先选：

```text
1 个 scene
clean 图像
10 到 20 对相邻帧
```

目标是先验证：

- FlowSeek 可以批量输出 `F_0`。
- 自适应采样可以输出查询点。
- CoTracker3 可以输出轨迹。
- 阶段四对齐检查正常。
- 阶段五可以生成 `G_track`。
- 阶段六可以用真实 GT 监督训练 FusionNet。

## 7. 放置后快速检查

放好数据后，在仓库根目录运行：

```shell
find data/MPI-Sintel/training -maxdepth 3 -type f | head
```

应该能看到类似：

```text
data/MPI-Sintel/training/clean/alley_1/frame_0001.png
data/MPI-Sintel/training/clean/alley_1/frame_0002.png
data/MPI-Sintel/training/flow/alley_1/frame_0001.flo
```

也可以用 Python 检查 FlowSeek 数据读取类是否能找到样本：

```shell
/root/miniconda3/envs/flowseek/bin/python - <<'PY'
import sys
sys.path.append('core')
from datasets import MpiSintel

dataset = MpiSintel(split='training', dstype='clean', root='data/MPI-Sintel/')
print('num pairs:', len(dataset))
print('first image pair:', dataset.image_list[0])
print('first flow:', dataset.flow_list[0])
PY
```

如果 `num pairs` 大于 0，说明基础路径可读。

## 8. FusionNet 训练所需预计算产物

阶段六的 `train_track_guided.py` 不直接读取原始 Sintel 图片，而是读取预计算 manifest。

每个样本需要：

```json
{
  "initial_flow": "path/to/flowseek_flow.npy",
  "track_prior": "path/to/g_track.npy",
  "gt_flow": "path/to/gt_flow.npy",
  "valid": "path/to/valid.npy"
}
```

字段说明：

- `initial_flow`：FlowSeek 输出的初始光流 `F_0`，形状 `(H, W, 2)`。
- `track_prior`：阶段五输出的五通道轨迹先验图 `G_track`，形状 `(H, W, 5)`。
- `gt_flow`：Sintel 真实光流，形状 `(H, W, 2)`。
- `valid`：有效像素 mask，形状 `(H, W)`。Sintel 第一版可以全 1。

## 9. 预计算流程

对每个 Sintel 相邻帧样本执行：

1. 使用 `demo_pair.py` 或后续批处理脚本生成 FlowSeek 初始光流：

```text
flow.npy
flow_vis.png
```

2. 使用 `demo_adaptive_sampler.py` 从 `flow.npy` 采样点：

```text
queries.csv
points.npy
```

3. 使用 `demo_cotracker_pair.py` 跟踪采样点：

```text
tracks.npy
visibility.npy
confidence.npy
```

4. 使用 `demo_track_flow_alignment.py` 生成稀疏轨迹光流：

```text
track_flow.npy
valid_mask.npy
alignment_stats.json
```

5. 使用 `demo_track_prior_rasterizer.py` 生成轨迹先验图：

```text
g_track.npy
```

6. 将 Sintel `.flo` GT 转成 `.npy`，并为该样本写入 manifest。

## 10. 第一版训练命令

当 manifest 准备好后，运行：

```shell
/root/miniconda3/envs/flowseek/bin/python train_track_guided.py \
  --manifest path/to/sintel_mini_manifest.json \
  --output_dir demo_fusion_outputs/sintel_mini \
  --steps 1000 \
  --batch_size 1 \
  --hidden_dim 32 \
  --lambda_flow 1.0 \
  --lambda_track 0.2 \
  --lambda_smooth 0.01 \
  --device cuda
```

第一版主要看：

- 训练 loss 是否下降。
- `F_refined` 的 EPE 是否优于或不差于 `F_0`。
- 轨迹点附近 refined flow 是否更接近 CoTracker 位移。

## 11. 常见问题

### 11.1 README 是否提供数据集下载脚本？

没有。README 只提供数据集官网链接。需要手动下载数据集。

### 11.2 `scripts/get_weights.sh` 会下载数据集吗？

不会。它只下载 Depth Anything v2 和 FlowSeek 的预训练权重。

### 11.3 为什么先选 Sintel？

因为 Sintel 有稠密 GT，仓库内已有读取类，数据规模适中，最适合作为 FusionNet 的第一个真实监督实验。

### 11.4 是否必须同时使用 clean 和 final？

不是。最小版本可以先只用 `clean`。跑通后再把 `final` 加进来。

### 11.5 是否需要重新训练 FlowSeek 或 CoTracker3？

第一版不需要。当前阶段只训练 FusionNet，FlowSeek 和 CoTracker3 都作为预训练模型使用。
