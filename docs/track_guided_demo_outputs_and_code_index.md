# Track-Guided Flow Demo 产物与代码索引

本文档说明当前仓库中为“基于点跟踪改进 FlowSeek 光流”实验新增的代码入口、demo 输出目录、Sintel 预计算目录、训练结果和验证结果。重点解释每个图片文件来自哪一步、表达什么含义，以及每个脚本完成哪一阶段功能。

## 1. 阶段与代码对应关系

### 阶段一：FlowSeek 两帧推理

- `demo_pair.py`
  - 功能：读取两张连续帧，加载 FlowSeek 权重，输出初始稠密光流 `F_0`。
  - 输入：`--image1`、`--image2`、`--cfg`、`--model`。
  - 输出：`flow.npy` 和 `flow_vis.png`。
  - 注意：该脚本已修正 resize 还原逻辑，确保 Sintel 原图 `436 x 1024` 的输出光流仍为 `(436, 1024, 2)`，不会因为缩放舍入变成 `437 x 1024`。

### 阶段二：CoTracker3 点跟踪

- `demo_cotracker_pair.py`
  - 功能：读取两帧或短序列图像，以及 `[t, x, y]` 格式查询点，调用 CoTracker3 offline 模型输出轨迹。
  - 输入：`--frames`、可选 `--queries`。
  - 输出：`tracks.npy`、`visibility.npy`、`confidence.npy`、`track_0000.png`、`track_0001.png`。
  - 当前 `confidence.npy` 暂用 visibility 的 float 版本占位。

### 阶段三：FlowSeek 引导的自适应采样

- `core/track_guidance/sampler.py`
  - 功能：根据 FlowSeek 光流梯度、运动幅值和可选图像边缘，生成稀疏采样点。
  - 主要输出：采样点坐标、采样权重、完整采样权重图。

- `demo_adaptive_sampler.py`
  - 功能：命令行入口，读取 `flow.npy` 和可选图像，调用 sampler，保存 CoTracker 可直接读取的 `queries.csv`。
  - 输出：`points.npy`、`point_weights.npy`、`sampling_weight.npy`、`sampling_weight.png`、`sampled_points.png`、`queries.csv`。

### 阶段四：轨迹与 FlowSeek 光流对齐

- `core/track_guidance/cotracker_wrapper.py`
  - 功能：把 CoTracker 输出轨迹转换为相邻帧稀疏位移，双线性采样 FlowSeek 稠密光流，并计算二者 EPE。

- `demo_track_flow_alignment.py`
  - 功能：命令行入口，对同一点处的 CoTracker 稀疏位移和 FlowSeek 光流进行对齐检查。
  - 输出：`points.npy`、`next_points.npy`、`track_flow.npy`、`flowseek_at_points.npy`、`valid_mask.npy`、`confidence.npy`、`endpoint_error.npy`、`alignment_stats.json`、`alignment_overlay.png`。

### 阶段五：轨迹先验栅格化

- `core/track_guidance/rasterizer.py`
  - 功能：把稀疏轨迹位移写入与光流同尺寸的五通道图 `G_track`。
  - 当前通道顺序：`[dx, dy, confidence, visibility, distance_to_nearest_track]`。

- `demo_track_prior_rasterizer.py`
  - 功能：命令行入口，读取阶段四输出，生成 `G_track` 及可视化。
  - 输出：`g_track.npy`、`g_track_stats.json`、`g_track_magnitude.png`、`g_track_confidence.png`、`g_track_distance.png`。

### 阶段六：后处理式 FusionNet 训练

- `core/track_guidance/fusion_net.py`
  - 功能：轻量 U-Net 风格后处理网络。
  - 输入：`F_0` 和 `G_track`。
  - 输出：`Delta F`、`gate`、`F_refined = F_0 + gate * Delta F`。

- `core/track_guidance/losses.py`
  - 功能：定义监督光流 L1、EPE、轨迹点约束 loss 和 smoothness loss。

- `train_track_guided.py`
  - 功能：读取预计算 manifest，训练 FusionNet。
  - 输入 manifest 每个样本包含：`initial_flow`、`track_prior`、`gt_flow`、`valid`。
  - 输出：`fusion_net_smoke.pth`、`train_history.json`。

### Sintel 数据准备与批处理

- `scripts/prepare_sintel_track_guided_manifest.py`
  - 功能：枚举 Sintel 相邻帧样本，创建预计算目录，把 `.flo` 真值光流转成 `gt_flow.npy`，并生成 manifest。
  - 输出：`manifest.json` 和 `manifest_pairs.json`。
  - 关键参数：
    - `--scenes alley_1`：只取指定 scene。
    - `--start_index`：跳过前若干对样本，用于拆训练/验证。
    - `--max_pairs`：限制样本数量。

- `scripts/precompute_track_guided_from_pairs.py`
  - 功能：读取 `manifest_pairs.json`，按样本批量执行阶段一到阶段五。
  - 具体顺序：FlowSeek 推理 -> 自适应采样 -> CoTracker 跟踪 -> 轨迹对齐 -> 栅格化 `G_track`。
  - 默认支持断点续跑：如果某一步产物已经存在，会跳过该步骤。

### 训练结果验证

- `evaluate_track_guided.py`
  - 功能：加载训练好的 FusionNet checkpoint，对 manifest 中样本逐个比较 `F_0` 和 `F_refined` 相对 GT 的 EPE。
  - 输出：`metrics.json`、`refined_flow.npy`、`delta_flow.npy`、`gate.npy`、多种可视化图片。

## 2. 合成 smoke demo 产物

这些目录来自最早的两帧合成平移样例，主要用于确认每个阶段的最小链路正确。

### `demo_pair_inputs/`

- `frame_0001.png`
  - 来源：阶段一合成输入第一帧。
  - 含义：用于制造已知水平平移的测试图。

- `frame_0002.png`
  - 来源：阶段一合成输入第二帧。
  - 含义：由第一帧水平平移约 12 像素得到，用于检查 FlowSeek 和 CoTracker 输出方向是否合理。

### `demo_pair_outputs/stage1_smoke/`

- `flow.npy`
  - 来源：`demo_pair.py`。
  - 含义：FlowSeek 对合成两帧输出的初始稠密光流，形状为 `(216, 384, 2)`。

- `flow_vis.png`
  - 来源：`demo_pair.py` 对 `flow.npy` 做光流颜色编码。
  - 含义：阶段一结果图，用颜色显示光流方向和幅值；合成平移测试中应表现为主要水平运动。

### `demo_cotracker_outputs/stage2_smoke/`

- `tracks.npy`
  - 来源：`demo_cotracker_pair.py`。
  - 含义：手工指定少量查询点后，CoTracker 输出的轨迹，形状为 `(T, N, 2)`。

- `visibility.npy`
  - 来源：`demo_cotracker_pair.py`。
  - 含义：每个点在每帧是否可见。

- `confidence.npy`
  - 来源：`demo_cotracker_pair.py`。
  - 含义：当前版本为 visibility 的 float 占位，用于后续统一接口。

- `queries.csv`
  - 来源：`demo_cotracker_pair.py`。
  - 含义：CoTracker 查询点，列为 `[t, x, y]`。

- `track_0000.png`、`track_0001.png`
  - 来源：`demo_cotracker_pair.py`。
  - 含义：阶段二结果图；在第 0/1 帧上叠加点编号和轨迹线，用于目视检查跟踪是否跟随合成平移。

### `demo_sampler_outputs/stage3_smoke/`

- `points.npy`
  - 来源：`demo_adaptive_sampler.py`。
  - 含义：自适应采样得到的 `(x, y)` 点坐标。

- `point_weights.npy`
  - 来源：`demo_adaptive_sampler.py`。
  - 含义：每个采样点的权重，来自光流梯度和运动幅值融合。

- `sampling_weight.npy`
  - 来源：`demo_adaptive_sampler.py`。
  - 含义：整张图的采样权重图。

- `queries.csv`
  - 来源：`demo_adaptive_sampler.py`。
  - 含义：把 `points.npy` 转成 CoTracker 查询格式 `[t, x, y]`。

- `sampling_weight.png`
  - 来源：`demo_adaptive_sampler.py`。
  - 含义：阶段三结果图；权重热力图，颜色越强表示越容易被采样。

- `sampled_points.png`
  - 来源：`demo_adaptive_sampler.py`。
  - 含义：阶段三结果图；在输入图上画出最终采样点，用于检查覆盖范围和点密度。

### `demo_cotracker_outputs/stage3_sampler_check/`

- `tracks.npy`、`visibility.npy`、`confidence.npy`
  - 来源：把阶段三的 `queries.csv` 输入 `demo_cotracker_pair.py`。
  - 含义：验证自适应采样点可以直接进入 CoTracker。

- `track_0000.png`、`track_0001.png`
  - 来源：`demo_cotracker_pair.py`。
  - 含义：阶段三到阶段二联通性检查图，显示自动采样点的跟踪结果。

### `demo_alignment_outputs/stage4_smoke/`

- `points.npy`
  - 来源：`demo_track_flow_alignment.py`。
  - 含义：每条轨迹在当前帧的起点。

- `next_points.npy`
  - 来源：`demo_track_flow_alignment.py`。
  - 含义：同一轨迹在下一帧的位置。

- `track_flow.npy`
  - 来源：`next_points - points`。
  - 含义：CoTracker 轨迹转成的稀疏光流。

- `flowseek_at_points.npy`
  - 来源：在 `points.npy` 位置双线性采样 FlowSeek `flow.npy`。
  - 含义：与 `track_flow.npy` 对齐比较的 FlowSeek 稠密光流采样值。

- `valid_mask.npy`
  - 来源：可见性、置信度和边界检查。
  - 含义：哪些轨迹点可以作为有效先验。

- `endpoint_error.npy`
  - 来源：`track_flow` 与 `flowseek_at_points` 的逐点 EPE。
  - 含义：阶段四对齐误差。

- `alignment_stats.json`
  - 来源：`demo_track_flow_alignment.py`。
  - 含义：有效点数量、平均位移、EPE 均值/中位数/最大值等统计。

- `alignment_overlay.png`
  - 来源：`demo_track_flow_alignment.py`。
  - 含义：阶段四结果图；绿色箭头表示 CoTracker 轨迹位移，红色箭头表示 FlowSeek 采样光流，用于检查坐标、方向、尺度是否一致。

### `demo_rasterizer_outputs/stage5_smoke/`

- `g_track.npy`
  - 来源：`demo_track_prior_rasterizer.py`。
  - 含义：阶段五输出的五通道轨迹先验图，默认形状 `(H, W, 5)`。

- `g_track_stats.json`
  - 来源：`demo_track_prior_rasterizer.py`。
  - 含义：输入有效点数量、实际栅格化像素数量、距离图统计等。

- `g_track_magnitude.png`
  - 来源：`demo_track_prior_rasterizer.py`。
  - 含义：阶段五结果图；显示 `G_track` 中轨迹位移幅值，只有有效轨迹附近有非零响应。

- `g_track_confidence.png`
  - 来源：`demo_track_prior_rasterizer.py`。
  - 含义：阶段五结果图；显示轨迹先验置信度通道。

- `g_track_distance.png`
  - 来源：`demo_track_prior_rasterizer.py`。
  - 含义：阶段五结果图；显示每个像素到最近有效轨迹点的距离。

### `demo_fusion_outputs/stage6_smoke/`

- `stage6_smoke_manifest.json`
  - 来源：`train_track_guided.py` 在未提供 manifest 时自动生成。
  - 含义：合成 smoke 样本的训练 manifest。

- `fusion_net_smoke.pth`
  - 来源：`train_track_guided.py`。
  - 含义：阶段六合成样例训练得到的 FusionNet checkpoint。

- `train_history.json`
  - 来源：`train_track_guided.py`。
  - 含义：阶段六合成样例训练 loss 历史。

## 3. Sintel 真实数据预计算目录

真实训练和验证使用了 Sintel `training/clean/alley_1`。

### `precomputed/track_guided_sintel_clean_20/`

用途：训练集预计算目录。对应 `alley_1` 前 20 对相邻帧：

```text
frame_0001 -> frame_0002
...
frame_0020 -> frame_0021
```

- `manifest.json`
  - 来源：`scripts/prepare_sintel_track_guided_manifest.py`。
  - 含义：`train_track_guided.py` 和 `evaluate_track_guided.py` 使用的训练 manifest。每个样本记录 `initial_flow`、`track_prior`、`gt_flow`、`valid`。

- `manifest_pairs.json`
  - 来源：`scripts/prepare_sintel_track_guided_manifest.py`。
  - 含义：批处理预计算使用的样本清单，包含原始 `image1`、`image2`、`.flo` GT 路径和各阶段输出目录。

每个样本目录形如：

```text
precomputed/track_guided_sintel_clean_20/alley_1/alley_1_frame_0001/
```

其子目录含义：

- `gt_flow.npy`
  - 来源：Sintel `training/flow/alley_1/frame_0001.flo` 转换。
  - 含义：真实光流 GT，监督训练和 EPE 验证使用。

- `valid.npy`
  - 来源：`.flo` 中有限值检查。
  - 含义：有效像素 mask，Sintel 第一版基本为全有效。

- `flowseek/flow.npy`
  - 来源：`demo_pair.py`。
  - 含义：该帧对的 FlowSeek 初始稠密光流 `F_0`。

- `flowseek/flow_vis.png`
  - 来源：`demo_pair.py`。
  - 含义：真实 Sintel 帧对的 FlowSeek 初始光流可视化，是阶段一在真实数据上的结果图。

- `sampler/points.npy`
  - 来源：`demo_adaptive_sampler.py`。
  - 含义：基于该样本 `F_0` 自动采样的点。

- `sampler/sampling_weight.png`
  - 来源：`demo_adaptive_sampler.py`。
  - 含义：阶段三在真实数据上的采样权重热力图。

- `sampler/sampled_points.png`
  - 来源：`demo_adaptive_sampler.py`。
  - 含义：阶段三在真实 Sintel 图像上叠加采样点的结果图。

- `sampler/queries.csv`
  - 来源：`points.npy` 转换。
  - 含义：给 CoTracker 使用的 `[t, x, y]` 查询点。

- `cotracker/tracks.npy`
  - 来源：`demo_cotracker_pair.py`。
  - 含义：真实 Sintel 帧对上的 CoTracker 轨迹。

- `cotracker/visibility.npy`
  - 来源：`demo_cotracker_pair.py`。
  - 含义：轨迹点可见性。

- `cotracker/confidence.npy`
  - 来源：`demo_cotracker_pair.py`。
  - 含义：当前为 visibility float 占位。

- `cotracker/track_0000.png`、`cotracker/track_0001.png`
  - 来源：`demo_cotracker_pair.py`。
  - 含义：阶段二在真实数据上的轨迹可视化；分别显示第一帧和第二帧上的轨迹点/轨迹线。

- `alignment/track_flow.npy`
  - 来源：`demo_track_flow_alignment.py`。
  - 含义：由 CoTracker 轨迹转换得到的稀疏光流。

- `alignment/flowseek_at_points.npy`
  - 来源：`demo_track_flow_alignment.py`。
  - 含义：在相同轨迹起点处采样的 FlowSeek 光流。

- `alignment/alignment_stats.json`
  - 来源：`demo_track_flow_alignment.py`。
  - 含义：该样本轨迹和 FlowSeek 的对齐统计。训练集 20 对样本中，有效轨迹通常约为 `115-125 / 128`。

- `alignment/alignment_overlay.png`
  - 来源：`demo_track_flow_alignment.py`。
  - 含义：阶段四真实数据对齐图；绿箭头为轨迹位移，红箭头为 FlowSeek 位移。

- `rasterizer/g_track.npy`
  - 来源：`demo_track_prior_rasterizer.py`。
  - 含义：训练 FusionNet 的轨迹先验 `G_track`。

- `rasterizer/g_track_magnitude.png`
  - 来源：`demo_track_prior_rasterizer.py`。
  - 含义：真实数据轨迹先验中的位移幅值图。

- `rasterizer/g_track_confidence.png`
  - 来源：`demo_track_prior_rasterizer.py`。
  - 含义：真实数据轨迹先验中的置信度图。

- `rasterizer/g_track_distance.png`
  - 来源：`demo_track_prior_rasterizer.py`。
  - 含义：真实数据轨迹先验中的最近轨迹点距离图。

- `rasterizer/g_track_stats.json`
  - 来源：`demo_track_prior_rasterizer.py`。
  - 含义：该样本 `G_track` 的点数和距离统计。

### `precomputed/track_guided_sintel_clean_val10/`

用途：held-out 验证集预计算目录。对应 `alley_1` 后 10 对相邻帧：

```text
frame_0021 -> frame_0022
...
frame_0030 -> frame_0031
```

该目录结构与 `precomputed/track_guided_sintel_clean_20/` 完全一致。区别是这些样本没有参与 `demo_fusion_outputs/sintel_clean_20/fusion_net_smoke.pth` 的训练，用于检查同一 scene 后续帧上的泛化能力。

### `precomputed/track_guided_sintel_clean_smoke/`

用途：真实 Sintel 最小 smoke 测试目录。最早使用 `alley_1` 的 2 对样本检查 manifest、GT 转换、FlowSeek、采样、CoTracker、对齐、栅格化和单样本/双样本训练是否能跑通。

## 4. 训练输出目录

### `demo_fusion_outputs/sintel_clean_20/`

- `fusion_net_smoke.pth`
  - 来源：`train_track_guided.py`。
  - 含义：使用 `precomputed/track_guided_sintel_clean_20/manifest.json` 训练 500 步得到的 FusionNet checkpoint。
  - 训练配置要点：`hidden_dim=32`，`lambda_flow=1.0`，`lambda_track=0.2`，`lambda_smooth=0.001`，`batch_size=1`。

- `train_history.json`
  - 来源：`train_track_guided.py`。
  - 含义：500 步训练历史，每步记录 `flow_l1`、`epe`、`initial_epe`、`track_l1`、`smoothness`、`total`。
  - 当前训练摘要：
    - 初始 total loss：`0.325839`
    - 最终 total loss：`0.297172`
    - 最优 total loss：`0.219942`，出现在 step `421`

### 其他训练 smoke 目录

- `demo_fusion_outputs/stage6_smoke_rerun/`
  - 来源：重新运行合成样例阶段六 smoke。
  - 含义：确认 FusionNet 训练链路稳定，loss 从 `0.049833` 降到 `0.039587`。

- `demo_fusion_outputs/sintel_clean_smoke_one_sample/`
  - 来源：真实 Sintel 单样本监督训练 smoke。
  - 含义：验证 `gt_flow.npy`、`valid.npy`、`F_0`、`G_track` 能被训练脚本正确读取。

- `demo_fusion_outputs/sintel_clean_smoke_2samples/`
  - 来源：真实 Sintel 两样本监督训练 smoke。
  - 含义：验证 DataLoader 能批量读取多个真实预计算样本。

## 5. 训练集内验证输出

目录：

```text
demo_fusion_outputs/sintel_clean_20_eval/
```

来源：`evaluate_track_guided.py` 使用训练好的 checkpoint `demo_fusion_outputs/sintel_clean_20/fusion_net_smoke.pth`，在训练 manifest `precomputed/track_guided_sintel_clean_20/manifest.json` 上评估。

### 指标文件

- `metrics.json`
  - 来源：`evaluate_track_guided.py`。
  - 含义：训练集内 20 个样本逐样本 EPE 和整体均值。
  - 当前结果：
    - `mean_initial_epe = 0.420434`
    - `mean_refined_epe = 0.397553`
    - `mean_epe_delta = -0.022881`
    - `num_improved = 20`
    - `num_worse = 0`

### 可视化样本目录

目录形如：

```text
demo_fusion_outputs/sintel_clean_20_eval/sample_0000/
...
demo_fusion_outputs/sintel_clean_20_eval/sample_0005/
```

每个目录里的文件含义：

- `refined_flow.npy`
  - 来源：`evaluate_track_guided.py` 调用 FusionNet。
  - 含义：模型输出的 refined flow，即 `F_refined`。

- `delta_flow.npy`
  - 来源：FusionNet 的 `delta_head`。
  - 含义：模型预测的残差光流 `Delta F`。

- `gate.npy`
  - 来源：FusionNet 的 `gate_head`。
  - 含义：门控图，控制 `Delta F` 对原始光流的影响强度。

- `initial_flow_vis.png`
  - 来源：对预计算 `F_0` 做光流颜色编码。
  - 含义：验证前的 FlowSeek 初始光流图。

- `refined_flow_vis.png`
  - 来源：对 `F_refined` 做光流颜色编码。
  - 含义：FusionNet 修正后的光流图。

- `gt_flow_vis.png`
  - 来源：对 Sintel GT flow 做光流颜色编码。
  - 含义：真实光流参考图。

- `initial_error.png`
  - 来源：计算 `||F_0 - F_gt||_2` 后做热力图。
  - 含义：FlowSeek 初始光流误差图，颜色越强误差越大。

- `refined_error.png`
  - 来源：计算 `||F_refined - F_gt||_2` 后做热力图。
  - 含义：FusionNet 修正后误差图。

- `improvement.png`
  - 来源：计算 `initial_error - refined_error`，只显示正向改善部分。
  - 含义：哪里 refined flow 比 initial flow 更接近 GT；颜色越强表示改善越大。

## 6. Held-Out 验证输出

目录：

```text
demo_fusion_outputs/sintel_clean_val10_eval/
```

来源：`evaluate_track_guided.py` 使用同一个训练 checkpoint，在未参与训练的验证 manifest `precomputed/track_guided_sintel_clean_val10/manifest.json` 上评估。

### 指标文件

- `metrics.json`
  - 来源：`evaluate_track_guided.py`。
  - 含义：held-out 验证集 10 个样本逐样本 EPE 和整体均值。
  - 当前结果：
    - `mean_initial_epe = 0.526242`
    - `mean_refined_epe = 0.512189`
    - `mean_epe_delta = -0.014054`
    - `num_improved = 10`
    - `num_worse = 0`

这说明当前模型不仅在训练集内有效，在同一 scene 后续未训练帧上也有小幅改善。因为验证集仍来自 `alley_1`，该结果还不能代表跨场景泛化。

### 可视化样本目录

目录形如：

```text
demo_fusion_outputs/sintel_clean_val10_eval/sample_0000/
...
demo_fusion_outputs/sintel_clean_val10_eval/sample_0005/
```

每个目录中的文件含义与训练集内验证一致：

- `initial_flow_vis.png`：验证样本的 FlowSeek 初始光流图。
- `refined_flow_vis.png`：FusionNet 修正后的光流图。
- `gt_flow_vis.png`：Sintel GT 光流图。
- `initial_error.png`：初始光流相对 GT 的误差热力图。
- `refined_error.png`：修正后光流相对 GT 的误差热力图。
- `improvement.png`：误差下降区域图。
- `refined_flow.npy`：修正后的光流数组。
- `delta_flow.npy`：预测残差。
- `gate.npy`：门控图。

## 7. 推荐查看顺序

如果只想快速确认整条方法是否有效，建议按下面顺序查看：

1. `demo_pair_outputs/stage1_smoke/flow_vis.png`
   - 确认 FlowSeek 初始光流推理链路正常。

2. `demo_sampler_outputs/stage3_smoke/sampled_points.png`
   - 确认自适应采样点覆盖图像，并偏向运动边界或高权重区域。

3. `demo_alignment_outputs/stage4_smoke/alignment_overlay.png`
   - 确认 CoTracker 稀疏位移和 FlowSeek 光流坐标系一致。

4. `demo_rasterizer_outputs/stage5_smoke/g_track_magnitude.png`
   - 确认轨迹先验已被正确写成图像网格张量。

5. `precomputed/track_guided_sintel_clean_20/alley_1/alley_1_frame_0001/flowseek/flow_vis.png`
   - 查看真实 Sintel 样本上的 FlowSeek 初始光流。

6. `precomputed/track_guided_sintel_clean_20/alley_1/alley_1_frame_0001/alignment/alignment_overlay.png`
   - 查看真实 Sintel 样本上的轨迹/光流对齐。

7. `demo_fusion_outputs/sintel_clean_20_eval/sample_0000/initial_error.png` 与 `refined_error.png`
   - 对比训练集内修正前后的误差。

8. `demo_fusion_outputs/sintel_clean_val10_eval/sample_0000/initial_error.png` 与 `refined_error.png`
   - 对比 held-out 验证集修正前后的误差。

9. `demo_fusion_outputs/sintel_clean_val10_eval/metrics.json`
   - 查看未参与训练样本上的定量改善。

## 8. 当前结论

- 合成 smoke：阶段一到阶段六全部跑通。
- Sintel 训练集 20 对：预计算、训练和训练集内验证全部跑通。
- Sintel held-out 验证集 10 对：全部样本 refined EPE 低于 initial EPE。
- 当前结果说明轨迹先验在小规模同 scene 实验中有效。
- 下一步如果要验证泛化能力，应使用不同 scene 作为验证集，例如用 `alley_1` 训练，用 `ambush_2`、`bamboo_1` 或其他 scene 做 held-out 验证。
