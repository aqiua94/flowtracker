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

## 8. 跨 Scene 小实验产物

本节记录一次更严格的跨 scene 阶段六小实验：使用 `bamboo_1` 训练，使用 `ambush_2` 验证。该实验用于判断基础 FusionNet 是否具备初步跨场景泛化能力，以及是否需要在大规模训练前添加安全机制。

### 训练预计算目录：`precomputed/track_guided_sintel_clean_bamboo1_train20/`

- 来源：`scripts/prepare_sintel_track_guided_manifest.py` 和 `scripts/precompute_track_guided_from_pairs.py`。
- 样本：Sintel `training/clean/bamboo_1` 前 20 对相邻帧。
- 目录结构与 `precomputed/track_guided_sintel_clean_20/` 一致，每个样本包含：
  - `flowseek/flow.npy`：FlowSeek 初始光流 `F_0`。
  - `flowseek/flow_vis.png`：初始光流可视化。
  - `sampler/sampled_points.png`：自适应采样点图。
  - `cotracker/track_0000.png`、`track_0001.png`：CoTracker 轨迹可视化。
  - `alignment/alignment_overlay.png`：CoTracker 稀疏位移与 FlowSeek 采样光流的对齐图。
  - `alignment/alignment_stats.json`：有效轨迹数和对齐 EPE 统计。
  - `rasterizer/g_track.npy`：训练用轨迹先验。
  - `rasterizer/g_track_magnitude.png`、`g_track_confidence.png`、`g_track_distance.png`：轨迹先验三类可视化。

### 验证预计算目录：`precomputed/track_guided_sintel_clean_ambush2_val10/`

- 来源：同上。
- 样本：Sintel `training/clean/ambush_2` 前 10 对相邻帧。
- 用途：跨 scene 验证。该 scene 存在明显大位移和遮挡，部分样本有效轨迹数较少，例如约 `67/128` 到 `113/128`，轨迹/FlowSeek 对齐 EPE 也明显高于 `alley_1` 和 `bamboo_1`。
- 典型文件含义与训练预计算目录一致。

### 训练输出：`demo_fusion_outputs/sintel_clean_bamboo1_train20/`

- `fusion_net_smoke.pth`
  - 来源：`train_track_guided.py`。
  - 含义：使用 `bamboo_1` 20 对样本训练 500 步得到的 FusionNet checkpoint。

- `train_history.json`
  - 来源：`train_track_guided.py`。
  - 含义：训练 loss 历史。该训练比 `alley_1` 更难，loss 有下降但波动明显。

### 训练 Scene 内评估：`demo_fusion_outputs/sintel_clean_bamboo1_train20_eval/`

- `metrics.json`
  - 来源：`evaluate_track_guided.py`。
  - 含义：在 `bamboo_1` 训练样本上评估 `F_0` 和 `F_refined`。
  - 当前结果：
    - `mean_initial_epe = 0.589425`
    - `mean_refined_epe = 0.542909`
    - `mean_epe_delta = -0.046516`
    - `num_improved = 20`
    - `num_worse = 0`

### 跨 Scene 评估：`demo_fusion_outputs/sintel_clean_bamboo1_to_ambush2_val10_eval/`

- `metrics.json`
  - 来源：`evaluate_track_guided.py`。
  - 含义：使用 `bamboo_1` 训练得到的 checkpoint，在未参与训练的 `ambush_2` 样本上评估。
  - 当前结果：
    - `mean_initial_epe = 3.370809`
    - `mean_refined_epe = 3.361017`
    - `mean_epe_delta = -0.009792`
    - `num_improved = 8`
    - `num_worse = 2`

该结果说明基础 FusionNet 已有一点跨 scene 泛化，但在大位移、遮挡、低有效轨迹比例场景下还不够稳。`ambush_2` 中有两个样本 refined EPE 高于 initial EPE，后续需要安全过滤或更强的轨迹建模。

### 新增安全机制代码

- `demo_track_prior_rasterizer.py`
  - 新增参数 `--endpoint_error` 和 `--max_endpoint_error`。
  - 功能：在栅格化 `G_track` 前，按阶段四得到的轨迹/FlowSeek 对齐 EPE 过滤明显 outlier 轨迹点。

- `scripts/precompute_track_guided_from_pairs.py`
  - 新增参数 `--max_alignment_epe`。
  - 功能：批量预计算时自动把 `alignment/endpoint_error.npy` 传入栅格化脚本，超过阈值的轨迹点不会写入 `G_track`。

- `scripts/prepare_sintel_track_guided_manifest.py`
  - 新增参数 `--max_pairs_per_scene`。
  - 功能：多 scene 训练时，每个 scene 取固定数量样本，避免某个 scene 因排序或数量问题占据过多训练样本。

## 9. 中等规模多 Scene 阶段六产物

本节记录第一次中等规模多 scene 阶段六训练：训练集使用 5 个 scene、每个 scene 20 对相邻帧，共 100 对；验证集使用 2 个未参与训练的 scene、每个 scene 10 对，共 20 对。该实验同时启用了阶段四 alignment EPE 过滤，阈值为 `max_alignment_epe = 3.0`。

### 训练预计算目录：`precomputed/track_guided_sintel_clean_multiscene_train100_safe/`

- 来源：`scripts/prepare_sintel_track_guided_manifest.py` 和 `scripts/precompute_track_guided_from_pairs.py`。
- 训练 scene：`alley_1`、`bamboo_1`、`bamboo_2`、`market_2`、`shaman_2`。
- 样本数：`100`。
- 安全过滤：`scripts/precompute_track_guided_from_pairs.py --max_alignment_epe 3.0`，栅格化前丢弃 `alignment/endpoint_error.npy > 3.0` 的轨迹点。
- 典型文件：
  - `manifest.json`：阶段六训练和评估读取的样本索引。
  - `manifest_pairs.json`：阶段一到阶段五批量预计算读取的帧对索引。
  - 每个样本目录中的 `flowseek/flow.npy`：FlowSeek 初始光流。
  - 每个样本目录中的 `alignment/endpoint_error.npy`：轨迹位移与 FlowSeek 采样光流的逐点 EPE，用于安全过滤。
  - 每个样本目录中的 `rasterizer/g_track.npy`：过滤后的五通道轨迹先验。

训练预计算中观察到 `market_2` 比较困难，部分样本 alignment 最大 EPE 可超过 `30`，过滤后有效栅格化点数会从约 `118/128` 降到约 `99/128`。这说明安全过滤确实挡住了明显不一致的轨迹 outlier。

### 验证预计算目录：`precomputed/track_guided_sintel_clean_multiscene_val20_safe/`

- 来源：同上。
- 验证 scene：`ambush_2`、`temple_2`。
- 样本数：`20`。
- 用途：跨 scene 压力测试，不参与训练。
- 典型现象：
  - `ambush_2` 大位移、遮挡明显，部分样本有效轨迹比例只有约 `0.52-0.88`。
  - 过滤后的轨迹 prior 覆盖率明显低于训练集：训练集平均约 `0.000262`，验证集平均约 `0.000173`。

### 训练输出：`demo_fusion_outputs/sintel_clean_multiscene_train100_safe/`

- `fusion_net_smoke.pth`
  - 来源：`train_track_guided.py`。
  - 含义：使用 100 对多 scene 样本训练 1000 step 得到的阶段六 FusionNet checkpoint。
  - 训练配置要点：`hidden_dim=32`，`lambda_flow=1.0`，`lambda_track=0.2`，`lambda_smooth=0.001`，`batch_size=1`。

- `train_history.json`
  - 来源：`train_track_guided.py`。
  - 含义：1000 step 训练历史。由于每次日志对应随机样本，loss 有明显波动，不能只看第一行和最后一行判断泛化。

### 训练集评估：`demo_fusion_outputs/sintel_clean_multiscene_train100_safe_eval/`

- `metrics.json`
  - 来源：`evaluate_track_guided.py`。
  - 当前结果：
    - `mean_initial_epe = 0.629587`
    - `mean_refined_epe = 0.601544`
    - `mean_epe_delta = -0.028043`
    - `num_improved = 95`
    - `num_worse = 5`

该结果说明中等规模训练在训练分布内有效，绝大多数样本 refined flow 优于 FlowSeek 初始光流。

### 原始跨 Scene 验证：`demo_fusion_outputs/sintel_clean_multiscene_val20_safe_eval/`

- `metrics.json`
  - 来源：`evaluate_track_guided.py`，未启用推理时回退。
  - 当前结果：
    - `mean_initial_epe = 2.209927`
    - `mean_refined_epe = 2.221544`
    - `mean_epe_delta = 0.011617`
    - `num_improved = 4`
    - `num_worse = 16`

该结果说明仅靠训练时/预计算时的 alignment EPE 过滤还不够。模型在训练分布内有效，但遇到 `ambush_2` 和 `temple_2` 这种低 prior 覆盖率、强运动验证集时，直接应用 refinement 会产生轻微平均劣化。

### 推理时安全回退代码：`evaluate_track_guided.py`

- 新增参数：
  - `--safe_refinement`：开启推理时安全回退。
  - `--min_prior_coverage`：要求 `G_track` 置信度通道非零像素比例达到阈值，否则回退到 FlowSeek 初始光流。
  - `--min_gate_mean`：门控均值低于阈值时回退，当前实验未使用。
  - `--max_delta_mean_abs`：预测残差过大时回退，当前实验未使用。

- 新增指标：
  - `prior_coverage`：每个样本 `G_track[..., 2] > 0` 的像素比例。
  - `used_fallback`：该样本是否触发回退。
  - `fallback_reason`：回退原因。
  - `num_fallback`：整体触发回退的样本数量。

### 宽松回退评估：`demo_fusion_outputs/sintel_clean_multiscene_val20_safe_eval_guarded/`

- 配置：`--safe_refinement --min_prior_coverage 0.00022`。
- 当前结果：
  - `mean_initial_epe = 2.209927`
  - `mean_refined_epe = 2.211249`
  - `mean_epe_delta = 0.001322`
  - `num_fallback = 19`
  - `num_improved = 0`
  - `num_worse = 1`

该阈值挡住了 19/20 个低覆盖验证样本，只剩 1 个样本仍有轻微劣化。

### 严格回退评估：`demo_fusion_outputs/sintel_clean_multiscene_val20_safe_eval_guarded_strict/`

- 配置：`--safe_refinement --min_prior_coverage 0.000225`。
- 当前结果：
  - `mean_initial_epe = 2.209927`
  - `mean_refined_epe = 2.209927`
  - `mean_epe_delta = 0.000000`
  - `num_fallback = 20`
  - `num_improved = 0`
  - `num_worse = 0`

该结果说明推理时安全回退可以保证当前困难验证集上不劣化。但它也比较保守：在该验证集上全部回退，等价于暂时不使用 FusionNet refinement。

### 严格回退训练集评估：`demo_fusion_outputs/sintel_clean_multiscene_train100_safe_eval_guarded_strict/`

- 配置：同样使用 `--safe_refinement --min_prior_coverage 0.000225`。
- 当前结果：
  - `mean_initial_epe = 0.629587`
  - `mean_refined_epe = 0.605227`
  - `mean_epe_delta = -0.024360`
  - `num_improved = 87`
  - `num_worse = 5`

该结果说明严格回退没有完全抹掉训练分布内的收益，但会比无回退版本更保守。

## 10. 阶段七轨迹注意力产物

阶段七第一版采用可靠性引导的跨轨迹注意力增强。它不是端到端训练的 Transformer，而是一个可解释的轻量模块：根据轨迹点之间的空间距离、运动相似度、置信度和 alignment EPE 形成注意力权重，让低可靠轨迹向附近高可靠轨迹的运动共识靠拢，高可靠轨迹基本保持原始 CoTracker 位移。

### 新增代码：`core/track_guidance/trajectory_attention.py`

- `trajectory_attention_enhance(...)`
  - 输入：阶段四输出的 `points.npy`、`track_flow.npy`、`valid_mask.npy`、`confidence.npy`，可选 `endpoint_error.npy`。
  - 输出：增强后的 `enhanced_track_flow`、`enhanced_confidence`、`enhanced_valid_mask`、`attention` 和统计信息。
  - 功能：对每个轨迹点计算跨点注意力。注意力源点由可靠性加权，可靠性来自 `valid_mask * confidence * exp(-endpoint_error / endpoint_error_scale)`。
  - 设计目的：减少 outlier 轨迹对阶段六的误导，并为低可靠点提供更平滑的局部运动估计。

- `TrajectoryAttentionStats`
  - 输出统计结构。
  - 字段包括输入/输出有效点数、平均置信度、平均可靠性、平均轨迹位移改变量和最大轨迹位移改变量。

### 新增脚本：`demo_trajectory_attention.py`

- 来源：阶段七 demo 入口。
- 功能：
  - 读取阶段四 alignment 产物。
  - 调用 `trajectory_attention_enhance(...)`。
  - 保存增强轨迹和注意力矩阵。
  - 生成两张可视化：
    - `attention_matrix.png`：轨迹点之间的注意力权重矩阵。
    - `flow_delta_points.png`：每个轨迹点被阶段七修改的位移幅值。

输出文件：

```text
enhanced_track_flow.npy
enhanced_confidence.npy
enhanced_valid_mask.npy
attention.npy
trajectory_attention_stats.json
attention_matrix.png
flow_delta_points.png
```

### 批处理接入：`scripts/precompute_track_guided_from_pairs.py`

新增参数：

- `--use_trajectory_attention`
  - 在阶段四 alignment 后、阶段五 rasterizer 前插入阶段七轨迹注意力。

- `--attention_spatial_sigma`
  - 空间注意力尺度，默认 `96.0`。

- `--attention_motion_sigma`
  - 运动相似度注意力尺度，默认 `8.0`。

- `--attention_endpoint_error_scale`
  - alignment EPE 可靠性衰减尺度，默认 `3.0`。

- `--attention_self_weight`
  - 自身轨迹保留偏置，默认 `1.0`。

- `--attention_promote_invalid`
  - 是否允许低置信/不可见点被注意力结果重新提升为有效点。当前默认关闭，避免盲目增加错误先验。

启用后，每个样本会新增：

```text
alignment/trajectory_attention/enhanced_track_flow.npy
alignment/trajectory_attention/enhanced_confidence.npy
alignment/trajectory_attention/enhanced_valid_mask.npy
alignment/trajectory_attention/attention.npy
alignment/trajectory_attention/trajectory_attention_stats.json
alignment/trajectory_attention/attention_matrix.png
alignment/trajectory_attention/flow_delta_points.png
```

随后 `rasterizer/g_track.npy` 会使用增强后的 `enhanced_track_flow.npy`、`enhanced_confidence.npy` 和 `enhanced_valid_mask.npy` 生成。

### 合成 Smoke：`demo_trajectory_attention_outputs/stage7_smoke/`

- 来源：使用 `demo_alignment_outputs/stage4_smoke/` 的合成平移样本运行 `demo_trajectory_attention.py`。
- 当前结果：
  - 输入有效点：`212/256`
  - 输出有效点：`212/256`
  - 平均可靠性：`0.974131`
  - 平均位移改变量：`0.002986 px`
  - 最大位移改变量：`0.110568 px`

解释：合成样本本身非常可靠，阶段七只做极小修正，说明模块不会在高可靠轨迹上过度改动。

### 合成 Smoke 栅格化：`demo_trajectory_attention_outputs/stage7_smoke_rasterizer/`

- 来源：使用阶段七增强后的 `enhanced_track_flow.npy` 运行 `demo_track_prior_rasterizer.py`。
- 输出：
  - `g_track.npy`
  - `g_track_stats.json`
  - `g_track_magnitude.png`
  - `g_track_confidence.png`
  - `g_track_distance.png`
- 当前结果：
  - `G_track shape = (216, 384, 5)`
  - 有效输入点：`212/256`
  - 栅格化像素：`212`

该结果说明阶段七输出可以无缝回到阶段五栅格化格式，也就是可以直接作为阶段六 FusionNet 输入。

### 困难样本 Demo：`demo_trajectory_attention_outputs/ambush2_frame0002/`

- 来源：使用 `precomputed/track_guided_sintel_clean_multiscene_val20_safe/ambush_2/ambush_2_frame_0002/alignment/` 运行阶段七。
- 当前结果：
  - 输入有效点：`67/128`
  - 输出有效点：`67/128`
  - 平均可靠性：`0.636699`
  - 平均位移改变量：`2.498192 px`
  - 最大位移改变量：`19.174435 px`

与 FlowSeek 采样光流的对齐 EPE 对比：

- 原始轨迹：
  - mean `5.490172`
  - median `1.184931`
  - max `77.626709`
- 阶段七增强轨迹：
  - mean `5.058599`
  - median `1.280508`
  - max `69.529526`

解释：困难样本中阶段七会明显修改低可靠轨迹，并降低平均和最大 outlier 对齐误差。median 略升，说明第一版更偏向压制大 outlier，而不是保证每个点都更贴近 FlowSeek。后续如果接入 GT 或训练式注意力，可以继续优化该权衡。

### 真实 Sintel 批处理 Smoke：`precomputed/track_guided_sintel_clean_stage7_smoke/`

- 来源：
  - `scripts/prepare_sintel_track_guided_manifest.py`
  - `scripts/precompute_track_guided_from_pairs.py --use_trajectory_attention`
- 样本：`ambush_2 frame_0001 -> frame_0002`。
- 当前结果：
  - alignment 有效轨迹：`81/128`
  - alignment EPE：mean `2.3997`，median `1.1169`，max `48.2313`
  - 阶段七平均可靠性：`0.629381`
  - 阶段七平均位移改变量：`2.467888 px`
  - 阶段七最大位移改变量：`34.389809 px`
  - 栅格化前叠加 `max_alignment_epe = 3.0` 后，有效输入点：`70/128`
  - 输出 `rasterizer/g_track.npy` 尺寸：`(436, 1024, 5)`

该目录证明阶段七已经接入完整预计算流水线。后续要验证它是否真正帮助阶段六，需要用 `--use_trajectory_attention` 重新生成训练/验证预计算目录，再训练阶段六 FusionNet 并比较 EPE。

## 11. 中等规模 Trajectory Attention 阶段六产物

本节记录使用 `--use_trajectory_attention` 重新生成的中等规模多 scene 预计算，并在该预计算上重训阶段六 FusionNet。训练/验证 split 与第 9 节 baseline 保持一致，便于直接比较：

- 训练集：`alley_1`、`bamboo_1`、`bamboo_2`、`market_2`、`shaman_2`，每个 scene 20 对，共 `100` 对。
- 验证集：`ambush_2`、`temple_2`，每个 scene 10 对，共 `20` 对。
- 预计算参数：`flowseek_max_size=384`、`num_points=128`、`max_alignment_epe=3.0`、`--use_trajectory_attention`、`attention_spatial_sigma=128`、`attention_motion_sigma=24`、`attention_endpoint_error_scale=3.0`。

### 训练预计算：`precomputed/track_guided_sintel_clean_multiscene_train100_attn/`

- 来源：`scripts/prepare_sintel_track_guided_manifest.py` 生成 manifest，随后 `scripts/precompute_track_guided_from_pairs.py --use_trajectory_attention` 逐样本运行阶段一到阶段五，并在阶段四 alignment 和阶段五 rasterizer 之间插入阶段七。
- 关键文件：
  - `manifest.json`：阶段六训练和评估读取的 `100` 个样本索引。
  - `manifest_pairs.json`：预计算脚本读取的相邻帧对。
  - 每个样本目录下的 `alignment/trajectory_attention/enhanced_track_flow.npy`：阶段七增强后的稀疏轨迹位移。
  - 每个样本目录下的 `alignment/trajectory_attention/enhanced_confidence.npy`：阶段七重估后的轨迹可靠性。
  - 每个样本目录下的 `alignment/trajectory_attention/enhanced_valid_mask.npy`：阶段七输出有效点掩码。
  - 每个样本目录下的 `alignment/trajectory_attention/trajectory_attention_stats.json`：该样本的有效点数、平均可靠性、平均/最大位移改变量。
  - 每个样本目录下的 `rasterizer/g_track.npy`：使用阶段七增强轨迹生成的阶段六输入先验。

训练集中大多数样本的阶段七修正较温和；`market_2` 中若干帧出现更明显修正，例如部分样本平均位移改变量约 `1.0-2.1 px`，最大改变量可超过 `25 px`，说明阶段七主要在疑似 outlier 或局部运动不一致区域发挥作用。

### 验证预计算：`precomputed/track_guided_sintel_clean_multiscene_val20_attn/`

- 来源：同样使用 `scripts/precompute_track_guided_from_pairs.py --use_trajectory_attention`，但 scene 换为未参与训练的 `ambush_2` 和 `temple_2`。
- 关键文件结构与训练预计算一致。
- 典型困难样本：
  - `ambush_2_frame_0001`：阶段七平均可靠性 `0.629381`，平均位移改变量 `2.467888 px`，最大位移改变量 `34.389809 px`，栅格化有效点 `70/128`。
  - `ambush_2_frame_0002`：阶段七平均可靠性 `0.636699`，平均位移改变量 `2.498192 px`，最大位移改变量 `19.174435 px`，栅格化有效点 `58/128`。
  - `temple_2_frame_0010`：阶段七平均可靠性 `0.632488`，平均位移改变量 `2.299739 px`，最大位移改变量 `28.536127 px`，栅格化有效点 `93/128`。

解释：验证集里 `ambush_2` 和后段 `temple_2` 的 attention 修正幅度明显大于多数训练样本，说明阶段七确实感知到了低可靠或强运动样本；但这也会放大训练/验证分布差异，需要通过评估确认是否真正提升最终 EPE。

### 阶段六训练：`demo_fusion_outputs/sintel_clean_multiscene_train100_attn/`

- 来源：使用 `precomputed/track_guided_sintel_clean_multiscene_train100_attn/manifest.json` 训练 `1000` step。
- 关键文件：
  - `fusion_net_smoke.pth`：使用 trajectory attention 预计算重训得到的阶段六 FusionNet checkpoint。
  - `train_log.json`：训练过程 loss 历史。
- 训练日志摘要：
  - step `0001` total loss `0.370710`
  - step `0500` total loss `0.496725`
  - step `1000` total loss `0.455053`

训练 loss 有波动，因此该实验主要以完整 train/val EPE 作为判断依据。

### 原始评估：`demo_fusion_outputs/sintel_clean_multiscene_train100_attn_eval/`

- 来源：使用 trajectory attention checkpoint 在 train100 上评估，不启用推理回退。
- 结果：
  - initial EPE：`0.629587`
  - refined EPE：`0.598920`
  - delta：`-0.030667`
  - improved/worse：`95/5`

与第 9 节 baseline train100 原始评估相比，trajectory attention 版本从 `0.601544` 降到 `0.598920`，训练分布内略有提升。

### 原始跨 Scene 验证：`demo_fusion_outputs/sintel_clean_multiscene_val20_attn_eval/`

- 来源：使用同一个 trajectory attention checkpoint 在 held-out val20 上评估，不启用推理回退。
- 结果：
  - initial EPE：`2.209927`
  - refined EPE：`2.227108`
  - delta：`+0.017182`
  - improved/worse：`2/18`

与第 9 节 baseline val20 原始评估相比，trajectory attention 版本从 `2.221544` 变为 `2.227108`，跨 scene 验证更差。这说明第一版阶段七增强了训练内收益，但还没有解决阶段六的跨 scene 泛化问题。

### 严格安全评估：`*_attn_eval_guarded_strict/`

- 来源：使用 `evaluate_track_guided.py --safe_refinement --min_prior_coverage 0.000225` 评估，与第 9 节严格安全阈值一致。
- 训练集结果：`demo_fusion_outputs/sintel_clean_multiscene_train100_attn_eval_guarded_strict/`
  - initial EPE：`0.629587`
  - refined EPE：`0.602582`
  - delta：`-0.027005`
  - improved/worse：`87/5`
  - fallback：`8/100`
- 验证集结果：`demo_fusion_outputs/sintel_clean_multiscene_val20_attn_eval_guarded_strict/`
  - initial EPE：`2.209927`
  - refined EPE：`2.209927`
  - delta：`0.000000`
  - improved/worse：`0/0`
  - fallback：`20/20`

严格安全机制能完全挡住 val20 上的退化，但代价是验证集全部回退到原始 FlowSeek，没有产生实际提升。

### 与 Baseline 对比

| 实验 | Split | Guard | Refined EPE | Delta | Improved/Worse | Fallback |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline | train100 | no | `0.601544` | `-0.028043` | `95/5` | - |
| trajectory attention | train100 | no | `0.598920` | `-0.030667` | `95/5` | `0/100` |
| baseline | val20 | no | `2.221544` | `+0.011617` | `4/16` | - |
| trajectory attention | val20 | no | `2.227108` | `+0.017182` | `2/18` | `0/20` |
| baseline | train100 | strict | `0.605227` | `-0.024360` | `87/5` | `8/100` |
| trajectory attention | train100 | strict | `0.602582` | `-0.027005` | `87/5` | `8/100` |
| baseline | val20 | strict | `2.209927` | `0.000000` | `0/0` | `20/20` |
| trajectory attention | val20 | strict | `2.209927` | `0.000000` | `0/0` | `20/20` |

结论：`--use_trajectory_attention` 对训练分布内阶段六有小幅正收益，但第一版还不能直接支持大批量训练，因为 held-out scene 原始验证更差。当前更稳的路线是保留阶段七预计算能力和推理安全回退，同时继续增强阶段六的可靠性建模，而不是只扩大数据规模。

## 12. 当前结论

- 合成 smoke：阶段一到阶段六全部跑通。
- Sintel 训练集 20 对：预计算、训练和训练集内验证全部跑通。
- Sintel held-out 验证集 10 对：全部样本 refined EPE 低于 initial EPE。
- 跨 scene 小实验：`bamboo_1 -> ambush_2` 平均 EPE 有小幅改善，但存在 `2/10` 个样本变差。
- 中等规模多 scene 训练：训练集内有效，`100` 对样本中 `95/100` 改善；但原始跨 scene 验证 `20` 对样本中 `16/20` 变差，说明阶段六基础 FusionNet 的跨 scene 泛化仍不稳定。
- 中等规模 trajectory attention 训练：训练集内 EPE 从 baseline 的 `0.601544` 降到 `0.598920`，但 held-out val20 从 `2.221544` 升到 `2.227108`，说明第一版阶段七增强了训练内收益但没有改善跨 scene 泛化。
- 安全机制：预计算阶段的 alignment EPE 过滤能挡住轨迹 outlier；推理阶段的 prior coverage 回退能在困难验证集上避免劣化。
- 当前建议：不要直接进入大批量阶段六训练。应保留阶段七预计算与推理回退，并优先增强阶段六的可靠性输入、coverage/gate 约束或训练时安全损失，让模型学会在低覆盖、强遮挡、大位移 scene 中少改或不改。
