# 基于点跟踪改进光流算法的实施步骤

## 1. 实施总原则

本方案的实现应采用分阶段验证方式，不建议一开始就端到端联合训练。推荐顺序是：

1. 跑通 FlowSeek，确认可以稳定输出初始稠密光流。
2. 跑通 CoTracker3，确认可以对指定稀疏点输出跨帧轨迹、可见性和置信度。
3. 实现自适应采样，把 FlowSeek 的光流结果转化为 CoTracker3 的输入点。
4. 对齐 FlowSeek 光流与 CoTracker3 轨迹，构建稀疏轨迹先验。
5. 先训练一个后处理式融合网络 FusionNet，验证轨迹先验是否能提升光流。
6. 再加入跨轨迹注意力模块，增强遮挡和低置信轨迹。
7. 最后考虑把轨迹先验深度接入 FlowSeek 的迭代更新模块，做端到端微调。

该路线可以降低调试难度，每个阶段都有独立可验收的结果。

## 2. 阶段一：跑通 FlowSeek

### 2.1 目标

确认当前 FlowSeek 代码可以完成基本推理，输出相邻帧之间的稠密光流。

### 2.2 需要做的事

1. 配置 FlowSeek 运行环境。
2. 下载或放置 FlowSeek 所需模型权重。
3. 准备一小段测试视频或连续图片序列。
4. 运行现有 `demo.py`，得到初始光流结果。
5. 保存光流结果为 `.flo`、`.npy` 或可视化图片。

当前阶段建议优先使用两帧图片推理脚本：

```text
demo_pair.py
```

该脚本不依赖 Sintel、KITTI、Spring 等数据集，只需要两张连续帧和官方公开权重，更适合阶段一快速验收。

### 2.3 产出

- 可运行的 FlowSeek 推理命令。
- 一组输入帧。
- 对应的初始稠密光流 `F_0^{t -> t+1}`。
- 光流可视化结果。

新增阶段一推理入口：

```text
demo_pair.py
```

输出：

```text
demo_pair_outputs/flow.npy
demo_pair_outputs/flow_vis.png
```

### 2.4 验收标准

- demo 可以完整运行，不报模型加载、显存或依赖错误。
- 输出光流尺寸与输入图像尺寸一致。
- 光流可视化中主要运动方向合理。
- 对静态区域，光流幅值应整体较小。

### 2.5 注意事项

- 先使用少量帧测试，避免一开始处理长视频导致问题难以定位。
- 如果 FlowSeek 依赖 Depth Anything v2 或其他深度模型，需要先确认权重路径和配置文件正确。
- 保存中间特征不是第一阶段必须项，先确保最终光流可用。

### 2.6 官方权重放置要求

以 `config/eval/flowseek-T.json` 为例，最小推理闭环需要：

```text
weights/depth_anything_v2_vits.pth
weights/flowseek_T_CT.pth
```

其中：

- `depth_anything_v2_vits.pth` 对应 Depth Anything v2 small。
- `flowseek_T_CT.pth` 对应 FlowSeek-T 官方公开权重。

官方脚本 `scripts/get_weights.sh` 会下载多组 T/M 权重。阶段一只需要先下载上述两个文件即可。

当前本机阶段一已使用以下实际权重完成烟雾测试：

```text
weights/depth_anything_v2_vits.pth
weights/flowseek_T_TartanC.pth
```

其中 `flowseek_T_TartanC.pth` 也是官方公开 FlowSeek-T 权重，可用于阶段一验证推理链路。

### 2.7 阶段一推荐运行命令

准备两张连续帧后运行：

```shell
python demo_pair.py \
  --cfg config/eval/flowseek-T.json \
  --model weights/flowseek_T_CT.pth \
  --image1 path/to/frame_0001.png \
  --image2 path/to/frame_0002.png \
  --output_dir demo_pair_outputs \
  --max_size 960
```

运行成功后检查：

```text
demo_pair_outputs/flow.npy
demo_pair_outputs/flow_vis.png
```

### 2.8 当前执行状态

当前已完成：

- 阅读 FlowSeek `README.md`、`demo.py`、`config/eval/flowseek-T.json` 和模型加载逻辑。
- 确认 FlowSeek-T 使用 `da_size = vits`，对应权重路径为 `weights/depth_anything_v2_vits.pth`。
- 新增 `demo_pair.py`，用于两帧图片直接推理并保存 `.npy` 光流与可视化图。
- 创建 conda 环境 `flowseek`，Python 版本为 3.10。
- 由于当前 GPU 为 RTX 5090，使用 `torch==2.8.0` 与 `torchvision==0.23.0` 替代官方 README 中较旧的 `torch==2.4.1`，以保证 CUDA 兼容性。
- 使用阿里云 PyPI 镜像安装其余依赖，包括 `gdown`、`h5py`、`opencv-python`、`scikit-learn`、`huggingface_hub`。
- 使用合成平移测试帧完成 `demo_pair.py` 烟雾测试。

当前未完成：

- 尚未使用真实视频连续帧进行定性验收。

下一步：

- 准备真实视频中的两张连续测试帧。
- 运行 `demo_pair.py` 生成真实样例的 `flow.npy` 和 `flow_vis.png`。
- 检查真实样例光流方向和运动幅值是否合理。

### 2.9 当前阶段一烟雾测试结果

测试输入：

```text
demo_pair_inputs/frame_0001.png
demo_pair_inputs/frame_0002.png
```

测试命令：

```shell
/root/miniconda3/envs/flowseek/bin/python demo_pair.py \
  --cfg config/eval/flowseek-T.json \
  --model weights/flowseek_T_TartanC.pth \
  --image1 demo_pair_inputs/frame_0001.png \
  --image2 demo_pair_inputs/frame_0002.png \
  --output_dir demo_pair_outputs/stage1_smoke \
  --max_size 384
```

输出文件：

```text
demo_pair_outputs/stage1_smoke/flow.npy
demo_pair_outputs/stage1_smoke/flow_vis.png
```

输出光流尺寸：

```text
(216, 384, 2)
```

该烟雾测试由第一帧水平平移 12 像素生成第二帧，FlowSeek 输出统计为：

```text
u_mean  = 11.9948
v_mean  = 0.0253
mag_mean = 11.9948
```

说明阶段一的模型加载、CUDA 推理、光流输出和可视化保存链路已经跑通。

## 3. 阶段二：跑通 CoTracker3

### 3.1 目标

确认 CoTracker3 可以接收视频帧和指定查询点，并输出点轨迹、可见性和置信度。

### 3.2 需要做的事

1. 配置 CoTracker3 运行环境。
2. 下载或放置 CoTracker3 模型权重。
3. 使用同一段视频帧作为输入。
4. 先手工指定少量点，例如图像中心点、角点、运动物体上的点。
5. 运行 CoTracker3 推理，得到轨迹结果。
6. 将轨迹绘制到视频或图片序列上，检查跟踪是否合理。

当前阶段二优先使用 PyTorch Hub 加载官方 CoTracker3 offline 模型：

```python
torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
```

该方式会自动缓存官方仓库和 `scaled_offline.pth` 权重。

### 3.3 产出

- 可运行的 CoTracker3 推理命令。
- 查询点输入格式说明。
- 点轨迹 `Q in R^{T x N x 2}`。
- 可见性标签 `V in {0,1}^{T x N}`。
- 跟踪置信度 `C in R^{T x N}`。
- 轨迹可视化视频或图片。

当前新增阶段二推理入口：

```text
demo_cotracker_pair.py
```

输出：

```text
demo_cotracker_outputs/stage2_smoke/tracks.npy
demo_cotracker_outputs/stage2_smoke/visibility.npy
demo_cotracker_outputs/stage2_smoke/confidence.npy
demo_cotracker_outputs/stage2_smoke/queries.csv
demo_cotracker_outputs/stage2_smoke/track_*.png
```

说明：

- `tracks.npy`：形状为 `(T, N, 2)`，坐标顺序为 `(x, y)`。
- `visibility.npy`：形状为 `(T, N)`，bool 类型。
- `confidence.npy`：第一版暂用 visibility 的 float 版本占位，后续若需要未阈值化置信度，需要进一步读取 CoTracker 内部输出。
- `queries.csv`：查询点格式为 `[t, x, y]`。

### 3.4 验收标准

- CoTracker3 可以对指定点输出完整轨迹。
- 可见点轨迹应跟随物体运动。
- 遮挡或离开画面的点，可见性应下降。
- 输出坐标与输入图像坐标系一致。

### 3.5 注意事项

- 需要明确 CoTracker3 的坐标顺序是 `(x, y)` 还是 `(y, x)`。
- 需要确认输入图像是否经过 resize，若有 resize，输出轨迹要能映射回原图尺寸。
- 如果 CoTracker3 使用滑动窗口推理，需要记录窗口长度和重叠方式。

当前已确认：

- CoTracker3 手工查询点输入格式为 `(t, x, y)`。
- CoTracker3 输出轨迹坐标格式为 `(x, y)`。
- 当前使用 offline 模型，官方权重缓存位置为 PyTorch Hub cache。
- 当前 checkpoint 为 `scaled_offline.pth`，本机缓存路径为 `/root/.cache/torch/hub/checkpoints/scaled_offline.pth`。

### 3.6 阶段二当前烟雾测试结果

测试输入复用阶段一的合成平移帧：

```text
demo_pair_inputs/frame_0001.png
demo_pair_inputs/frame_0002.png
```

测试命令：

```shell
/root/miniconda3/envs/flowseek/bin/python demo_cotracker_pair.py \
  --frames demo_pair_inputs/frame_0001.png demo_pair_inputs/frame_0002.png \
  --output_dir demo_cotracker_outputs/stage2_smoke \
  --max_size 384 \
  --device cuda
```

默认手工点：

```text
[0,  96,  54]
[0, 192, 108]
[0, 288, 108]
[0, 192, 162]
```

输出轨迹尺寸：

```text
tracks:     (2, 4, 2)
visibility: (2, 4)
```

该测试中第二帧由第一帧水平平移 12 像素得到。CoTracker3 输出的四个点位移为：

```text
[12.0555, -0.0090]
[12.0987,  0.0213]
[12.0317,  0.0233]
[12.1258,  0.0083]
```

平均位移：

```text
[12.0779, 0.0110]
```

说明阶段二的模型加载、CUDA 推理、手工点查询、轨迹保存和轨迹可视化链路已经跑通。

## 4. 阶段三：实现自适应稀疏点采样

### 4.1 目标

利用 FlowSeek 初始光流，从图像中自动选择适合点跟踪的稀疏点。

### 4.2 需要做的事

1. 读取 FlowSeek 输出的初始光流。
2. 计算光流梯度幅值。
3. 计算光流运动幅值。
4. 可选：计算图像边缘响应。
5. 融合得到采样权重图。
6. 将图像划分为网格，在每个网格中按权重分配采样点。
7. 加入最小点间距或 NMS，避免点过密。
8. 输出 CoTracker3 可直接使用的查询点。

### 4.3 产出

- `sampler.py`。
- 采样点坐标 `P_t = {p_i^t}`。
- 每个点的采样权重。
- 采样点可视化图。

当前新增阶段三模块：

```text
core/track_guidance/sampler.py
demo_adaptive_sampler.py
```

当前输出：

```text
demo_sampler_outputs/stage3_smoke/points.npy
demo_sampler_outputs/stage3_smoke/point_weights.npy
demo_sampler_outputs/stage3_smoke/sampling_weight.npy
demo_sampler_outputs/stage3_smoke/sampling_weight.png
demo_sampler_outputs/stage3_smoke/sampled_points.png
demo_sampler_outputs/stage3_smoke/queries.csv
```

说明：

- `points.npy`：形状为 `(N, 2)`，坐标顺序为 `(x, y)`。
- `point_weights.npy`：每个采样点对应的采样权重。
- `sampling_weight.npy`：完整图像采样权重图。
- `queries.csv`：CoTracker3 可直接读取的查询点，列格式为 `[t, x, y]`。

### 4.4 验收标准

- 采样点覆盖图像主要区域。
- 运动边界和大位移区域点更密集。
- 静态背景仍保留少量点。
- 点数量可由配置控制，例如 512、1024、2048。
- 采样过程不依赖人工标注。

### 4.5 建议先实现的简单版本

第一版可以先不做复杂策略，只实现：

- 光流梯度权重。
- 网格内 top-k 采样。
- 每格最少点数。
- 全局最大点数限制。

等主流程跑通后，再加入运动幅值、图像边缘和概率采样。

当前已实现第一版自适应采样：

- 光流梯度权重。
- 光流运动幅值权重。
- 可选图像边缘权重。
- 按固定像素大小的网格分块采样。
- 每格最少点数和最多点数限制。
- 全局点数预算。
- 最小点间距约束。
- 采样点到 CoTracker3 查询格式转换。

### 4.6 阶段三当前烟雾测试结果

测试输入：

```text
demo_pair_outputs/stage1_smoke/flow.npy
demo_pair_inputs/frame_0001.png
```

测试命令：

```shell
/root/miniconda3/envs/flowseek/bin/python demo_adaptive_sampler.py \
  --flow demo_pair_outputs/stage1_smoke/flow.npy \
  --image demo_pair_inputs/frame_0001.png \
  --output_dir demo_sampler_outputs/stage3_smoke \
  --num_points 256 \
  --cell_size 32 \
  --min_points_per_cell 1 \
  --max_points_per_cell 8 \
  --min_distance 4 \
  --flow_gradient_weight 0.7 \
  --flow_magnitude_weight 0.3
```

输出统计：

```text
points:  (256, 2)
queries: (256, 3)
x 覆盖 12 个网格 bin
y 覆盖 7 个网格 bin
权重范围 [0.2180, 1.0000]
平均权重 0.5946
```

采样点范围：

```text
x=[0.0, 383.0]
y=[0.0, 215.0]
```

为了确认阶段三输出可以直接进入阶段二，已将 `queries.csv` 输入 CoTracker3：

```shell
/root/miniconda3/envs/flowseek/bin/python demo_cotracker_pair.py \
  --frames demo_pair_inputs/frame_0001.png demo_pair_inputs/frame_0002.png \
  --queries demo_sampler_outputs/stage3_smoke/queries.csv \
  --output_dir demo_cotracker_outputs/stage3_sampler_check \
  --max_size 384 \
  --device cuda
```

CoTracker3 输出：

```text
tracks:     (2, 256, 2)
visibility: (2, 256)
mean displacement frame0->frame1: [11.8877, 0.0155]
```

说明阶段三的采样点输出、查询格式和 CoTracker3 联通性已经跑通。

## 5. 阶段四：对齐 FlowSeek 光流与 CoTracker3 轨迹

### 5.1 目标

将 CoTracker3 的稀疏轨迹转换为与 FlowSeek 稠密光流同一坐标系下的稀疏运动先验。

### 5.2 需要做的事

1. 使用自适应采样点作为 CoTracker3 查询点。
2. 运行 CoTracker3 得到跨帧轨迹。
3. 对相邻帧计算轨迹位移：

```text
F_track^{t -> t+1}(p_i^t) = Q_{t+1,i} - Q_{t,i}
```

4. 检查轨迹位移与 FlowSeek 光流在采样点处是否大致一致。
5. 过滤不可见或低置信轨迹点。
6. 保存每对相邻帧的稀疏轨迹先验。

### 5.3 产出

- `cotracker_wrapper.py`。
- 稀疏轨迹光流。
- 稀疏可见性 mask。
- 稀疏置信度。
- FlowSeek 与 CoTracker3 对齐检查脚本。

### 5.4 验收标准

- 同一点的轨迹位移和 FlowSeek 采样光流方向基本一致。
- 坐标没有明显翻转、缩放或偏移错误。
- 不可见点不会作为高置信先验输入融合网络。
- 每个训练样本能同时读取图像、初始光流和稀疏轨迹先验。

### 5.5 重点排查

这一阶段最容易出现坐标错误，必须重点检查：

- 原图尺寸与模型输入尺寸是否一致。
- `x/y` 顺序是否一致。
- 光流单位是否是像素。
- resize 后光流是否按比例缩放。
- 轨迹点是否落在图像边界内。

## 6. 阶段五：构建轨迹先验图

### 6.1 目标

将稀疏点轨迹转换为稠密网络可读取的图像网格张量。

### 6.2 需要做的事

1. 为每个像素构建稀疏先验通道。
2. 将轨迹点位移写入对应像素位置。
3. 将置信度和可见性写入对应位置。
4. 可选：生成最近轨迹点距离图。
5. 可选：使用高斯核将稀疏点信息软扩散到局部邻域。
6. 输出与 FlowSeek 光流同尺寸的轨迹先验图。

### 6.3 产出

- `rasterizer.py`。
- 轨迹先验张量 `G_track`。
- 可视化的稀疏位移图、置信度图和距离图。

### 6.4 推荐通道设计

第一版建议使用 5 个通道：

```text
[dx, dy, confidence, visibility, distance_to_nearest_track]
```

后续可扩展：

```text
[track_density, gaussian_dx, gaussian_dy, forward_backward_error]
```

### 6.5 验收标准

- `G_track` 尺寸为 `H x W x C` 或 `C x H x W`，与训练代码约定一致。
- 轨迹点位置处能正确读取到轨迹位移。
- 非轨迹点区域不会产生错误高置信信息。
- 距离图和置信度图可视化合理。

## 7. 阶段六：训练后处理式 FusionNet

### 7.1 目标

先训练一个独立的融合网络，在不修改 FlowSeek 主干结构的情况下验证轨迹先验是否有效。

这是当前最推荐的第一版训练目标。

### 7.2 输入与输出

输入：

```text
[I_t, I_{t+1}, F_0, G_track]
```

第一版也可以不输入原图，只使用：

```text
[F_0, G_track]
```

输出：

```text
Delta F
F_refined = F_0 + Delta F
```

### 7.3 需要做的事

1. 准备训练数据读取器，能同时读取图像、FlowSeek 初始光流、轨迹先验和真实光流。
2. 实现轻量 FusionNet，例如小型 U-Net 或多层卷积残差网络。
3. 实现光流残差预测。
4. 实现置信度门控：

```text
F_refined = F_0 + g * Delta F
```

5. 加入基础损失函数。
6. 训练并与原始 FlowSeek 输出对比。

### 7.4 产出

- `fusion_net.py`。
- `losses.py`。
- `train_track_guided.py`。
- 第一版训练日志。
- FlowSeek 与 refined flow 的对比结果。

### 7.5 第一版损失函数

若有真实光流：

```text
L = lambda_flow * L_flow + lambda_track * L_track
```

其中：

```text
L_flow = ||F_refined - F_gt||_1
```

```text
L_track = ||sample(F_refined, Q_t) - (Q_{t+1} - Q_t)||_1
```

若没有真实光流，可先做弱监督验证：

```text
L = lambda_track * L_track + lambda_smooth * L_smooth
```

但正式实验仍建议使用带真实光流的数据集。

### 7.6 验收标准

- 训练 loss 能稳定下降。
- 在验证集上，`F_refined` 的 EPE 优于或不差于 `F_0`。
- 在轨迹点附近，refined flow 与轨迹位移更一致。
- 没有出现全图光流被错误轨迹大幅带偏的问题。

## 8. 阶段七：加入跨轨迹注意力模块

### 8.1 目标

在基础 FusionNet 跑通后，引入跨轨迹注意力机制，提升遮挡点和低置信点的运动推断能力。

### 8.2 需要做的事

1. 构建轨迹特征矩阵。
2. 使用位置、位移、置信度、可见性和局部视觉特征作为输入。
3. 实现多头自注意力或轻量 Transformer。
4. 使用高置信可见点增强低置信和遮挡点。
5. 输出增强后的轨迹位置、位移和置信度。
6. 将增强轨迹重新栅格化，输入 FusionNet。

### 8.3 产出

- `trajectory_attention.py`。
- 增强轨迹 `Q_enhanced`。
- 增强置信度 `C_enhanced`。
- 注意力权重可视化。

### 8.4 验收标准

- 遮挡点附近的轨迹位移更加连续。
- 低置信轨迹不会被盲目提升为高置信。
- 加入轨迹注意力后，遮挡区域 EPE 有改善。
- 计算开销仍可接受。

## 9. 阶段八：深度接入 FlowSeek

### 9.1 目标

当后处理式 FusionNet 已证明有效后，再将轨迹先验接入 FlowSeek 内部迭代优化过程，进一步提升性能上限。

### 9.2 需要做的事

1. 阅读 FlowSeek 的核心网络结构和 refinement block。
2. 找到光流迭代更新的特征输入位置。
3. 将 `G_track` 编码后拼接到上下文特征或更新模块输入中。
4. 设计轨迹置信度门控，避免错误先验干扰主网络。
5. 使用预训练 FlowSeek 权重初始化。
6. 进行小学习率微调。

### 9.3 产出

- 修改后的 FlowSeek 更新模块。
- 支持轨迹先验输入的新配置。
- 端到端微调脚本。

### 9.4 验收标准

- 深度融合版本优于后处理 FusionNet。
- 不使用轨迹先验时，模型仍可退化为原始 FlowSeek 或接近原始性能。
- 推理速度和显存增长在可接受范围内。

## 10. 推荐文件结构

建议新增：

```text
core/track_guidance/
    __init__.py
    sampler.py
    cotracker_wrapper.py
    rasterizer.py
    fusion_net.py
    trajectory_attention.py
    losses.py

train_track_guided.py
demo_track_guided.py
scripts/precompute_flowseek_outputs.py
scripts/precompute_cotracker_tracks.py
scripts/visualize_track_guidance.py
```

## 11. 推荐执行顺序清单

### Step 1：FlowSeek 单独推理

完成后应得到初始光流和可视化结果。

### Step 2：CoTracker3 单独推理

完成后应得到人工指定点的轨迹和可视化结果。

### Step 3：FlowSeek 输出驱动采样

完成后应得到自适应采样点图。

### Step 4：采样点输入 CoTracker3

完成后应得到由 FlowSeek 引导采样产生的稀疏轨迹。

### Step 5：轨迹转稀疏光流

完成后应得到每个相邻帧对的稀疏轨迹位移。

### Step 6：稀疏轨迹栅格化

完成后应得到融合网络输入张量 `G_track`。

### Step 7：训练后处理 FusionNet

完成后应得到第一版 refined flow。

### Step 8：加入轨迹注意力

完成后应验证遮挡和低置信区域是否改善。

### Step 9：端到端深度融合

完成后应得到最终版本模型。

## 12. 当前推荐的最小可行版本

为了尽快验证想法，最小可行版本只做以下内容：

1. 跑通 FlowSeek，保存 `F_0`。
2. 基于 `F_0` 的梯度做网格 top-k 采样。
3. 跑通 CoTracker3，保存轨迹。
4. 将轨迹转成 `[dx, dy, confidence, visibility, distance]` 五通道先验图。
5. 训练一个小型 U-Net 预测 `Delta F`。
6. 使用 `F_refined = F_0 + Delta F` 得到优化光流。
7. 用 EPE、轨迹点误差和可视化结果判断是否有效。

跨轨迹注意力和深度接入 FlowSeek 可以放到第二轮。

## 13. 后续变更记录

本文档用于记录实现该算法方案的执行步骤。后续若调整实现顺序、模块拆分、训练策略或验收标准，需要同步更新本文档。

| 日期 | 变更内容 | 备注 |
| --- | --- | --- |
| 2026-05-10 | 创建初版实施步骤，明确先跑通 FlowSeek 和 CoTracker3，再训练后处理式 FusionNet 的路线 | 初版 |
| 2026-05-10 | 阶段一新增两帧推理脚本 `demo_pair.py`，记录最小权重组合和验收命令 | 阶段一进行中 |
| 2026-05-10 | 创建 `flowseek` conda 环境，使用国内镜像安装依赖，并完成 FlowSeek 两帧烟雾测试 | 阶段一基础链路跑通 |
| 2026-05-10 | 新增 `demo_cotracker_pair.py`，通过 PyTorch Hub 加载官方 CoTracker3 offline 模型并完成手工点跟踪烟雾测试 | 阶段二基础链路跑通 |
| 2026-05-10 | 新增 `core/track_guidance/sampler.py` 和 `demo_adaptive_sampler.py`，完成基于 FlowSeek 光流的自适应稀疏点采样，并验证输出可直接输入 CoTracker3 | 阶段三基础链路跑通 |
