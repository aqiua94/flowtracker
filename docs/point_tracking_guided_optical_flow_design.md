# 基于点跟踪改进的光流算法设计方案

## 1. 研究目标

本方案旨在构建一种由稀疏点跟踪先验引导的稠密光流估计方法。核心思想是先利用 FlowSeek 生成高质量初始稠密光流，再从该光流场中自适应选择关键运动点，并使用 CoTracker3 获取跨帧稀疏轨迹。随后，通过跨轨迹注意力机制补全和增强轨迹信息，最终设计融合网络将稀疏轨迹先验注入光流估计过程，提升复杂运动、遮挡、边界区域和低纹理区域的光流精度与鲁棒性。

该方法结合两类方法的优势：

- 点跟踪：跨帧长时序一致性强，对局部关键点运动具有较高可靠性。
- 光流估计：输出稠密，能够覆盖全图像像素，但在遮挡、大位移和低纹理区域容易不稳定。

## 2. 总体流程

整体流程分为五个阶段：

1. 输入连续视频帧，使用 FlowSeek 预测相邻帧或多帧间的初始稠密光流。
2. 基于光流梯度、运动幅值和图像结构信息进行自适应稀疏点采样。
3. 使用 CoTracker3 对采样点进行跨帧跟踪，获得轨迹、坐标、可见性和置信度。
4. 构建轨迹特征矩阵，通过跨轨迹注意力扩展遮挡点和低置信度点的运动信息。
5. 设计轨迹引导的光流融合网络，将稀疏轨迹先验与 FlowSeek 稠密特征融合，输出优化后的稠密光流。

推荐将整体结构抽象为：

```text
Video Frames
    |
    v
FlowSeek Initial Dense Optical Flow
    |
    +--> Adaptive Sparse Point Sampling
              |
              v
        CoTracker3 Sparse Tracking
              |
              v
      Trajectory Attention Enhancement
              |
              v
Track-Guided Flow Fusion Network
              |
              v
Refined Dense Optical Flow
```

## 3. 模块一：初始稠密光流预测

### 3.1 输入与输出

输入：

- 视频帧序列：`I = {I_1, I_2, ..., I_T}`
- 每帧尺寸：`H x W`

输出：

- 初始稠密光流：`F_0^{t -> t+1} in R^{H x W x 2}`
- 可选中间特征：相关体特征、迭代更新特征、深度先验特征等

### 3.2 FlowSeek 的作用

FlowSeek 作为基础光流估计框架，用于提供初始稠密运动场。其优势在于利用单目深度大模型提供几何先验，并通过逆深度运动基约束运动结构，使光流预测在跨场景泛化、几何一致性和遮挡区域稳定性上优于普通光流网络。

在本方案中，FlowSeek 不仅输出初始光流，还建议保留其编码器特征和迭代更新隐状态，用于后续融合网络。

### 3.3 光流梯度计算

设初始光流为：

```text
F_0(x, y) = (u(x, y), v(x, y))
```

计算光流梯度幅值：

```text
G(x, y) = sqrt(
    |d u / d x|^2 + |d u / d y|^2 +
    |d v / d x|^2 + |d v / d y|^2
)
```

也可加入运动幅值：

```text
M(x, y) = sqrt(u(x, y)^2 + v(x, y)^2)
```

综合采样权重：

```text
S(x, y) = alpha * normalize(G(x, y))
        + beta  * normalize(M(x, y))
        + gamma * Edge(I_t)
```

其中 `Edge(I_t)` 可由图像梯度、Sobel 算子或特征图边界响应获得。

## 4. 模块二：自适应稀疏点采样

### 4.1 设计目标

采样策略需要满足：

- 在运动边界、大位移和快速运动区域提高采样密度。
- 在背景静止或低运动区域降低采样密度。
- 保持点的空间分布均衡，避免所有点集中在局部高响应区域。
- 控制点数量，保证 CoTracker3 的计算成本可控。

### 4.2 推荐采样策略

采用分块加权采样：

1. 将图像划分为 `K x K` 个局部网格。
2. 在每个网格内根据采样权重 `S(x, y)` 分配点数。
3. 对高权重区域执行 top-k 或概率采样。
4. 对低权重区域保留少量基础点，防止背景完全缺失。
5. 使用非极大值抑制或最小距离约束，避免点过密。

每个网格采样点数可设为：

```text
N_i = N_base + round(lambda * sum(S_i) / sum(S))
```

其中：

- `N_i`：第 `i` 个网格采样点数。
- `N_base`：每个网格最少保留点数。
- `lambda`：额外采样预算。

### 4.3 输出格式

采样输出：

```text
P_t = {p_i^t = (x_i^t, y_i^t)}_{i=1}^{N}
```

可附加每个点的采样权重：

```text
w_i = S(x_i, y_i)
```

## 5. 模块三：CoTracker3 稀疏点跟踪

### 5.1 输入与输出

输入：

- 视频帧序列：`I_1 ... I_T`
- 初始采样点：`P_t`

输出：

- 点轨迹：`Q in R^{T x N x 2}`
- 可见性标签：`V in {0,1}^{T x N}`
- 跟踪置信度：`C in R^{T x N}`

其中：

```text
Q_{t,i} = (x_{t,i}, y_{t,i})
```

表示第 `i` 个点在第 `t` 帧的位置。

### 5.2 与光流的关系

由轨迹可以得到稀疏光流监督或先验：

```text
F_track^{t -> t+1}(x_{t,i}, y_{t,i})
    = Q_{t+1,i} - Q_{t,i}
```

当 `V_{t,i} = 1` 且 `V_{t+1,i} = 1` 时，该轨迹位移可作为高置信稀疏运动约束。

## 6. 模块四：跨轨迹注意力增强

### 6.1 设计目标

CoTracker3 输出的轨迹在遮挡、快速运动或局部纹理缺失时可能出现断裂或低置信度。本模块使用可见高置信点的运动信息推断遮挡点或低置信度点的潜在运动状态，从而增强轨迹完整性。

### 6.2 轨迹特征构建

对每个点构建轨迹特征：

```text
z_{t,i} = [
    x_{t,i}, y_{t,i},
    dx_{t,i}, dy_{t,i},
    C_{t,i},
    V_{t,i},
    phi(I_t, x_{t,i}, y_{t,i})
]
```

其中：

- `(x, y)`：点位置。
- `(dx, dy)`：相邻帧位移。
- `C`：跟踪置信度。
- `V`：可见性。
- `phi`：从图像编码器或 FlowSeek 特征图中采样得到的局部视觉特征。

形成轨迹特征矩阵：

```text
Z_t in R^{N x D}
```

### 6.3 跨轨迹自注意力

使用可见点与低置信点之间的注意力传播：

```text
Q = Z_t W_q
K = Z_t W_k
V_attn = Z_t W_v
```

```text
A = softmax((Q K^T) / sqrt(d) + B_spatial + B_conf)
```

其中：

- `B_spatial`：空间邻近偏置，使相邻点更容易互相传播信息。
- `B_conf`：置信度偏置，使高置信点贡献更大。

增强后的轨迹特征：

```text
Z'_t = A V_attn
```

再通过 MLP 预测修正量：

```text
Delta Q_{t,i}, Delta C_{t,i} = MLP(Z'_t)
```

最终得到增强轨迹：

```text
Q'_{t,i} = Q_{t,i} + Delta Q_{t,i}
C'_{t,i} = sigmoid(C_{t,i} + Delta C_{t,i})
```

### 6.4 遮挡点处理

对遮挡点或低置信点使用软更新：

```text
Q_enhanced = m * Q + (1 - m) * Q'
```

其中 `m` 由可见性和置信度决定：

```text
m = V * C
```

可见高置信点更依赖原始 CoTracker3 输出，遮挡或低置信点更多依赖注意力传播结果。

## 7. 模块五：轨迹引导的光流融合网络

### 7.1 设计目标

融合网络用于将稀疏轨迹先验转换为对稠密光流的约束。它不应只在稀疏点处修正光流，而应通过局部传播、特征引导和迭代更新影响周围区域。

### 7.2 输入

推荐输入包括：

- FlowSeek 初始光流：`F_0`
- FlowSeek 图像/相关体/上下文特征：`E`
- 稀疏轨迹光流：`F_track`
- 稀疏轨迹置信度图：`C_track`
- 可见性图：`V_track`
- 点到像素距离图或局部邻域索引：`D_track`

### 7.3 稀疏先验栅格化

将轨迹点投影到图像网格上，构建稀疏先验图：

```text
G_track in R^{H x W x K}
```

通道可包含：

- 稀疏位移：`dx, dy`
- 置信度：`C`
- 可见性：`V`
- 到最近轨迹点的距离：`d`
- 局部轨迹密度：`rho`

可使用高斯核进行软栅格化：

```text
G_track(x) = sum_i exp(-||x - p_i||^2 / sigma^2) * f_i
```

### 7.4 融合结构

推荐采用残差式融合：

```text
Delta F = FusionNet([F_0, E, G_track])
F_refined = F_0 + Delta F
```

FusionNet 可采用轻量 U-Net、ConvGRU 迭代更新模块，或直接嵌入 FlowSeek 的 refinement block。

优先实现路线：

1. 第一阶段：在 FlowSeek 输出后追加后处理式 FusionNet，降低改动风险。
2. 第二阶段：将轨迹先验注入 FlowSeek 迭代更新模块，使其参与光流递归优化。
3. 第三阶段：联合训练采样、轨迹增强和光流融合模块。

### 7.5 置信度门控

为了避免错误轨迹误导稠密光流，融合时加入门控：

```text
g = sigmoid(Conv([C_track, V_track, D_track, E]))
F_refined = F_0 + g * Delta F
```

高置信、近轨迹点区域修正更强；低置信、远离轨迹点区域修正更弱。

## 8. 损失函数设计

### 8.1 稠密光流监督损失

若数据集提供真实光流 `F_gt`：

```text
L_flow = ||F_refined - F_gt||_1
```

可采用 robust loss：

```text
L_flow = (||F_refined - F_gt||_1 + epsilon)^q
```

### 8.2 稀疏轨迹一致性损失

约束稠密光流在轨迹点处与点跟踪位移一致：

```text
L_track = sum_i C'_{t,i} V'_{t,i}
    * || sample(F_refined, Q'_{t,i}) - (Q'_{t+1,i} - Q'_{t,i}) ||_1
```

### 8.3 时序一致性损失

使用前后向光流一致性：

```text
L_cycle = || F^{t -> t+1} + warp(F^{t+1 -> t}, F^{t -> t+1}) ||
```

遮挡区域可通过 forward-backward consistency mask 过滤。

### 8.4 边界保持损失

在图像边缘处保留运动边界：

```text
L_edge = || grad(F_refined) || * exp(-k * || grad(I) ||)
```

该损失鼓励非图像边缘区域平滑，同时允许图像边界处产生光流突变。

### 8.5 总损失

```text
L = lambda_flow  * L_flow
  + lambda_track * L_track
  + lambda_cycle * L_cycle
  + lambda_edge  * L_edge
```

## 9. 训练与推理策略

### 9.1 分阶段训练

建议按以下顺序训练：

1. 冻结 FlowSeek 和 CoTracker3，仅训练轨迹注意力增强模块与 FusionNet。
2. 解冻 FusionNet 与 FlowSeek 后端 refinement 模块，进行端到端微调。
3. 加入遮挡样本、长序列样本和大位移样本，提升鲁棒性。

### 9.2 推理流程

推理阶段流程：

1. 输入视频帧。
2. FlowSeek 预测初始稠密光流。
3. 根据光流梯度和运动幅值采样稀疏点。
4. CoTracker3 生成稀疏轨迹。
5. 轨迹注意力模块补全低置信或遮挡轨迹。
6. 将增强轨迹栅格化为先验图。
7. FusionNet 输出光流残差并得到最终光流。

## 10. 实现建议

### 10.1 推荐代码模块

后续实现可新增以下模块：

```text
core/track_guidance/
    sampler.py              # 自适应点采样
    cotracker_wrapper.py    # CoTracker3 调用封装
    trajectory_attention.py # 跨轨迹注意力增强
    rasterizer.py           # 稀疏轨迹先验栅格化
    fusion_net.py           # 轨迹引导光流融合网络
    losses.py               # 轨迹一致性与时序一致性损失
```

训练脚本可新增：

```text
train_track_guided.py
```

推理脚本可新增：

```text
demo_track_guided.py
```

### 10.2 配置项

建议新增配置：

```yaml
track_guidance:
  enabled: true
  num_points: 2048
  grid_size: 16
  min_points_per_cell: 1
  flow_gradient_weight: 0.5
  flow_magnitude_weight: 0.3
  image_edge_weight: 0.2
  nms_radius: 4
  raster_sigma: 8.0
  confidence_threshold: 0.5
  fusion_hidden_dim: 128
  use_trajectory_attention: true
```

## 11. 实验设计

### 11.1 对比实验

建议对比：

- 原始 FlowSeek。
- FlowSeek + 均匀采样点跟踪先验。
- FlowSeek + 自适应采样点跟踪先验。
- FlowSeek + 自适应采样 + 跨轨迹注意力。
- 完整方法。

### 11.2 消融实验

重点消融：

- 是否使用光流梯度采样。
- 是否使用运动幅值采样。
- 是否使用图像边缘采样。
- 是否使用跨轨迹注意力。
- 是否使用置信度门控。
- 不同采样点数量对性能与速度的影响。

### 11.3 评价指标

推荐指标：

- EPE：平均端点误差。
- Fl-all：异常光流像素比例。
- Occlusion EPE：遮挡区域误差。
- Boundary EPE：运动边界区域误差。
- Runtime：推理时间。
- Memory：显存占用。

## 12. 潜在风险与应对

### 12.1 点跟踪错误传播

风险：CoTracker3 在低纹理或严重遮挡区域可能输出错误轨迹，误导光流融合。

应对：

- 使用可见性和置信度门控。
- 引入前后向一致性检查。
- 对低置信轨迹降低融合权重。

### 12.2 采样点过密导致计算量升高

风险：点数量过多会增加 CoTracker3 和轨迹注意力模块开销。

应对：

- 设置全局点数预算。
- 分块采样并限制每块最大点数。
- 对长视频采用滑动窗口跟踪。

### 12.3 稀疏先验难以影响远距离像素

风险：轨迹点只覆盖部分像素，远离轨迹点的区域改善有限。

应对：

- 使用高斯软栅格化和距离图。
- 在 FusionNet 中使用多尺度特征传播。
- 结合 FlowSeek 原有上下文特征进行全局补偿。

## 13. 当前推荐实现里程碑

### Milestone 1：后处理式轨迹引导光流优化

目标：最小侵入式验证点跟踪先验是否能提升 FlowSeek 输出。

内容：

- 实现自适应采样。
- 接入 CoTracker3 输出轨迹。
- 实现轨迹栅格化。
- 训练轻量 FusionNet 输出光流残差。

### Milestone 2：加入跨轨迹注意力增强

目标：提升遮挡和低置信轨迹区域的鲁棒性。

内容：

- 构建轨迹特征矩阵。
- 实现跨轨迹自注意力。
- 预测遮挡点轨迹修正量。
- 加入轨迹一致性损失。

### Milestone 3：与 FlowSeek 迭代更新模块深度融合

目标：让稀疏轨迹先验参与光流网络内部更新。

内容：

- 将轨迹先验图接入 FlowSeek refinement block。
- 联合优化光流残差和轨迹一致性。
- 完成多数据集评估。

## 14. 后续变更记录

本文档作为当前算法设计方案的主文档。后续若更改采样策略、轨迹增强模块、融合网络结构、训练损失或实现路径，需要同步更新本文档。

| 日期 | 变更内容 | 备注 |
| --- | --- | --- |
| 2026-05-10 | 创建初版方案，明确 FlowSeek + CoTracker3 + 轨迹注意力 + FusionNet 的整体路线 | 初版 |
