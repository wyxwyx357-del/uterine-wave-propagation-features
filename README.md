# 子宫蠕动波传播特征

本仓库整理当前一轮的研究代码：**解剖布点 → 分段 CUDA LK 追踪 → 径向形变 / 内膜窄带曲率变化 → 传播方向模式分析 → 已知运动对照**。

目标是为幅度、频率、传播和协调性提供统一、可验证的运动测量基础。当前已有传播模式速度和双壁ER相似度候选；尚未完成逐波事件计数、事件速度和患者级妊娠预测/AUC。

当前新增固定区域/时段的患者汇总，以及**已复核事件**的次数、方向、到达速度和双壁到达差计算。自动判波尚未验证；未复核不会输出零次。主幅度称径向伸缩活动幅度，保留约2 s区间应变及频带的明确含义。当前定义、复核格式和命令见[特征定义](docs/FEATURE_DEFINITIONS.md)，其优先级高于旧事件分支建议。

## 当前实现

| 模块 | 做什么 |
|---|---|
| `geometry.py` | 用宫颈/宫底端确定解剖顺序；沿前后壁采样，建立显式径向点对和三条窄带曲线 |
| `tracking.py`、`_lk.py` | 保留当前 CUDA LK、增强图定位、原灰度 QC 和分段重定位逻辑，移除旧工程路径依赖 |
| `deformation.py` | A：同壁径向相对长度变化率；B：局部曲率变化率；禁止跨失追、重定位和校正阶段计算 |
| `amplitude.py` | 完整连续块内的0.5–5次/分零相位频带信号与区间应变SD；原始SD保留为QC |
| `patient.py`、`events.py` | 固定解剖区域/时段汇总；已复核事件到达顺序、计数、速度与双壁配对 |
| `spectrum.py` | 在实际空间坐标上计算方向能量比 ER；正值表示宫颈→宫底 |
| `wavefit.py`、`modes.py` | 连续双向时空波拟合细化模式速度；按连续支持计算滑窗模式和双壁ER相似度 |
| `controls.py` | 静止、整体平移、局部单次变化、正向传播和反向传播的已知图像变形 |
| `cli.py` | 布点、局部视频追踪、特征图和对照测试命令 |

固定候选参数：处理图上窗口 21 px，纵向间距目标 10.5 px，径向基准长 21 px，端点留边 21 px，窄带位置 0.25/0.5/0.75。原图默认缩放至 0.5。这些是本项目候选参数，**不是论文统一标准，也没有换算成毫米**。窗口有重叠，采样点不能当作独立样本。

## 安装与数值测试

Python ≥ 3.10；数值模块只依赖 NumPy。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

追踪、视频读取和图像对照还需要含 `cudaoptflow` 的 CUDA OpenCV。本项目原环境为 OpenCV 4.10.0。普通 `opencv-python` wheel 不提供本流程需要的 CUDA LK；本仓库不静默改用 CPU 算法。Windows 自行配置相应 OpenCV DLL 搜索目录，不在代码中写死本机路径。

已有 CUDA 环境可设置 `UTERINE_WAVE_CUDA_TESTS=1` 再运行测试，启用 CUDA 冒烟检查。

## 无患者数据的完整对照

```bash
python -m uterine_wave controls --output outputs/synthetic_controls.json
```

默认生成一张人工纹理，在同一张图上施加五种运动，各 501 个样本，真实时间间隔 0.12 s、跨度 60 s；追踪输入没有补帧。每条路线分别比较测量值与解析映射参考的 RMS、误差、相关及 ER。人工纹理用于软件复现，不是真实超声体模。

也可使用自己本地的首帧纹理与标注：

```bash
python -m uterine_wave controls --image data/first_gray_half.png --annotation data/first.json --scale 0.5 --output outputs/local_controls.json
```

图像应已在处理分辨率，标注仍在原图坐标。原始视频、标注、身份信息及结局不随仓库发布。

## 本地真实视频流程

首帧 LabelMe JSON 需包含唯一的 `ant_wall`、`post_wall` 折线及 `cervix_point`、`fundus_point` 单点。布点只读取这四类几何，忽略嵌入图像和其他标签；不自动使用后续帧标注作为材料运动真值。

```bash
python -m uterine_wave layout --annotation data/first.json --output outputs/layout.npz
python -m uterine_wave track --video data/video.mp4 --annotation data/first.json --poses data/poses.npz --time-factor 3 --output outputs/tracks.npz
python -m uterine_wave features --tracks outputs/tracks.npz --layout outputs/tracks_layout.npz --interval-s 2.0 --output outputs/maps_2s.npz
```

`--time-factor` 必须依据真实拍摄时间确定：确认是 3 倍速视频时填 3，原速填 1。保留解码帧及展示时间戳，不通过插帧恢复采样率。features默认选取最接近2.0秒的原生帧跨度，并保存实际dt与区间中心时间；31/35 fps分别约62/70帧；显式`--interval-s 0.96`保留敏感性对照。需要复现旧A/B输出时显式使用`--hop 8`；不能把不同帧率的8帧当成相同物理跨度。

`track` 要求显式提供背景校正文件：`poses` 为 `(T,3,3)` 的处理图坐标→校正参考坐标仿射矩阵，`time_s` 与解码后的真实时间一一对应，`pose_epoch` 为 `(T,)` 校正阶段编号。本轮没有把历史背景估计器重新移植进仓库，也不会把缺失姿态自动替换成恒等矩阵。已有追踪结果可直接按 [轨迹接口](docs/INTERFACES.md) 接入 `compute_maps`。

当前 `features` 默认输出A、epsilon、有效掩码、归一化空间轴和区间时间；`--hop`显式保留旧A/B输出。位姿缓存提供`pose_valid`时，track保留该字段供区间测量消费，缺失时不伪造为全有效。频带幅度可调用`amplitude.summarize_interval_strain`；ER可调用`spectrum.direction`；模式速度与相似度可调用`modes.analyze_modes`。**不能将ER或连续波拟合解释为事件计数**。

## 当前证据

此前在三张真实首帧纹理上完成了 15 组图像变形对照。A 路线波动 RMS 相对参考的平均绝对比例偏差约 3.2%，B 约 9.6%；正反传播的方向符号均恢复。整体晃动仍留下假形变，B 更明显。这是首帧图像变形对照结果，不是原视频真实波识别准确率。

本次整理另有数值测试、CUDA 检查、整理前后配对复算及人工纹理完整流程。具体范围见 [验证记录](docs/VALIDATION.md)。下一步是在固定测量定义下核对真实视频的运动模式与晃动残差，再研究事件、速度和协调性；不按妊娠结局调测量参数。

- [特征定义和轨迹接口](docs/INTERFACES.md)
- [文献对应及未复现部分](docs/LITERATURE.md)
- [共同目标](TRACKING_GOAL.md)
- [逐次事件冻结方案](docs/EVENT_DEFINITION_PLAN.md)

## 约2秒传播表征默认值

A0追踪参数及恢复逻辑保持冻结。约2秒仅作为传播模式表征默认有限跨度；A、epsilon及其SD/RMS、ER结果按跨度分别命名和保存。已知组合运动6组NRMSE全部改善，均值0.3836→0.2542；波幅RMS比例偏差4.70%→2.84%，相关0.9232→0.9667，方向12/12正确。该结果不验证真实事件计数或事件速度。
