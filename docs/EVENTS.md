# V0 逐波事件与患者级特征

本模块把现有 A/B 形变时空图继续推进到候选逐波事件。目的不是用整段 ER 代替波检测，而是显式识别 time-space 中的连续传播活动。

当前状态：**方法开发版，不是已验证的生理波检测器。** 默认门槛均为 V0 工程规则，必须先在真实视频人工候选事件和已知传播/刚体对照上固定；不得按妊娠结局调整。

## 输入

`event_detection.detect_events(signal, valid, time_s, space)`

- `signal`: T×S，来自 A 某一壁或 B 某一窄带的形变图。
- `valid`: 同形状有效掩码。
- `time_s`: 真实采集时间。
- `space`: 严格按宫颈→宫底递增的实际弧长坐标。
- 缺失保持缺失；不会跨无效样本补齐事件。

## V0 候选事件

每个空间位置用自身时间序列的中位数和 MAD 构造绝对 robust-z 活动。默认：

- z_threshold = 3.0
- min_component_cells = 6
- min_space_points = 4
- min_path_fraction = 0.15
- min_fit_r2 = 0.50
- synchronous_tolerance_s = 0.50

活动像素用 time-space 8 邻域连通；每个连通成分在每个空间列取活动最强时刻作为 ridge，并拟合：

`t = a * space + b`

方向定义：

- a > 0 → C2F
- a < 0 → F2C
- 传播跨越时间很短 → synchronous
- 空间覆盖不足 → local
- 覆盖足够但线性传播拟合差 → complex
- 无法形成可解释拟合 → uncertain

**ER 不参与“有没有事件”的判定。**

## 单次事件输出

每个事件包括：

- start/end time
- duration
- start/end space
- path fraction
- classification
- ridge space/time
- slope
- fit R²
- speed_space_per_s
- speed_path_per_s
- peak/rms amplitude

只有确认物理空间标定后才允许把速度报告为 mm/s。

## 前后壁协调

A 路线前后壁事件按时间中点一对一配对，默认最大时间差 2 s。匹配后单独报告方向一致性和时间差。

## 患者级候选特征

A 路线可汇总：

- traveling event count
- events/min
- C2F/F2C 比例
- 中位传播速度
- 中位传播范围
- 中位事件峰值形变
- 前后壁方向一致比例和时间差

有效观察时间不是视频总时长直接替代；默认要求某时刻至少 50% 空间位置有效，且相邻两个时间点都满足才累计该时间间隔。

## 当前必须做的验证

1. 长视频人工候选事件 vs 自动事件的时间、方向、范围；
2. 纯刚体条件的假事件率；
3. 已知 C2F/F2C 的事件检出、方向和速度偏差；
4. local_once 不应被强行解释成 travelling wave；
5. 比较累计追踪与 anatomical adjacent-frame tracking 后再冻结事件规则。

事件模块不能通过单纯提高阈值掩盖底层刚体假形变。
