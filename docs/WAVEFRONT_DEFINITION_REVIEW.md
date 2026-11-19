# TVUS显著传播活动：定义审查与失败定位

本轮为定义审查，不改正式算法、参数、计数或ownership。用户指令基于73a15dc；实际核对最新2c0f630，包含已补回的phase-fit。诊断入口`experiments/audit_wavefront_definition.py`，只读取既有保存结果、真值与源码。

## 1. 文献方法及可映射范围

原文DOI：10.1016/j.rbmo.2025.105282。期刊直连403，本轮通过检索索引取得作者正文的“IC-EHG uterine propagation analysis”完整方法段及Discussion，不是只读摘要。可核对[期刊正文](https://www.rbmojournal.com/article/S1472-6483%2825%2900489-4/fulltext)及[作者正文索引入口](https://www.rbmojournal.com/cms/10.1016/j.rbmo.2025.105282/attachment/200d572c-24a6-4302-a26f-7ad5dfbe2457/mmc7.docx)。

论文先分割五通道的局部活动，再按多通道连续支持与相邻传播速度建立wavefront；中间通道保持基线属于broken propagation。竞争归属先选传播范围较大的结构，再选相邻速度更均匀者；未归入者记为non-propagated。最终wavefront建立后才分类方向，时间标记用收缩质心。其至少3通道及>0.6 mm/s均属电信号方法参数。文中抽取200个wavefront与专家标注核对，不能代替本项目验证。

| IC-EHG概念 | 原文含义 | 当前TVUS量 | 映射边界 |
|---|---|---|---|
| local contraction | 单通道分割活动，不自动等于wavefront | 某section的有效显著相位区间 | 电活动分割不能当径向形变真值 |
| propagated wavefront | 多位置构成的连续传播活动整体 | 若干相位ridge/fragment及arrival轨迹 | 定义原则可借鉴，当前只实现局部拟合和成对匹配 |
| broken propagation | 中间位置维持基线，未体现连续活动 | 中间位置有效但无对应相位，或观测缺失 | 必须区分“测到无活动”和“不可观察”，不能等同 |
| propagation extent | 一个wavefront覆盖的位置范围 | s跨度、位置并集及可观察区段 | 通道数量比例不等于TVUS路径长度比例 |
| propagation timing | 收缩质心时间及相邻延迟 | 零交叉或相位峰值时间 | 不能直接当成相同的生理标记 |
| ownership | 在竞争wavefront间选择归属 | parent/child provenance、共享phase、arrival对应 | 缺少所有fragment上的全局分配 |
| non-propagated | 未归入最终wavefront的局部活动 | other_significant_activity及unresolved | 算法未证实传播不等于证明真实组织绝不传播 |

3/5通道、50%、0.6 mm/s、通道间距及电活动onset均不直接写入TVUS。原文明确在身份建立后分析方向；不能把这理解为建立wavefront时完全不需要任何时空进展约束。

## 2. 四层TVUS对象的形式化（待实现规格）

**Significant local activity**：输入有效A、时间轴、section、原始相位footprint；输出局部相位节点的区间、极性、峰、归一化峰值、有效段ID。0.6只表示显著性。现实现取整个相位footprint里的最大值，三个相邻位置可以在不同时间超过0.6；不能把这个空间条件自动解释为连续显著传播。保留未显著和不可观察节点作上下文，不能用它们充当已显著支持。

**Propagation fragment**：由连续native positions上的同一活动标记组成，输出到达顺序、相邻延迟、拟合残差、相位序列和有效性。已有≥3位置只是最低拟合支持。保持相位角色与segment连续，不跨unsupported区直接画直线。单相只能提供单相传播证据；不能伪装为双相互证。若有效但静息、缺失、相位角色冲突，分别记录broken/unknown/conflict，不能压成一种失败。

**Final wavefront**：一个或多个fragment组成的唯一活动身份。共享原始phase/provenance或同位置相位与arrival唯一相符，可支持同一身份；不同完整活动序列、同时可观察的时间冲突或明确不同周期应保持独立。无共享section意味着缺乏直接身份比较，不意味着已证明不同。互补fragment只可作为待组合关系：需要原始路径中间支持、边界相位兼容及可辨识的时间延续，禁止仅按时间近或跨gap外推合并。当前输入不够时保持unresolved。

**Other significant activity**：显著但尚未组成被支持wavefront的局部、同步、角色不清或归属不明活动；保留具体原因。不能把所有other都宣称为生理non-propagated。

## 3. 当前代码实际在数什么

- `normalized_activity.py:28–52`：在已有候选相位区间内找局部最大显著值，保留≥3相邻section。
- `activity_counting.py:28–41`：先按已有候选做跨壁唯一成对匹配；没有建立显式fragment对象或全局wavefront图。
- `activity_counting.py:44–59`：每个连续retained block拟合arrival与可唯一对应的phase peaks，调用原分类器；只要一个block被分类C2F/F2C，整个候选身份就计一次。标签名称被用于判传播，但原解剖方向不作为身份门槛。
- `radial_events.py:44–61`：先识别近同步；非单调且拟合弱/相位冲突才成为complex，其余斜率结构保留传播分类。不是“所有条件都严格通过”的传播验证。
- `direction_identity_trial.py:52–81`：跨壁需≥3共同相邻section；`shared_sections=[]`最后没有对应证据，但调用层仍把未配对节点当各自计数单位。

因此当前单位是“至少一个局部block通过arrival分类的已有候选身份”，还不是“全局唯一wavefront”。方向硬否决已取消，但对象层次仍未完成。

## 4. local_once的两次误计

`controls.py:58–60`定义局部场为空间Gaussian乘共同sin²时间包络，没有沿s移动的注入相位。因此真值为局部同步伸缩。测量A来自图像warp后的LK与区间距离，不能把它的伪延迟直接当成真实传播。

| 记录 | 来源 | 支持区块 / s | footprint秒 | arrival秒 | 序列 |
|---|---|---|---|---|---|
| 前壁001 | base candidate001 / single_phase |9–11 / 0.4086–0.4817|1.02–28.74|16.86,17.58,17.94|仅正相位|
| 后壁002 | base candidate002 / single_phase |10–12 / 0.4452–0.5183|29.10–58.98|42.06,42.18,43.26|仅负相位|

两者均有3个连续位置、arrival跨度约1.08/1.20秒，拟合R²约0.964/0.824，高于现有拟合参照。原分类器据此判传播。arrival标记为phase_peak，phase-fit与arrival完全同源，所以补回拟合不能提供独立确认。两个长footprint分别接近一次伸长和回程的半周期，不代表28秒中一直有一条运动前沿。当前缺少对“同一局部活动的不同phase角色”和全周期活动身份的表达。

已证明：已知同步局部输入被测成了窄区段上的有序峰延迟；未证明：误差具体有多少来自定位、图像插值或带通滤形。不可未经对照断言全是噪声或全是FFT。本轮不做这些额外实验，也不因只有单相就删除全部证据。

## 5. 10秒双波的六条来源

| 壁 / proposal | 来源 | 类型 / phase sequence | footprint秒 | retained sections | 既有truth对应 |
|---|---|---|---|---|---|
| 前002 | base candidate004 | paired / −→+ |20.70–58.98|2–7、9–23|unassigned|
| 前005 | child candidate002_1 | paired / +→− |6.06–30.06|12–18|truth1|
| 前006 | child candidate002_2 | paired / +→− |24.30–37.74|12–18|truth2|
| 后004 | base candidate005 | paired / −→+ |1.02–29.94|10–14、18–21|unassigned|
| 后008 | child candidate007_1 | paired / +→− |25.02–38.10|2–6|truth1|
| 后009 | child candidate007_2 | paired / +→− |34.74–55.50|2–6|truth2|

全部逐section arrival、phase peaks、phase-to-truth标签和匹配失败原因保存在诊断JSON，不把宽时间范围当作单个到达时间。

**A. 重复表示**：truth1和truth2各有前后壁两份表示，4份表示含2份额外重复。对应child没有共同sections，不能按当前规则配对；不能仅凭时间近合并，另一真实相邻活动也可能时间近。

**B. additional/unassigned**：两条额外base均为负→正，且都有有序双相peak轨迹。`test_double_event_signals.py:158–166`对当前Gaussian长度隆起真值明确把负→正作为谷部/回程，不作为新的长度最大活动，因此它们不能归到truth1/2。计数层仍把这种斜向回程结构当独立事件；这比“只是未匹配、来源未知”定位更具体。它们不一定是纯随机伪影，也不能把负→正在真实TVUS中统一禁止。仍缺“收缩/恢复属于同一次活动”的可验证角色定义。

2个真值 → 4份有效但跨壁重复表示 + 2份谷部/回程候选 = 当前6个计数。空间span更大的前002包含一个约0.512路径跨度的连续block，说明单加50%也不保证去掉额外活动。

## 6. 最小后续设计（本轮不实现）

1. 给现有显著局部证据明确local_activity_id、phase角色和segment，保留幅度与原始provenance。
2. 将每个连续block表示为fragment；arrival、相位完整性、相邻进展及可观察缺口作为证据，不新增阈值。禁止用拟合斜率本身证明独立活动身份。
3. 建立wavefront候选及冲突图。共享相位和完整活动角色优先；互补片段允许提议连续组合，但没有中间证据不能强并。单phase、paired-phase和恢复phase不能仅凭标签数成不同wavefront。
4. 统一ownership：同一局部活动最多归属一个最终wavefront。先比较不跨gap的传播范围，再看相邻时序一致性；相位完整性作项目的次级证据（不是宣称原论文同样采用）；仍竞争则保留unresolved，不使用单总分或患者专用阈值。
5. 最后计唯一wavefront / 有效分钟。可观察性不足与测到局部不传播分开。方向仅QC；暂不宣称能无参数地唯一解决互补片段或单phase歧义。

不能先承诺这套结构必然使local_once归零或10秒双波=2；先用现有失败及正确对照验证，允许结论为不可辨识。

## 7. 是否需要50%实验

当前不需要。先解决活动角色与唯一身份的定义缺层；50%会排除窄的真实fragment，也不能排除跨度较大的谷部候选。当前诊断只报告s-span、同壁在完整phase窗口持续有效的sections及其span/比例；这些是QC描述，分母的包络不代表填上中间缺失。以后若单独研究“大范围传播”才预注册比较coverage，不把它改名为全部传播的生理定义。

## 本轮交付和范围

没有改正式流程或患者输出；未读取结局；未扫阈值；未重跑tracking。诊断文件位于`outputs/wavefront_definition_audit_20261119/`，包含local_once两条与双波六条的完整节点。所有诊断truth只用于解释已知fixture，不进入检测或ownership。
