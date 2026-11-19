# 保留证据类型与身份范围：固定87组沙盒

## 结论

审查指出的接口问题可以在当前源码确认：上游保留`evidence_kind`，`candidate_nodes()`未传出；正式计数以宽松arrival分类器的结果决定传播支持，并未区分单相与双相证据。单相的`single_phase_event_identity_unresolved`警告不能被当成已经确定的活动身份。

本轮只测试两项：传递既有provenance；双相进入主候选、单相保留为unresolved；另外恢复已有跨壁替代匹配的条件范围。不做新的cycle pairing，不改变分类器、tracking、A、bandpass、0.6、方向或coverage参数。

**单相误计可以消除，必要的单强波、15/20秒双强波保持；但10秒双波重复仍在，而且一条原来支持真实第二活动的单相被移出主候选。两项修改不足以冻结次数，不接入正式计数，不覆盖患者结果。**

## 实验与接口

`experiments/provenance_count_trial.py`是独立沙盒，正式`activity_counting.py`和`candidate_nodes()`未改。

1. 沿用原有periodic+partial、local split、无方向硬否决的ownership和固定0.6。复用既有87组，不新增fixture、不读取患者或结局。
2. 从canonical source传递`evidence_kind`、source kind、parent/entity和phase-ridge provenance。原始类型不从峰数重新猜测，原始测量也不丢弃。
3. 先完整复算原计数；87组逐组与当前保存v2结果的**完整baseline字典相等**后，才比较实验结果。
4. 实验仅将`paired_phase`节点送入同一个计数器。single/unknown保留完整profile、arrival、来源和原传播QC，标记`unresolved_phase_identity`；未判无活动，也没有删除证据。双相只说明候选的配对来源，不证明周期角色或生理事件身份成立。
5. 跨壁沿用现有比较器及span+dt容差。已经唯一对应者维持合并。剩余记录的shared/unresolved边用历史最大二分匹配计算条件范围`[候选行数-最大可匹配数, 候选行数]`，不强制选其中任一配对。
6. **没有共同native sections的候选不能因为时间近而新增匹配边。**记录其原相位时间范围是否重叠，但此信息只作QC，不修改归属或范围。条件范围不覆盖同壁歧义或互补区段身份，不能称真波数上下界、置信区间或确定计数。

所有输出仍有`event_count=null`、`event_frequency=null`。`primary_candidate_row_count`表示算法候选行数，不能升级为全部蠕动次数。区间端点不是已确认次数。

## 固定条件结果

表中每行是该类全部条件合计；不是单个患者。强波幅度0.04、弱第二活动0.01为原有fixture设置，未调整。

| 条件 | 条件数 | 原计数 | 双相主候选行数 | 记录内条件范围 |
|---|---:|---:|---:|---:|
| local_once |3|2|0|0–0|
| 单强波 |6|6|6|6–6|
| 15s双强波 |6|12|12|12–12|
| 20s双强波 |6|12|12|12–12|
| 10s双强波 |6|24|23|21–23|
| 15s强→弱 |6|6|6|6–6|
| 20s强→弱 |6|6|6|6–6|
| local/global synchronous |6|0|0|0–0|
| rigid / rigid_only |6|0|0|0–0|

全部87组原计数126→实验主候选118。27条显著节点被保留到单相/未定记录。候选变少本身不是准确率提高。

### local_once：修复的是单相自动计数

三个纹理原合计2次、实验主候选0次；原误计的前壁正单相和后壁负单相均完整留在unresolved。没有重新配正负相位，也没有凭已知局部真值加排除规则。这说明区分证据类型有作用，不能据此认为所有局部假传播都已解决。

### 指定10秒双波：6仍为6，范围也不能替代身份

`texture0_double_F2C_0.04_gap10`的6条显著传播记录原本全部为paired-phase，所以本轮都保留：

- truth1有前后壁各一份，共2份；truth2也各一份，共2份。
- 另外2条base是负→正恢复/谷部候选。对于该已知长度隆起fixture，它们不对应新的收缩最大值；算法没有据truth删掉负→正。
- 两壁同truth的显著支持落在互补section，原比较器没有可建立的shared/unresolved匹配边。
- 因而记录内条件范围仍是`[6,6]`，但`identity_remains_unresolved=true`、`range_is_global_identity_bound=false`。这**不表示6个独立真波**，也不能将未被规则覆盖的歧义冒充已确定不同。

其他两组10秒双C2F存在已有匹配替代，条件范围分别从3行得到2–3、4行得到3–4；不是所有双波均能得到有意义的缩小范围。

### 非回退反例：单相可能是真实第二活动的唯一证据

`texture1_double_F2C_0.04_gap10`原3条→实验2条，看似接近真值2，但不能称成功：

- 原前壁`event_proposal_003`为single-phase，在17/17个native sections对应truth2同极性阶段，峰时RMSE为1.607秒。
- 被移到unresolved后，truth2主候选表示数从1→0；truth1原已无被归属的主支持，本轮未修复。
- 剩下的2条均未按独立raw phase真值归入任一正确活动。**行数恰好为2不等于恢复两个真实活动。**

评价在候选及范围产生后才附加raw truth；没有将正→负的fixture角色或truth身份用于筛选。按相同signed phase归属及既有3连续section支持，66组信号阳性条件中，原80个被表示的observable identities变为79个；这只是结构支持数量，不是准确率。arrival误差和空间支持逐条保留。

### 边界与图像层对照

现有87组里，带有原periodic边界相位标记且能归属raw truth的44个身份均仍有双相主支持；移出的节点含33条边界phase观测。这些是现有缓存的边界信息，不是专门截短真波测试，不能推广为“边界波不会丢”。上述损失的truth2没有边界标记。

6组图像层传播条件原候选合计29→24。逐组为：texture0 C2F 5→5、F2C 4→4；texture1 C2F 6→4、F2C 5→4；texture2 C2F 4→3、F2C 5→4。该旧fixture为连续周期已知运动，本轮未建立它的独立逐周期身份标签，因此这些减少**不自动记作去假成功或真波漏检**。局部与刚体阴性仍归上述对照。

## 必要审查与停止

- single-phase仍可支持运动/局部传播证据；不计入双相主候选不等于判假。恢复provenance可以保留，但不能声称纯双相主候选已经覆盖所有显著活动。
- 双相不等于事件角色已经确定，负→正base及两壁互补身份仍是10秒失败来源。
- 旧条件范围只诚实表达已有匹配边的歧义，不能自行补齐缺少空间对应的信息；本轮未把范围内任一值当患者正式特征。
- **停在这里，不继续加配对阈值、极性特例或时间合并规则，不重启cycle枚举，不因本轮结果直接重算12例。**正式次数仍不冻结。

## 复现

```text
python -m unittest tests.test_provenance_count_trial -v
python experiments/test_provenance_count_trial.py
```

本地结果：`outputs/provenance_count_trial_20261119/summary.json`及87组详细JSON。保存基准代码、输入与原正式输出的SHA256，并在运行末尾核验未修改。只提交实验代码、接口测试和本文档，不提交数据、患者信息或输出文件。

