# 全局事件归属：执行、判定与验收协议

基准：`bcef4d290ba35cd32179b432ca099227d9bd1c4c`。
流程固定为：periodic＋partial父候选 → 已有局部split提案 → 全局归属协调 → event proposals。
全局指同一病例、同一壁的全部候选；双壁之间不因时间相近就自动认作同一生理事件。
正式`event_count`、`event_frequency`始终为空。

## 判定优先级

1. **先检查明确冲突和连续性。** 同一次split的两个child必须有不同身份。
   不跨失追/重定位支持断裂合并。相位序列或已知传播方向冲突时，不强制归为同一次。
2. **原始有序相位节点完全重合，优先作为重复证据。** 同时保留原ridge、section和时间来源。
   共享一个相位不能自动证明两个完整活动相同。
3. **完整时空结构一致，才判定完整重复。** 在共同的相邻native sections上，
   比较有序正负相位、相位时间范围、同标记arrival轨迹和方向；不能只看时间包围框。
4. **部分相位证据只能在归属唯一时补充已有身份。** 完整匹配优先于不完整的次级重合。
   多个相互冲突的完整身份不能靠一个局部child“搭桥”合并。
5. **仍无法唯一判断时保留歧义。** 不为减少行数选择一个owner，也不把不确定等同于零次。

唯一时间匹配尺度固定为“实际measurement span＋一个native sample interval”。
还必须满足局部相位范围互相包含峰值（允许一帧离散误差）及空间/顺序条件。
相邻支持沿用现有3个section，不继续扫描。该尺度是工程归属容差，不是临床最短事件间隔。

## 合并、保留和歧义输出

| 情况 | 动作 | 必须保留 |
| --- | --- | --- |
| child由已有event唯一表示 | `reuse_existing`，指向已有event ID，不再新建一份 | 原测量及child补充证据 |
| 两个child分别已有不同owner | 复用两个已有身份，原parent不被重复拆出两份 | 全部来源与别名映射 |
| child尚无owner，局部独立身份成立且来源容器可协调分解 | `accept_split`，新增必要身份；其他等价child统一owner | 父来源、两个child及拆分理由 |
| 更完整原事件与较小child表示同一活动 | 同一身份，完整相位优先，再按native支持范围选择canonical；同等支持优先原事件 | 原完整测量，不以小child覆盖 |
| 多个owner、顺序/方向矛盾，或来源仍被其他完整身份独立占用 | `ambiguous`，保留原事件，child仅作待定证据 | 冲突对象、原因、两个child证据 |

合并的是event identity，不平均轨迹、不拼接缺失、不重新计算A/幅度/ER。
每条最终提案保存canonical来源、所有parent/child来源、sections、arrival、direction、
phase/ridge provenance、冲突与处理原因。原始证据不删除。

原`propose_groups()`中的未定单相碎片不等同于独立完整事件。容器被有效拆分后，
剩余单相碎片若无法归属，另存`unresolved_retired_fragments`，不能凭空赋予第三个身份，
也不能悄悄丢弃。原始完整事件仍有独立归属时则不能退役。

## 明确执行流程

1. 冻结父候选、局部split及源文件哈希；不重新生成父候选来扩大范围。
2. 固定相邻支持3及原约2秒span，建立原事件和child证据图；禁止同胞child合一。
3. 先协调原始证据，再给child找已有owner，再协调真正未归属的child。
4. 以整个split事务处理依赖：不能保留两个新增child又留下一个无法解释的第三来源身份。
   被拒绝的提案不能成为其他合并唯一的桥。证据不足时转歧义。
5. 复用原87组，分别报告确认分离、歧义保留、重复、弱活动支持及负对照新增身份。
6. 满足提案复核条件后停止合成方法开发，处理原12例；只输出待人工核对提案。

```bash
python experiments/test_global_event_ownership.py \
  --splits outputs/candidate_split_trial \
  --parents outputs/event_input_smoothing \
  --controls-root outputs/frozen_image_warp_controls \
  --output outputs/global_ownership_trial

python experiments/apply_global_ownership_real.py \
  --validation outputs/global_ownership_trial \
  --source outputs/cohort_review \
  --parents outputs/partial_phase_trial \
  --output outputs/global_ownership_real
```

真实对照图需Matplotlib。第二条命令检查已通过的提案复核验收及实现哈希。
真实视频只链接现有一倍速文件，不重新编码；辅助处理仍仅在每个parent内部进行。
数据、临床衍生结果、复核页和CSV全部留在本地ignored outputs/，不发布。

## 验收定义与本轮控制结果

- **分离**：分别报告“确认保留”和“证据完整保留但歧义”。原两个真活动不得被合成一个身份，
  歧义不算成功确认，也不能因未确认就删除其双活动证据。
- **重复**：真值评价同时检查原始phase identity、至少3个相邻native支持及arrival误差，
  时间尺度复用上述span＋sample容差。另保留不限制时间的原始相位归属统计，
  不能把落在长相位尾部、到达误差很大的假候选直接称为同一真实事件。
- **保留**：比较弱第二活动的实际支持和时序，不能用“仍有一个匹配”替代。
- **安全性**：单活动及非传播对照不得新增确认误拆或方向身份。
- **边界**：不得改A0、正式特征、noise floor、band或connector来换取通过。

本轮10秒双强：原局部可恢复6个壁条件，协调后5个确认、1个完整保留为歧义，
没有强行合为一个。严格的“确认数完全不下降”审计为false，单独保留在输出，
不把歧义冒充成功。15秒局部拆分额外重复20份降为0；20秒弱第二活动最差支持
恢复至原91.7%；单波和负对照没有新增确认误拆/方向身份。

因此只进入**proposal人工复核**，不宣称自动确认能力无回退或算法已冻结。
不为把最后一个歧义变成自动确认继续调参。真实病例效果需单独人工核对，不能从这些
相关合成/半合成条件推出临床准确率，更不能输出正式患者波次数。
