# 有效时长修复与特征计算冻结

## 已确认并修复的问题

`d49450c`的`patient._duration(mask,time_s)`计算相邻有效中心之间的距离，少算测量区间两端，也无法完整表达孤立有效measurement interval的覆盖。此前“有效时长可以冻结”的结论过早。

当前改为`_duration(mask,interval_start_s,interval_end_s)`：按start排序，只合并重叠/相接的选中区间，再相加各段长度。未选区间不被补值，端点之间的正长度缺口不会被填。区间端点继续由`compute_spacetime`保存或由旧缓存的center±实际dt/2恢复，原完整区间containment规则不变。

仅影响每壁`common_region_duration_s`、复核`effective_duration_s`和以该时长为分母的复核频率。输出增加`duration_definition=union_of_supported_measurement_intervals`。有效比例仍是原有效样本比例，没有改为另一个定义。

显著活动计数实验已有独立的`observation_union()`，本轮不修改其计数、分母或0.6；不重启事件检测优化。

## 必要测试

| 条件 | 旧结果 | 修复结果 |
|---|---:|---:|
| 0–60 s完整2 s测量支持，复核一次事件 |58 s，60/58次/分|60 s，1次/分|
| 10–50 s完整测量支持 |38 s|40 s|
| 支持为[0,20]及[30,60]，双壁均有效 |中心跨度会少算两段端部|50 s，保留10 s缺口|
| 复核[0,40]和[20,60] |不另作计数|并集60 s，不是80 s|

补充乱序、重复区间和空支持验证。原缺失事件、相同event_id双壁去重、时间裁剪及CLI测试继续通过。全部154项测试中152通过，2项CUDA检查跳过；本轮未改GPU追踪。

## 缓存配对复算

对既有三份真实A0径向图，用相同s=0.2–0.8范围，分别重放旧中心计时和新端点计时。验证每个wall除时长外的全部字段，以及完整modes/events/event_summary相等；未提供事件复核，次数和频率继续为null。

三份输入中，两个完整固定区域的两壁时长分别增加实际测量跨度约2秒；有断裂支持的长片后壁增加2.04秒，前壁仍为0。幅度、有效比例、ER、RMS、energy fraction和CC/MSE均逐值不变。长片前壁主幅度仍为null，未用局部结果填主值。

结果只说明bug修复未改变其他测量，不构成新的临床准确性证据。输入缓存SHA256核对未变。旧患者输出不覆盖；修复后的结果单独保存在本地`outputs/duration_union_fix_20261119/`。

```text
python -m unittest tests.test_patient tests.test_modes -v
python -m unittest discover -s tests -p "test_*.py" -q
python experiments/validate_duration_union.py --maps PATH_1 PATH_2 PATH_3 --output OUTPUT_DIRECTORY
```

## 谱能量接口说明

只增加metadata，不改任何谱公式。`forward_energy/backward_energy/directional_energy`明确标记为未归一化FFT power、仅作QC、不可直接跨不同时间样本数/空间列数比较。包括模式候选内的方向谱能量，都继承这一说明。ER、directional RMS及directional band energy fraction不变。

## 冻结边界

- 冻结A/epsilon、约2秒跨度、频带幅度、有效支持及修复后的区间并集计时。
- 冻结window/block级ER、directional RMS/energy fraction、block级双壁ER CC/MSE计算。
- raw谱能量只作QC。保留重标敏感性、短窗分辨率、低活动和重叠窗口的既有说明。
- 不把window ER升级为全片global ER；没有新增唯一患者CC/MSE聚合。
- 事件次数沿用用户指定的暂冻结状态，速度仍为候选。

本轮不修改A0、A(s,t)、幅度/ER/CC/MSE公式、bandpass、任何阈值或患者专用参数。

