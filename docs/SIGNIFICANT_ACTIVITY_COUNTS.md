# 显著传播活动次数：不使用解剖方向

## 补回phase-fit验证（v2）

在每个连续保留block上拟合正、负相位峰值并传给原arrival分类器；沿用原min_fit_r2、同步容差和相位冲突规则，不改跨壁匹配或0.6。缺失或同号多峰不能任意挑选，记录phase_fit_availability，只有每位置唯一峰的相位进入拟合。

87组及12例均重新运行。12组必要强波仍12/12；texture0_local_once仍误计2次，10秒双F2C仍6次。真实病例由17次降至16次，仅1040由5降为4；其他显著活动由5增为6。这是相位证据接入后的输出变化，不作为准确率提高的证明。

局部误计的两条均为single-phase：前壁正相位约1.02–28.74秒、后壁负相位约29.10–58.98秒。arrival marker是phase_peak，所以单相位的phase fit与arrival fit相同，无法提供独立一致性检验。补回证据没有消除它们，不能声称本轮修复成功；也不因此新增“双相才计波”的排除规则。

10秒双F2C六条来源如下（时间为相位证据范围）：

| 壁 / proposal | 来源 | 类型 | 时间秒 | retained sections | 既有truth匹配 |
|---|---|---|---|---|---|
| 前壁002 | base candidate004 | paired |20.70–58.98|2–7、9–23|未分配|
| 前壁005 | child candidate002_1 | paired |6.06–30.06|12–18|truth1|
| 前壁006 | child candidate002_2 | paired |24.30–37.74|12–18|truth2|
| 后壁004 | base candidate005 | paired |1.02–29.94|10–14、18–21|未分配|
| 后壁008 | child candidate007_1 | paired |25.02–38.10|2–6|truth1|
| 后壁009 | child candidate007_2 | paired |34.74–55.50|2–6|truth2|

额外两条是base双相结构，不能归因于不完整单相证据；未分配只代表既有真值匹配没有认领，不直接证明来源是何种伪影。同一truth的两壁child位于互补sections，没有共同section，现规则无法配对。所有6条的phase fit依旧支持其斜向结构，本轮不改跨壁规则、不为失败fixture调参。

v1输出保留；v2输出在`outputs/significant_activity_counts_20261119_v2/`。复现运行`python experiments/run_activity_counts.py`，默认输出v2；可传一个输出文件夹名称保存对照。

用户确认主目标为次数，不再验证C2F/F2C。A0、约2秒A(s,t)、periodic+partial及固定0.6均不变。实验计数入口为`experiments/run_activity_counts.py`，输出`significant_propagation_activity_count`和`significant_propagation_activity_frequency_cpm`。

同壁ownership的方向硬否决已取消，方向冲突保留为QC。重新从已保存parents和split proposals执行同壁ownership，再计算0.6显著支持；跨壁按相位、arrival、至少3个相邻位置及唯一一对一归属去重。计数只检查连续保留区块是否具有原有arrival分类器支持的传播结构，不要求与旧方向一致。方向字段仅为QC；没有新增传播阈值。旧方向页面保留为历史输出。

## 同参数87组对照与12例结果

- 12组单强波/20秒双强波计数均正确。
- 12例程序检测显著传播活动合计17次，其他显著活动5次。逐例为：1040=5、1101=2、1209=1、305=0、353=2、437=0、527=2、611=1、698=2、785=1、860=0、955=1。
- 15组非传播条件中，rigid和同步条件均未计波；texture0_local_once误计2次，故局部活动排除验收未通过。
- texture0_double_F2C_0.04_gap10实际输出6次：两次真实活动各在两壁互补区段表示而未能配对，另有两条未分配到已有真值的活动。不能写成已经修复为2次，也不能通过调方向或强制按真值合并解决。
- 已有node审查证实该重复不是方向硬否决造成的同壁重复，因此移除方向条件没有解决它。互补区段缺乏共享section证据，当前保留未合并状态。

## 验收结论

方向解耦接口及必要强波对照通过；相邻双波和局部非传播两项失败。次数路线尚未冻结，不自动追加阈值或算法分支。17次是当前固定0.6算法输出，不代表全部生理蠕动次数；弱活动漏检仍存在。所有真实数据仅在本地，代码及结论可以远端审核。

```text
python experiments/run_activity_counts.py
python -m unittest tests.test_activity_counting tests.test_global_event_ownership -v
```

本地结果：`outputs/significant_activity_counts_20261119_v1/summary.json`。本轮未读取妊娠结局或临床表，没有根据患者数值调参。
