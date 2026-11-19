# 显著传播活动次数：不使用解剖方向

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
