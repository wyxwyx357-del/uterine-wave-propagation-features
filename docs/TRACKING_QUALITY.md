# 已有NCC与论文PCC：定义、支持和接口

当前`_lk.match`在原灰度块上去均值后计算：

`NCC = dot(x-mean(x), y-mean(y)) / (norm(x-mean(x))*norm(y-mean(y)))`

这就是块内Pearson相关系数。LK定位仍使用既有增强图；质量相关仍来自既有原灰度图，本文不更改追踪。

Huang 2022在方法C中以相邻帧模板相关评价记录质量，并以平均PCC>0.8纳入分析。原文：[方法C，期刊页2053](https://pure.tue.nl/ws/files/217021262/Characterization_of_Uterine_Peristaltic_Waves_by_Ultrasound_Strain_Analysis.pdf)。其筛选不等于逐次蠕动识别。

## 当前汇总的含义

`uterine_wave.quality.tracking_quality(tracks,grid)`复用已存`target_ncc`，报告所有采样点、径向点、两壁inner/outer的样本均值、P05、最小值、逐点均值的空间平均、相关性覆盖率及状态计数。

现有算法在flow/FB通过后才计算相关，并以NCC>=0.75接受位置。因此汇总明确命名为**已有筛选后、有效原生步骤上的条件PCC**，不能称为独立重现论文全记录质量筛选。覆盖率以全部请求的原生点步骤为分母；缺失不填零或-1，不利用重定位较长跨度的分数冒充相邻帧质量。无支持时均值为None。`point_mean_average`使每个有测量的点等权，`mean`使每个有测量的点步骤等权；两者在覆盖不均时可能不同。

排除首帧种子、重定位跳跃、轨迹段边界和位姿阶段边界；重定位确认后的第一个正常相邻帧步骤仍可计入。位姿有效性与块外观相关性不同，完整形变支持继续由既有`valid_mask`决定。

0.8只作为来源明确的描述性参照，不增加追踪门槛、不删除候选、不排除患者，`independent_recording_quality_pass`保持None。高PCC不证明材料身份、形变准确或生理蠕动存在。

## 与径向图和候选对齐

`radial_quality_maps(tracks,grid,spacetime)`输入既有`compute_spacetime`输出。对每个[start,end]径向区间，质量统计使用两成员各自的原生步骤start+1到end；首端种子的质量不属于这个测量。检查相同hop和原生起止时间，保留原中心时刻与s坐标。

输出`interval_pcc_mean`、`interval_pcc_min`、`interval_pcc_coverage_fraction`，形状均为`时间×两壁×section`；同时保留原始`deformation_valid_mask`。相关性部分缺失时，均值仅描述已有观测，覆盖率明确降低；不以此填补形变。

`candidate_quality(quality_maps,candidates)`描述既有候选起止时间与section包围区域内、原有效形变区间的质量，记录实际源时间范围。相邻形变区间重叠，统计不是独立重复观测；不将局部均值解释为检测概率。它不修改候选分类、相位配对、计数或正式患者结果。

## 本地审查与发布边界

`experiments/audit_tracking_quality.py --review-root <existing-review> --output <new-local-output>`读取现有`screening_summary.json`及各病例缓存，独立输出质量JSON/NPZ/CSV/报告，核对原文件哈希未改变。不读取临床表或妊娠结局，不解码视频、不重追踪。旧缓存没有PCC时明确记录不可用，不编造值。

新运行的`cohort_wave_review.py`同样旁路保存质量文件。既有结果不会因再次运行而被自动重算；旧缓存使用上述审查脚本。新代码、此方法文档及合成单元测试可发布；患者数据和所有本地输出不发布。
