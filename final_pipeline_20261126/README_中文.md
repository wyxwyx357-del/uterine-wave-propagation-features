# 子宫超声最终动态特征批处理 V1（2026-11-26）

本目录只用于**无结局批量测量**。正式预测候选动态特征固定为3个：

1. `R2_frequency_cpm`：R2 V1.1 后壁局部组织主导频率。
2. `m1_prop_rate_min_per_min`：`m1-global-pairing-unseen-v1` 显著传播活动率下界。
3. `record_MSE_time_equal`：20 s / 1 s 双壁 ER 差异的记录级时间等权 MSE。

辅助/QC会同时输出幅度、ER、directional RMS、MSE分解、有效时长、窗口数、空间支持、M1上界等。F01–F20、传播速度、曲率、传统CC、宫底–宫颈差异不进入本轮最终批处理。

## 冻结边界

- 不读取妊娠结局，不运行AUC/分类模型。
- 不接入 nnU-Net 失追恢复。
- 不启用实验性 pose-reset。
- A0 沿用原冻结恢复：最后可靠纹理 + 已有 background pose 预测 + 下一原生帧确认；恢复后新 `segment_id`，下游不跨失追/重定位边界计算形变。
- M1：原FFT 0.5–5 cpm + M1联合幅值配对 + 全视频双壁 Min-Max `>0.6` + Step1 v2计数。
- MSE：固定 `s_range=[0.2,0.8]`，20 s窗/1 s步；没有共同有效窗则缺失，不填0。
- R2：完整保留 `R2_LONGITUDINAL_POSTERIOR_FROZEN_CANDIDATE_V1_1_20261106` 的测量与可信度规则；本包不包含读取临床结局的 `evaluate_frozen_model.py`。
- 所有无法计算的病例保留在最终表，数值留空，并保存失败原因。

## 两条追踪链不要混用

### A0 → M1/MSE
需要以下之一：

- 已有冻结 `spacetime_2s.npz`：填 `a0_spacetime_cache`；或
- `video + annotation + pose`：由本包重新运行冻结A0。

`pose` 必须是现有 background pose NPZ，至少含 `time_s / poses / pose_epoch`，可含 `pose_valid`。**本项目当前冻结仓库不负责生成该pose，本包不会新写一个pose算法。**

### R2
R2是独立追踪链，需要：

- `video`
- 历史初始 `tracking` NPZ（在总清单中叫 `r2_tracking`）
- `nodes` JSON（在总清单中叫 `r2_nodes`）

R2自己执行CUDA稀疏LK、背景刚体估计与扣除、纵向投影、可信度、重复性和中可信共识。

因此某病例可以出现：R2可算但M1/MSE缺失，或反过来。不要因为某个动态特征缺失删除患者。

## 1. 下载与环境

建议直接使用你已经配置好 CUDA OpenCV 的 `frequency_calculation` 环境。

```powershell
conda activate frequency_calculation
pip install -r final_pipeline_20261126/requirements.txt
python final_pipeline_20261126/verify_environment.py
```

**不要执行 `pip install opencv-python`**，避免覆盖已经配置好的 CUDA OpenCV。

## 2. 准备总清单

复制：

`final_pipeline_20261126/manifest_template.csv`

字段：

- `case_id`：唯一检查ID。
- `video`：视频路径。
- `annotation`：A0首帧四标签LabelMe JSON：`ant_wall/post_wall/cervix_point/fundus_point`。
- `pose`：A0 background pose NPZ。
- `time_factor`：普通原速度视频填1；只有已确认3x加速视频才填3，程序不根据文件名猜。
- `a0_spacetime_cache`：已有可信冻结A0 `spacetime_2s.npz`时可填，填后A0不重新追踪。
- `r2_tracking`：R2历史初始tracking NPZ。
- `r2_nodes`：R2 nodes JSON。
- `mm_per_processed_pixel`：可空。

缺什么就留空。程序会尽量计算可用分支，并把缺失原因写进最终表。

## 3. 正式运行

在仓库根目录执行：

```powershell
python final_pipeline_20261126/run_final_batch.py `
  --manifest D:/study/final_manifest.csv `
  --output D:/study/final_dynamic_20261126 `
  --resume `
  --r2-workers 1
```

也可以：

```powershell
final_pipeline_20261126\run_windows.bat D:\study\final_manifest.csv D:\study\final_dynamic_20261126
```

GTX1080 8GB环境默认建议 `--r2-workers 1`。确认GPU显存和稳定性后再自行增加；不要为提速修改算法参数。

## 4. 断点续跑

重复执行同一条带 `--resume` 的命令即可。

- A0/M1/MSE：病例已有 `cases/<case_id>/final_dynamic_row.json` 时直接复用。
- R2 tracking：原冻结 `run_tracking_batch.py` 自己支持同一 `run-dir` 的断点续跑。
- 后续R2步骤若已存在完整输出会跳过。

如果输入文件或算法源码发生变化，请用新的输出目录，不要覆盖冻结结果。

## 5. 最终输出

主文件：

`FINAL_DYNAMIC_FEATURES.csv`

三个正式预测候选字段：

- `R2_frequency_cpm`
- `m1_prop_rate_min_per_min`
- `record_MSE_time_equal`

重要辅助/QC包括：

- `m1_prop_rate_max_per_min`
- `effective_observation_seconds`
- `m1_prop_count_min/max`
- `raw_valid_fraction`
- `unique_window_count`
- `mean_spatial_columns`
- 前/后壁幅度
- 前/后壁配对ER均值
- 前/后壁 directional RMS均值
- `record_mean_bias_sq`
- `record_sd_scale_sq`
- `record_time_shape_term`
- `record_MSE_z`
- R2保留后壁区域数
- 各分支状态与失败原因

另有：`FINAL_RUN_SUMMARY.json`。

## 6. 结果解释底线

- 缺失永远不是0。
- M1叫“算法检测的显著传播活动率”，不叫真实蠕动波次数。
- MSE叫“双壁ER差异”，不直接叫协调性。
- R2叫“局部组织主导频率”，不叫蠕动波次数。
- 本轮只完成测量与质控，不看妊娠结局。
