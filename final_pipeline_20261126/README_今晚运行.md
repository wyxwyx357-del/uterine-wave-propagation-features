# 今晚正式全流程运行

完整链条：

`已有缓存盘点 → A0追踪/复用 → 约2 s A(s,t) → amplitude/ER/directional_rms → M1 → MSE → 独立R2 → RAW表 → 中位数补齐表`

明确不启用：nnU-Net失追恢复、experimental pose-reset、CPU LK fallback、妊娠结局/AUC。

## 1. 找正确CUDA环境

当前VS Code解释器如果没有 `cv2`，不要安装普通 `opencv-python` 覆盖环境。运行：

```powershell
python final_pipeline_20261126\find_cuda_env.py
```

优先使用既有 `frequency_calculation`：

```powershell
conda run -n frequency_calculation python final_pipeline_20261126\verify_environment.py
conda run -n frequency_calculation python final_pipeline_20261126\preflight_tests.py
```

两步通过后再批量运行。

## 2. Manifest复用优先级

字段：

`case_id,video,annotation,pose,time_factor,a0_spacetime_cache,a0_tracks_cache,a0_layout_cache,r2_tracking,r2_nodes,mm_per_processed_pixel`

1. 已有可信 `spacetime_2s.npz` → 填 `a0_spacetime_cache`，不追踪。
2. 只有可信 `tracks.npz + layout.npz` → 填两列，只补算spacetime。
3. 都没有 → 有 `video + annotation + pose` 才运行冻结A0。
4. 缺输入 → 保留病例并记录失败原因。

R2是独立追踪链，不使用A0轨迹替代。

## 3. 正式运行

```powershell
conda run -n frequency_calculation python final_pipeline_20261126\run_final_batch_full.py ^
  --manifest D:\study\final_manifest.csv ^
  --output D:\study\final_dynamic_20261126 ^
  --resume ^
  --r2-workers 1
```

## 4. 输出

- `FINAL_DYNAMIC_FEATURES_RAW.csv`：原始测量，缺失保持NaN。
- `FINAL_DYNAMIC_FEATURES_MEDIAN_IMPUTED.csv`：指定数值特征中位数补齐并增加 `<feature>__missing`。
- `MEDIAN_IMPUTATION_VALUES.json`：补齐值与缺失数。
- `FINAL_RUN_SUMMARY.json`。
- 病例级A0/M1/MSE/R2中间结果和错误日志。

全队列中位数表仅用于探索/准备。正式CV或时间验证时必须在训练折内重新计算median。
