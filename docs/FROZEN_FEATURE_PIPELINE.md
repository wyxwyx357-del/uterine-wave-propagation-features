# 本地输入整理与冻结特征流水线

基于`4aafb8a9fbb24fb9cf256a25d66eb6f14d911384`，仅增加输入整理、病例编排和输出。不会运行事件检测、自动计数、妊娠模型或旧patient_features聚合。

## 入口

`run_frozen_features.py`分为两步，先核对输入，之后才显式请求少量或全部病例运行。`audit`不追踪；`run`必须提供`s_range`，且显式选择`--limit`或`--all`，没有隐含的队列启动。

```text
python run_frozen_features.py audit --cohort COHORT.xlsx --data-root VIDEO_ROOT --annotation-root LABELS.zip --pose-root POSE_ROOT --time-factor 1 --output PREPARED
python run_frozen_features.py run --manifest PREPARED/input_manifest.json --s-range .2 .8 --limit 2 --output TECHNICAL_TEST
```

Windows CUDA OpenCV需要额外DLL目录时，将`--opencv-dll-dir DLL_DIRECTORY`放在`audit/run`之前。原始播放速度的`time_factor`必须明确配置；不从文件名猜倍速。

固定ROI不是新标注：现有首帧壁线定义完整0–1路径，s_range只规定从中取哪一段。技术试跑使用此前的0.2–0.8，未将其升级为正式患者级ROI。

## 输入整理

只读取表格的视频编号、完整路径、首帧时间、估计时长、帧数/FPS，以及用于标注对应的姓名/拼音。临床指标、蠕动次数和妊娠结局列不读取、不用于筛选；原始表格不放入数据包，改提供白名单输入清单。

视频：检查能打开、帧数/FPS、尺寸、首尾能解码及实际首尾时间。核对阶段不宣称完整解码全部视频；少量运行时再次完整解码，严格核对全部时间与pose。

LabelMe：使用已有标签，不要求重画。支持完整检查编号目录，以及日期＋拼音/中文姓名的首帧命名。四个解剖标签必须能由原`read_labelme`读取，尺寸必须对应；同次检查的多份不同几何不会任意选一份。相同几何的重复备份可保留一个及来源记录。首帧参考图的PCC仅记录为QC，不据此引入新的特征门槛。

pose：必须有time_s、poses、pose_epoch；pose_valid存在则原样保留。旧缓存的epoch键仅在显式`--allow-legacy-epoch`时原值复制成pose_epoch，不读取旧R2特征或目标轨迹。可用`--pose-index`指定现有case_id/cache_path注册表，否则扫描pose根目录。核对完整检查标识、原视频大小（有记录时）、全部pose时刻与视频帧率/首尾时间。多个不同pose解不按质量猜选。原视频的绝对呈现时间起点会按冻结read_video的原规则归零。

数据文件原地保留。整理出的JSON、归一化键名的pose和清单写入新输出目录，原始数据不修改。

## 顺序计算

1. `read_video`读取所有真实帧；`layout/read_labelme`使用原默认0.5坐标尺度和21 px径向布点。
2. `tracking.track_segments`不传入新的跟踪调参，仅传原pose、pose_epoch、pose_valid。默认LK、FB/NCC、恢复/确认逻辑完全沿用。
3. 保存tracks/layout，立即调用`compute_spacetime(...,interval_s=2.0)`，保存真实跨度、hop、区间端点及mask，不插值。
4. 立即调用`patient.summarize_patient(...,s_range=显式配置,include_mode_velocity=False)`。无事件review，不调用radial_events、candidate_evidence或旧patient_features。

### 必要接口扩展

原患者入口默认会附带调用谱速度。为满足本轮“连速度也不算”，在patient/modes增加可关闭速度的接口参数，默认True保留旧行为；新流水线显式False，调用已经存在的`spectrum.direction`，跳过mode_velocity。共用ER、能量、RMS、幅度和协调性的公式不改。测试中将mode_velocity替换成报错函数，确认关闭分支时从未调用，并逐字段比较共用特征相等。

每个幅度连续段额外保存原本已经计算的positive_frequency_bins，仅补QC字段。A0、A/epsilon、频带、任何阈值、ROI采样规则及最新区间并集计时算法均未改。

## 输出

```text
PREPARED/
  input_manifest.json
  inputs/CASE_ID/first_frame.json
  inputs/CASE_ID/poses.npz
TECHNICAL_TEST/
  manifest.json
  manifest.csv
  CASE_ID/
    processing.json
    layout.npz
    tracks.npz
    spacetime_2s.npz
    features.json
    mode_windows.json
    section_support.json
    failure.json                 # 仅失败时
```

features保留window/block级ER与CC/MSE完整列表，不自行平均为患者global ER或唯一CC/MSE。raw谱能量只作QC。主幅度缺section仍为null，局部值只在QC中；失败病例所有未计算特征为null，CSV中为空。

每例阶段状态、明确失败原因和traceback单独记录；异常只在病例边界捕获一次，无回退算法。一个病例失败继续处理后面的病例。成功只代表流水线技术完成，不代表该病例有波或已验证妊娠相关性。

## 验证与数据包

`tests/test_frozen_pipeline.py`验证自动衔接、失败后继续、缺section主幅度null、禁算速度且共用数值一致、表格不读取结局值，以及两种现有标签命名。少量真实病例另外从原视频重新执行完整A0，而非只复用旧轨迹。

本地压缩包包含按病例整理的输入、白名单清单、代码/说明和已完成的少量技术输出。未计算全队列正式特征；清单明确区分输入异常、输入就绪未运行和少量已完成。不会打包原表的临床/结局字段，数据包不上传GitHub。

