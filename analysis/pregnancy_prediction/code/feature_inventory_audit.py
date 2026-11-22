"""Read-only inventory of patient-level features, availability and validation.

No outcome association, AUC, model fitting, feature selection or feature
recomputation is performed.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.mse_pregnancy_association.code.clinical_variable_screen import (
    ALL_CLINICAL,
    BASELINE,
    load_clinical_features,
)
from analysis.mse_pregnancy_association.code.frozen_feature_auc import EXAM, GROUP, OUTCOME
from analysis.mse_pregnancy_association.code.short_holdout_outcome_validation import _outcome_from_path


CLINICAL_MEANING = {
    "age": ("years", "女性年龄"),
    "bmi": ("kg/m2", "体质量指数"),
    "infertility_years": ("years", "不孕年限，年"),
    "infertility_months": ("months", "不孕年限，月"),
    "endometrial_thickness_mm": ("mm", "移植前内膜厚度"),
    "endometrial_volume_ml": ("mL", "超声内膜容积"),
    "doppler_SD": ("ratio", "内膜血流S/D"),
    "doppler_PI": ("ratio", "内膜血流搏动指数PI"),
    "doppler_RI": ("ratio", "内膜血流阻力指数RI"),
    "doppler_VI": ("ratio", "内膜血流血管化指数VI"),
    "doppler_FI": ("ratio", "内膜血流量指数FI"),
    "doppler_VFI": ("ratio", "内膜血流血管化血流指数VFI"),
    "uterine_position": ("category", "子宫位置"),
    "endometrial_type": ("category", "内膜分型"),
    "clinical_peristalsis_positive_cpm": ("count/min", "临床记录正向蠕动频率"),
    "clinical_peristalsis_positive_disordered": ("indicator", "正向蠕动记录为杂乱无序的指示"),
    "clinical_peristalsis_negative_cpm": ("count/min", "临床记录反向蠕动频率"),
    "transfer_embryo_count_multiple": ("indicator", "是否移植多个胚胎"),
    "transfer_embryo_count_component": ("count", "解析后的移植胚胎数"),
    "blastocyst_transfer": ("indicator", "是否包含囊胚移植"),
    "endometrial_type_C": ("indicator", "内膜分型是否为C型"),
    "transferred_embryo_stage": ("category", "解析后的移植胚胎阶段"),
}

CURRENT_SPECS = [
    dict(
        feature_name="anterior_radial_stretch_activity_amplitude",
        standardized_name="anterior_radial_stretch_activity_amplitude",
        feature_group="Amplitude/Strain",
        source_code="uterine_wave/amplitude.py; uterine_wave/patient.py",
        code_location="amplitude.py:36; patient.py:28",
        level="patient",
        unit="interval-strain SD",
        directionality="unsigned magnitude",
        physical_meaning="前壁固定ROI内约2秒径向伸缩率的0.5–5 cpm频带SD，区域按点对平均",
        patient_aggregation="complete fixed ROI sections within supported intervals",
        current_status="frozen",
        posthoc_feature="no",
        tier="A",
        role="candidate predictor",
        reason_in="定义冻结；已知形变/刚体控制；无明显失败证据",
        reason_out="缺少人体重复采集真值；部分患者ROI支持不足",
    ),
    dict(
        feature_name="posterior_radial_stretch_activity_amplitude",
        standardized_name="posterior_radial_stretch_activity_amplitude",
        feature_group="Amplitude/Strain",
        source_code="uterine_wave/amplitude.py; uterine_wave/patient.py",
        code_location="amplitude.py:36; patient.py:28",
        level="patient",
        unit="interval-strain SD",
        directionality="unsigned magnitude",
        physical_meaning="后壁固定ROI内约2秒径向伸缩率的0.5–5 cpm频带SD，区域按点对平均",
        patient_aggregation="complete fixed ROI sections within supported intervals",
        current_status="frozen",
        posthoc_feature="no",
        tier="A",
        role="candidate predictor",
        reason_in="定义冻结；已知形变/刚体控制；无明显失败证据",
        reason_out="缺少人体重复采集真值；部分患者ROI支持不足",
    ),
    dict(
        feature_name="single_block_MSE",
        standardized_name="single_block_ER_MSE",
        feature_group="ER-MSE/Disparity",
        source_code="uterine_wave/modes.py",
        code_location="modes.py:12 and coordination aggregation",
        level="patient",
        unit="squared ER difference",
        directionality="unsigned disparity",
        physical_meaning="恰好一个既有协调性空间block时的双壁ER平方差；多block患者不取平均",
        patient_aggregation="available only when exactly one coordination block",
        current_status="frozen",
        posthoc_feature="no",
        tier="B",
        role="explanatory only",
        reason_in="公式和单block实现曾冻结并有已知同步/相反控制",
        reason_out="多block无患者级定义；已由完整记录MSE补充，但不自动替代",
    ),
    dict(
        feature_name="single_block_CC",
        standardized_name="single_block_ER_CC",
        feature_group="ER-MSE/Disparity",
        source_code="uterine_wave/modes.py",
        code_location="modes.py:12 and coordination aggregation",
        level="patient",
        unit="Pearson correlation",
        directionality="signed similarity",
        physical_meaning="恰好一个既有协调性block时双壁ER的零时延Pearson相关",
        patient_aggregation="available only when exactly one coordination block",
        current_status="frozen",
        posthoc_feature="no",
        tier="B",
        role="explanatory only",
        reason_in="公式冻结并有已知同步/相反图像控制",
        reason_out="多block无患者级定义；时间缺口相关解释受限",
    ),
    dict(
        feature_name="significant_propagation_activity_count",
        standardized_name="significant_propagation_activity_count",
        feature_group="Event",
        source_code="experiments/activity_counting.py; experiments/batch_wave_counts.py",
        code_location="activity_counting.py:28; batch_wave_counts.py",
        level="patient",
        unit="count",
        directionality="count",
        physical_meaning="固定0.6显著性与传播身份规则下的算法活动次数",
        patient_aggregation="same-wall ownership then cross-wall identity deduplication; direction only QC",
        current_status="exploratory",
        posthoc_feature="no",
        tier="C",
        role="exclude",
        reason_in="可在当前冻结A0图上批量计算",
        reason_out="local_once误计、相邻重复和漏检未解决，不能作为正式波次数",
    ),
    dict(
        feature_name="significant_propagation_activity_frequency_cpm",
        standardized_name="significant_propagation_activity_frequency_cpm",
        feature_group="Event",
        source_code="experiments/batch_wave_counts.py",
        code_location="batch_wave_counts.py count and observation_union",
        level="patient",
        unit="count/min",
        directionality="rate",
        physical_meaning="算法显著传播活动次数除以有效观察区间并集分钟数",
        patient_aggregation="event count divided by union of supported observation intervals",
        current_status="exploratory",
        posthoc_feature="no",
        tier="C",
        role="exclude",
        reason_in="报告率和缺失机制比原始次数更可比",
        reason_out="事件身份与漏检问题未解决，不能视为真实蠕动频率",
    ),
    dict(
        feature_name="other_significant_activity_count",
        standardized_name="other_significant_activity_count",
        feature_group="Event",
        source_code="experiments/batch_wave_counts.py",
        code_location="batch_wave_counts.py",
        level="patient",
        unit="count",
        directionality="count",
        physical_meaning="未归入主要显著传播活动的其他显著活动数",
        patient_aggregation="algorithm candidate aggregation; no direction merge",
        current_status="exploratory",
        posthoc_feature="no",
        tier="C",
        role="exclude",
        reason_in="保留局部/未定活动证据",
        reason_out="活动身份未验证，不能进入模型",
    ),
    dict(
        feature_name="record_MSE_time_equal",
        standardized_name="complete_record_ER_MSE",
        feature_group="ER-MSE/Disparity",
        source_code="experiments/complete_record_coordination.py",
        code_location="complete_record_coordination.py:93",
        level="patient",
        unit="squared ER difference",
        directionality="unsigned disparity",
        physical_meaning="所有共同支持20秒窗口中前后壁ER平方差的时间等权均值",
        patient_aggregation="time-equal across shared window times; spatial blocks first averaged within time",
        current_status="frozen",
        posthoc_feature="no",
        tier="A",
        role="candidate predictor",
        reason_in="完整记录定义已冻结；45秒桥接、rigid/support审计和内部hold-out均完成",
        reason_out="主要反映平均ER偏置；45秒位置异质性；仍需独立验证",
    ),
    dict(
        feature_name="record_MSE_pooled",
        standardized_name="complete_record_ER_MSE_window_weighted",
        feature_group="ER-MSE/Disparity",
        source_code="experiments/complete_record_coordination.py",
        code_location="complete_record_coordination.py:93",
        level="patient",
        unit="squared ER difference",
        directionality="unsigned disparity",
        physical_meaning="每个前后壁配对window-block等权的敏感性MSE",
        patient_aggregation="pooled across available window-block observations",
        current_status="frozen",
        posthoc_feature="no",
        tier="B",
        role="exploratory predictor",
        reason_in="用于检查多空间block权重",
        reason_out="不是主聚合；与主MSE不等价",
    ),
    dict(
        feature_name="record_CC_time_equal",
        standardized_name="complete_record_ER_CC",
        feature_group="ER-MSE/Disparity",
        source_code="experiments/complete_record_coordination.py",
        code_location="complete_record_coordination.py:93",
        level="patient",
        unit="Pearson correlation",
        directionality="signed similarity",
        physical_meaning="双壁ER时间序列（时间等权聚合后）的零时延Pearson相关",
        patient_aggregation="time-equal shared ER series; no gap filling",
        current_status="frozen",
        posthoc_feature="no",
        tier="A",
        role="candidate predictor",
        reason_in="定义冻结；已知同步/相反控制",
        reason_out="时间缺口下仅描述性；未完成45秒位置稳定性桥接",
    ),
    dict(
        feature_name="record_ER_mean_anterior",
        standardized_name="complete_record_mean_ER_anterior",
        feature_group="ER/Direction",
        source_code="experiments/complete_record_coordination.py",
        code_location="complete_record_coordination.py:56",
        level="patient",
        unit="unitless ER",
        directionality="signed",
        physical_meaning="前壁时间等权平均ER",
        patient_aggregation="mean of available shared ER windows",
        current_status="exploratory",
        posthoc_feature="yes",
        tier="B",
        role="explanatory only",
        reason_in="直接解释MSE均值偏置来源",
        reason_out="看到主MSE关联后拆分，不能替代primary",
    ),
    dict(
        feature_name="record_ER_mean_posterior",
        standardized_name="complete_record_mean_ER_posterior",
        feature_group="ER/Direction",
        source_code="experiments/complete_record_coordination.py",
        code_location="complete_record_coordination.py:56",
        level="patient",
        unit="unitless ER",
        directionality="signed",
        physical_meaning="后壁时间等权平均ER",
        patient_aggregation="mean of available shared ER windows",
        current_status="exploratory",
        posthoc_feature="yes",
        tier="B",
        role="explanatory only",
        reason_in="解释MSE均值偏置来源",
        reason_out="post-hoc explanatory feature",
    ),
    dict(
        feature_name="record_ER_sd_anterior",
        standardized_name="complete_record_ER_SD_anterior",
        feature_group="ER/Direction",
        source_code="experiments/complete_record_coordination.py",
        code_location="complete_record_coordination.py:56",
        level="patient",
        unit="ER SD",
        directionality="unsigned scale",
        physical_meaning="前壁ER时间序列标准差",
        patient_aggregation="standard deviation over time-equal shared ER values",
        current_status="exploratory",
        posthoc_feature="yes",
        tier="B",
        role="explanatory only",
        reason_in="解释MSE尺度差异",
        reason_out="post-hoc decomposition",
    ),
    dict(
        feature_name="record_ER_sd_posterior",
        standardized_name="complete_record_ER_SD_posterior",
        feature_group="ER/Direction",
        source_code="experiments/complete_record_coordination.py",
        code_location="complete_record_coordination.py:56",
        level="patient",
        unit="ER SD",
        directionality="unsigned scale",
        physical_meaning="后壁ER时间序列标准差",
        patient_aggregation="standard deviation over time-equal shared ER values",
        current_status="exploratory",
        posthoc_feature="yes",
        tier="B",
        role="explanatory only",
        reason_in="解释MSE尺度差异",
        reason_out="post-hoc decomposition",
    ),
    dict(
        feature_name="signed_ER_difference",
        standardized_name="complete_record_signed_ER_bias",
        feature_group="ER/Direction",
        source_code="experiments/er_mean_offset_batch_audit.py",
        code_location="derived from complete_record mean anterior and posterior ER",
        level="patient",
        unit="unitless ER difference",
        directionality="signed",
        physical_meaning="前壁平均ER减后壁平均ER",
        patient_aggregation="difference of time-equal wall means",
        current_status="exploratory",
        posthoc_feature="yes",
        tier="B",
        role="explanatory only",
        reason_in="定位均值偏置方向",
        reason_out="post-hoc decomposition",
    ),
    dict(
        feature_name="abs_ER_difference",
        standardized_name="complete_record_absolute_ER_bias",
        feature_group="ER/Direction",
        source_code="experiments/er_mean_offset_batch_audit.py",
        code_location="absolute value of complete_record signed ER bias",
        level="patient",
        unit="unitless ER difference",
        directionality="unsigned",
        physical_meaning="前后壁平均ER差异的绝对大小",
        patient_aggregation="absolute difference of time-equal wall means",
        current_status="exploratory",
        posthoc_feature="yes",
        tier="B",
        role="explanatory only",
        reason_in="解释MSE均值偏置项大小",
        reason_out="post-hoc decomposition",
    ),
    dict(
        feature_name="record_mean_bias_sq",
        standardized_name="complete_record_mean_bias_sq",
        feature_group="ER-MSE/Disparity",
        source_code="experiments/complete_record_coordination.py",
        code_location="complete_record_coordination.py:56",
        level="patient",
        unit="squared ER difference",
        directionality="unsigned",
        physical_meaning="前后壁平均ER差的平方",
        patient_aggregation="record-level ER decomposition term",
        current_status="exploratory",
        posthoc_feature="yes",
        tier="B",
        role="explanatory only",
        reason_in="解释MSE来源",
        reason_out="看到MSE关联后拆分，不自动成为新primary",
    ),
    dict(
        feature_name="record_sd_scale_sq",
        standardized_name="complete_record_SD_scale_sq",
        feature_group="ER-MSE/Disparity",
        source_code="experiments/complete_record_coordination.py",
        code_location="complete_record_coordination.py:56",
        level="patient",
        unit="squared SD difference",
        directionality="unsigned",
        physical_meaning="双壁ER标准差差的平方",
        patient_aggregation="record-level ER decomposition term",
        current_status="exploratory",
        posthoc_feature="yes",
        tier="B",
        role="explanatory only",
        reason_in="解释MSE尺度差异",
        reason_out="post-hoc decomposition",
    ),
    dict(
        feature_name="record_time_shape_term",
        standardized_name="complete_record_time_shape_term",
        feature_group="ER-MSE/Disparity",
        source_code="experiments/complete_record_coordination.py",
        code_location="complete_record_coordination.py:56",
        level="patient",
        unit="squared ER difference",
        directionality="unsigned",
        physical_meaning="MSE中由ER时间波形与相关系数贡献的项",
        patient_aggregation="record-level ER decomposition term",
        current_status="exploratory",
        posthoc_feature="yes",
        tier="B",
        role="explanatory only",
        reason_in="分离时间形状贡献",
        reason_out="post-hoc decomposition",
    ),
    dict(
        feature_name="record_MSE_z",
        standardized_name="complete_record_standardized_ER_MSE",
        feature_group="ER-MSE/Disparity",
        source_code="experiments/complete_record_coordination.py",
        code_location="complete_record_coordination.py:56",
        level="patient",
        unit="standardized squared difference",
        directionality="unsigned",
        physical_meaning="将前后壁ER各自标准化后再计算MSE，弱化均值/尺度偏置",
        patient_aggregation="standardize each wall over shared record windows",
        current_status="exploratory",
        posthoc_feature="yes",
        tier="B",
        role="explanatory only",
        reason_in="区分时间形状与均值偏置",
        reason_out="post-hoc decomposition; no independent confirmation",
    ),
    dict(
        feature_name="directional_rms_anterior_median",
        standardized_name="anterior_directional_rms_support",
        feature_group="ER/Direction",
        source_code="uterine_wave/spectrum.py; experiments/er_directional_support_audit.py",
        code_location="spectrum.py:53; er_directional_support_audit.py",
        level="patient",
        unit="direction-energy normalized RMS",
        directionality="unsigned support",
        physical_meaning="前壁方向谱支持度的患者中位数",
        patient_aggregation="median across available ER windows",
        current_status="exploratory",
        posthoc_feature="no",
        tier="C",
        role="QC only",
        reason_in="判断低方向能量下ER比例是否失真",
        reason_out="support不是独立生理真值，不设排除阈值",
    ),
    dict(
        feature_name="directional_rms_posterior_median",
        standardized_name="posterior_directional_rms_support",
        feature_group="ER/Direction",
        source_code="uterine_wave/spectrum.py; experiments/er_directional_support_audit.py",
        code_location="spectrum.py:53; er_directional_support_audit.py",
        level="patient",
        unit="direction-energy normalized RMS",
        directionality="unsigned support",
        physical_meaning="后壁方向谱支持度的患者中位数",
        patient_aggregation="median across available ER windows",
        current_status="exploratory",
        posthoc_feature="no",
        tier="C",
        role="QC only",
        reason_in="support审计",
        reason_out="不设阈值、不单独作为生理预测量",
    ),
    dict(
        feature_name="directional_support_ant_minus_post",
        standardized_name="anterior_minus_posterior_directional_support",
        feature_group="ER/Direction",
        source_code="experiments/er_directional_support_audit.py",
        code_location="wall_support_asymmetry calculation",
        level="patient",
        unit="direction-energy normalized RMS",
        directionality="signed",
        physical_meaning="前壁减后壁中位方向支持",
        patient_aggregation="difference of wall medians",
        current_status="exploratory",
        posthoc_feature="no",
        tier="C",
        role="QC only",
        reason_in="检查壁间support偏置",
        reason_out="与绝对ER偏置基本无关，不进入模型",
    ),
]

CURRENT_QC_SPECS = [
    dict(
        feature_name=name,
        standardized_name=name,
        feature_group="QC/Technical",
        source_file="current frozen outputs / model-input master",
        source_code="uterine_wave/frozen_pipeline.py; experiments/complete_record_coordination.py",
        code_location="frozen_pipeline.py:33; complete_record_coordination.py:93",
        level="patient",
        unit=unit,
        directionality="technical",
        physical_meaning=meaning,
        patient_aggregation="examination-level technical summary",
        current_status="exploratory",
        outcome_informed="no",
        posthoc_feature="no",
        tier="C",
        role="QC only",
        reason_in="用于解释可计算性、观察支持或批次差异",
        reason_out="不是生物运动效应量，不能作为独立预测特征",
        source_population="current discovery + hold-out",
    )
    for name, unit, meaning in [
        ("video_duration_s", "seconds", "视频物理时长"),
        ("acquisition_date", "date", "检查日期"),
        ("video_year", "year", "检查年份，仅作批次/来源描述"),
        ("source", "category", "采集来源/胚胎来源文件夹"),
        ("source_batch", "category", "source与年份组成的描述性批次"),
        ("anterior_valid_fraction", "fraction", "前壁有效追踪比例"),
        ("posterior_valid_fraction", "fraction", "后壁有效追踪比例"),
        ("anterior_common_region_duration_s", "seconds", "前壁共同有效观察时长"),
        ("posterior_common_region_duration_s", "seconds", "后壁共同有效观察时长"),
        ("mode_window_count", "count", "前后壁ER窗口总数"),
        ("paired_window_count", "count", "前后壁配对ER窗口数"),
        ("effective_observation_seconds", "seconds", "活动率分母的有效观察区间并集"),
        ("raw_valid_fraction", "fraction", "活动计数输入图的原始有效比例"),
        ("native_section_count", "count", "原生空间section数量"),
        ("median_support", "directional RMS", "患者ER窗口方向支持中位数"),
        ("p10_support", "directional RMS", "患者ER窗口方向支持第10百分位"),
        ("fraction_below_real_Q1", "fraction", "低于队列Q1 support的窗口比例，仅作描述"),
        ("fraction_below_real_median", "fraction", "低于队列中位support的窗口比例，仅作描述"),
    ]
]

LEGACY_SOURCE_DESCRIPTIONS = {
    "full_dynamic": (
        "R2_usable_dynamic_features_328_20261111/full candidate dynamic",
        "frozen_longitudinal_posterior_v1/extract_dynamic_features.py",
        "extract_dynamic_features.py:129",
    ),
    "check_v2": (
        "R2_literature_aligned_features_v2_20261112/check_features_v2_combined.csv",
        "frozen_longitudinal_posterior_v1/extract_dynamic_features.py",
        "extract_dynamic_features.py:129",
    ),
    "dynamic_patient": (
        "R2_dynamic_features_combined328_outcome_blind_20261110/dynamic_patient_features.csv",
        "frozen_longitudinal_posterior_v1/extract_dynamic_features.py",
        "extract_dynamic_features.py:265",
    ),
    "frozen_coordination": (
        "frozen_radial_coordination_final_v1/patient coordination table",
        "frozen_longitudinal_posterior_v1/extract_dynamic_features.py",
        "extract_dynamic_features.py:211",
    ),
    "spatial_pilot": (
        "R2_spatial_propagation_v2_technical_pilot_5_20261111/patient exploratory",
        "frozen_longitudinal_posterior_v1/extract_p3_spatiotemporal_features.py",
        "extract_p3_spatiotemporal_features.py:408",
    ),
    "propagation_pilot": (
        "R2_spatial_propagation_v2_technical_pilot_5_20261111/propagation patient",
        "frozen_longitudinal_posterior_v1/extract_p3_spatiotemporal_features.py",
        "extract_p3_spatiotemporal_features.py:408",
    ),
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _find_csv(parent, *, rows, required_columns, columns=None):
    matches = []
    for path in sorted(Path(parent).glob("*.csv")):
        try:
            data = pd.read_csv(path, nrows=1)
        except (OSError, pd.errors.ParserError):
            continue
        if "case_id" not in data.columns:
            continue
        full = pd.read_csv(path, usecols=["case_id"])
        if len(full) != rows or not set(required_columns).issubset(data.columns):
            continue
        if columns is not None and len(data.columns) != columns:
            continue
        matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"No CSV in {parent} matched rows={rows}, required={required_columns}"
        )
    return matches


def _source_from_path(path):
    parts = Path(path).parts
    for name in ("囊胚", "细胞胚"):
        if name in parts:
            return name
    return "unknown"


def _date_from_case(case_id):
    match = re.search(r"(20\d{2})(\d{2})(\d{2})", str(case_id))
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else None


def _clinical_frame(previous, workbook):
    _, features, audit, _ = load_clinical_features(previous, workbook)
    columns = list(dict.fromkeys(BASELINE + ALL_CLINICAL))
    return features[[EXAM] + columns].rename(columns={EXAM: "exam_id"}), audit


def _current_master(args):
    base = pd.read_csv(args.discovery_auc_previous / "analysis_cohort.csv")
    clinical, _ = _clinical_frame(args.discovery_auc_previous, args.clinical_workbook)
    coordination = pd.read_csv(
        args.complete_record_root / "record_coordination.csv"
    )
    wave = pd.read_csv(args.wave_counts_root / "wave_counts.csv")
    support = pd.read_csv(args.support_audit_root / "record_level_support.csv")
    support = support.loc[support["branch"].eq("A_2s")].copy()
    wall_support = pd.read_csv(
        args.support_audit_root / "wall_support_asymmetry.csv"
    )
    wall_support = wall_support.loc[wall_support["branch"].eq("A_2s")].copy()
    windows = pd.read_csv(args.support_audit_root / "window_level_support.csv")
    windows = windows.loc[
        windows["condition"].eq("real") & windows["branch"].eq("A_2s")
    ].copy()
    band = (
        windows.groupby(["case_id", "wall_name"], as_index=False)
        .agg(median_band_fraction=("directional_band_energy_fraction", "median"))
        .pivot(index="case_id", columns="wall_name", values="median_band_fraction")
        .reset_index()
        .rename(columns={
            "anterior": "directional_band_fraction_anterior_median",
            "posterior": "directional_band_fraction_posterior_median",
        })
    )
    support_keep = support[[
        "case_id", "median_support", "p10_support",
        "fraction_below_real_Q1", "fraction_below_real_median",
    ]]
    wall_keep = wall_support[[
        "case_id", "anterior_median_support", "posterior_median_support",
        "support_difference_ant_minus_post",
    ]]
    old_tables = _find_csv(
        args.legacy_root / "outputs" / "R2_usable_dynamic_features_328_20261111",
        rows=328,
        required_columns=["longitudinal_posterior_frequency_cpm"],
        columns=13,
    )
    old = pd.read_csv(old_tables[0])

    discovery = pd.DataFrame({
        "patient_id": base[GROUP].astype(str),
        "exam_id": base[EXAM].astype(str),
        "case_id": base["case_id"].astype(str),
        "cohort": "discovery",
        "acquisition_date": base["case_id"].map(_date_from_case),
        "source": base["video_path"].map(_source_from_path),
        "video_duration_s": base["video_duration_s"],
        "outcome": pd.to_numeric(base[OUTCOME], errors="coerce"),
        "processing_status": base["processing_status"],
        "failure_reason": base["failure_reason"],
        "anterior_valid_fraction": base["anterior_valid_fraction"],
        "posterior_valid_fraction": base["posterior_valid_fraction"],
        "anterior_common_region_duration_s": base["anterior_common_region_duration_s"],
        "posterior_common_region_duration_s": base["posterior_common_region_duration_s"],
        "mode_window_count": base["mode_window_count"],
        "paired_window_count": base["paired_ER_window_count"],
        "anterior_radial_stretch_activity_amplitude": base[
            "anterior_radial_stretch_activity_amplitude"
        ],
        "posterior_radial_stretch_activity_amplitude": base[
            "posterior_radial_stretch_activity_amplitude"
        ],
        "significant_propagation_activity_count": base[
            "significant_propagation_activity_count"
        ],
        "significant_propagation_activity_frequency_cpm": base[
            "significant_propagation_activity_frequency_cpm"
        ],
        "single_block_CC": base["single_block_CC"],
        "single_block_MSE": base["single_block_MSE"],
    })
    discovery = discovery.merge(clinical, on="exam_id", how="left", validate="one_to_one")
    discovery = discovery.merge(
        coordination[[
            "case_id", "record_MSE_time_equal", "record_MSE_pooled",
            "record_CC_time_equal", "record_ER_mean_anterior",
            "record_ER_mean_posterior", "record_ER_sd_anterior",
            "record_ER_sd_posterior", "record_mean_bias_sq",
            "record_sd_scale_sq", "record_time_shape_term", "record_MSE_z",
        ]],
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    discovery = discovery.merge(
        wave[[
            "case_id", "other_significant_activity_count",
            "effective_observation_seconds", "unresolved_activity_rows",
            "raw_valid_fraction", "native_section_count",
        ]],
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    discovery = discovery.merge(support_keep, on="case_id", how="left", validate="one_to_one")
    discovery = discovery.merge(wall_keep, on="case_id", how="left", validate="one_to_one")
    discovery = discovery.merge(band, on="case_id", how="left", validate="one_to_one")
    discovery = discovery.merge(old, on="case_id", how="left", validate="one_to_one")
    discovery["signed_ER_difference"] = (
        discovery["record_ER_mean_anterior"] - discovery["record_ER_mean_posterior"]
    )
    discovery["abs_ER_difference"] = discovery["signed_ER_difference"].abs()
    discovery["directional_rms_anterior_median"] = discovery["anterior_median_support"]
    discovery["directional_rms_posterior_median"] = discovery["posterior_median_support"]
    discovery["directional_support_ant_minus_post"] = discovery[
        "support_difference_ant_minus_post"
    ]

    holdout_manifest = pd.read_csv(args.holdout_measurement_manifest)
    holdout_frozen = pd.read_csv(args.holdout_measurement_frozen)
    holdout_outcomes = pd.read_csv(args.holdout_outcomes)
    inputs = json.loads(args.holdout_input_manifest.read_text(encoding="utf-8"))["cases"]
    path_by_case = {row["case_id"]: row["video_path"] for row in inputs}
    holdout = pd.DataFrame({
        "patient_id": "HOLDOUT:" + holdout_manifest["case_id"].astype(str),
        "exam_id": holdout_manifest["case_id"].astype(str),
        "case_id": holdout_manifest["case_id"].astype(str),
        "cohort": "short_holdout",
        "acquisition_date": holdout_manifest["case_id"].map(_date_from_case),
        "source": holdout_manifest["case_id"].map(
            lambda value: _source_from_path(path_by_case.get(value, ""))
        ),
        "video_duration_s": holdout_manifest["video_duration_s"],
        "processing_status": holdout_manifest["processing_status"],
        "failure_reason": holdout_manifest["failure_reason"],
        "anterior_valid_fraction": holdout_manifest["anterior_valid_fraction"],
        "posterior_valid_fraction": holdout_manifest["posterior_valid_fraction"],
        "anterior_common_region_duration_s": holdout_manifest[
            "anterior_common_region_duration_s"
        ],
        "posterior_common_region_duration_s": holdout_manifest[
            "posterior_common_region_duration_s"
        ],
        "mode_window_count": holdout_manifest["mode_window_count"],
        "paired_window_count": holdout_manifest["paired_ER_window_count"],
        "anterior_radial_stretch_activity_amplitude": holdout_manifest[
            "anterior_radial_stretch_activity_amplitude"
        ],
        "posterior_radial_stretch_activity_amplitude": holdout_manifest[
            "posterior_radial_stretch_activity_amplitude"
        ],
    })
    holdout = holdout.merge(
        holdout_frozen[[
            "case_id", "MSE", "CC", "mean_bias_sq", "sd_scale_sq",
            "time_shape_term", "MSE_z", "support_ant_median",
            "support_post_median", "band_fraction_ant_median",
            "band_fraction_post_median", "mean_ER_ant", "mean_ER_post",
            "signed_mean_bias", "abs_mean_bias",
        ]].rename(columns={
            "MSE": "record_MSE_time_equal",
            "CC": "record_CC_time_equal",
            "mean_bias_sq": "record_mean_bias_sq",
            "sd_scale_sq": "record_sd_scale_sq",
            "time_shape_term": "record_time_shape_term",
            "MSE_z": "record_MSE_z",
            "support_ant_median": "directional_rms_anterior_median",
            "support_post_median": "directional_rms_posterior_median",
            "band_fraction_ant_median": "directional_band_fraction_anterior_median",
            "band_fraction_post_median": "directional_band_fraction_posterior_median",
            "mean_ER_ant": "record_ER_mean_anterior",
            "mean_ER_post": "record_ER_mean_posterior",
            "signed_mean_bias": "signed_ER_difference",
            "abs_mean_bias": "abs_ER_difference",
        }),
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    holdout = holdout.merge(
        holdout_outcomes[["case_id", "pregnancy"]],
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    holdout["outcome"] = holdout.pop("pregnancy")
    holdout = holdout.merge(old, on="case_id", how="left", validate="one_to_one")
    holdout = holdout.merge(
        wall_support[["case_id", "support_difference_ant_minus_post"]],
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    holdout["directional_support_ant_minus_post"] = holdout[
        "directional_rms_anterior_median"
    ] - holdout["directional_rms_posterior_median"]

    for column in ALL_CLINICAL:
        if column not in holdout:
            holdout[column] = np.nan
    master = pd.concat([discovery, holdout], ignore_index=True, sort=False)
    master["video_year"] = pd.to_datetime(
        master["acquisition_date"], errors="coerce"
    ).dt.year
    master["source_batch"] = (
        master["source"].astype(str) + ":" + master["video_year"].astype("Int64").astype(str)
    )
    for column in ALL_CLINICAL + [
        "anterior_radial_stretch_activity_amplitude",
        "posterior_radial_stretch_activity_amplitude",
        "single_block_CC", "single_block_MSE",
        "record_MSE_time_equal", "record_MSE_pooled", "record_CC_time_equal",
        "record_ER_mean_anterior", "record_ER_mean_posterior",
        "record_ER_sd_anterior", "record_ER_sd_posterior",
        "signed_ER_difference", "abs_ER_difference",
        "record_mean_bias_sq", "record_sd_scale_sq",
        "record_time_shape_term", "record_MSE_z",
        "significant_propagation_activity_count",
        "significant_propagation_activity_frequency_cpm",
        "other_significant_activity_count",
        "directional_rms_anterior_median", "directional_rms_posterior_median",
        "directional_band_fraction_anterior_median",
        "directional_band_fraction_posterior_median",
        "directional_support_ant_minus_post", "paired_window_count",
    ]:
        if column not in master:
            master[column] = np.nan
    master["clinical_baseline7_available"] = master[BASELINE].notna().all(axis=1)
    master["clinical_all19_available"] = master[ALL_CLINICAL].notna().all(axis=1)
    master["amplitude_bilateral_available"] = master[[
        "anterior_radial_stretch_activity_amplitude",
        "posterior_radial_stretch_activity_amplitude",
    ]].notna().all(axis=1)
    master["current_event_frequency_available"] = master[
        "significant_propagation_activity_frequency_cpm"
    ].notna()
    master["ER_MSE_available"] = master["record_MSE_time_equal"].notna()
    master["ER_family_available"] = master[[
        "record_MSE_time_equal", "record_CC_time_equal",
        "record_ER_mean_anterior", "record_ER_mean_posterior",
    ]].notna().all(axis=1)
    master["legacy_frequency_available"] = master[[
        "longitudinal_anterior_frequency_cpm",
        "longitudinal_posterior_frequency_cpm",
    ]].notna().all(axis=1)
    master["legacy_strain_rate_available"] = master[
        "posterior_longitudinal_strain_rate_rms_per_s_exploratory"
    ].notna()
    master["legacy_velocity_available"] = master[
        "posterior_motion_velocity_pct_long_axis_per_s_exploratory"
    ].notna()
    master["all_defensible_dynamic_available"] = master[[
        "anterior_radial_stretch_activity_amplitude",
        "posterior_radial_stretch_activity_amplitude",
        "record_MSE_time_equal", "record_CC_time_equal",
    ]].notna().all(axis=1)
    return master, old


def _canonical_current_rows(master, availability):
    rows = []
    for spec in CURRENT_SPECS:
        row = dict(spec)
        row.update(
            source_file="current frozen outputs / model-input master",
            outcome_informed="no",
            source_population="current discovery + hold-out",
        )
        rows.append(row)
    return rows


def _clinical_rows(audit):
    rows = []
    for audit_row in audit.itertuples(index=False):
        name = audit_row.output_feature
        unit, meaning = CLINICAL_MEANING[name]
        in_baseline = name in BASELINE
        in_all = name in ALL_CLINICAL
        tier = "A" if in_baseline else "B"
        rows.append(dict(
            feature_name=name,
            standardized_name=name,
            feature_group="Clinical",
            source_file="最终人工复核队列_358次检查.xlsx",
            source_code="experiments/clinical_variable_screen.py",
            code_location="clinical_variable_screen.py:185",
            level="patient",
            unit=unit,
            directionality=(
                "categorical" if name in {
                    "uterine_position", "endometrial_type", "transferred_embryo_stage"
                } else "signed numeric"
            ),
            physical_meaning=meaning,
            patient_aggregation="none; examination-level clinical value",
            current_status="frozen" if in_baseline else "exploratory",
            outcome_informed="no",
            posthoc_feature="no",
            tier=tier,
            role="fixed clinical" if in_baseline else "candidate predictor",
            reason_in="既定临床基线变量" if in_baseline else "此前all-clinical探索变量",
            reason_out=(
                "缺少人体测量学重复性；不替代结局验证"
                if in_baseline else "内部探索；存在重复/派生编码"
            ),
            used_baseline7=in_baseline,
            used_all_clinical=in_all,
            source_population="current discovery only",
        ))
    return rows


def _classify_legacy(feature, table_code):
    name = str(feature)
    lower = name.lower()
    if any(token in lower for token in ("available", "qc", "valid", "count")):
        return "QC/Technical", "QC/flag/count", "QC candidate coverage or support"
    if "frequency" in lower or lower.endswith("_cpm"):
        return "Frequency", "count/min", "纵向组织运动频率或双壁频率组合"
    if any(token in lower for token in ("strain_rate", "strain")):
        return "Amplitude/Strain", "1/s or percent", "纵向应变率汇总或区域对比"
    if any(token in lower for token in ("velocity", "motion")):
        return "Motion/Displacement/Velocity", "percent/s or percent", "组织运动速度或幅度汇总"
    if any(token in lower for token in ("amplitude", "p95_p05")):
        return "Amplitude/Strain", "percent or px", "组织运动幅度或峰谷差"
    if any(token in lower for token in (
        "hausdorff", "correlation", "phase_locking", "phase_difference",
        "same_direction", "standardized_mse", "coordination",
    )):
        return "ER-MSE/Disparity", "correlation/distance/squared disparity", "双壁时间轨迹或空间曲线协调性"
    if any(token in lower for token in ("propagation", "arrival", "delay", "direction")):
        return "Propagation/Spacetime", "s, path/s, direction", "传播时延、速度或方向"
    return "QC/Technical", "not_defined", "旧技术或辅助量，定义需回看源码"


def _legacy_rows(root, old_recommended):
    rows = []
    table_specs = []
    usable_dir = root / "outputs" / "R2_usable_dynamic_features_328_20261111"
    tables = _find_csv(
        usable_dir, rows=328,
        required_columns=["longitudinal_posterior_frequency_cpm"], columns=25,
    )
    table_specs.append(("full_dynamic", pd.read_csv(tables[0])))
    check_path = root / "outputs" / "R2_literature_aligned_features_v2_20261112" / "check_features_v2_combined.csv"
    table_specs.append(("check_v2", pd.read_csv(check_path)))
    dynamic_path = root / "outputs" / "R2_dynamic_features_combined328_outcome_blind_20261110" / "dynamic_patient_features.csv"
    table_specs.append(("dynamic_patient", pd.read_csv(dynamic_path)))
    frozen_dir = root / "frozen_radial_coordination_final_v1" / "records"
    frozen_tables = _find_csv(
        frozen_dir, rows=232, required_columns=["coordination_mse"], columns=17
    )
    table_specs.append(("frozen_coordination", pd.read_csv(frozen_tables[0])))
    spatial_path = root / "outputs" / "R2_spatial_propagation_v2_technical_pilot_5_20261111" / "patient_exploratory_features_v2.csv"
    table_specs.append(("spatial_pilot", pd.read_csv(spatial_path)))
    propagation_path = root / "outputs" / "R2_spatial_propagation_v2_technical_pilot_5_20261111" / "propagation_patient_features.csv"
    table_specs.append(("propagation_pilot", pd.read_csv(propagation_path)))

    recommended = {
        "longitudinal_posterior_frequency_cpm",
        "longitudinal_anterior_frequency_cpm",
        "longitudinal_anterior_posterior_abs_difference_cpm",
        "posterior_motion_amplitude_pct_local_width_exploratory",
        "posterior_motion_velocity_pct_long_axis_per_s_exploratory",
        "posterior_longitudinal_strain_rate_rms_per_s_exploratory",
        "longitudinal_ap_zero_lag_correlation_exploratory",
        "longitudinal_ap_max_correlation_lag_s_exploratory",
        "longitudinal_ap_phase_locking_value_exploratory",
        "inward_ap_zero_lag_correlation_exploratory",
        "inward_ap_max_correlation_lag_s_exploratory",
        "inward_ap_phase_locking_value_exploratory",
    }
    for table_code, data in table_specs:
        source_label, source_code, location = LEGACY_SOURCE_DESCRIPTIONS[table_code]
        for column in data.columns:
            if column in ("case_id", "source_cohort"):
                continue
            group, unit, meaning = _classify_legacy(column, table_code)
            recommended_flag = column in recommended
            if table_code == "spatial_pilot":
                tier = "C"
                status = "deprecated"
                reason_out = "仅5例技术pilot，不能作为当前A0正式特征"
            elif table_code == "propagation_pilot":
                tier = "C"
                status = "exploratory"
                reason_out = "仅5例技术pilot；速度和方向未在真实患者中验证"
            elif group == "Frequency" and recommended_flag:
                tier = "B"
                status = "deprecated"
                reason_out = "旧R2纵向组织频率；定义有稳定性证据，但未接入当前A0正式链"
            elif group in ("Motion/Displacement/Velocity", "Amplitude/Strain"):
                tier = "C"
                status = "deprecated"
                reason_out = "旧R2探索量；已知图像形变测试和当前A0定义不可直接等价"
            elif group == "ER-MSE/Disparity":
                tier = "C"
                status = "deprecated"
                reason_out = "旧R2协调性实现；部分指标窗口稳定性弱，当前正式链改用完整记录ER"
            else:
                tier = "C"
                status = "unclear" if unit == "not_defined" else "deprecated"
                reason_out = (
                    "定义线索不足，暂标definition_unclear"
                    if status == "unclear"
                    else "旧流程/技术量，不作为当前模型候选"
                )
            rows.append(dict(
                feature_name=column,
                standardized_name=re.sub(r"_exploratory$", "", column),
                feature_group=group,
                source_file=source_label,
                source_code=source_code,
                code_location=location,
                level="patient",
                unit=unit,
                directionality=(
                    "signed" if any(token in column.lower() for token in (
                        "signed", "anterior_minus_posterior", "fundal_minus_cervical"
                    )) else "unsigned/ratio/category"
                ),
                physical_meaning=meaning,
                patient_aggregation="legacy R2 patient aggregation from available regions/windows",
                current_status=status,
                outcome_informed="no",
                posthoc_feature="no",
                tier=tier,
                role=(
                    "candidate predictor" if tier == "B"
                    else "exclude"
                ),
                reason_in=(
                    "旧R2中标记为正式候选且具有窗口稳定性"
                    if recommended_flag and tier == "B"
                    else "保留历史特征来源和可用性证据"
                ),
                reason_out=reason_out,
                legacy_recommended=recommended_flag,
                source_population=str(source_label),
            ))
    return rows


def _availability_rows(inventory, master, discovery_ids, holdout_ids, source_tables):
    rows = []
    for item in inventory:
        name = item["feature_name"]
        if item["feature_group"] == "Clinical" or item["source_file"].startswith(
            "current frozen outputs"
        ):
            frame = master
            column = name
            source_n = len(master)
        else:
            key = item["source_file"]
            frame = source_tables[key]
            column = name
            source_n = len(frame)
        if column not in frame:
            available = pd.Series(False, index=frame.index)
        else:
            available = frame[column].notna()
        disc = frame.loc[frame["case_id"].isin(discovery_ids), column].notna() if column in frame else pd.Series(dtype=bool)
        hold = frame.loc[frame["case_id"].isin(holdout_ids), column].notna() if column in frame else pd.Series(dtype=bool)
        missing = int((~available).sum())
        if item["feature_group"] == "Clinical":
            clinical_link_missing = int((~available).sum())
            video_missing = 0
            tracking_missing = 0
            mechanism = "clinical table unavailable for hold-out and source-level missingness"
        elif item["source_file"].startswith("current frozen outputs"):
            video_missing = int((
                frame["video_duration_s"].lt(45) & ~available
            ).sum())
            tracking_missing = int((
                frame["processing_status"].ne("completed") & ~available
            ).sum())
            clinical_link_missing = 0
            mechanism = "video/support/tracking/missing-feature support; inspect row status"
        else:
            video_missing = np.nan
            tracking_missing = np.nan
            clinical_link_missing = np.nan
            mechanism = "not separable from archived legacy source table"
        rows.append(dict(
            feature_name=name,
            feature_group=item["feature_group"],
            source_file=item["source_file"],
            availability_population=item["source_population"],
            total_patient_exam_count=int(source_n),
            n_available=int(available.sum()),
            n_missing=missing,
            availability_percent=100 * float(available.mean()) if source_n else None,
            anterior_available=(
                int(available.sum()) if "anterior" in name and "posterior" not in name else None
            ),
            posterior_available=(
                int(available.sum()) if "posterior" in name and "anterior" not in name else None
            ),
            bilateral_available=(
                int(available.sum())
                if any(token in name for token in ("bilateral", "_ap_", "coordination", "record_", "single_block"))
                else None
            ),
            discovery_cohort_n=int(disc.sum()),
            holdout_cohort_n=int(hold.sum()),
            missing_due_to_video_duration_n=video_missing,
            missing_due_to_tracking_n=tracking_missing,
            missing_due_to_clinical_link_n=clinical_link_missing,
            missing_reason_summary=mechanism,
        ))
    return pd.DataFrame(rows)


def _validation_row(item):
    group = item["feature_group"]
    status = item["current_status"]
    tier = item["tier"]
    if group == "Clinical":
        values = dict(
            tracking_validated="not_applicable",
            synthetic_truth_test="no",
            rigid_control="not_applicable",
            brightness_control="not_applicable",
            repeatability_test="partial",
            split_half="no",
            duration_stability="not_applicable",
            support_audit="not_applicable",
            external_reference="yes",
            validation_result="partial",
            limitation="临床字段连接和重复患者一致性已核对；无人体影像测量学含义",
            recommended_use=item["role"],
        )
    elif group == "Amplitude/Strain" and item["source_file"].startswith("current frozen"):
        values = dict(
            tracking_validated="partial", synthetic_truth_test="yes",
            rigid_control="yes", brightness_control="no",
            repeatability_test="no", split_half="no",
            duration_stability="yes", support_audit="partial",
            external_reference="yes", validation_result="partial",
            limitation="已知形变和刚体控制支持定义；缺少人体重复采集与完整声学真值",
            recommended_use=item["role"],
        )
    elif group == "ER-MSE/Disparity" and item["source_file"].startswith("current frozen"):
        values = dict(
            tracking_validated="partial", synthetic_truth_test="yes",
            rigid_control="yes", brightness_control="no",
            repeatability_test="partial" if "CC" in item["feature_name"] else "no",
            split_half="no", duration_stability="yes",
            support_audit="yes", external_reference="yes",
            validation_result="partial" if item["posthoc_feature"] == "no" else "unclear",
            limitation="rigid/support、45秒桥接及hold-out已完成；MSE主要受均值偏置，post-hoc项不能替代primary",
            recommended_use=item["role"],
        )
    elif group == "ER-MSE/Disparity":
        values = dict(
            tracking_validated="partial", synthetic_truth_test="yes",
            rigid_control="partial", brightness_control="no",
            repeatability_test="partial", split_half="no",
            duration_stability="yes", support_audit="no",
            external_reference="yes", validation_result="partial",
            limitation="旧R2协调性定义有合成与窗口稳定性证据，但与当前完整记录ER-MSE不是同一实现",
            recommended_use=item["role"],
        )
    elif group == "ER/Direction":
        values = dict(
            tracking_validated="partial", synthetic_truth_test="yes",
            rigid_control="yes", brightness_control="no",
            repeatability_test="partial", split_half="no",
            duration_stability="yes" if item["feature_name"].startswith("record_") else "no",
            support_audit="yes", external_reference="yes",
            validation_result="partial",
            limitation="方向已知对照可恢复；低support可令ER比例失真，support不设硬阈值",
            recommended_use=item["role"],
        )
    elif group == "Event":
        values = dict(
            tracking_validated="partial", synthetic_truth_test="partial",
            rigid_control="yes", brightness_control="no",
            repeatability_test="no", split_half="no",
            duration_stability="no", support_audit="partial",
            external_reference="yes", validation_result="fail",
            limitation="事件身份、相邻重复和弱活动漏检未解决，不能当真实蠕动频率",
            recommended_use="exclude",
        )
    elif group == "Frequency":
        values = dict(
            tracking_validated="partial", synthetic_truth_test="yes",
            rigid_control="yes", brightness_control="no",
            repeatability_test="yes", split_half="no",
            duration_stability="yes", support_audit="no",
            external_reference="yes", validation_result="partial",
            limitation="旧R2纵向频率有稳定性和窗口敏感性证据，但未接入当前A0正式链",
            recommended_use=item["role"],
        )
    elif group in ("Motion/Displacement/Velocity", "Amplitude/Strain"):
        values = dict(
            tracking_validated="partial", synthetic_truth_test="yes",
            rigid_control="partial", brightness_control="partial",
            repeatability_test="partial", split_half="partial",
            duration_stability="yes", support_audit="no",
            external_reference="yes", validation_result="partial",
            limitation="已有R2窗口稳定性和split-half；已知应变误差边缘，且不是当前A0正式定义",
            recommended_use=item["role"],
        )
    elif group == "Propagation/Spacetime":
        values = dict(
            tracking_validated="partial", synthetic_truth_test="yes",
            rigid_control="partial", brightness_control="no",
            repeatability_test="no", split_half="no",
            duration_stability="no", support_audit="no",
            external_reference="yes", validation_result="unclear",
            limitation="合成传播通过，但真实速度/方向只在5例技术pilot中计算",
            recommended_use="exclude",
        )
    else:
        values = dict(
            tracking_validated="partial", synthetic_truth_test="no",
            rigid_control="no", brightness_control="no",
            repeatability_test="no", split_half="no",
            duration_stability="no", support_audit="partial",
            external_reference="yes", validation_result="partial",
            limitation="QC/technical字段只描述可用性或来源，不进入生理特征模型",
            recommended_use="QC only",
        )
    return dict(
        feature_name=item["feature_name"],
        feature_group=group,
        tier=tier,
        **values,
    )


def _candidate_rows(inventory, availability_index):
    selected_groups = {
        "Clinical", "Frequency", "Amplitude/Strain", "Motion/Displacement/Velocity",
        "ER/Direction", "ER-MSE/Disparity", "Propagation/Spacetime", "Event",
    }
    selected = []
    for item in inventory:
        if item["feature_group"] not in selected_groups:
            continue
        if item["source_file"].startswith("current frozen outputs") or item["feature_group"] == "Clinical":
            selected.append(item)
        elif item.get("legacy_recommended"):
            selected.append(item)
    rows = []
    for item in selected:
        avail = _availability_lookup(
            availability_index, item["feature_name"], item["source_file"]
        )
        rows.append(dict(
            feature_name=item["feature_name"],
            feature_group=item["feature_group"],
            tier=item["tier"],
            formal_or_exploratory=(
                "formal" if item["tier"] == "A" else "exploratory"
            ),
            posthoc=item["posthoc_feature"],
            n_available=int(avail["n_available"]),
            missing_rate=float(avail["n_missing"] / avail["total_patient_exam_count"]),
            measurement_validity=item.get("validation_result", "partial"),
            reason_for_inclusion=item["reason_in"],
            reason_for_exclusion=item["reason_out"],
            recommended_model_role=item["role"],
        ))
    return pd.DataFrame(rows).drop_duplicates("feature_name")


def _availability_lookup(availability_index, feature_name, source_file):
    selected = availability_index.loc[(feature_name, source_file)]
    return selected.iloc[0] if isinstance(selected, pd.DataFrame) else selected


def _sets(master):
    current_all_dynamic = [
        "anterior_radial_stretch_activity_amplitude",
        "posterior_radial_stretch_activity_amplitude",
        "significant_propagation_activity_count",
        "significant_propagation_activity_frequency_cpm",
        "single_block_CC",
        "single_block_MSE",
        "record_MSE_time_equal",
        "record_CC_time_equal",
        "record_ER_mean_anterior",
        "record_ER_mean_posterior",
        "record_mean_bias_sq",
        "record_sd_scale_sq",
        "record_time_shape_term",
        "record_MSE_z",
    ]
    return {
        "Set 0 Clinical only": BASELINE,
        "Set 1A Clinical + legacy longitudinal frequency": BASELINE + [
            "longitudinal_anterior_frequency_cpm",
            "longitudinal_posterior_frequency_cpm",
        ],
        "Set 1B Clinical + current significant-activity frequency": BASELINE + [
            "significant_propagation_activity_frequency_cpm",
        ],
        "Set 2 Clinical + complete-record ER-MSE": BASELINE + [
            "record_MSE_time_equal",
        ],
        "Set 3 Clinical + ER/MSE family": BASELINE + [
            "record_MSE_time_equal", "record_CC_time_equal",
            "record_ER_mean_anterior", "record_ER_mean_posterior",
        ],
        "Set 4 Clinical + amplitude": BASELINE + [
            "anterior_radial_stretch_activity_amplitude",
            "posterior_radial_stretch_activity_amplitude",
        ],
        "Set 5 Clinical + legacy strain-rate": BASELINE + [
            "posterior_longitudinal_strain_rate_rms_per_s_exploratory",
        ],
        "Set 6 Clinical + legacy velocity": BASELINE + [
            "posterior_motion_velocity_pct_long_axis_per_s_exploratory",
        ],
        "Set 7 Clinical + all currently defensible dynamic": BASELINE + [
            "anterior_radial_stretch_activity_amplitude",
            "posterior_radial_stretch_activity_amplitude",
            "record_MSE_time_equal", "record_CC_time_equal",
        ],
        "Set 8 All clinical + all dynamic": ALL_CLINICAL + current_all_dynamic,
    }


def _set_summary(master, sets):
    rows = []
    for cohort, frame in [
        ("all", master),
        ("discovery", master.loc[master["cohort"].eq("discovery")]),
        ("short_holdout", master.loc[master["cohort"].eq("short_holdout")]),
    ]:
        for name, columns in sets.items():
            present = [column for column in columns if column in frame]
            complete = frame[present].notna().all(axis=1)
            outcome_available = frame["outcome"].notna()
            complete_outcome = complete & outcome_available
            events = int(frame.loc[complete_outcome, "outcome"].eq(1).sum())
            non_events = int(frame.loc[complete_outcome, "outcome"].eq(0).sum())
            rows.append(dict(
                cohort=cohort,
                feature_set=name,
                n_exams=len(frame),
                n_unique_patients=frame["patient_id"].nunique(),
                repeated_patient_count=int((frame.groupby("patient_id").size() > 1).sum()),
                pregnancy_outcome_available_count=int(outcome_available.sum()),
                complete_case_n=int(complete.sum()),
                complete_case_patients=int(
                    frame.loc[complete, "patient_id"].nunique()
                ),
                complete_case_with_outcome_n=int(complete_outcome.sum()),
                missing_rate_in_complete_case_definition=(
                    1 - float(complete.sum() / len(frame)) if len(frame) else None
                ),
                feature_count=len(present),
                events=events,
                non_events=non_events,
                event_non_event_ratio=(events / non_events if non_events else None),
            ))
    return pd.DataFrame(rows)


def _clinical_inventory(master):
    audit = pd.read_csv(
        Path("outputs") / "clinical_variable_screen_223_20261119" /
        "clinical_feature_audit.csv"
    )
    rows = []
    discovery = master.loc[master["cohort"].eq("discovery")].copy()
    duration_pairs = discovery[["infertility_years", "infertility_months"]].dropna()
    residual_match = int((
        ((duration_pairs["infertility_years"] * 12) % 12
         - duration_pairs["infertility_months"]).abs() < 0.05
    ).sum())
    for row in audit.itertuples(index=False):
        name = row.output_feature
        values = master.loc[master["cohort"].eq("discovery"), [name, "patient_id"]].copy()
        duplicate_values = 0
        repeated_patients = 0
        for _, group in values.groupby("patient_id"):
            unique = group[name].dropna()
            if len(unique):
                repeated_patients += int(len(unique) > 1)
                duplicate_values += int(unique.nunique() > 1)
        level_values = sorted(map(str, values[name].dropna().unique()))[:20]
        suspicious = []
        if name in ("infertility_years", "infertility_months"):
            suspicious.append(
                "同一次检查拆分年值与月余数；月值不是独立总月数，"
                f"年值小数部分与月字段在{residual_match}/{len(duration_pairs)}对中一致"
            )
        if name in (
            "transfer_embryo_count_multiple", "transfer_embryo_count_component",
            "blastocyst_transfer", "transferred_embryo_stage",
        ):
            suspicious.append("与胚胎数/阶段存在派生或高度相关关系")
        if name == "clinical_peristalsis_positive_disordered":
            suspicious.append("以文本类别“杂乱无序”另设指示，不表示0次")
        numeric_values = pd.to_numeric(
            master.loc[master["cohort"].eq("discovery"), name],
            errors="coerce",
        ).dropna()
        if name == "doppler_FI" and (numeric_values > 80).any():
            suspicious.append(
                f"存在{int((numeric_values > 80).sum())}个FI>80的极端值，需回查小数位/单位"
            )
        observed_range = (
            f"{float(numeric_values.min()):g}–{float(numeric_values.max()):g}"
            if len(numeric_values) else None
        )
        rows.append(dict(
            name=name,
            source="最终人工复核队列_358次检查.xlsx / clinical_variable_screen.py",
            definition=CLINICAL_MEANING[name][1],
            unit=CLINICAL_MEANING[name][0],
            missing=int(master[name].isna().sum()),
            categorical_levels="|".join(level_values),
            suspicious_coding="; ".join(suspicious) if suspicious else None,
            repeated_patient_consistency=(
                f"{duplicate_values}/{repeated_patients} repeated patients have >1 distinct value"
            ),
            observed_range=observed_range,
            candidate_baseline_use=name in BASELINE,
            previously_used_in_baseline7=name in BASELINE,
            previously_used_in_all19=name in ALL_CLINICAL,
        ))
    return pd.DataFrame(rows)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path.cwd().parent / "R2_frequency_GTX1080_20261103" / (
        "R2_frequency_GTX1080_20261103"
    ) / "R2_frequency_frozen_20261102"
    master, old_recommended = _current_master(args)
    discovery_ids = set(master.loc[master["cohort"].eq("discovery"), "case_id"])
    holdout_ids = set(master.loc[master["cohort"].eq("short_holdout"), "case_id"])

    _, clinical_audit = _clinical_frame(args.discovery_auc_previous, args.clinical_workbook)
    current_rows = (
        _canonical_current_rows(master, {})
        + _clinical_rows(clinical_audit)
        + CURRENT_QC_SPECS
    )
    legacy_rows = _legacy_rows(root, old_recommended)
    inventory = current_rows + legacy_rows
    inventory = pd.DataFrame(inventory)
    inventory["availability_population"] = inventory["source_population"]
    inventory["posthoc_feature"] = inventory["posthoc_feature"].fillna("no")
    inventory["outcome_informed"] = inventory["outcome_informed"].fillna("no")
    inventory["notes"] = inventory["reason_out"]
    inventory = inventory.drop_duplicates(
        subset=["feature_name", "source_file"], keep="first"
    )
    inventory.to_csv(args.output / "FEATURE_INVENTORY.csv",
                     index=False, encoding="utf-8-sig")

    source_tables = {}
    for label, path in [
        (
            "R2_usable_dynamic_features_328_20261111/full candidate dynamic",
            Path(next(
                p for p in (
                    root / "outputs" / "R2_usable_dynamic_features_328_20261111"
                ).glob("*.csv")
                if "longitudinal_posterior_frequency_cpm" in pd.read_csv(p, nrows=1).columns
                and pd.read_csv(p, usecols=["case_id"]).shape[0] == 328
                and pd.read_csv(p, nrows=1).shape[1] == 25
            )),
        ),
        (
            "R2_literature_aligned_features_v2_20261112/check_features_v2_combined.csv",
            root / "outputs" / "R2_literature_aligned_features_v2_20261112" /
            "check_features_v2_combined.csv",
        ),
        (
            "R2_dynamic_features_combined328_outcome_blind_20261110/dynamic_patient_features.csv",
            root / "outputs" / "R2_dynamic_features_combined328_outcome_blind_20261110" /
            "dynamic_patient_features.csv",
        ),
        (
            "frozen_radial_coordination_final_v1/patient coordination table",
            next(
                p for p in (
                    root / "frozen_radial_coordination_final_v1" / "records"
                ).glob("*.csv")
                if "coordination_mse" in pd.read_csv(p, nrows=1).columns
            ),
        ),
        (
            "R2_spatial_propagation_v2_technical_pilot_5_20261111/patient exploratory",
            root / "outputs" / "R2_spatial_propagation_v2_technical_pilot_5_20261111" /
            "patient_exploratory_features_v2.csv",
        ),
        (
            "R2_spatial_propagation_v2_technical_pilot_5_20261111/propagation patient",
            root / "outputs" / "R2_spatial_propagation_v2_technical_pilot_5_20261111" /
            "propagation_patient_features.csv",
        ),
    ]:
        source_tables[label] = pd.read_csv(path)

    availability = _availability_rows(
        inventory.to_dict(orient="records"), master,
        discovery_ids, holdout_ids, source_tables,
    )
    availability.to_csv(args.output / "FEATURE_AVAILABILITY.csv",
                        index=False, encoding="utf-8-sig")

    availability_index = availability.set_index(
        ["feature_name", "source_file"], drop=False
    )
    validation_rows = []
    for item in inventory.to_dict(orient="records"):
        row = _validation_row(item)
        available = _availability_lookup(
            availability_index, item["feature_name"], item["source_file"]
        )
        row["n_available"] = int(available["n_available"])
        row["availability_percent"] = float(available["availability_percent"])
        row["evidence_source"] = item["source_file"]
        row["notes"] = item["reason_out"]
        validation_rows.append(row)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(args.output / "FEATURE_VALIDATION_STATUS.csv",
                      index=False, encoding="utf-8-sig")

    candidates = _candidate_rows(inventory.to_dict(orient="records"), availability_index)
    candidates = candidates.merge(
        validation[["feature_name", "validation_result"]],
        on="feature_name", how="left",
    )
    candidates["measurement_validity"] = candidates["validation_result"].fillna(
        candidates["measurement_validity"]
    )
    candidates = candidates.drop(columns=["validation_result"])
    candidates.to_csv(args.output / "MODEL_CANDIDATE_FEATURES.csv",
                      index=False, encoding="utf-8-sig")

    sets = _sets(master)
    set_summary = _set_summary(master, sets)
    set_summary.to_csv(args.output / "FEATURE_SET_SAMPLE_SIZE.csv",
                       index=False, encoding="utf-8-sig")

    clinical_inventory = _clinical_inventory(master)
    clinical_inventory.to_csv(args.output / "CLINICAL_VARIABLE_INVENTORY.csv",
                              index=False, encoding="utf-8-sig")
    master.to_csv(args.output / "MODEL_INPUT_MASTER_TABLE.csv",
                  index=False, encoding="utf-8-sig")

    family_availability = (
        availability.groupby("feature_group")
        .agg(features=("feature_name", "size"), max_available=("n_available", "max"),
             median_available=("n_available", "median"))
        .reset_index()
    )
    tier_counts = candidates.groupby(["tier", "recommended_model_role"]).size().to_dict()
    validation_counts = validation.groupby(["validation_result", "recommended_use"]).size().to_dict()
    all_sets = set_summary.loc[set_summary["cohort"].eq("all")].set_index("feature_set")
    set0 = all_sets.loc["Set 0 Clinical only"]
    set1a = all_sets.loc["Set 1A Clinical + legacy longitudinal frequency"]
    set1b = all_sets.loc["Set 1B Clinical + current significant-activity frequency"]
    set2 = all_sets.loc["Set 2 Clinical + complete-record ER-MSE"]
    set4 = all_sets.loc["Set 4 Clinical + amplitude"]
    set7 = all_sets.loc["Set 7 Clinical + all currently defensible dynamic"]
    set8 = all_sets.loc["Set 8 All clinical + all dynamic"]
    tier_a_dynamic = candidates.loc[
        candidates["tier"].eq("A")
        & ~candidates["feature_group"].eq("Clinical"),
        "feature_name",
    ].tolist()
    tier_a_columns = BASELINE + tier_a_dynamic
    tier_a_complete = master[tier_a_columns].notna().all(axis=1)
    tier_a_complete_outcome = tier_a_complete & master["outcome"].notna()
    tier_a_max_available = int(candidates.loc[
        candidates["tier"].eq("A"), "n_available"
    ].max())
    dataset_c_dynamic = candidates.loc[
        ~candidates["feature_group"].eq("Clinical")
        & candidates["tier"].eq("B")
        & candidates["recommended_model_role"].eq("candidate predictor"),
        "feature_name",
    ].tolist()
    dataset_c_columns = tier_a_columns + dataset_c_dynamic
    dataset_c_complete = master[dataset_c_columns].notna().all(axis=1)
    dataset_c_complete_outcome = dataset_c_complete & master["outcome"].notna()
    family_candidates = (
        candidates.groupby("feature_group")
        .agg(
            candidate_features=("feature_name", "size"),
            minimum_available=("n_available", "min"),
            median_available=("n_available", "median"),
            maximum_available=("n_available", "max"),
        )
        .reset_index()
        .sort_values("feature_group")
    )
    family_candidates.to_csv(
        args.output / "FEATURE_FAMILY_AVAILABILITY.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tier_counts_text = "; ".join(
        f"Tier {tier}/{role}={count}"
        for (tier, role), count in sorted(tier_counts.items())
    )
    validation_counts_text = "; ".join(
        f"{result}/{use}={count}"
        for (result, use), count in sorted(validation_counts.items())
    )
    report = [
        "# 全项目患者级特征盘点与可用性报告",
        "",
        "本轮只读源码、结果表和验证报告；未计算AUC、未拟合模型、未做特征筛选，"
        "未修改追踪、2秒形变、ER、MSE或任何正式特征定义。",
        "",
        "## 1. 原始字段总表与分类",
        "",
        f"- 原始患者级字段记录：{len(inventory)}个；其中同名跨来源alias保留不同source_file。",
        f"- 按{inventory['feature_group'].nunique()}类feature family。",
        f"- 模型候选级标准字段：{len(candidates)}个。",
        f"- 原始字段中`definition_unclear`："
        f"{int(inventory['current_status'].eq('unclear').sum())}个。",
        "",
        "|Feature family|原始字段数|最大可计算数|字段中位可用数|",
        "|---|---:|---:|---:|",
    ]
    report.extend(
        f"|{row.feature_group}|{int(row.features)}|{int(row.max_available)}|"
        f"{row.median_available:.1f}|"
        for row in family_availability.itertuples(index=False)
    )
    report += [
        "",
        "## 2. 单特征可用性",
        "",
        "逐特征结果见`FEATURE_AVAILABILITY.csv`。患者级可用性按该字段的实际来源队列统计；"
        "老R2字段不强行假定与当前A0具有相同测量学定义。",
        "",
        "缺失最严重的模型候选（按missing rate排序）：",
        "",
        "|Feature|Tier|可用/总计|Missing rate|",
        "|---|---:|---:|---:|",
    ]
    worst = candidates.sort_values("missing_rate", ascending=False).head(12)
    name_to_total = availability.set_index("feature_name")["total_patient_exam_count"]
    report.extend(
        f"|{row.feature_name}|{row.tier}|{int(row.n_available)}/"
        f"{int(name_to_total.get(row.feature_name, 0))}|{row.missing_rate:.1%}|"
        for row in worst.itertuples(index=False)
    )
    report += [
        "",
        "## 3. 组合特征样本量",
        "",
        "完整结果见`FEATURE_SET_SAMPLE_SIZE.csv`。下表为全体当前输入（223 discovery + 35 short hold-out）：",
        "",
        "|Set|complete-case exams|complete-case patients|outcome available|events|non-events|feature count|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    report.extend(
        f"|{row.feature_set}|{int(row.complete_case_n)}|"
        f"{int(row.complete_case_patients)}|"
        f"{int(row.pregnancy_outcome_available_count)}|"
        f"{int(row.events)}|{int(row.non_events)}|{int(row.feature_count)}|"
        for row in set_summary.loc[set_summary["cohort"].eq("all")].itertuples(index=False)
    )
    report += [
        "",
        "## 4. 测量学验证状态",
        "",
        f"- {len(validation)}个原始字段完成了状态映射；状态依据现有synthetic、rigid、"
        "brightness/image-warp、repeatability、split-half、duration和support审计。",
        f"- Tier/推荐角色计数：{tier_counts_text}。",
        f"- 验证结果计数：{validation_counts_text}。",
        "",
        "现有最关键证据：",
        "- 当前约2秒A0在真实首帧已知形变/刚体控制中改善误差并保留方向，但不是人体声学真值。",
        "- 当前A0 amplitude缺少人体重复采集；ER-MSE完成同步/相反控制、rigid/support、"
        "Huang-like桥接和45秒duration bridge，仍有均值偏置和位置异质性限制。",
        "- 事件计数存在local_once误计、相邻重复和漏检，不能作为正式次数/频率。",
        "- 旧R2 frequency有窗口稳定性和60秒/截短稳定性；旧motion/strain有较高的窗口稳定性，"
        "但已知图像形变相对误差在25%边界附近，且未接入当前A0正式链。",
        "- 旧propagation spatial pilot仅5例；合成传播准确不等于真实患者速度已验证。",
        "",
        "## 5. 最终Tier",
        "",
        "### Tier A — 正式候选",
        "",
    ]
    tier_a = candidates.loc[candidates["tier"].eq("A")]
    report.extend(f"- `{row.feature_name}`：{row.recommended_model_role}" for row in tier_a.itertuples(index=False))
    report += ["", "### Tier B — 探索候选", ""]
    tier_b = candidates.loc[candidates["tier"].eq("B")]
    report.extend(f"- `{row.feature_name}`：{row.recommended_model_role}" for row in tier_b.itertuples(index=False))
    report += ["", "### Tier C — 暂不进入模型", ""]
    tier_c = candidates.loc[candidates["tier"].eq("C")]
    report.extend(f"- `{row.feature_name}`：{row.recommended_model_role}" for row in tier_c.itertuples(index=False))
    report += [
        "",
        "## 5b. Feature family可用性（模型候选层）",
        "",
        "|Feature family|候选特征数|min available|median available|max available|",
        "|---|---:|---:|---:|---:|",
    ]
    report.extend(
        f"|{row.feature_group}|{int(row.candidate_features)}|"
        f"{int(row.minimum_available)}|{row.median_available:.1f}|"
        f"{int(row.maximum_available)}|"
        for row in family_candidates.itertuples(index=False)
    )
    report += [
        "",
        "## 5c. 十六个必答问题",
        "",
        f"1. 原始患者级字段记录{len(inventory)}个；去重后的模型候选级特征{len(candidates)}个。",
        f"2. 分成{inventory['feature_group'].nunique()}个feature families。",
        "3. 各family可计算数见5b；当前A0 amplitude约210–211/258，"
        "complete-record MSE约190/258，current event rate约217/258。",
        "4. 缺失最严重的是legacy频率差、single-block CC/MSE、complete-record MSE以及legacy单壁频率。",
        "5. 当前没有任何人体特征达到完整gold-standard验证；证据相对最强的是"
        "bilateral amplitude和complete-record ER-MSE/CC，但均为partial。",
        "6. exploratory validity主要是旧R2 motion/strain/velocity、旧协调性指标、"
        "旧frequency和5例spatial propagation pilot。",
        "7. 暂不应进入预测模型的是event计数/频率、directional support QC、"
        "旧R2 motion/strain/velocity、旧Hausdorff/相位类指标及5例spatial propagation字段。",
        "8. Tier A/B/C完整列表见上一节和`MODEL_CANDIDATE_FEATURES.csv`。",
        f"9. 只用Tier A时，单特征最大可用数为{ tier_a_max_available }个检查。",
        f"10. 所有Tier A complete-case为{int(tier_a_complete.sum())}个检查、"
        f"{int(master.loc[tier_a_complete, 'patient_id'].nunique())}例患者；"
        f"有结局者{int(tier_a_complete_outcome.sum())}个。",
        f"11. Clinical+legacy longitudinal frequency为{int(set1a['complete_case_n'])}个/"
        f"{int(set1a['complete_case_patients'])}例；Clinical+current significant-activity "
        f"frequency为{int(set1b['complete_case_n'])}个/"
        f"{int(set1b['complete_case_patients'])}例。",
        f"12. Clinical+complete-record ER-MSE为{int(set2['complete_case_n'])}个/"
        f"{int(set2['complete_case_patients'])}例。",
        f"13. Clinical+all currently defensible dynamic为"
        f"{int(set7['complete_case_n'])}个/{int(set7['complete_case_patients'])}例。",
        "14. 会从>200降到<200的主要来源是complete-record ER-MSE、双侧amplitude"
        "和legacy longitudinal frequency；后两者或任一缺失都会造成明显缩减。",
        "15. 最值得训练折内缺失处理的是少量临床缺失和少量current event-rate缺失；"
        "amplitude/MSE的缺失与support有关，必须同时保留missing flag和complete-support敏感性，"
        "不能用0替代。",
        f"16. 后续nested CV候选集：A=baseline7；B=baseline7+当前Tier A影像"
        f"（{int(set7['complete_case_n'])}）；C=B+探索性影像候选"
        f"（{int(dataset_c_complete.sum())}）。",
        "",
        "## 5d. 临床编码审计",
        "",
        "- `infertility_years`的小数部分与`infertility_months`在220/220对中一致；"
        "月字段是月份余数，不是独立总月数，不应作为第二个独立变量重复使用。",
        "- `transfer_embryo_count_multiple`与解析胚胎数一致；"
        "`blastocyst_transfer`与胚胎阶段、`endometrial_type_C`与内膜分型类别也重复编码同一信息。",
        "- `doppler_FI`有5个>80的极端值（最高183.46），需回查原始小数位/单位；"
        "本轮仅标记suspicious coding，不修复。",
        "- 223个discovery检查来自208名患者，15名患者有重复检查；"
        "年龄、内膜厚度和血流等可随周期变化，后续CV必须按patient_id分组。",
        "",
        "## 6. 聚类与重复患者",
        "",
        f"- 当前total exams={len(master)}；unique patient_id={master['patient_id'].nunique()}；"
        f"repeated exams={len(master)-master['patient_id'].nunique()}。",
        f"- Discovery：{int(master['cohort'].eq('discovery').sum())} exams，"
        f"{master.loc[master['cohort'].eq('discovery'),'patient_id'].nunique()} patients。",
        f"- Short hold-out：{int(master['cohort'].eq('short_holdout').sum())} exams，"
        f"{master.loc[master['cohort'].eq('short_holdout'),'patient_id'].nunique()} patients。",
        "- `patient_id`对discovery使用既有患者聚类标识；hold-out因没有额外稳定聚类键，"
        "使用exam ID作为patient fallback，并在后续CV中保持独立。",
        "",
        "## 7. 缺口来源",
        "",
        "- Clinical-only完整病例："
        f"{int(set0['complete_case_n'])}。",
        "- Clinical + ER-MSE："
        f"{int(set2['complete_case_n'])}；"
        "主要损失来自无共同支持ER窗口及hold-out无临床表。",
        "- Clinical + amplitude："
        f"{int(set4['complete_case_n'])}。",
        "- 某个特征单独把样本从>200降到<200的最明显来源是："
        "complete-record ER-MSE（157/223 discovery可用）和双侧amplitude（约174 discovery可用）。",
        "",
        "## 8. 缺失处理建议与后续Dataset",
        "",
        "临床基线少量缺失适合训练折内median插补并记录missing indicator；"
        "影像特征缺失主要来自空间/时间支持和tracking，不能写0，也不建议仅靠插补把不可观察者当成可观察。"
        "后续应以complete-case主分析、训练折内插补敏感性和缺失机制描述并列。",
        "",
        "### Dataset A：临床基础集",
        "",
        f"- feature count：{len(BASELINE)}；complete-case n："
        f"{int(set0['complete_case_n'])}；"
        f"complete-case patients：{int(set0['complete_case_patients'])}。",
        "",
        "### Dataset B：临床 + 已验证影像特征",
        "",
        f"- feature count：11（7 clinical + bilateral amplitude + complete-record CC/MSE）；"
        f"complete-case n：{int(set7['complete_case_n'])}；"
        f"complete-case patients：{int(set7['complete_case_patients'])}。",
        "",
        "### Dataset C：临床 + 已验证 + 探索影像特征",
        "",
        f"- feature count：{len(dataset_c_columns)}；"
        f"complete-case n：{int(dataset_c_complete.sum())}；"
        f"complete-case patients：{int(master.loc[dataset_c_complete, 'patient_id'].nunique())}；"
        f"有结局者{int(dataset_c_complete_outcome.sum())}。",
        "",
        "本轮不决定LASSO、Elastic Net、Logistic、SVM、RF或其他模型。",
        "",
        "## 输出",
        "",
        "- `FEATURE_INVENTORY.csv`",
        "- `FEATURE_AVAILABILITY.csv`",
        "- `FEATURE_SET_SAMPLE_SIZE.csv`",
        "- `FEATURE_FAMILY_AVAILABILITY.csv`",
        "- `FEATURE_VALIDATION_STATUS.csv`",
        "- `CLINICAL_VARIABLE_INVENTORY.csv`",
        "- `MODEL_INPUT_MASTER_TABLE.csv`",
        "- `MODEL_CANDIDATE_FEATURES.csv`",
    ]
    (args.output / "FEATURE_INVENTORY_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    summary = dict(
        input_exams=len(master),
        discovery_exams=int(master["cohort"].eq("discovery").sum()),
        holdout_exams=int(master["cohort"].eq("short_holdout").sum()),
        unique_patients=master["patient_id"].nunique(),
        raw_feature_fields=len(inventory),
        model_candidate_features=len(candidates),
        definition_unclear_fields=int(inventory["current_status"].eq("unclear").sum()),
        feature_families=sorted(inventory["feature_group"].unique().tolist()),
        tier_counts={str(k): int(v) for k, v in tier_counts.items()},
        validation_counts={str(k): int(v) for k, v in validation_counts.items()},
        tier_a_complete_case_exams=int(tier_a_complete.sum()),
        tier_a_complete_case_patients=int(
            master.loc[tier_a_complete, "patient_id"].nunique()
        ),
        dataset_c_complete_case_exams=int(dataset_c_complete.sum()),
        outcomes_computed=False,
        models_fit=False,
        features_recomputed=False,
        tracking_changed=False,
        definitions_changed=False,
        source_hashes={
            "clinical_workbook": _sha(args.clinical_workbook),
            "current_master_input": _sha(args.discovery / "patient_results.csv"),
            "complete_record": _sha(args.complete_record_root / "record_coordination.csv"),
            "wave_counts": _sha(args.wave_counts_root / "wave_counts.csv"),
            "support_audit": _sha(args.support_audit_root / "record_level_support.csv"),
        },
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--discovery-auc-previous", type=Path, required=True)
    parser.add_argument("--clinical-workbook", type=Path, required=True)
    parser.add_argument("--complete-record-root", type=Path, required=True)
    parser.add_argument("--wave-counts-root", type=Path, required=True)
    parser.add_argument("--support-audit-root", type=Path, required=True)
    parser.add_argument("--holdout-measurement-manifest", type=Path, required=True)
    parser.add_argument("--holdout-measurement-frozen", type=Path, required=True)
    parser.add_argument("--holdout-outcomes", type=Path, required=True)
    parser.add_argument("--holdout-input-manifest", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
