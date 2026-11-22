# Existing prediction results, including negative findings

These are **copies of already frozen aggregate tables**, not a new model-selection exercise. Exact source relative paths/hashes and all uploaded columns are in `../RESULT_SOURCE_MAP.json`. Cohorts differ; use paired within-cohort estimates rather than subtracting across rows with different n. All AUC95% intervals below are the previously stored fixed-OOF conditional intervals.

## Primary Ridge result

| cohort | model | n_exams | n_unique_patients | AUC | Brier | logloss |
| --- | --- | --- | --- | --- | --- | --- |
| Full306 | Clinical6 | 306 | 288 | 0.549574 | 0.229124 | 0.651289 |
| Full306 | Clinical9 | 306 | 288 | 0.554667 | 0.229035 | 0.651259 |
| MSE185 | Clinical6_historical | 185 | 179 | 0.511985 | 0.231530 | 0.657416 |
| MSE185 | Clinical6_MSE_historical | 185 | 179 | 0.580963 | 0.228325 | 0.648795 |
| MSE185 | Clinical9 | 185 | 179 | 0.491969 | 0.230831 | 0.655122 |
| MSE185 | Clinical9_MSE | 185 | 179 | 0.575988 | 0.230065 | 0.653657 |
| MSE185 | Clinical9_Position | 185 | 179 | 0.497050 | 0.231453 | 0.657123 |
| MSE185 | Clinical9_MSE_Position | 185 | 179 | 0.581807 | 0.230447 | 0.654452 |

| cohort | role | Delta_AUC | CI_low | CI_high | Delta_Brier | Delta_logloss |
| --- | --- | --- | --- | --- | --- | --- |
| Full306 | baseline_completion | 0.005093 | -0.025774 | 0.034526 | -0.000088 | -0.000030 |
| MSE185 | historical_MSE | 0.068978 | -0.002702 | 0.141739 | -0.003205 | -0.008621 |
| MSE185 | primary_MSE | 0.084019 | 0.019419 | 0.150828 | -0.000766 | -0.001465 |
| MSE185 | position_adjusted_MSE | 0.084756 | 0.021687 | 0.149368 | -0.001006 | -0.002671 |
| MSE185 | adding_position_to_MSE_model | 0.005819 | -0.006431 | 0.017532 | 0.000383 | 0.000795 |

The main question remains Clinical9→Clinical9_MSE on185 examinations/179 patients. Clinical9's low AUC on this subset is not hidden. The full Clinical9 cohort has306 examinations/288 patients and is a different prediction population. Position-adjusted results retain the paired clinical-position control. No highest-AUC model is declared primary.

## Nonlinear model-form sensitivity only

| classifier | model | n_exams | n_unique_patients | AUC | Brier | logloss |
| --- | --- | --- | --- | --- | --- | --- |
| Ridge Logistic | Ridge_Clinical9 | 185 | 179 | 0.491969 | 0.230831 | 0.655122 |
| Ridge Logistic | Ridge_Clinical9_MSE | 185 | 179 | 0.575988 | 0.230065 | 0.653657 |
| Tiny MLP | MLP_Clinical9 | 185 | 179 | 0.495506 | 0.234751 | 0.663652 |
| Tiny MLP | MLP_Clinical9_MSE | 185 | 179 | 0.589725 | 0.228176 | 0.649645 |

| role | Delta_AUC | CI_low | CI_high | Delta_Brier | Delta_logloss |
| --- | --- | --- | --- | --- | --- |
| primary_MLP_MSE_increment | 0.094219 | 0.015858 | 0.173373 | -0.006575 | -0.014006 |
| classifier_change_Clinical9 | 0.003537 | -0.055041 | 0.060990 | 0.003921 | 0.008530 |
| classifier_change_Clinical9_MSE | 0.013737 | -0.016151 | 0.045353 | -0.001888 | -0.004012 |
| historical_Ridge_MSE_increment | 0.084019 | 0.019419 | 0.150828 | -0.000766 | -0.001465 |

The Tiny MLP Clinical9 baseline changed AUC by only+0.003537 with a paired interval spanning0. The augmented model differed from Ridge by+0.013737, also with an interval spanning0. There is no clear paired evidence of Tiny MLP superiority. MSE's positive internal increment in this sensitivity does not constitute neural-network validation. Repeats are not independent studies.

## Baseline sensitivities and exploratory Enhanced model

| analysis | cohort | model | n_exams | n_unique_patients | AUC | Brier | logloss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Clinical9_primary | Full306 | Clinical6 | 306 | 288 | 0.549574 | 0.229124 | 0.651289 |
| Clinical9_primary | Full306 | Clinical9 | 306 | 288 | 0.554667 | 0.229035 | 0.651259 |
| Clinical9_primary | MSE185 | Clinical6_historical | 185 | 179 | 0.511985 | 0.231530 | 0.657416 |
| Clinical9_primary | MSE185 | Clinical6_MSE_historical | 185 | 179 | 0.580963 | 0.228325 | 0.648795 |
| Clinical9_primary | MSE185 | Clinical9 | 185 | 179 | 0.491969 | 0.230831 | 0.655122 |
| Clinical9_primary | MSE185 | Clinical9_MSE | 185 | 179 | 0.575988 | 0.230065 | 0.653657 |
| Clinical9_primary | MSE185 | Clinical9_Position | 185 | 179 | 0.497050 | 0.231453 | 0.657123 |
| Clinical9_primary | MSE185 | Clinical9_MSE_Position | 185 | 179 | 0.581807 | 0.230447 | 0.654452 |
| Clinical11_sensitivity | Blood287 | Clinical11 | 287 | 270 | 0.531669 | 0.228029 | 0.649137 |
| Clinical11_sensitivity | MSE_Blood180 | Clinical11 | 180 | 174 | 0.518315 | 0.231020 | 0.655596 |
| Clinical11_sensitivity | MSE_Blood180 | Clinical11_MSE | 180 | 174 | 0.593692 | 0.231691 | 0.657987 |
| Enhanced_exploratory | Full306 | Enhanced | 306 | 288 | 0.519157 | 0.231793 | 0.657676 |
| Enhanced_exploratory | MSE185 | Enhanced | 185 | 179 | 0.480965 | 0.233841 | 0.662194 |
| Enhanced_exploratory | MSE185 | Enhanced_MSE | 185 | 179 | 0.601240 | 0.226318 | 0.644189 |
| Clinical_v2_sensitivity | Full306 | Clinical_v2 | 306 | 288 | 0.560964 | 0.229378 | 0.652347 |
| Clinical_v2_sensitivity | MSE185 | Clinical_v2 | 185 | 179 | 0.501942 | 0.230969 | 0.655418 |
| Clinical_v2_sensitivity | MSE185 | Clinical_v2_MSE | 185 | 179 | 0.593572 | 0.229279 | 0.651379 |

Clinical11 uses observed PI/VFI on287/270 or180/174, not the306/288 or185/179 cohorts. Its cleaning differs from the later Enhanced exploration. Clinical v2 contains treatment preparation context plus IVF-only/ICSI-involved and retains306/288 and185/179 with training-fold imputation. It did not establish a clearly stronger baseline: on306 the Clinical9→v2 ΔAUC was+0.006298[-0.006233,0.019078] and Brier/log-loss slightly worsened. A larger ΔAUC when an exploratory baseline is weaker does not establish a superior final combined model.

## Other dynamic features: retain negative/inconclusive increments

These historical analyses use Clinical6, not Clinical9; sample sizes are feature-availability-specific.

| module | feature_group | n_exams | n_unique_patients | Clinical_AUC | Augmented_AUC | Delta_AUC | Delta_CI_low | Delta_CI_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | 显著传播活动频率 | 246 | 231 | 0.533516 | 0.541311 | 0.007795 | -0.020206 | 0.034522 |
| AMP | 前后壁幅度 | 196 | 187 | 0.498311 | 0.488782 | -0.009529 | -0.060205 | 0.037773 |
| CC | ER-CC | 183 | 177 | 0.507581 | 0.517366 | 0.009785 | -0.031250 | 0.051203 |
| IMG_ACT | 幅度＋MSE＋CC＋活动频率 | 180 | 174 | 0.506543 | 0.548636 | 0.042093 | -0.038645 | 0.120761 |
| IMG | 幅度＋MSE＋CC | 180 | 174 | 0.506543 | 0.554781 | 0.048238 | -0.029381 | 0.124252 |
| MSE | ER-MSE | 185 | 179 | 0.511985 | 0.580963 | 0.068978 | -0.002702 | 0.141739 |

ActivityRate, amplitude and CC did not show a clear stable increment. ActivityRate is algorithm-detected significant propagation activity, not validated individual physiological wave count. The old R2 longitudinal frequency is excluded from the current candidate route.

## Static analyses: final explorations retained

Static PCA uses the common180/174 cohort with Clinical6/five dynamic features:

| augmented | baseline | n_exams | n_unique_patients | Delta_AUC | CI_low | CI_high | crosses_zero | interval_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clinical_Static | Clinical | 180 | 174 | -0.014105 | -0.054164 | 0.023563 | True | exploratory marginal95%, no multiplicity correction; fixed OOF |
| Clinical_Static_Dynamic | Clinical_Dynamic | 180 | 174 | -0.015375 | -0.040093 | 0.008251 | True | exploratory marginal95%, no multiplicity correction; fixed OOF |
| Clinical_Dynamic | Clinical | 180 | 174 | 0.042093 | -0.039635 | 0.120565 | True | exploratory marginal95%, no multiplicity correction; fixed OOF |
| Clinical_Static_Dynamic | Clinical_Static | 180 | 174 | 0.040823 | -0.031506 | 0.111471 | True | exploratory marginal95%, no multiplicity correction; fixed OOF |
| Clinical_Static_Dynamic | Clinical | 180 | 174 | 0.026718 | -0.058526 | 0.108345 | True | exploratory marginal95%, no multiplicity correction; fixed OOF |

The one final supervised static Elastic Net exploration:

| augmented | baseline | n_exams | n_unique_patients | Delta_AUC | CI_low | CI_high | crosses_zero | Delta_Brier | Delta_logloss | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clinical_Static_EN | Clinical | 180 | 174 | -0.003758 | -0.009159 | 0.000895 | True | 0.001205 | 0.003077 | exploratory marginal95% fixed OOF; no full training-refit or independent validation claim |
| Clinical_Dynamic_Static_EN | Clinical_Dynamic | 180 | 174 | -0.002869 | -0.008974 | 0.002751 | True | 0.000419 | 0.001826 | exploratory marginal95% fixed OOF; no full training-refit or independent validation claim |

Both had intervals including no static increment; probability-score improvements were not demonstrated. These observations do not prove every original radiomics feature is uninformative. No new feature/method search is undertaken; the saved static feature selection frequencies are also retained.

## Interpretation

ER-MSE retains a positive candidate internal incremental signal across several overlapping baseline/model-form sensitivities. This does not prove biological validity, clinical utility, mechanism independence, or external performance. The data have been repeatedly explored and evaluation aggregation was revised after audit; the complete development uncertainty is not covered by fixed-OOF intervals. Measurement validation and patients untouched by development remain the next evidence needs.
