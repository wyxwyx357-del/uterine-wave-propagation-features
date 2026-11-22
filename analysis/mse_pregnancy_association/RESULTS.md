# Existing MSE evidence, without reanalysis

Every table in `results/` is copied from saved aggregate CSVs; source relative paths and hashes are in `../RESULT_SOURCE_MAP.json`. No new AUC, OR, patient selection or feature extraction was run for this package.

Review clarification: the historical numeric MSE formula is label-independent, but its full aggregation script already held clinical/outcome columns. It was not a strictly outcome-blind run. This provenance correction changes no stored result. CC pools supported segments and remains auxiliary; the new measurement-only boundary is a future workflow, not a new validation result.

## Association and decomposition

The earlier223-examination development cohort had157 available complete-record MSE measurements. Non-pregnancy n57: mean MSE0.203217; clinical pregnancy n100: mean0.308610. The fixed7-clinical-variable complete-case association used154 examinations/151 patients and gave adjusted OR/SD2.291151,95%CI1.449546–3.621393. These are association results, not Clinical9 test-fold prediction estimates.

The observed higher-MSE/pregnancy direction conflicts with Huang2022; no beneficial physiological interpretation is assigned. The Huang-like adjacent-frame bridge retained this direction on the same support, so the approximately2s interval alone did not explain it. This remains an internal, overlapping-cohort comparison.

The post-hoc group-average decomposition is:

| Outcome | n | MSE | Mean-bias² | SD-scale² | Time-shape |
| --- | --- | --- | --- | --- | --- |
| Non-pregnancy |57|0.203217|0.067156|0.015513|0.120548|
| Clinical pregnancy |100|0.308610|0.140815|0.017354|0.150440|

Mean-bias² explains much of the difference; standardized MSE/CC did not show the corresponding clear adjusted signal. This supports the name inter-wall ER disparity, not a validated temporal coordination biomarker. Full saved associations, CC and explanatory variables are retained, including non-significant results.

## Technical and measurement evidence

- `mse_technical_adjustment_summary.csv`: saved adjustments for video duration, anterior/posterior valid fraction and optional source. These are different historical technical adjustment models, not a claim of all clinical and technical covariates adjusted simultaneously.
- `technical_correlation_summary.csv`: archived technical relationships.
- `rigid_and_synthetic_control_summary.csv` and `support_direction_accuracy_summary.csv`: outcome-blind controls. Known-direction controls support sign implementation, but rigid motion can yield nonzero ER-MSE, and this limitation has not been removed or hidden.
- The published Huang bridge code preserves the separate outcome-blind measurement phase and later association phase. It is a Huang-like comparison, not exact literature replication.

## Prediction bridge

See `mse_incremental_prediction_summary.csv` and the separate [prediction results](../pregnancy_prediction/RESULTS.md). Primary Ridge Clinical9 AUC0.491969→0.575988; paired ΔAUC+0.084019, conditional95%CI[0.019419,0.150828], same185 examinations/179 patients. The Clinical v2 sensitivity is0.501942→0.593572; Δ+0.091631[0.026156,0.157863]. These are not independent confirmations of one another.

Neither positive association nor incremental AUC establishes measurement validity. Repeat segmentation, repeat acquisition, transportability and genuinely new patient evaluation remain necessary research work.
