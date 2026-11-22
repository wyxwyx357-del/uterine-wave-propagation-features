# Historical private-data assembly sources

These are relocated original scripts, not a newly specified cohort or re-extraction. They have top-level historical source-read/write operations: **inspect them as source evidence; do not import them as utility modules.** Input/output defaults now use DATA_ROOT and CLINICAL_SOURCE_PATH rather than a user's desktop. Their original private dependency files and exact row order are required. This packaging task does not execute them.

The assembly chain is:

1. `../feature_inventory_audit.py:_current_master`: reads complete-record coordination and joins existing case keys; short-cohort MSE is renamed to the same `record_MSE_time_equal` field without changing its values.
2. `reconcile_feature_inventory.py`: aligns to the desktop-duration cohort and preserves existing verified clinical patient keys where available. Name-only provisional keys do not become verified IDs by assumption.
3. `../short_activity_auc.py`: preserves the separately audited short35 clinical-outcome/identity joins and MSE fields.
4. `build_expanded_sample_registry.py`: concatenates the original276 and35 without overlapping exam/case keys; preserves the distinction between clinical outcome and directory labels;311 registered rows remain.
5. `build_clinical_baseline_design.py`: obtains the existing358-source normalized clinical values without outcome-based selection.
6. `freeze_extended6_increment_plan.py`: existing source-aligned Clinical6 values, including the previously reviewed year/month and BMI missing policy; the image fields are retained.
7. `../clinical9_data_audit.py`: adds frozen volume, pure-C indicator and explicitly parsed days before transfer; creates the canonical311-row Clinical9 master. The306 eligible examinations and185 eligible MSE-available examinations are properties of the retained private inputs and flags, not new selection rules in this archive.
8. `../first_pass_extended6.py` originally creates global grouped outer/inner maps; `../clinical9_frozen_mse.py` reuses those maps and the historical MSE185 subset.

The portable prediction entry verifies exact canonical master/job SHA256 before fitting. It intentionally does not reconstruct a substitute cohort from anonymized public data. Private assembly/source hashes are evidence of which data were used, not authorization to publish their contents. No patient key/value, source workbook, join table, original job membership or master is included here.
