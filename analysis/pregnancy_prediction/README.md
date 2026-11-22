# Frozen pregnancy-prediction archive

**Primary:** Clinical9 versus Clinical9+ER-MSE using Ridge Logistic Regression. **Tiny MLP:** nonlinear model-form sensitivity analysis only. Clinical6, Clinical11, Enhanced and Clinical v2 remain baseline sensitivity/exploratory analyses.

- [Model definitions](MODEL_DEFINITIONS.md)
- [Exact internal validation and evaluation](VALIDATION_PROTOCOL.md)
- [All results, including negative findings](RESULTS.md)
- [Limitations](LIMITATIONS.md)
- Actual Ridge implementation: `code/clinical9_models.py`, `code/clinical9_frozen_mse.py`.
- Actual Tiny MLP: `code/tiny_mlp_models.py`, `code/tiny_mlp_sensitivity.py`.
- Actual within-fold evaluator/bootstrap: `code/frozen_foldwise_auc.py`.
- Original grouped split generation: `code/first_pass_extended6.py:prepare` (global maps), seeds fixed in that source.

The existing implementations were relocated, not replaced with a newly optimized model. The primary published result is AUC0.491969→0.575988 on185 examinations/179 patients, not the largest AUC found among sensitivity analyses.

## Private-input replay

Install `../requirements.txt`. Original inputs are not in Git. An authorized holder must supply the canonical311-row Clinical9 master and the original `FIT_JOBS_LOCAL.json`. The saved row order, patient IDs, labels and patient-grouped assignments are required; no substitute random split is silently generated.

```sh
python -m analysis.pregnancy_prediction.code.replay_prediction --help
python -m analysis.pregnancy_prediction.code.replay_prediction --master data/private/master.csv --jobs data/private/folds.json --output outputs/private_replay
```

The entry defaults to the primary Ridge pair on the historical MSE185 jobs; it calls the archived fit implementation. `--classifier tiny_mlp` calls the fixed archived Tiny MLP and needs separately installed PyTorch2.6.0 CPU. This packaging task does not run either replay on patient data. Input/output paths are explicit; protect replay outputs and never commit them.

Original audit/source/report-generation scripts can require the original private artifact tree and source hashes. They are preserved for provenance; the portable entry bypasses report housekeeping, not model/preprocessing/epoch selection rules.
