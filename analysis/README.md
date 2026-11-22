# Frozen ER-MSE evidence and pregnancy prediction

This archive packages existing implementations and aggregate results. It introduces no new patient analysis, feature definition, model search, threshold or training run.

```
Dynamic measurement
├── A0 tracking → finite-interval radial A → directional ER
├── ER-MSE measurement and pregnancy association
│   └── mse_pregnancy_association/
└── propagation/activity analyses (historical, separate construct)

Pregnancy prediction
├── Clinical9 Ridge (primary)
├── Clinical9 + ER-MSE Ridge (primary comparison)
└── Tiny MLP (nonlinear model-form sensitivity only)
    └── pregnancy_prediction/
```

- [MSE definition, implementation and evidence](mse_pregnancy_association/README.md)
- [Clinical9 prediction and complete sensitivity results](pregnancy_prediction/README.md)
- [Freeze status and limitations](FREEZE_STATUS.md)
- [Content-based sandbox inventory](SANDBOX_CODE_INVENTORY.md)
- [Code relocation provenance](SOURCE_RELOCATION_MAP.json)
- [Aggregate result provenance](RESULT_SOURCE_MAP.json)
- [Manifest](FREEZE_MANIFEST.json)

Primary comparison: Clinical9 versus Clinical9+ER-MSE, Ridge Logistic Regression, on the same 185 examinations/179 patients. Clinical6, Clinical11, Enhanced and Clinical v2 remain baseline sensitivity/exploratory analyses; their different cohorts must be retained.

## What is reproducible

The published code covers saved A0 tracking arrays → supported radial A → ER windows → block summaries and record_MSE_time_equal, and private Clinical9/MSE inputs plus the original fold jobs → Ridge/Tiny-MLP OOF evaluation. Original patient data, original fold membership and OOF probabilities are not public. Therefore this repository supports implementation review and synthetic reproduction, **not numerical reproduction of the patient results from public data alone**. An authorized holder of the original private inputs can replay the chain. CUDA-enabled OpenCV is required to rerun A0 tracking; no CPU substitute is introduced.

The source measurement snapshot is byte-identical to the active local package. Original upstream `uterine_wave/` and historical scripts have not been replaced by this archive. Analysis copies change import namespaces and local input/resource paths only; no numerical constants, equations, windows, thresholds or patient-selection rules were optimized. Some archived audit scripts expect their original private relative artifact layout and are retained as source evidence; use the documented portable entry points for replay.

## Synthetic checks

Install `requirements.txt`, then from the repository root:

```sh
python -m unittest discover -s analysis/tests -v
```

The tests use artificial observations only. They test formulas, aggregation weights, unsupported intervals, grouped folds, training-only preprocessing, within-fold AUC and paired comparisons. They are implementation checks, not clinical validation.

Only code, documentation, aggregate/fold results and synthetic tests are published. No patient identifiers, patient-level features/outcomes, OOF rows, clinical workbooks, videos, ROIs, tracking arrays or local absolute paths are included in this change. Private input/output paths are supplied via `DATA_ROOT`, `INPUT_PATH`, `OUTPUT_PATH` or command-line arguments. Do not commit private replay outputs.

These results are based on internally developed and repeatedly explored data and do not constitute independent external validation.
