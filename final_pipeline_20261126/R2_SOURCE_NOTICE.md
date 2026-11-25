# R2 frozen source notice

The final batch runner expects `final_pipeline_20261126/r2_frozen/`.

The exact R2 source is the previously frozen package:

- Version: `R2_LONGITUDINAL_POSTERIOR_FROZEN_CANDIDATE_V1_1_20261106`
- Frequency rule: `R2_LONGITUDINAL_TISSUE_FREQUENCY_V1_1_20261106`
- Source files are SHA-256 locked in `SOURCE_LOCK.json`.
- The downloadable installation package supplied for this branch vendors these unchanged R2 files into `r2_frozen/`.
- `evaluate_frozen_model.py` is deliberately excluded because it reads clinical outcomes and is outside this outcome-blind measurement batch.

Do not substitute another R2 directory or modify the locked files. `verify_environment.py` checks the vendored source hashes before a formal run.
