# Audited freeze status

| Component | Status | Qualification |
| --- | --- | --- |
| A0 tracking parameters/interface | frozen snapshot | source archived; no new tracking or claim of complete material identity validation |
| Approximately2s radial A | frozen definition | finite-interval relative distance rate, not adjacent-frame RSR; actual hop recorded |
| ER and20s/1s nominal windows | frozen definition | support/rigid artifacts remain; no new physiological threshold |
| Block-level MSE | frozen definition | same walls/block/windows; gaps not filled |
| record_MSE_time_equal | frozen, audited source identified | per-time mean of block squared errors, then equal observed-window-time mean; not a simple mean of block MSEs |
| MSE decomposition | post-hoc explanatory only | undefined with multiple simultaneous blocks; no promotion to primary feature |
| Clinical9 | frozen primary baseline | source values/coding unchanged |
| Ridge classifier and CV | frozen primary | exact original grouped folds retained privately |
| Tiny MLP | sensitivity only | one fixed architecture; no primary-model replacement |
| Clinical6/11/v2/Enhanced | sensitivity/exploratory only | distinct sample sizes and cleaning rules retained |
| Static PCA/Elastic Net | final exploratory results retained | no new feature/method search |
| Activity/event identity | not established as true discrete-wave counting | historical negative/inconclusive prediction results retained |
| Model/feature search | closed | no patient reanalysis in this package |
| Independent evaluation | pending | new patients uninvolved in development required |

Audit scope: actual source code, saved aggregate tables and historical reports. Formal record-aggregation documentation agrees with the executable weighting. One old inline code comment describes the spatial-average ER series; METHOD explicitly distinguishes that CC series from the actual squared-error MSE path. No numerical discrepancy was found that required replacing a feature or stopping its publication.

Public reproducibility covers algorithms/synthetic checks. Exact historical patient results need authorized private arrays, labels, canonical order and original grouped jobs; these are deliberately not uploaded. Source and result hashes use relative paths only. The freeze manifest does not hash itself, avoiding a self-reference; `uploaded_to_git` describes this publication's allowlist and the final commit/PR provide transport verification.
