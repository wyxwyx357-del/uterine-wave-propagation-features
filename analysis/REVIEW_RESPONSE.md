# PR1 review corrections — no patient reanalysis

This update corrects provenance wording and adds future measurement/CI boundaries. It does not recalculate a patient, change MSE/ER/window/support definitions, modify Ridge/Tiny-MLP fitting, change cohorts/folds, or merge main.

| Review point | Resolution |
| --- | --- |
| Historical aggregation already held outcomes | Accepted. METHOD/README/RESULTS/FREEZE_STATUS distinguish a label-independent numeric function from the historical mixed-cohort runtime, which was not strictly outcome-blind. |
| Strict future measurement entry | Added `mse_pregnancy_association/code/record_mse_measurement.py`. Only case_id, paired_ER_windows and measurement_metadata are accepted; unknown clinical/label fields are rejected at every input level. |
| Avoid association imports in measurement | Added numeric-only `record_mse_core.py`, copying the four historical functions verbatim. AST/output equality tests and a fresh-process dependency check preserve that separation. The original mixed script stays unchanged. |
| Later label join | Added `code/mse_outcome_association.py`, a separate exact-key left join of saved measurement artifacts and private Clinical9/outcome inputs. It preserves unmatched/missing records, rejects ambiguous keys and recodes no outcome. It performs no model/AUC/OR analysis. |
| Protocol consistency for new records | Required version, ROI0.2–0.8, nominal2s interval,20s window,1s step,0.5–5cpm band are checked before aggregation. Mismatch is an input-protocol error, not a biological exclusion threshold. Legacy mixed scripts are not approved future label-free entry points. |
| Saved-tracks interface | `replay_measurement` now requires a local case key, calls the strict boundary and emits its three-field phase1 artifact. Python replay retains diagnostic modes. This is an explicit CLI output-schema change inside the unmerged PR. |
| Equal-window-time MSE and decomposition | Unchanged. Tests still cover unequal block counts, pooled versus time-equal weighting, segment duration, missing support and decomposition identity. |
| Gap/reanchor handling | Unchanged. Saved-tracks and gap tests remain; new tests verify the same original numeric outputs. |
| CC interpretation | Clarified as correlation of supported ER observations pooled across segments, potentially influenced by segment mean shifts; auxiliary only, not continuous-time synchrony. |
| AUC terminology | Explicitly fold-stratified / pair-weighted outer-fold AUC. The post-hoc aggregation audit/revision remains documented; no AUC is recalculated and no evaluation rule changes. |
| Tiny MLP automated coverage | Added dedicated `tiny-mlp-verification` Actions job, CPU torch2.6.0, invoking the unchanged verification() on synthetic data only. It remains a model-form sensitivity. |
| Overlapping-window sensitivity | Deferred. The proposed sliding1s versus non-overlapping nominal20s comparison belongs to a separately frozen measurement-robustness task. No such patient comparison, pregnancy-direction analysis or AUC-based definition selection runs here. |

The proposed next robustness task must freeze window anchoring, segment/support treatment and cohort rules before calculation, apply one protocol across cases, report measurement/rank/support differences, and keep any later outcome-direction assessment separate from measurement generation. Current primary MSE is not replaced by whichever alternative looks favorable.

The new boundary cannot retroactively blind the historical cohort or establish independent validity. Future independent evaluation still needs patients and access separation uninvolved in development. All patient-level input/output artifacts remain private.
