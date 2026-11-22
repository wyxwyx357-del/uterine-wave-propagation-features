# ER-MSE measurement and pregnancy association

`record_MSE_time_equal` measures **inter-wall ER disparity**, not a pure temporal-synchrony measurement. Its true implementation is [summarize_paired_windows](code/complete_record_coordination.py). The decomposition is post-hoc explanatory analysis, not a new prespecified primary feature.

Read [METHOD](METHOD.md), [RESULTS](RESULTS.md) and [LIMITATIONS](LIMITATIONS.md). Patient-level inputs/results remain private; aggregate source tables are in `results/`.

## Source chain

| Stage | Frozen source |
| --- | --- |
| A0 tracklets and reanchors | `code/measurement/tracking.py:track_segments`, `_lk.py` |
| Explicit grid/pair identities | `code/measurement/geometry.py` |
| Valid intervals, approximately2s radial A | `code/measurement/deformation.py:interval_support,rate_A,compute_spacetime` |
| Directional spectrum and ER | `code/measurement/spectrum.py:_spectrum,_direction_summary,direction` |
| Complete 20s windows/nominal1s step, paired ER, block MSE | `code/measurement/modes.py:analyze_modes` |
| Frozen0.2–0.8 anatomical region | `code/measurement/patient.py:summarize_patient` |
| Equal-window-time patient/exam aggregate | `code/complete_record_coordination.py:summarize_paired_windows` |
| Post-hoc bias/scale/shape decomposition | `code/complete_record_coordination.py:_decompose` |
| Clinical-adjusted association | `code/complete_record_coordination.py:_association`, `feature_association.py:fit_cluster_logit` |
| Mean-offset/technical/source audit | `code/er_mean_offset_batch_audit.py` |
| Outcome-blind spectral-support controls | `code/er_directional_support_audit.py` |
| Huang-like measurement/association bridge | `code/huang_bridge_phase_a.py`, `huang_bridge_phase_b.py` |

Important: record aggregation uses the **paired ER window observations**, not a simple average of previously emitted block MSE values. Block MSE is a parallel descriptive output; unequal windows/block support require the explicit record weighting in METHOD.

Portable saved-tracks entry: `python -m analysis.mse_pregnancy_association.code.replay_measurement --help`. No tracking/videos or clinical data are run during this packaging task.

## Evidence statement

ER-MSE showed a positive internal association with clinical pregnancy within the current development dataset. The incremental prediction signal remained positive across several clinical-baseline sensitivity analyses. These overlapping internal analyses do not establish independent reproducibility. **These results are based on internally developed and repeatedly explored data and do not constitute independent external validation.**

## Literature boundaries

- [Huang et al.,2022](https://doi.org/10.1109/TUFFC.2022.3165688): strain-rate,20s space-time spectrum,ER and inter-wall CC/MSE analyses. Our approximately2s finite-interval A, segmented support and exam-level aggregation are project implementations, not a line-by-line Huang reproduction. Current pregnancy association direction differs from Huang; this conflict remains visible.
- [Sammali et al.,2019](https://doi.org/10.1109/TUFFC.2018.2867098): tracked distance/strain methodology. This does not validate our particular point spacing, LK settings or clinical feature.
- [Wang et al.,2022](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.983177/full): activity/event detection branch; it is not the source of our MSE definition.

No papers, patient figures or copyrighted full texts are bundled.
