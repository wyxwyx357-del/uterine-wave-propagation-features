# Frozen measurement and aggregation method

This documents the actual code, without changing definitions or recomputing patient results. `record` denotes an examination/patient-cycle, not an averaging of multiple cycles from the same patient.

## A0 output to approximately2s radial A

For each anatomical radial pair, let L(t) be its corrected Euclidean separation. The implemented input is

`A(t-h,t) = [L(t)-L(t-h)] / [L(t-h) * (t-t_h)]`, units s^-1.

`compute_spacetime` chooses an integer hop nearest nominal2s on the acquisition clock and records actual interval endpoints/centres. It is finite-interval relative length change, not Huang's adjacent-frame strain-rate. Explicit grid pair IDs are retained. No millimetre calibration is inferred.

`interval_support` requires valid endpoints and every intervening step, consistent segment IDs and correction/pose epochs. Invalid/reanchor transitions and pose-invalid intervals cannot contribute across the break. A new reanchored segment can contribute later once its own entire measurement interval is supported; it is never stitched to the old segment for differentiation.

## 20s space-time windows and ER

`summarize_patient` fixes the anatomical range0.2–0.8 and passes the selected columns to `analyze_modes`. Nominal20s windows advance nominal1s; both are rounded to sample counts on native uniform timestamps. Thus the span between first/last samples is `(n-1)*dt`, while the spectral period is `n*dt`. Exact spans/steps are returned as metadata; nominal20s does not assert an exact20s endpoint span.

Every FFT input is a complete contiguous time×space rectangle. Unsupported time/space is not interpolated. At least3 supported spatial columns are the existing numerical minimum; this is not a physiological wave criterion. A window crossing a tracking gap fails complete support and does not enter as a zero-valued observation. The two walls are paired only on the same original spatial block and identical start/end/centre times.

The temporal mean of each spatial column is removed. The existing weighted spatial transform uses anatomically ordered cervix→fundus coordinates. Within0.5–5cpm, forward energy E_F sums spectral cells with `f_t*f_s<0`; backward E_B uses `f_t*f_s>0`. Zero spatial/temporal direction cells do not contribute to that difference:

`ER = (E_F-E_B)/(E_F+E_B)`.

Positive ER denotes C2F. The existing numerical energy floor is unchanged; it is not a physiological noise/activity threshold. Rigid/noisy motion can still create a directional ER and inter-wall disparity, as the archived controls show. ER is not proof of a discrete wave event.

## Block MSE and temporal runs

For a fixed common spatial block b and its supported paired-window run R,

`MSE_block,R = mean_{t in R} [(ER_ant,b,t-ER_post,b,t)^2]`.

`modes.analyze_modes` splits runs when centre spacing exceeds1.5 times the known nominal-step sample spacing. `_segments` in record aggregation uses1.5 times the median positive centre spacing for its descriptive per-block run summary. These existing heuristics describe runs; they do not fill missing times or change the main record MSE weighting. Their gap-run counts need not be a definitive count of material tracking segments when valid observations are very sparse. No threshold is changed here.

## Exact record_MSE_time_equal

Let T be all distinct valid `(window_start_s,window_end_s,window_center_s)` tuples. Let B_t be the available, bilaterally paired spatial blocks at t. The true code computes

`e_t = (1/|B_t|) * sum_{b in B_t} (ER_ant,b,t-ER_post,b,t)^2`

`record_MSE_time_equal = (1/|T|) * sum_{t in T} e_t`.

This answers the aggregation questions explicitly:

1. Simultaneous blocks receive equal weight **within that time**.
2. Each unique observed window time then receives equal weight, regardless of how many blocks it has.
3. All supported runs/segments contribute their valid observations; segments are not first equally averaged. A longer run therefore contributes more valid window times.
4. Overlapping20s windows are retained, one observation per nominal1s window time. This is equal-window-time weighting, not independent20s measurements and not unique-raw-frame-duration weighting.
5. Unsupported gaps contribute no observation, never zero; no cross-gap filter/FFT/difference is computed.
6. Spatial support is not filled; disconnected blocks are not merged into one spectrum.
7. With no shared ER windows the output remains `None`/missing.
8. `record_MSE_pooled` instead gives every window-block observation equal weight and is an existing sensitivity output.
9. With one block per time, record MSE equals the ordinary mean squared ER difference, weighted by window count across runs. It equals the old single-block value only where that old and new observation support is identical.
10. The name `time_equal` refers to unique **observed window tuples**, not to all original video seconds or equally weighted segments.

`record_CC_time_equal` first averages ER across blocks at each observed time and then computes one Pearson correlation over **all supported times pooled across segments**. Gaps are not interpolated, but segment mean shifts can affect this pooled correlation. It is neither continuous-sequence synchrony nor an average of within-segment correlations. CC remains auxiliary/descriptive. MSE instead averages squared differences at each observed time, without introducing cross-segment lags or temporal relationships.

The nearby historical inline comment about first averaging ER describes the CC series; the actual MSE implementation uses `error_by_time`, as the equations above show. The code is preserved and the documentation follows the executable formula. The old formal METHOD's stated same-time squared-difference weighting agrees with the actual implementation.

## Post-hoc explanatory analysis

`_decompose` returns means, population SDs, squared mean bias, squared SD difference, time-shape term and MSE_z. For aligned nonconstant wall series with the same observations/weights and population SD (ddof=0):

`MSE = (mu_A-mu_P)^2 + (sigma_A-sigma_P)^2 + 2*sigma_A*sigma_P*(1-CC)`

`MSE_z = 2*(1-CC)` up to floating-point error.

The code requires at least3 observations and non-negligible variance for CC/MSE_z. Constant series leave standardized MSE undefined; the unstandardized decomposition can still use its residual shape term. At record times with multiple spatial blocks, the code deliberately marks the decomposition undefined (`not_defined_for_multiple_blocks_per_time`) rather than substituting a decomposition of the spatially averaged ER series. That alternative would generally measure a different MSE.

Signed `delta_ER = record_ER_mean_anterior-record_ER_mean_posterior`; absolute difference is its absolute value. These and `record_mean_bias_sq`, `record_sd_scale_sq`, `record_time_shape_term`, `record_MSE_z` are **post-hoc explanatory analysis**. They are not substituted for primary ER-MSE or promoted using their pregnancy associations.

## Historical label access and future measurement separation

The mathematical tracking/A/ER/MSE functions do not use pregnancy labels. However, historical `complete_record_coordination.py:run()` reads a cohort CSV that already contains OUTCOME, CLINICAL and GROUP, copies these fields into the aggregated records, then performs association/prediction in the same invocation. **That historical run was not strictly outcome-blind.** Formula-level label independence and runtime label access are different claims. No outcome-dependent branch was found in the numeric MSE function; this provenance correction changes no stored value/result.

Future `record_mse_measurement.py:measure_record` accepts exactly `case_id`, `paired_ER_windows` and `measurement_metadata`. Nested schemas are allowlisted: pregnancy, outcome and clinical fields are rejected rather than silently dropped. `record_mse_core.py` copies the four historical numeric functions verbatim; tests compare their full ASTs and outputs. This new measurement path imports no clinical/model modules and opens no clinical table.

Before aggregation, metadata must specify `definition_version=radial_wave_features_v2`, `requested_s_range=[0.2,0.8]`, `nominal_interval_s=2`, `nominal_window_s=20`, `nominal_step_s=1`, `band_cpm=[0.5,5]`. Missing/mismatched metadata fail explicitly; no protocol is guessed, missing value replaced with0, patient silently deleted, or new physiological threshold imposed. Native sample-rounded durations are carried separately. This verifies declared protocol consistency, not authenticity of arbitrary external metadata.

`replay_measurement` constructs/checks metadata from its fixed measurement pipeline and calls that boundary. Its CLI now requires `--case-id` and writes the strict phase1 artifact only; its Python function retains diagnostic modes alongside the artifact. The unchanged historical mixed-cohort script is an archive, not the entry for future label-free validation.

Only the separate phase2 `mse_outcome_association.py` loads clinical/outcome data and joins exact case keys after measurement files are saved/frozen. It preserves unmatched/missing records and performs no fitting, AUC, OR, variable selection or label recoding. Further association/prediction still needs its specified protocol; the older7-variable association model and Clinical9 prediction remain distinct.

The new interface cannot retroactively blind development. Future independent validation requires patients/data access uninvolved in development. No patient measurement or association is rerun by this review fix.
