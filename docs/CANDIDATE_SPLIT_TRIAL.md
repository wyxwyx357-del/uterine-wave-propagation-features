# Parent-constrained split verifier: sandbox experiment

Baseline: `5f9e3e945b2e124dc29eec2c5be415aa7e97d55f`.
The only new operation is proposing two children inside an EXISTING periodic
FFT + observable-partial-phase candidate. A0, A(s,t), the periodic band, noise
reference, all connector/ownership function bodies, and formal features remain
unchanged. This trial is not adopted and does not output patient counts.

## Implementation

`experiments/candidate_split_verifier.py` receives an immutable parent event,
native maps/validity, the frozen noise floor and the actual measurement span.
It does not generate parents. Auxiliary processing reads only the parent's
sections and its original time bounds plus one measurement span of context.
It does not scan other sections, merge parents or use injected truth.

Each continuous valid local run is processed with the already tested
`measurement_span_mean`, at the existing approximately 2 s span. Missing runs
remain separate. Child arrival marks must lie inside the original unexpanded
parent time bounds; context only supports phase measurement. Bounding-box
membership does not prove physiological ownership; this becomes a failure
mechanism in the experiment, below.

Local signed phase peaks use the existing absolute noise floor. Two independent
positive-to-negative phase sequences must share adjacent native sections,
have disjoint phase nodes and preserve chronological rank and arrival order.
This is the frozen length-bump fixture's observable-maximum definition, not a
claim that all real physiological events must have one polarity. Conflicting
identities, more than one plausible pair or insufficient shared support remain
ambiguous. A single peak/section cannot create a child identity.

The unchanged `_ridges()`, `_phase_edges()`, `_measure_pair()` and restricted
`_pair_phases()` are called for child measurement. Child direction/speed are
null when propagation is unsupported, even if activity identity is supported.
The parent direction is not overwritten and no global auxiliary detector runs.

Every parent has exactly one status:

- `keep_single`: preserve its original evidence.
- `split_two`: two traceable local child proposals.
- `split_ambiguous`: preserve original evidence, no automatic replacement/count.

Each output retains the complete original parent, context bounds, reason,
support, local curves, original phase references and child measurement. Formal
`event_count` and `event_frequency` stay null for every status.

## Fixed validation

```bash
python experiments/test_candidate_split_verifier.py \
  --source outputs/event_input_smoothing \
  --controls-root outputs/frozen_image_warp_controls \
  --noise-reference outputs/noise_reference.json \
  --output outputs/candidate_split_trial
```

The runner reuses exactly the existing 87 records: 48 double, 27 single/signal
negative and 12 image-level controls. Parent JSON is used verbatim, with source
and frozen-producer hash checks. No outcome data or additional fixtures are
introduced. Scope/interface unit tests are not new benchmark cases.

Smoothing remains the existing measurement span. Adjacent support 3/4/5 is
compared on the SAME prepared evidence; no window or amplitude optimization.
Support=3 is the reference closest to the existing connector, not an adopted
clinical minimum. The larger supports are sensitivity comparisons.

Evaluation uses the preceding filter-independent observable raw truth and
one-to-one assignments. Latent components that form one observable activity
are not forced into two. Report each child's support/error, weak-event support,
false splits, duplicate truth support across parents, and ambiguous proposals.
An assigned event is not automatically recovered; counts alone are insufficient.

## Controlled outcome and decision

At support=3, true two-child splits were recovered in 6/12 wall conditions of
the 10 s equal-strength case. Support=4/5 recovered 5/12. No confirmed false
splits appeared in the 41 single-wave parents or the 31 nonpropagating parents
at any support setting. However, 34/41 single-wave parents remained ambiguous;
zero confirmed false splits does not establish a definitive detector.

The experiment failed the preservation requirement. At 15 s equal strength,
20 parents with an existing single raw-event attribution were split again.
The proposed second activity was real but already represented by a different
parent. At 20 s strong-to-weak, both raw identities remained assignable, but
the weak second event's minimum native-section support fell from 91.7% to
29.4% under provisional replacement. Parent-local evidence is not exclusive
event ownership. Larger adjacent-support settings did not fix this problem.

Therefore no setting is selected for adoption, no real twelve-case run is
performed, and no default or formal event API is changed. The result supports
the usefulness of local evidence but not independent per-parent replacement.

The minimal next step is an ownership check before accepting a split: establish
whether both children belong to the parent and whether the second child already
has an existing separate parent identity. Do not silently broaden/merge parents
to improve the 10 s result, retune smoothing, or raise the amplitude threshold.
That ownership change is NOT implemented in this trial.

All condition reports, local curves, matching details and source fingerprints
stay local under ignored `outputs/`. No patient accuracy or validated automatic
frequency is claimed.
