# Event-filter boundary and response diagnostic

This experiment changes only the prefilter used by the experimental event
detector. A0, finite-interval radial deformation, formal band-SD amplitude,
ER, coordination and reviewed-event patient interfaces remain unchanged.

`experiments/audit_event_filtering.py` compares three prespecified responses:

- Original periodic FFT, 0.5–5 cpm.
- Even-reflected FFT at the same nominal cutoffs, cropped to observed time.
- Fixed second-order Butterworth SOS with SciPy default odd padding and a
  forward/backward pass. The nominal cutoffs agree; its effective response
  does not equal the original rectangular frequency mask.

Run both the original detector and the observable-partial-phase detector for
every response. Censored-peak retention must not be a hidden difference between
filter comparisons. Do not compare one version's counts with another's.

The order, cutoffs and padding are engineering ablations, not published uterine
wave decision rules. No parameter search, amplitude veto or patient-specific
threshold is introduced. Keep the existing absolute noise thresholds frozen;
these are response-sensitivity tests, not independently noise-calibrated new
clinical detectors. SciPy is optional for this experiment and is not added to
the production package dependencies.

## Reproduction

```bash
python experiments/audit_event_filtering.py \
  --review-root outputs/cohort_review \
  --controls-root outputs/frozen_image_warp_controls \
  --noise-reference outputs/noise_reference.json \
  --partial-trial outputs/partial_phase_trial \
  --output outputs/event_filter_audit
```

The runner checks source hashes against the earlier partial-phase trial. It
uses all twelve development cases, twelve original image-warp controls and the
same twenty-seven fixed-noise signal conditions. Additional prespecified pulse
centers at 22 and 38 s test reliance on the original 30 s center: same three
textures, two directions, three amplitudes and two wall residuals. No samples
wrap around and no gaps are filled. These are related fixtures, not independent
clinical validation cases.

Each column's continuous valid run is filtered separately. Reflected samples
are boundary assumptions, never observations or extensions of valid support.
Reflection changes the FFT grid and admits additional frequency bins; it does
not improve the frequency resolution supported by the acquisition. Therefore
this experiment cannot isolate boundary effects from all spectral-grid effects.

## Evaluation and output contract

Keep source-specific JSON with every candidate and its measured support.
For known injections, reproduce the existing whole-path matching convention:
highest retained spatial extent, then lowest arrival error. Also report timing
error and the minimum retained path fraction; a direction hit alone is not
recovery of the whole injected activity. The fraction is relative to the
retained sampled spatial interval, not necessarily the complete anatomical
0–1 path.

Report nonpropagating directional structures separately. Real-case candidate
counts have no accuracy interpretation without a reference. Unknown ground
truth, unreviewed event count and frequency stay null, not zero. Partial-phase
structures include support-only evidence; their counts are not the anchored
identity-proposal counts from the preceding trial.

Controlled results support keeping reflected FFT plus observable partial phases
as a sandbox candidate: fewer extra signal-layer structures with preserved
direction and spatial support, including shifted pulses. A gradual response
can reduce counts while shortening the supported weak-wave path and increasing
nonpropagating directional structures. Thus fewer candidates is insufficient
grounds for replacement.

The original image-level negative controls still contain directional structures
under the reflected/partial combination. Single-pulse fixtures do not establish
correct separation of successive events. Formal default replacement and
automatic patient wave counting remain unvalidated. Patient data and all
derived outputs stay local under ignored `outputs/`.

Follow-up: [double-activity separation](DOUBLE_EVENT_VALIDATION.md) adds
filter-independent raw references, one-to-one identity matching and clean
causal controls. The single-phase reference in the runner now uses raw isolated
peaks; historical filtered-reference timing outputs are not retroactively
corrected.
