# Local event-input smoothing: separation/noise experiment

The preceding double-activity test showed that the existing 0.5–5 cpm
rectangular bandpass can erase distinguishing inner phases. This experiment
tests one minimal alternative at the event INPUT only, not another parameter
sweep or a new physiological feature.

`experiments/event_input_smoothing.py` demeans each continuous valid run and
uses an odd centered moving average near the existing radial measurement
interval (about 2 s). The short kernel is reflected at run edges. These padded
values do not extend valid support. There is no frequency-band selection or
polynomial detrending. Native timestamps, point spacing, A0 and all formal
amplitude/ER/coordination calculations remain unchanged.

## Necessary paired checks

`experiments/test_event_input_smoothing.py` reuses:

- 48 fixed double-activity conditions, including the known weak-tail failure;
- 27 single-activity and nonpropagating fixed-residual conditions;
- 12 original image-warp controls.

Three input/reference combinations are compared with the same partial-phase
detector and unchanged connection/ownership logic:

1. Original periodic FFT and original absolute noise reference.
2. Span-mean input and the same frozen reference, isolating the input change.
3. Span-mean input and the SAME noise-calibration rule applied to its residuals:
   three times P95 of per-section temporal SD pooled over the other textures.

The original periodic thresholds are reproduced numerically before computing
the alternative thresholds. This is a prespecified calibration comparison,
not a search for a threshold that reduces event counts. It does not establish
a clinical amplitude veto. Ground truth, one-to-one matching and latent versus
observable activity definitions come from the preceding double-event test.

```bash
python experiments/test_event_input_smoothing.py \
  --controls-root outputs/frozen_image_warp_controls \
  --noise-reference outputs/noise_reference.json \
  --source-protocol outputs/partial_phase_trial/protocol.json \
  --output outputs/event_input_smoothing
```

The runner writes per-condition reports, noise references, source hashes and
summary results locally. It needs NumPy, not SciPy. No patient data or derived
outputs are published, and no clinical outcome is used.

## Interpretation and decision

Local averaging preserves the 10 s double-activity separation evidence and
substantially improves timing in that test. Direct substitution, however,
exposes additional residual fluctuations and degrades weak-event support.
Keeping the old reference increases nonpropagating candidates; applying the
same calibration rule to the changed response reduces those candidates but
also loses weak signals. Fewer candidates after calibration is therefore not
sufficient evidence of improvement.

This alternative is not adopted as the default. The result supports preserving
raw separation evidence while retaining noise control, rather than simply
weakening the sole event prefilter. A possible next targeted experiment is to
use less-filtered evidence to check/split existing activity proposals rather
than independently generating all events from it. That refinement is not
implemented or claimed validated here.

The established periodic-plus-partial experiment and formal feature code remain
unchanged. Counts remain unreviewed candidates, not patient wave frequencies;
assigned truth matches must be read alongside spatial support and timing error.
