# Double-activity separation: necessary validation

The experiment in `experiments/test_double_event_signals.py` addresses event
identity rather than overall candidate count. It does not alter A0, formal
amplitude/ER or the reviewed patient-event interface.

## Frozen design

Use three original rigid residual textures, both propagation directions,
strong-to-strong (.04/.04) and strong-to-weak (.04/.01), and fixed spacings
5, 10, 15 s. Include 20 s to reproduce the previously observed weak-tail
failure, not to search for a favorable filter setting. This gives 48 conditions
and two wall residuals per condition. Related textures, walls and spacings are
not independent clinical validation observations.

Gaussian length width=4 s, speed=.04 normalized paths/s, native timestamps,
finite measurement interval and leave-one-texture-out absolute thresholds all
remain fixed. Sum the two LENGTH changes first, then calculate the finite
interval relative rate. Adding two independently normalized rates is different.

Compare periodic FFT + observable-partial phases with reflected FFT + the
same phase retention. No cutoff/order/noise-floor sweep and no outcome data.

```bash
python experiments/test_double_event_signals.py \
  --controls-root outputs/frozen_image_warp_controls \
  --noise-reference outputs/noise_reference.json \
  --source-protocol outputs/partial_phase_trial/protocol.json \
  --output outputs/double_event_signals
```

NumPy suffices; this runner does not use the Butterworth branch or SciPy.
Data/outputs remain local and ignored. Source hashes must match the preceding
trial. JSON records and compressed arrays retain raw/filtered clean A, filtered
noisy A, coordinates, layer traces and independent truth.

## Filter-independent truth and matching

Keep both latent component arrival lines and observable maxima of the RAW
composite length signal. The latter are positive-to-negative transitions of
its unfiltered finite-interval rate. Signed extrema and phase bounds also come
from this raw signal, never from the filter under evaluation.

With these fixed waveforms, 5 s spacing yields one composite maximum; 10 s
strong-to-weak also yields one. They must not be scored as failures to recover
two visible events. The 10 s equal-strength and 15/20 s conditions contain two
observable maxima. This is a property of these fixtures, not a clinical
minimum-spacing rule or permission to delete short patient activities.

For evaluation only, attribute candidate phase nodes by membership in raw
same-sign phase intervals. Out-of-phase nodes remain unassigned. Match groups
one-to-one to observable events, maximizing supported native sections and then
minimizing raw-reference timing error. A candidate cannot match two events.
Always report coverage and error: an assigned group is not necessarily a
correctly recovered event. In particular, a fragment with a large timing error
must not inflate a recall/accuracy claim. Single-phase candidates retain their
signed peak marker; paired candidates use the zero-crossing marker.

`audit_event_filtering.known_match()` was also corrected to use unfiltered
isolated peaks for single-phase references. Old artifacts remain historical;
their filtered-reference timing metrics should not be treated as corrected
independent-reference results.

## Necessary layer checks

- Filter: retain raw phase bounds and measure clean-filter signed peak
  retention, displacement, and relation to the existing noise floor.
- Lobe/ridge: record observable nodes before `_ridges()` and the supported
  sections in each surviving path afterward.
- Pairing: record raw-identity switches and phase pairs spanning different
  observable activities. In these positive length-bump fixtures a trough
  between activities is not a new maximum; this is not a universal rule for
  physiological polarity.
- Ownership: evaluate actual complementary attachments and record joins
  between distinct pure raw identities. Potential duplicate support is not
  automatically a confirmed split.

Use clean/no-noise controls under each filter, plus raw clean input under the
same connector and threshold, to distinguish upstream information loss from
connection failure. Raw clean input is a causal control, NOT a proposed noisy
patient detector.

## Result and implementation decision

At 15 s both responses retained both observable activities, including weak
second activities. At 20 s, reflected FFT reproduced the weak second-wave
support loss across textures/directions. It therefore remains auxiliary rather
than replacing periodic FFT.

At 10 s equal strength, both bandpasses suppressed the distinguishing inner
phases before connection. Their raw peaks exceeded the existing floor, while
all corresponding clean-filter inner-phase peaks fell below it. The same
connector on unfiltered clean input separated both observable maxima. This
supports addressing the event-input loss before modifying ridge/ownership
logic; it does not establish an optimal replacement filter or patient accuracy.

No new physiological veto, default filter replacement or automatic patient
count was introduced. The next event-layer change should preserve separation
evidence in its input while retaining the established band-limited amplitude
and ER definitions. The experiment does not justify applying a 10/15 s rule
to real events.
