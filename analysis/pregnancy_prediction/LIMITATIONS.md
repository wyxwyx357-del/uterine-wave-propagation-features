# Prediction limitations and closed search

- Already explored internal development data. These results do not constitute independent external validation.
- Fixed-OOF conditional bootstrap omits full training/development uncertainty and the effects of prior model/feature/evaluation exploration.
- Primary Clinical9 is weak in the MSE-available subset. A positive incremental AUC does not imply an adequate absolute discrimination/calibration or clinical utility.
- Clinical11 uses a different blood-observed subset. Enhanced has different blood cleaning. Comparisons across those different n are not paired evidence of better predictors.
- Static PCA showed no incremental benefit under its fixed representation; this does not prove all101 raw static features are useless. The one final supervised EN exploration also showed no clear increment. Further search remains closed.
- Tiny MLP has no clear paired advantage over Ridge; it is sensitivity only, regardless of which point AUC is largest.
- Pregnancy-label source dashes, missing clinical information, treatment coding, measurement availability and case mix remain limitations. A failed treatment-context extension does not establish that all useful clinical information has been exhausted.
- Standardized embryo quality, true prior reproductive/transfer history and AMH/AFC remain unavailable or unreliable. No later-outcome information is backfilled.
- No risk cutoff, final full-data patient classifier, external performance or clinical deployment claim is added.
- Patient data/OOF are withheld. Synthetic checks verify code behavior, not the archived patient results from publicly available data alone.

Model search is closed: no alternative classifier, feature screening, category regrouping, seed search or replacement of the primary Clinical9/Ridge analysis is part of this publication.
