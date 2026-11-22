# Measurement and association limitations

- Repeatedly explored internal development data; no independent external validation and no causal inference.
- The numeric MSE function is label-independent, but the historical aggregation run already held clinical/outcome columns and was not strictly outcome-blind. The new measurement-only interface applies to future work, not retrospective blinding.
- Current association direction conflicts with Huang2022. Neither study establishes that higher ER-MSE is biologically beneficial.
- Approximately2s finite-interval radial A, pixel-based point spacing, pose correction, segmented support and equal-window-time aggregation differ from Huang's implementation.
- Rigid/technical motion and weak spectral support can yield direction and nonzero MSE. No post-hoc hard support threshold is introduced or used to discard all local evidence.
- MSE contains mean and scale differences as well as temporal shape. Its post-hoc bias decomposition must remain explanatory. Decomposition is explicitly unavailable for multiple blocks at the same time.
- Overlapping windows are dependent. Reanchors cannot be crossed by interval differentiation/FFT; separately supported windows still contribute to the record summary.
- Record CC pools supported observations across disconnected segments and can reflect segment mean shifts. It is auxiliary, not continuous-time synchrony. Non-overlapping-window measurement sensitivity remains deferred; no alternative definition is selected here.
- Clinical association and prediction cohorts differ. Measurement availability can select a different case mix; missing measurements are never assumed to be zero.
- Patient cycles repeat. Grouping/clustering addresses this dependence in the specified analyses, not every source of development bias.
- Patient data are withheld. Public synthetic tests establish implementation consistency, not the numerical patient results or clinical generalizability.
- Archived per-block gap summaries use existing spacing heuristics. They are not evidence of globally persistent material-point identity.
