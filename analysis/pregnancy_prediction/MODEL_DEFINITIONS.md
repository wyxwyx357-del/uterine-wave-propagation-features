# Frozen model definitions

Clinical9 retains age, BMI, infertility years, actual transferred embryo count, blastocyst indicator, endometrial thickness, endometrial volume, pure-C endometrial pattern indicator and days before transfer. Actual embryo count/stage were confirmed known by scan completion. No outcome-driven re-selection is performed.

Continuous variables use training-fold medians and training-fold population mean/SD scaling. Blastocyst/pure-C indicators use the frozen encoding and training-fold mode, unscaled. All nine predictors stay in Ridge. ER-MSE is `record_MSE_time_equal`, observed on the specified imaging cohort, standardized using the training fold and fixed into the augmented model.

Ridge: LogisticRegression L2,lbfgs,max_iter10000,tol1e-4. C grid0.001,0.01,0.1,1,10,100; select equal-inner-fold mean log-loss minimum, exact ties first ascending C. No class weighting/threshold search. Position-adjusted comparison uses its paired clinical-position control, not just an unadjusted baseline.

Tiny MLP:9→8→1 or10→8→1; Linear/ReLU/Dropout0.20; output logit; BCEWithLogitsLoss; AdamW lr1e-3/weight_decay1e-4; batch32; max200 epochs; inner-validation BCE patience20. Within each outer training set and initialization, take the median of5 inner best epochs, refit from scratch on all outer training data for that epoch count, then evaluate outer test. Average probabilities over exactly20261122/20261123/20261124. No architecture/hyperparameter/seed search. Primary Ridge is not replaced.

## Sensitivity-only baselines

- Clinical6: historical six-variable clinical baseline, no rewriting of its outcomes/folds.
- Clinical11: Clinical9+PI+VFI, its own observed-blood cohort287/270 and MSE/blood180/174. Keep its raw-reported blood audit rules distinct from Enhanced cleaning.
- Enhanced:12 candidate clinical constructs/13 encoded columns; one bounded Elastic Net exploration, with target MSE L2-only. It is not the primary Ridge result and is not a selected fixed final model.
- Clinical v2: Clinical9+preparation/treatment background+IVF-only versus ICSI-involved. HRT/downregulated HRT merge to HRT-based while raw labels remain private; natural, FET induction and fresh stimulation remain distinct. Four preparation levels yield3 indicators; with the ICSI indicator,11 constructs become13 columns,14 with MSE. Any ICSI includes mixed/Rescue, not pure-IVF versus pure-ICSI. Three uncertain pre-scan fertilization timings remain raw missing; training mode only, no later-data backfill.
- Static PCA and static Elastic Net: final exploratory analyses; original Clinical6 and five dynamic inputs, not Clinical9, on the common180/174 cohort. No further method search is authorized by this archive.
