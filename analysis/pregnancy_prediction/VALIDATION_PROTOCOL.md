# Actual frozen internal validation

Outcome1=clinical pregnancy,0=non-pregnancy; probabilities are P(outcome1). Historical source-dash records remain coded0 under the unchanged source policy and explicitly flagged locally:5 on306,1 on185. They remain an unresolved source-label limitation, not confirmed negatives.

Outer5-fold patient-grouped CV ×10 repeats; inner5-fold patient-grouped CV for C selection. All cycles of a patient remain together at both levels. Original global fold maps and canonical row order are reused, including in baseline/augmented comparisons. Original outer seeds are20260531…20260540; inner seed rule is20260531+1000+repeat*100+outer_fold, with original1-based outer-fold indexing. The archived generation source is authoritative; its exact original input ordering is needed to regenerate identical assignments.

Imputation, continuous scaling and categorical modes are learned independently in each inner training split; the selected model is then refitted with parameters learned on the outer training set only. Outer tests choose no variable, C, epoch or initialization. Exact split membership is private, so public examples cannot reproduce the historical patient assignments without authorized data.

## Frozen AUC aggregation

The primary reported quantity is **fold-stratified / pair-weighted outer-fold AUC**, not pooled-OOF AUC. Existing result columns named `AUC` are retained unchanged.

For repeat r,test fold f, calculate AUC_rf within that test fold and pair weight `w_rf = n_positive_rf*n_negative_rf`:

`AUC_r = sum_f(w_rf*AUC_rf)/sum_f(w_rf)`

`reported AUC = mean_r(AUC_r)`.

For ΔAUC, augmented and baseline use identical patients/exams/folds/repeats, so differences are paired within each fold before the same weighted aggregation. Never compare folds' raw probability scales as the primary pooled ranking. Simple fold means/pooled values in older audits are separate sensitivity/diagnostic results.

This rule was adopted **after a pooled-OOF audit revealed cross-model/fold probability-scale concerns**, then frozen. It is a post-hoc evaluation revision, not a rule prespecified at study inception. Older pooled values remain historical diagnostics, not the current primary AUC. No evaluation method is chosen again for this package.

Brier and log-loss are exam-average OOF probability scores per repeat, then averaged across10 repeats. Lower is better. Do not weight these by AUC's positive-negative-pair weights. Where an archive includes per-fold scores they remain descriptive.

## Conditional paired intervals

The archived evaluator uses10000 patient-cluster bootstrap draws,seed20261120. The same global patient multiplicities apply to every model/repeat, preserve all cycles and labels, and recompute within-fold positive-negative-pair weights after each draw. Bootstrap joint model draws support paired ΔAUC.

**The95% intervals condition on fixed OOF predictions. They do not repeat fold creation, imputation, scaling, C/epoch selection, training or prior exploratory choices.** They therefore do not represent full development-process uncertainty or independent external validation. Repeats/folds overlap and cannot be treated as independent studies. No new patient bootstrap/refitting is run in this packaging task.

Patient clusters, case pairing and train-only preprocessing are tested on synthetic fixtures. Public tables contain aggregate/fold results only, never the private patient OOF matrix.
