"""Within-fold AUC and paired patient-cluster bootstrap for frozen OOF scores."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits


def evaluate_frozen_scores(frame, score_columns, iterations=10000, seed=20261120):
    """One exam per repeat; score columns paired on the same rows and folds.

    Return joint bootstrap draws so all requested contrasts use the same patients.
    Fold pair weights are recomputed after each global patient-cluster draw.
    """
    keys = ['repeat', 'exam_id']
    assert not frame.duplicated(keys).any()
    assert set(frame.outcome.unique()) == {0, 1}
    assert np.isfinite(frame[score_columns].to_numpy()).all()
    assert frame[score_columns].ge(0).all().all() and frame[score_columns].le(1).all().all()
    repeats = sorted(frame['repeat'].unique())
    assert len(repeats) == 10
    reference = frame[frame['repeat'].eq(repeats[0])].sort_values('exam_id').reset_index(drop=True)
    patients = sorted(reference.patient_id.unique())
    patient_index = {p: i for i, p in enumerate(patients)}
    nr, nm, ng = len(repeats), len(score_columns), len(patients)
    kernels = np.zeros((nr, nm + 1, ng, ng))
    fold_data, fold_rows, repeat_rows = [], [], []
    for ri, repeat in enumerate(repeats):
        q = frame[frame['repeat'].eq(repeat)].sort_values('exam_id').reset_index(drop=True)
        assert q[['exam_id', 'patient_id', 'outcome']].equals(reference[['exam_id', 'patient_id', 'outcome']])
        assert q.groupby('patient_id').fold.nunique().eq(1).all()
        assert q.fold.nunique() == 5
        pooled = {c: float(roc_auc_score(q.outcome, q[c])) for c in score_columns}
        for fold, part in q.groupby('fold', sort=True):
            y = part.outcome.to_numpy(int)
            assert set(y) == {0, 1}
            g = part.patient_id.map(patient_index).to_numpy(int)
            pp = part[score_columns].to_numpy().T
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            ix = (g[pos, None], g[neg][None, :])
            np.add.at(kernels[ri, nm], ix, np.ones((len(pos), len(neg))))
            for j, name in enumerate(score_columns):
                diff = pp[j, pos, None] - pp[j, neg][None, :]
                np.add.at(kernels[ri, j], ix, (diff > 0).astype(float) + .5 * (diff == 0))
                fold_rows.append(dict(repeat=int(repeat), fold=int(fold), model=name,
                    n_exams=len(part), n_unique_patients=part.patient_id.nunique(),
                    pregnancy_n=len(pos), non_pregnancy_n=len(neg), pair_weight=len(pos) * len(neg),
                    AUC=float(roc_auc_score(y, pp[j])), mean_probability=float(pp[j].mean())))
            fold_data.append((ri, y, g, pp))
        weighted = kernels[ri, :nm].sum(axis=(1, 2)) / kernels[ri, nm].sum()
        for j, name in enumerate(score_columns):
            repeat_rows.append(dict(repeat=int(repeat), model=name,
                weighted_fold_AUC=float(weighted[j]), pooled_AUC=pooled[name]))

    def evaluate(weights):
        result = np.zeros((len(weights), nm))
        for kernel in kernels:
            denom = np.einsum('bi,ij,bj->b', weights, kernel[nm], weights, optimize=True)
            assert np.all(denom > 0), 'Undefined draw: do not silently discard.'
            for j in range(nm):
                numerator = np.einsum('bi,ij,bj->b', weights, kernel[j], weights, optimize=True)
                result[:, j] += numerator / denom / nr
        return result

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(ng, np.full(ng, 1 / ng), size=iterations).astype(float)
    assert np.all(weights.sum(axis=1) == ng)
    with threadpool_limits(limits=1):
        points = evaluate(np.ones((1, ng)))[0]
        for w in [np.ones(ng), *weights[:2]]:
            num, den = np.zeros((nr, nm)), np.zeros(nr)
            for ri, y, g, pp in fold_data:
                ew = w[g]
                pair_weight = ew[y == 1].sum() * ew[y == 0].sum()
                if pair_weight == 0:
                    continue
                den[ri] += pair_weight
                for j in range(nm):
                    num[ri, j] += pair_weight * roc_auc_score(y, pp[j], sample_weight=ew)
            np.testing.assert_allclose(evaluate(w[None, :])[0], (num / den[:, None]).mean(axis=0), atol=1e-12)
        draws = evaluate(weights)
    assert np.isfinite(draws).all()
    return dict(points=points, draws=draws, folds=pd.DataFrame(fold_rows), repeats=pd.DataFrame(repeat_rows),
        n_exams=len(reference), n_unique_patients=ng,
        repeated_patient_groups=int(reference.groupby('patient_id').size().gt(1).sum()),
        pregnancy_n=int(reference.outcome.sum()), non_pregnancy_n=int(reference.outcome.eq(0).sum()),
        score_columns=score_columns, valid_draws=iterations, invalid_draws=0)
