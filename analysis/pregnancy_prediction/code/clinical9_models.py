"""Frozen Clinical6/Clinical9 Ridge pipelines; no learned feature selection."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from analysis.pregnancy_prediction.code.clinical9_data_audit import NUM6, NUM9, MSE

GRID = [.001, .01, .1, 1., 10., 100.]
FULL_MODELS = ['Clinical6', 'Clinical9']
MSE_MODELS = ['Clinical9', 'Clinical9_MSE', 'Clinical9_Position', 'Clinical9_MSE_Position']
POSITIONS = ['anterior', 'posterior', 'horizontal']


def specification(model):
    assert model in set(FULL_MODELS + MSE_MODELS)
    numeric = NUM6 if model == 'Clinical6' else NUM9
    names = numeric + ['blastocyst_transfer']
    if model != 'Clinical6':
        names += ['endometrial_pattern_pure_C']
    if 'MSE' in model:
        names += [MSE]
    if 'Position' in model:
        names += ['uterine_position_clean']
    return numeric, names


def pipeline(model, C):
    numeric, names = specification(model)
    parts = [('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                                  ('scale', StandardScaler())]), numeric),
             ('stage', SimpleImputer(strategy='most_frequent', keep_empty_features=True), ['blastocyst_transfer'])]
    if model != 'Clinical6':
        parts.append(('pattern', SimpleImputer(strategy='most_frequent', keep_empty_features=True), ['endometrial_pattern_pure_C']))
    if 'MSE' in model:
        parts.append(('MSE', StandardScaler(), [MSE]))
    if 'Position' in model:
        position = Pipeline([('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('encode', OneHotEncoder(categories=[POSITIONS], drop='first', handle_unknown='error', sparse_output=False))])
        parts.append(('position', position, ['uterine_position_clean']))
    return Pipeline([('preprocess', ColumnTransformer(parts)),
        ('model', LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=10000, tol=1e-4))])


def preprocessing_state(fitted, train, model):
    numeric, _ = specification(model)
    prep = fitted.named_steps['preprocess']
    block = prep.named_transformers_['numeric']
    imputer, scale = block.named_steps['imputer'], block.named_steps['scale']
    np.testing.assert_allclose(imputer.statistics_, np.nanmedian(train[numeric], axis=0))
    filled = imputer.transform(train[numeric])
    np.testing.assert_allclose(scale.mean_, filled.mean(axis=0))
    np.testing.assert_allclose(scale.scale_, np.where(filled.std(axis=0) == 0, 1, filled.std(axis=0)))
    state = dict(n_fit=len(train), numeric_names=numeric, medians=imputer.statistics_.tolist(),
        means=scale.mean_.tolist(), scales=scale.scale_.tolist(),
        stage_mode=prep.named_transformers_['stage'].statistics_.tolist())
    feature_names = numeric + ['blastocyst_transfer']
    if model != 'Clinical6':
        state['pattern_mode'] = prep.named_transformers_['pattern'].statistics_.tolist()
        feature_names += ['endometrial_pattern_pure_C']
    if 'MSE' in model:
        s = prep.named_transformers_['MSE']
        np.testing.assert_allclose(s.mean_, train[[MSE]].mean(axis=0))
        state['MSE_mean'], state['MSE_scale'] = s.mean_.tolist(), s.scale_.tolist()
        feature_names += [MSE]
    if 'Position' in model:
        pos = prep.named_transformers_['position']
        state['position_mode'] = pos.named_steps['imputer'].statistics_.tolist()
        assert pos.named_steps['encode'].categories_[0].tolist() == POSITIONS
        state['position_categories'], state['position_reference'] = POSITIONS, 'anterior'
        feature_names += ['uterine_position_posterior', 'uterine_position_horizontal']
    np.testing.assert_array_equal(fitted.named_steps['model'].classes_, [0, 1])
    return state, feature_names


def reconstruct(frame, state, names):
    """Independent NumPy transform for checking exported coefficients/probabilities."""
    raw = frame[state['numeric_names']].to_numpy(float)
    raw = np.where(np.isnan(raw), state['medians'], raw)
    blocks = [(raw - state['means']) / state['scales']]
    stage = frame[['blastocyst_transfer']].to_numpy(float)
    blocks.append(np.where(np.isnan(stage), state['stage_mode'], stage))
    if 'endometrial_pattern_pure_C' in names:
        pattern = frame[['endometrial_pattern_pure_C']].to_numpy(float)
        blocks.append(np.where(np.isnan(pattern), state['pattern_mode'], pattern))
    if MSE in names:
        blocks.append((frame[[MSE]].to_numpy(float) - state['MSE_mean']) / state['MSE_scale'])
    if 'uterine_position_posterior' in names:
        position = frame.uterine_position_clean.fillna(state['position_mode'][0])
        blocks.append(np.column_stack([position.eq('posterior'), position.eq('horizontal')]).astype(float))
    return np.column_stack(blocks)
