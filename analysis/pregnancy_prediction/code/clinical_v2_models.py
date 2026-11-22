"""One locked treatment-context extension of Clinical9, without feature selection."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from analysis.pregnancy_prediction.code.clinical9_models import (
    GRID, pipeline as clinical9_pipeline,
    preprocessing_state as clinical9_state, reconstruct as clinical9_reconstruct,
)
from analysis.pregnancy_prediction.code.clinical9_data_audit import NUM9, MSE

PREP = 'preparation_context_v2'
ICSI = 'ICSI_involved_before_scan_v2'
PREP_LEVELS = ['HRT_based', 'natural_recorded', 'FET_ovulation_induction', 'fresh_cycle_stimulation']
PREP_MAP = {'激素替代周期': 'HRT_based', '降调节+激素替代周期': 'HRT_based',
            '自然周期': 'natural_recorded', '诱导排卵周期': 'FET_ovulation_induction',
            '刺激周期': 'fresh_cycle_stimulation'}
METHOD_MAP = {'conventional_IVF_only': 0., 'ICSI_involved_including_mixed_or_rescue': 1.}
BASE = NUM9 + ['blastocyst_transfer', 'endometrial_pattern_pure_C']
MODELS = {'Full306': ['Clinical_v2'], 'MSE185': ['Clinical_v2', 'Clinical_v2_MSE']}


def inputs(model):
    assert model in {'Clinical_v2', 'Clinical_v2_MSE'}
    return BASE + ([MSE] if model.endswith('_MSE') else []) + [PREP, ICSI]


def pipeline(model, C):
    # Reuse the untouched frozen Clinical9 transformations and Ridge settings.
    fitted = clinical9_pipeline('Clinical9_MSE' if model.endswith('_MSE') else 'Clinical9', C)
    preparation = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ('encode', OneHotEncoder(categories=[PREP_LEVELS], drop='first',
                                 handle_unknown='error', sparse_output=False)),
    ])
    fitted.named_steps['preprocess'].transformers += [
        ('preparation', preparation, [PREP]),
        ('ICSI', SimpleImputer(strategy='most_frequent', keep_empty_features=True), [ICSI]),
    ]
    return fitted


def preprocessing_state(fitted, train, model):
    state, names = clinical9_state(fitted, train,
        'Clinical9_MSE' if model.endswith('_MSE') else 'Clinical9')
    blocks = fitted.named_steps['preprocess'].named_transformers_
    prep = blocks['preparation']
    state['preparation_mode'] = prep.named_steps['imputer'].statistics_.tolist()
    state['preparation_categories'] = prep.named_steps['encode'].categories_[0].tolist()
    state['ICSI_mode'] = blocks['ICSI'].statistics_.tolist()
    assert state['preparation_categories'] == PREP_LEVELS
    assert state['preparation_mode'][0] == sorted(train[PREP].dropna().mode())[0]
    assert state['ICSI_mode'][0] == train[ICSI].mode().iloc[0]
    names += ['preparation_' + level for level in PREP_LEVELS[1:]] + [ICSI]
    assert len(names) == (14 if model.endswith('_MSE') else 13)
    return state, names


def reconstruct(frame, state, names):
    clinical = clinical9_reconstruct(frame, state, names[:-4])
    preparation = frame[PREP].fillna(state['preparation_mode'][0])
    nominal = np.column_stack([preparation.eq(level).to_numpy(float) for level in PREP_LEVELS[1:]])
    method = frame[ICSI].fillna(state['ICSI_mode'][0]).to_numpy(float)[:, None]
    return np.column_stack([clinical, nominal, method])
