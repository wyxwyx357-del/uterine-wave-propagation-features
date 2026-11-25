"""Fixed four-model comparison; only the static block receives train-only PCA."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLINICAL = ['age', 'transfer_embryo_count_component', 'blastocyst_transfer',
            'bmi', 'infertility_years', 'endometrial_thickness_mm']
NUMERIC = ['age', 'transfer_embryo_count_component', 'bmi', 'infertility_years', 'endometrial_thickness_mm']
STAGE = 'blastocyst_transfer'
DYNAMIC = ['record_MSE_time_equal', 'record_CC_time_equal',
           'anterior_radial_stretch_activity_amplitude', 'posterior_radial_stretch_activity_amplitude',
           'significant_propagation_activity_frequency_cpm']
MODELS = ['Clinical', 'Clinical_Static', 'Clinical_Dynamic', 'Clinical_Static_Dynamic']
C_GRID = [.001, .01, .1, 1., 10., 100.]
K_GRID = [1, 2, 3, 5]


def has_static(model):
    return model in ['Clinical_Static', 'Clinical_Static_Dynamic']


def has_dynamic(model):
    return model in ['Clinical_Dynamic', 'Clinical_Static_Dynamic']


def pipeline(model, static_columns):
    assert model in MODELS
    numerical = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                          ('scale', StandardScaler())])
    blocks = [('clinical_numeric', numerical, NUMERIC),
              ('stage', SimpleImputer(strategy='most_frequent', keep_empty_features=True), [STAGE])]
    if has_dynamic(model):
        blocks.append(('imaging', StandardScaler(), DYNAMIC))
    if has_static(model):
        static = Pipeline([('input_scale', StandardScaler()),
                           ('pca', PCA(n_components=1, svd_solver='full', whiten=False)),
                           ('score_scale', StandardScaler())])
        blocks.append(('static', static, static_columns))
    return Pipeline([('preprocess', ColumnTransformer(blocks)),
                     ('model', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000))])


def grid(model):
    if has_static(model):
        # First exact tie: smaller k, then smaller C, before outer results exist.
        return [{'preprocess__static__pca__n_components': [k], 'model__C': C_GRID} for k in K_GRID]
    return {'model__C': C_GRID}


def training_isolation_check():
    import pandas as pd
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(20261121)
    static_columns = [f'static_{i}' for i in range(12)]
    names = NUMERIC + [STAGE] + DYNAMIC + static_columns
    data = pd.DataFrame(rng.normal(size=(48, len(names))), columns=names)
    data[STAGE] = np.arange(len(data)) % 2
    tr, te = data.iloc[:36].copy(), data.iloc[36:].copy()
    tr.loc[0, 'bmi'] = np.nan
    model = pipeline('Clinical_Static_Dynamic', static_columns)
    model.set_params(preprocess__static__pca__n_components=3)
    with threadpool_limits(limits=1):
        model.fit(tr, np.arange(len(tr)) % 2)
    prep = model.named_steps['preprocess']
    block = prep.named_transformers_['static']
    np.testing.assert_allclose(block.named_steps['input_scale'].mean_, tr[static_columns].mean())
    assert block.named_steps['pca'].n_samples_ == len(tr)
    z = block.transform(tr[static_columns])
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-12)
    before = block.named_steps['pca'].components_.copy()
    te.loc[:, NUMERIC + DYNAMIC + static_columns] += 1e6
    model.predict_proba(te)
    np.testing.assert_array_equal(before, block.named_steps['pca'].components_)
    np.testing.assert_allclose(block.named_steps['input_scale'].mean_, tr[static_columns].mean())
    return dict(training_only_scale_and_PCA=True, component_scale_unit_SD=True,
                extreme_heldout_values_do_not_change_fitted_basis=True)
