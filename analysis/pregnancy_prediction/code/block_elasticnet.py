"""Logistic Elastic Net on static coefficients; fixed base covariates receive L2 only.

Sum-loss objective: logloss_sum + lambda/2*||base||²
  + lambda*(1-alpha)/2*||static||² + lambda*alpha*||static||1.
The intercept is unpenalized. lambda=1/C preserves the original base Ridge scale.
L-BFGS-B solves the equivalent smooth problem static=u-v, u,v>=0.
Accelerated proximal polishing enforces the same objective's KKT tolerance
if relative objective reduction stops L-BFGS-B prematurely.
"""
import numpy as np
from scipy.optimize import minimize, check_grad
from scipy.special import expit, logit

ALPHAS = [.9, .5, .1]
LAMBDAS = [1000., 100., 10., 1., .1, .01]
OPTIONS = dict(maxiter=10000, maxfun=100000, maxls=50, maxcor=20, ftol=1e-13, gtol=1e-8)
KKT_TOLERANCE = 1e-5


def objective_gradient(theta, X, y, n_base, alpha, lam):
    ns = X.shape[1] - n_base
    base = theta[1:1 + n_base]
    positive = theta[1 + n_base:1 + n_base + ns]
    negative = theta[1 + n_base + ns:]
    static = positive - negative
    coefficient = np.r_[base, static]
    eta = theta[0] + X @ coefficient
    inverse_n = 1. / len(y)
    strength = lam * inverse_n
    value = np.mean(np.logaddexp(0., eta) - y * eta)
    value += .5 * strength * (base @ base + (1 - alpha) * (static @ static))
    value += strength * alpha * (positive.sum() + negative.sum())
    error = (expit(eta) - y) * inverse_n
    gradient = X.T @ error
    gb = gradient[:n_base] + strength * base
    gs = gradient[n_base:] + strength * (1 - alpha) * static
    g = np.r_[error.sum(), gb, gs + strength * alpha, -gs + strength * alpha]
    return float(value), g


def kkt_residual(X, y, n_base, intercept, coef, alpha, lam):
    error = (expit(intercept + X @ coef) - y) / len(y)
    gradient = X.T @ error
    strength = lam / len(y)
    base = gradient[:n_base] + strength * coef[:n_base]
    static = coef[n_base:]
    gs = gradient[n_base:] + strength * (1 - alpha) * static
    residual = np.where(static == 0, np.maximum(np.abs(gs) - strength * alpha, 0),
                        np.abs(gs + strength * alpha * np.sign(static)))
    return float(np.max(np.r_[abs(error.sum()), np.abs(base), residual]))


def proximal_polish(X, y, n_base, intercept, coef, alpha, lam):
    """Same EN objective; soft threshold only static coordinates, never base.

    The global smooth-gradient Lipschitz bound uses logistic curvature <=1/4.
    Adaptive restart accelerates numerical convergence without selecting models.
    """
    design = np.column_stack([np.ones(len(y)), X])
    strength = lam / len(y)
    l2 = np.r_[0., np.full(n_base, strength),
               np.full(X.shape[1] - n_base, strength * (1 - alpha))]
    lipschitz = np.linalg.eigvalsh(design.T @ design / (4 * len(y)) + np.diag(l2))[-1]
    step = 1. / lipschitz
    threshold = step * strength * alpha
    current = np.r_[intercept, coef]
    extrapolated, momentum = current.copy(), 1.
    for iteration in range(1, 100001):
        gradient = design.T @ ((expit(design @ extrapolated) - y) / len(y)) + l2 * extrapolated
        candidate = extrapolated - step * gradient
        static = candidate[1 + n_base:]
        candidate[1 + n_base:] = np.sign(static) * np.maximum(np.abs(static) - threshold, 0.)
        if iteration % 10 == 0:
            residual = kkt_residual(X, y, n_base, candidate[0], candidate[1:], alpha, lam)
            if residual <= KKT_TOLERANCE:
                return float(candidate[0]), candidate[1:], iteration, residual
        next_momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
        if np.dot(extrapolated - candidate, candidate - current) > 0:
            extrapolated, next_momentum = candidate.copy(), 1.
        else:
            extrapolated = candidate + (momentum - 1) / next_momentum * (candidate - current)
        current, momentum = candidate, next_momentum
    raise RuntimeError(f'Same-objective proximal polishing did not converge: KKT={residual:.3g}')


def fit(X, y, n_base, alpha, lam, warm=None):
    X, y = np.asarray(X, float), np.asarray(y, float)
    assert X.ndim == 2 and len(X) == len(y) and np.isfinite(X).all()
    assert set(np.unique(y)) == {0, 1} and 0 <= n_base < X.shape[1]
    assert 0 < alpha < 1 and lam > 0
    ns = X.shape[1] - n_base
    initial = np.zeros(1 + n_base + 2 * ns)
    initial[0] = logit(y.mean())
    if warm is not None:
        initial[0] = warm['intercept']
        coefficient = np.asarray(warm['coef'])
        initial[1:1 + n_base] = coefficient[:n_base]
        initial[1 + n_base:1 + n_base + ns] = np.maximum(coefficient[n_base:], 0)
        initial[1 + n_base + ns:] = np.maximum(-coefficient[n_base:], 0)
    bounds = [(None, None)] * (1 + n_base) + [(0., None)] * (2 * ns)
    result = minimize(objective_gradient, initial, args=(X, y, n_base, alpha, lam),
                      method='L-BFGS-B', jac=True, bounds=bounds, options=OPTIONS)
    base = result.x[1:1 + n_base]
    coef = np.r_[base, result.x[1 + n_base:1 + n_base + ns] - result.x[1 + n_base + ns:]]
    intercept = float(result.x[0])
    residual = kkt_residual(X, y, n_base, intercept, coef, alpha, lam)
    initial_residual, polish_iterations = residual, 0
    if np.isfinite(coef).all() and np.isfinite(intercept) and residual > KKT_TOLERANCE:
        intercept, coef, polish_iterations, residual = proximal_polish(X, y, n_base, intercept, coef, alpha, lam)
    if not np.isfinite(coef).all() or not np.isfinite(intercept) or residual > KKT_TOLERANCE:
        raise RuntimeError(f'Block EN numerical convergence failure:status={result.status},'
                           f' KKT={residual:.3g}, iterations={result.nit}; {result.message}')
    eta = intercept + X @ coef
    objective = np.mean(np.logaddexp(0., eta) - y * eta) + lam / len(y) * (
        .5 * (coef[:n_base] @ coef[:n_base]) + .5 * (1 - alpha) * (coef[n_base:] @ coef[n_base:])
        + alpha * np.abs(coef[n_base:]).sum())
    return dict(intercept=intercept, coef=coef.tolist(), alpha=float(alpha),
                lambda_sum_loss=float(lam), lambda_mean_loss=float(lam / len(y)),
                selected_static=int(np.count_nonzero(coef[n_base:])),
                iterations=int(result.nit) + polish_iterations, objective=float(objective), KKT_residual=residual,
                lbfgsb_success=bool(result.success), lbfgsb_initial_KKT=initial_residual,
                proximal_polish_iterations=polish_iterations,
                converged=True)


def predict(result, X):
    return expit(result['intercept'] + np.asarray(X) @ np.asarray(result['coef']))


def verification():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(20261122)
    X = StandardScaler().fit_transform(rng.normal(size=(100, 8)))
    y = (rng.random(100) < expit(.2 + X[:, 0] - .7 * X[:, 2])).astype(int)
    theta = rng.normal(size=1 + 2 + 2 * 6)
    theta[3:] = np.abs(theta[3:])
    error = check_grad(lambda b: objective_gradient(b, X, y, 2, .5, 2.)[0],
                       lambda b: objective_gradient(b, X, y, 2, .5, 2.)[1], theta)
    assert error < 1e-5
    with threadpool_limits(limits=1):
        no_static = fit(X, y, 2, .9, 1000.)
        assert no_static['selected_static'] == 0
        ridge = LogisticRegression(C=.001, penalty='l2', solver='lbfgs', tol=1e-11, max_iter=10000).fit(X[:, :2], y)
        np.testing.assert_allclose(predict(no_static, X), ridge.predict_proba(X[:, :2])[:, 1], atol=2e-6, rtol=2e-6)
        full_en = fit(X, y, 0, .5, 2.)
        reference = LogisticRegression(C=.5, penalty='elasticnet', l1_ratio=.5, solver='saga',
                                       tol=1e-11, max_iter=100000, random_state=20261122).fit(X, y)
        np.testing.assert_allclose(predict(full_en, X), reference.predict_proba(X)[:, 1], atol=3e-6, rtol=3e-6)
        polished_intercept, polished_coef, polish_steps, polish_kkt = proximal_polish(
            X, y, 0, 0., np.zeros(8), .5, 2.)
        np.testing.assert_allclose(expit(polished_intercept + X @ polished_coef),
                                   reference.predict_proba(X)[:, 1], atol=1e-4, rtol=1e-4)
        # Correlated p>n synthetic design exercises the weak-penalty numerical case.
        latent = rng.normal(size=(100, 10))
        correlated = StandardScaler().fit_transform(latent @ rng.normal(size=(10, 107))
                                                     + .03 * rng.normal(size=(100, 107)))
        cy = (rng.random(100) < expit(.2 + latent[:, 0] - latent[:, 1])).astype(int)
        weak = fit(correlated, cy, 6, .1, .01)
        assert weak['KKT_residual'] <= KKT_TOLERANCE
        # Held-out values do not enter the fit; coefficients are unchanged by predict.
        saved = np.asarray(full_en['coef']).copy()
        predict(full_en, X + 1e6)
        np.testing.assert_array_equal(saved, full_en['coef'])
    return dict(analytic_gradient_finite_difference_error=float(error),
        zero_static_solution_allowed_and_matches_Ridge=True,
        all_static_special_case_matches_sklearn_ElasticNet_on_synthetic_data=True,
        proximal_polishing_matches_sklearn_reference=True,
        weak_penalty_correlated_p_greater_n_KKT=float(weak['KKT_residual']),
        L1_applied_only_to_static=True, fixed_base_features_receive_L2=True,
        test_prediction_does_not_refit=True, solver_KKT_tolerance=KKT_TOLERANCE)
