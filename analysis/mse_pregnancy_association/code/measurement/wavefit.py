"""Continuous travelling-plane fit for mode speed, without event counting.

Two counterpropagating components are fitted jointly. Per-position offsets and
linear trends are nuisance terms, projected out of both data and model. This
refines the speed measurement; the original directional-energy ER is separate.
"""
import numpy as np


def _minimize(objective, start, *, iterations=240):
    """Bounded Nelder-Mead in unit coordinates; small 2/4-parameter problem."""
    start = np.asarray(start, float)
    simplex = np.tile(start, (len(start)+1, 1))
    for j in range(len(start)):
        simplex[j+1, j] += .06 if start[j] <= .94 else -.06
    values = np.array([objective(p) for p in simplex])
    for _ in range(iterations):
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        if np.max(np.abs(simplex[1:]-simplex[0])) < 1e-6:
            break
        center = simplex[:-1].mean(0)
        reflected = np.clip(2*center-simplex[-1], 0, 1)
        reflected_value = objective(reflected)
        if reflected_value < values[0]:
            expanded = np.clip(center+2*(reflected-center), 0, 1)
            expanded_value = objective(expanded)
            simplex[-1], values[-1] = ((expanded, expanded_value)
                if expanded_value < reflected_value else (reflected, reflected_value))
        elif reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
        else:
            outside = reflected_value < values[-1]
            contracted = np.clip(center+.5*((reflected if outside else simplex[-1])-center), 0, 1)
            contracted_value = objective(contracted)
            if contracted_value < (reflected_value if outside else values[-1]):
                simplex[-1], values[-1] = contracted, contracted_value
            else:
                simplex[1:] = (simplex[1:]+simplex[0])/2
                values[1:] = [objective(p) for p in simplex[1:]]
    best = int(np.argmin(values))
    converged = bool(np.max(np.abs(simplex-simplex[best])) < 1e-6)
    return simplex[best], float(values[best]), converged


def fit_wave_modes(y, time_s, space, band_cpm=(.5, 5.)):
    """Measure signed mode speeds using actual coordinates, not FFT grid bins.

    Inputs are already validated complete rectangular blocks by spectrum.py.
    Fit quality and parameter boundaries are diagnostics, not biological gates.
    A spatially uniform component has no resolved propagation speed. The only
    energy cutoff is numerical; this routine never declares an event present.
    """
    y, time, space = np.asarray(y, float), np.asarray(time_s, float), np.asarray(space, float)
    t = time-time.mean()
    x = space-space.mean()
    span = float(np.ptp(space))
    trend = t/np.linalg.norm(t)
    gaps = np.diff(space)
    weights = np.r_[gaps[0], (gaps[:-1]+gaps[1:])/2, gaps[-1]]
    weights = np.sqrt(weights/weights.mean())

    def project(values):
        values = values-values.mean(0)
        return (values-trend[:, None]*(trend@values)[None, :])*weights

    data = project(y).ravel()
    energy = float(data@data)
    floor = max(1e-25, float(np.sum((y-y.mean(0))**2))*1e-12)
    if energy <= floor:
        return dict(status='no_residual_motion_after_linear_trend', residual_fraction=None,
                    components={name: dict(velocity=None, temporal_frequency_hz=None,
                                           spatial_frequency=None, status='no_resolved_motion')
                                for name in ('C2F', 'F2C')})
    low, high = np.asarray(band_cpm)/60
    nyquist = .5/np.mean(gaps)

    def parameters(unit, sign):
        return low+unit[0]*(high-low), sign*unit[1]*nyquist

    def basis(unit, sign):
        frequency, wavenumber = parameters(unit, sign)
        phase = 2*np.pi*(frequency*t[:, None]+wavenumber*x[None, :])
        return np.column_stack((project(np.cos(phase)).ravel(), project(np.sin(phase)).ravel()))

    def solve(unit, signs, details=False):
        matrix = np.column_stack([basis(unit[2*j:2*j+2], sign) for j, sign in enumerate(signs)])
        coefficients = np.linalg.lstsq(matrix, data, rcond=np.sqrt(np.finfo(float).eps))[0]
        residual = data-matrix@coefficients
        loss = float(residual@residual/energy)
        return (loss, matrix, coefficients) if details else loss

    starts = []
    for sign in (-1, 1):
        objective = lambda unit: solve(unit, [sign])
        candidates = [(objective([f, k]), [f, k])
                      for f in np.linspace(0, 1, 7)
                      for k in np.linspace(0, 1, 2*len(space)+1)]
        initial = min(candidates, key=lambda item: item[0])[1]
        fitted, _, _ = _minimize(objective, initial, iterations=160)
        starts.extend(fitted)
    fitted, _, converged = _minimize(lambda unit: solve(unit, [-1, 1]), starts)
    loss, matrix, coefficients = solve(fitted, [-1, 1], details=True)
    singular = np.linalg.svd(matrix, compute_uv=False)
    separable = bool(singular[-1] > singular[0]*np.sqrt(np.finfo(float).eps))
    components = {}
    for j, (name, sign) in enumerate([('C2F', -1), ('F2C', 1)]):
        unit = fitted[2*j:2*j+2]
        frequency, wavenumber = parameters(unit, sign)
        component = matrix[:, 2*j:2*j+2]@coefficients[2*j:2*j+2]
        fraction = float(component@component/energy)
        uniform = abs(wavenumber)*span <= 1e-5
        present = fraction > 1e-10
        components[name] = dict(
            velocity=float(-frequency/wavenumber) if present and not uniform and separable else None,
            temporal_frequency_hz=float(frequency), spatial_frequency=float(wavenumber),
            component_energy_fraction=fraction if separable else None,
            temporal_boundary=bool(unit[0] < 1e-4 or unit[0] > 1-1e-4),
            spatial_boundary=bool(unit[1] < 1e-4 or unit[1] > 1-1e-4),
            status=('unresolved_counterpropagating_components' if not separable else
                    'no_resolved_component' if not present else
                    'spatially_uniform_component' if uniform else 'continuous_mode_candidate'))
    return dict(status='continuous_two_direction_fit', residual_fraction=loss,
                optimizer_converged=converged,
                component_condition_number=float(singular[0]/singular[-1]) if singular[-1] > 0 else None,
                nuisance='per_position_offset_and_linear_trend', components=components)
