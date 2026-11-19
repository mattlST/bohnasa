"""Unit tests for `mean_field_flow` after the solve_ivp switch (issue #24).

The adaptive DOP853 integration must reproduce the fixed-step RK4 reference
it replaced on the anchor points (saddle egress and stationarity) to well
below 1e-6, keep the documented input semantics (unsorted and duplicate
sample times, flow starting at t = 0) and raise on malformed input. The
RK4 reference below is the replaced implementation, frozen here so the
agreement stays testable.

All bands come from measurements on the CI matrix stacks (py3.10/scipy1.15.3
to py3.14/scipy1.18.1, numpy 2.2-2.5): the anchor-point RK4-vs-DOP853 max
difference was 9.723121863203232e-09, bit-identical on every stack tested,
and +-1 ulp jitter on the trigonometric right-hand side (a backend libm
surrogate) perturbs it by less than 1e-12.
"""

import numpy as np
import pytest

from bohnasa import classical_mean_field as mf

THETA = 1.35  # unstable-saddle regime, same anchor point as the physics suite


def _rk4_reference(
    theta: float, initial: tuple[float, float], times: np.ndarray, n_steps: int = 4000
) -> np.ndarray:
    """Fixed-step RK4 replaced by solve_ivp in issue #24 (verbatim)."""
    times = np.asarray(times, dtype=float)
    order = np.argsort(times)
    out = np.empty((len(times), 2))
    s, phi = float(initial[0]), float(initial[1])
    t_now = 0.0
    h = times.max() / n_steps if times.max() > 0 else 1.0
    targets = times[order]
    k_idx = 0
    for i in range(n_steps + 1):
        while k_idx < len(targets) and targets[k_idx] <= t_now + 1e-12:
            out[order[k_idx]] = (s, phi)
            k_idx += 1
        if k_idx >= len(targets) or i == n_steps:
            break
        k1 = mf._mf_fields(s, phi, theta)
        k2 = mf._mf_fields(s + 0.5 * h * k1[0], phi + 0.5 * h * k1[1], theta)
        k3 = mf._mf_fields(s + 0.5 * h * k2[0], phi + 0.5 * h * k2[1], theta)
        k4 = mf._mf_fields(s + h * k3[0], phi + h * k3[1], theta)
        s += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        phi += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        t_now = i * h + h
    while k_idx < len(targets):
        out[order[k_idx]] = (s, phi)
        k_idx += 1
    return out


def _saddle_egress_state() -> tuple[np.ndarray, np.ndarray]:
    """1e-9 unstable-eigenvector kick at (0, pi), as in the physics anchor."""
    jac = mf.dimer_jacobian(THETA, 0.0, np.pi)
    eigval, eigvec = np.linalg.eig(jac)
    udir = eigvec[:, np.argmax(eigval.real)]
    y0 = np.array([1e-9 * udir[0].real, np.pi + 1e-9 * udir[1].real])
    return y0, np.linspace(0.0, 10.0, 500)


def test_flow_matches_rk4_reference_on_saddle_egress_anchor():
    y0, times = _saddle_egress_state()
    new = mf.mean_field_flow(THETA, y0, times)
    ref = _rk4_reference(THETA, y0, times)
    # observed max difference 9.723121863203232e-09, bit-identical on all
    # tested stacks
    assert np.abs(new - ref).max() < 1e-6
    assert new.shape == (len(times), 2)


def test_flow_matches_rk4_reference_on_stationarity_anchor():
    # Same anchor and horizon as the module doctest (t <= 5). Past it the
    # two paths separate: the RK4 grid keeps its sin(pi) round-off drift on
    # a slowly drifting track (2.1e-15 from (0, pi) at t = 20), while DOP853
    # tolerance noise acquires an unstable-manifold projection and grows
    # with e^(lambda t) (3.3e-9 at t = 20 - still within 1e-6 of RK4, but
    # no longer within the 1e-12 stationarity claim asserted here).
    times = np.array([0.0, 1.0, 5.0])
    new = mf.mean_field_flow(THETA, (0.0, np.pi), times)
    ref = _rk4_reference(THETA, (0.0, np.pi), times)
    # observed: new-vs-reference max 9.552e-15, new-vs-(0, pi) max 9.016e-15
    assert np.abs(new - ref).max() < 1e-6
    assert np.abs(new - np.array([0.0, np.pi])).max() < 1e-12


def test_flow_accepts_unsorted_and_duplicate_times():
    y0, _ = _saddle_egress_state()
    sorted_times = np.array([0.0, 0.5, 3.5, 7.0, 12.0])
    shuffled = np.array([7.0, 0.0, 12.0, 3.5, 3.5, 0.5])  # unsorted, 3.5 twice
    base = mf.mean_field_flow(THETA, y0, sorted_times)
    out = mf.mean_field_flow(THETA, y0, shuffled)
    assert out.shape == (len(shuffled), 2)
    # same sample set and horizon -> the deduplicated integration follows
    # an identical path, so the permuted rows match bit for bit
    for k, t in enumerate(shuffled):
        assert np.array_equal(out[k], base[np.flatnonzero(sorted_times == t)[0]])


def test_flow_starts_at_time_zero():
    """Sample times at or below zero report the initial state (flow starts at
    t = 0), matching the fixed-step reference the solver replaced."""
    y0 = np.array([0.1, 2.0])
    out = mf.mean_field_flow(THETA, y0, np.array([-4.0, 0.0]))
    assert np.allclose(out, np.tile(y0, (2, 1)))


def test_flow_rejects_bad_times():
    y0 = (0.1, 2.0)
    with pytest.raises(ValueError, match="non-empty 1D"):
        mf.mean_field_flow(THETA, y0, np.array([]))
    with pytest.raises(ValueError, match="finite"):
        mf.mean_field_flow(THETA, y0, np.array([0.0, np.inf]))
    with pytest.raises(ValueError, match="non-empty 1D"):
        mf.mean_field_flow(THETA, y0, np.ones((2, 2)))


def test_n_steps_parameter_removed():
    """Regression pin (issue #24): the fixed-step knob is gone from the API."""
    y0, times = _saddle_egress_state()
    with pytest.raises(TypeError):
        mf.mean_field_flow(THETA, y0, times, n_steps=500)
    with pytest.raises(TypeError):
        mf.mean_field_flow(THETA, y0, times, 500)
