"""Unit tests for `mean_field_flow` after the solve_ivp switch (issue #24).

The adaptive DOP853 must reproduce the fixed-step RK4 reference it replaced
on both anchors well below 1e-6, keep the documented input semantics
(unsorted/duplicate times, flow starts at t = 0), honor the caller-
adjustable rtol/atol, and raise on malformed input. The reference is the
verbatim replaced implementation, NOT scipy's RK45: the acceptance compares
against *the scheme that was removed*, and RK45 is a different adaptive
method that would pin nothing about that regression. Bands are CI-matrix
measurements (py3.10/scipy 1.15.3 to py3.14/scipy 1.18.1): anchor
difference 9.723121863203232e-09, bit-identical on every tested stack.
"""

import numpy as np
import pytest

from bohnasa.mean_field import dimer as mf

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
    eigval, eigvec = np.linalg.eig(mf.dimer_jacobian(THETA, 0.0, np.pi))
    udir = eigvec[:, np.argmax(eigval.real)]
    return np.array([1e-9 * udir[0].real, np.pi + 1e-9 * udir[1].real]), np.linspace(0.0, 10.0, 500)


@pytest.mark.parametrize(
    ("initial", "times", "band"),
    [
        # observed max diff vs reference: 9.723121863203232e-09
        pytest.param(*_saddle_egress_state(), 1e-6, id="saddle-egress"),
        # stationarity horizon t <= 5 (module doctest); past it, RK4's
        # sin(pi) round-off and DOP853 tolerance noise separate as e^lambda t
        # (diffs 9.552e-15 on-grid; 3.2988e-09 vs 2.1456e-15 at t = 20)
        pytest.param((0.0, np.pi), np.array([0.0, 1.0, 5.0]), 1e-6, id="stationarity"),
    ],
)
def test_flow_matches_rk4_reference(initial, times, band):
    new = mf.mean_field_flow(THETA, initial, times)
    assert new.shape == (len(times), 2)
    assert np.abs(new - _rk4_reference(THETA, initial, times)).max() < band
    if initial[1] == np.pi and initial[0] == 0.0:  # stationarity anchor only
        assert np.abs(new - np.array([0.0, np.pi])).max() < 1e-12


def test_flow_sample_time_semantics():
    y0, _ = _saddle_egress_state()
    sorted_times = np.array([0.0, 0.5, 3.5, 7.0, 12.0])
    # unsorted, 3.5 twice: same sample set and horizon, so the deduplicated
    # integration follows an identical path and permuted rows match bitwise
    shuffled = np.array([7.0, 0.0, 12.0, 3.5, 3.5, 0.5])
    base = mf.mean_field_flow(THETA, y0, sorted_times)
    out = mf.mean_field_flow(THETA, y0, shuffled)
    for k, t in enumerate(shuffled):
        assert np.array_equal(out[k], base[np.flatnonzero(sorted_times == t)[0]])
    # decreasing is just unsorted: rows follow the same permutation
    assert np.array_equal(mf.mean_field_flow(THETA, y0, sorted_times[::-1]), base[::-1])
    # times at or below the start report the initial state (start is
    # max(0, min(times)): zero here)
    assert np.allclose(
        mf.mean_field_flow(THETA, [0.1, 2.0], np.array([-4.0, 0.0])), [[0.1, 2.0], [0.1, 2.0]]
    )
    # a positive start anchors the initial pair at the first sample
    positive = mf.mean_field_flow(THETA, [0.1, 2.0], np.array([2.0, 3.0]))
    assert np.array_equal(positive[0], [0.1, 2.0])
    assert not np.array_equal(positive[1], [0.1, 2.0])


def test_tolerances_are_honored():
    """Loosening rtol/atol trades accuracy monotonically (issue #24 review).

    Measured on the bounded orbit below, deviations from the default-tight
    solution over t in [0, 20], x10-x100 per tolerance decade (no chaos,
    so the ordering is stable across the CI scipy matrix).
    """
    times = np.linspace(0.0, 20.0, 201)
    tight = mf.mean_field_flow(THETA, (0.1, 1.0), times)
    devs = [
        np.abs(mf.mean_field_flow(THETA, (0.1, 1.0), times, rtol=rt, atol=rt / 100.0) - tight).max()
        for rt in (1e-8, 1e-6, 1e-4, 1e-3)
    ]
    assert all(d > 0.0 for d in devs)  # observed 4.22e-09 ... 4.78e-03
    assert all(a < b for a, b in zip(devs, devs[1:], strict=False))
    assert devs[0] < 1e-7 < devs[-1]  # observed margins, orders away


def test_integration_methods_agree_with_reference():
    """Keyword-only method selection (issue #24 review round 2).

    Measured max deviation from the DOP853 default at the same tolerances:
    RK23 6.23e-09, RK45 1.61e-09, Radau 1.20e-10, BDF 4.60e-08, LSODA
    6.08e-09 on the bounded orbit; <= 2.39e-08 on the exact saddle point
    over t in [0, 20].
    """
    times = np.linspace(0.0, 20.0, 201)
    for probe in ((0.1, 1.0), (0.0, np.pi)):
        ref = mf.mean_field_flow(THETA, probe, times)
        for method in mf._INTEGRATION_METHODS[1:]:
            other = mf.mean_field_flow(THETA, probe, times, method=method)
            assert other.shape == ref.shape
            assert np.abs(other - ref).max() < 1e-6


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        pytest.param((np.array([]),), {}, "non-empty 1D", id="empty-times"),
        pytest.param((np.ones((2, 2)),), {}, "non-empty 1D", id="2d-times"),
        pytest.param((np.array([0.0, np.inf]),), {}, "finite", id="inf-times"),
        pytest.param((np.array([0.0, 1.0]),), {"rtol": 0.0}, "positive", id="rtol-0"),
        pytest.param((np.array([0.0, 1.0]),), {"atol": -1.0}, "positive", id="atol-neg"),
        pytest.param((np.array([0.0, 1.0]),), {"rtol": np.nan}, "positive", id="rtol-nan"),
        pytest.param(
            (np.array([0.0, 1.0]),), {"method": "verlet"}, "method must be", id="bad-method"
        ),
    ],
)
def test_flow_rejects_bad_input(args, kwargs, match):
    with pytest.raises(ValueError, match=match):
        mf.mean_field_flow(THETA, (0.1, 2.0), *args, **kwargs)


def test_n_steps_parameter_removed():
    """Regression pin (issue #24): the fixed-step knob is gone from the API."""
    y0, times = _saddle_egress_state()
    with pytest.raises(TypeError):
        mf.mean_field_flow(THETA, y0, times, n_steps=500)
    with pytest.raises(TypeError):
        mf.mean_field_flow(THETA, y0, times, 500)
