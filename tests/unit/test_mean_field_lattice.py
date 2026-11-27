"""Unit tests for MeanFieldLattice input validation and structure (issue #26)."""

import numpy as np
import pytest

import bohnasa
from bohnasa import MeanFieldLattice


@pytest.fixture
def lat2():
    return MeanFieldLattice(2, 1.35)


@pytest.fixture
def lat4():
    return MeanFieldLattice(4, 1.35)


def test_constructor_validation():
    with pytest.raises(TypeError):
        MeanFieldLattice(2.5, 1.0)
    with pytest.raises(ValueError):
        MeanFieldLattice(1, 1.0)
    with pytest.raises(ValueError):
        MeanFieldLattice(np.int64(3), np.nan)
    with pytest.raises(ValueError):
        MeanFieldLattice(3, 1.0, mu=np.inf)
    with pytest.raises(ValueError):
        MeanFieldLattice(3, 1.0, connections="open")
    with pytest.raises((TypeError, ValueError)):
        MeanFieldLattice(3, 1.0, mu=1.0j)


def test_properties(lat2):
    assert lat2.L == 2
    assert lat2.theta == 1.35
    assert lat2.mu == 0.0
    assert lat2.connections == "pbc"
    assert lat2.dim == 4
    assert lat2.filling == 2.0


def test_state_validation(lat4):
    bad = [
        (np.zeros(7), "wrong length"),
        (np.zeros((4, 2)), "wrong shape"),
        (np.ones(8) * np.nan, "non-finite"),
    ]
    for arr, _ in bad:
        with pytest.raises(ValueError):
            lat4.flow(arr)
        with pytest.raises(ValueError):
            lat4.hamiltonian(arr)
    with pytest.raises(ValueError):  # imaginary parts are rejected, not dropped
        lat4.flow(np.ones(8, dtype=complex) * (1 + 1e-3j))
    lat4.flow(np.ones(8, dtype=complex) * (1 + 1e-15j))  # negligible imag accepted


def test_times_validation(lat2):
    with pytest.raises(ValueError):
        lat2.trajectory([1.0, 1.0, 0.0, 0.0], np.array([]))
    with pytest.raises(ValueError):
        lat2.trajectory([1.0, 1.0, 0.0, 0.0], np.ones((2, 2)))
    with pytest.raises(ValueError):
        lat2.trajectory([1.0, 1.0, 0.0, 0.0], np.array([0.0, np.inf]))
    single = lat2.trajectory([1.4, 0.6, 0.1, 0.2], np.array([3.0]))
    assert single.shape == (1, 4)
    assert np.allclose(single[0], [1.4, 0.6, 0.1, 0.2])


def test_trajectory_sample_time_semantics(lat2):
    """Issue #45: unsorted/duplicate times are accepted family-wide."""
    start = np.array([1.4, 0.6, 0.1, 0.2])
    sorted_times = np.array([0.0, 0.5, 1.5, 2.5])
    shuffled = np.array([2.5, 0.0, 1.5, 0.5, 1.5])
    base = lat2.trajectory(start, sorted_times)
    out = lat2.trajectory(start, shuffled)
    for k, t in enumerate(shuffled):
        assert np.array_equal(out[k], base[np.flatnonzero(sorted_times == t)[0]])
    # decreasing is just unsorted: rows follow the same permutation
    assert np.array_equal(lat2.trajectory(start, sorted_times[::-1]), base[::-1])
    # times at or below the start report the initial state
    below = lat2.trajectory(start, np.array([-4.0, 0.0]))
    assert np.array_equal(below, np.array([start, start]))
    positive = lat2.trajectory(start, np.array([2.0, 3.0]))
    assert np.array_equal(positive[0], start)


def test_trajectory_shape_and_shift(lat4):
    times = np.linspace(2.0, 5.0, 13)
    traj = lat4.trajectory(np.ones(8), times)
    assert traj.shape == (13, 8)
    # starting at t0 != 0 is the same flow translated in time
    shift = lat4.trajectory(np.ones(8), times - 2.0)
    assert np.abs(traj - shift).max() < 1e-12


def test_trajectory_accepts_explicit_method(lat4):
    times = np.linspace(0.0, 2.0, 21)
    a = lat4.trajectory(np.ones(8), times, method="Radau")
    b = lat4.trajectory(np.ones(8), times, method="DOP853", rtol=1e-10, atol=1e-12)
    assert np.abs(a - b).max() < 1e-6  # same flow, different quadrature


def test_lyapunov_validation(lat4):
    z = np.random.default_rng(0).normal(size=8)
    times = np.linspace(0.0, 4.0, 9)
    with pytest.raises(ValueError):
        lat4.lyapunov(z, times, method="benettin ")  # typo, no trailing magic
    with pytest.raises(ValueError):
        lat4.lyapunov(z, np.linspace(0.0, 4.0, 1))
    with pytest.raises(ValueError):
        lat4.lyapunov(z, times, delta=0.0)
    with pytest.raises(ValueError):
        lat4.lyapunov(z, times, v0=np.ones(7))


def test_fixed_point_structure(lat4):
    pts = lat4.fixed_points()
    assert pts.ndim == 2 and pts.shape[1] == 8
    energies = [lat4.hamiltonian(z) for z in pts]
    assert energies == sorted(energies)  # sorted by energy
    for z in pts:
        assert lat4.particle_number(z) == pytest.approx(4.0, abs=1e-12)  # shell
        assert z[4] == 0.0  # gauge p_0 = 0
        assert z[0] > 0.0  # gauge q_0 > 0
        # fixed point of the reduced flow: F(z*) = gamma * (-p, q)
        generator = np.concatenate((-z[4:], z[:4]))
        gamma = lat4.rotation_frequency(z)
        assert np.abs(lat4.flow(z) - gamma * generator).max() < 1e-8
    with pytest.raises(ValueError):
        lat4.fixed_points(n_random=-1)


def test_fixed_points_deterministic(lat4):
    a = lat4.fixed_points()
    b = lat4.fixed_points()
    assert np.array_equal(a, b)


def test_open_chain_has_no_wraparound():
    lat = MeanFieldLattice(3, 1.0, connections="obc")
    # a kick on site 2 must not move the momenta of site 0 (no bond 2-0)
    z = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    dq = lat.flow(z)[:3]
    assert dq[0] == 0.0
    assert dq[2] == pytest.approx(np.sin(1.0) * 0.5)  # on-site term only
    # the closed ring in the same situation does couple site 0
    ring = MeanFieldLattice(3, 1.0)
    assert ring.flow(z)[0] == pytest.approx(-np.cos(1.0))


def test_pbc_two_site_bond_counts_twice(lat2):
    # roll-sum convention: dphi/dt carries the factor 2 from the double bond
    z = np.array([np.sqrt(2.0), np.sqrt(2.0), 0.0, 0.0])
    dp = lat2.flow(z)[2:]
    assert dp[0] == pytest.approx(np.sqrt(2.0) * (2 * np.cos(1.35) - np.sin(1.35)))
    assert dp[1] == dp[0]


def test_zero_state_rejects_rotation_frequency(lat2):
    with pytest.raises(ValueError):
        lat2.rotation_frequency(np.zeros(4))


def test_overflowing_state_raises(lat2):
    # the integrator's intermediate states pass _state: an overflowing
    # initial condition is caught as a non-finite RHS state
    times = np.linspace(0.0, 1.0, 5)
    huge = np.full(4, 1e200)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError):
            lat2.trajectory(huge, times)


def test_solver_failure_is_surfaced(lat2, monkeypatch):
    # all three failure branches (trajectory, Benettin segment, QR segment)
    # must raise RuntimeError when the solver reports no success
    class _Failed:
        success = False
        message = "stubbed failure"
        y = np.zeros((4, 1))

    # trajectory routes through mean_field.solver; the Lyapunov segments
    # keep their own direct solve_ivp call in the lattice module
    monkeypatch.setattr(bohnasa.mean_field.solver, "solve_ivp", lambda *a, **k: _Failed())
    monkeypatch.setattr(bohnasa.mean_field.lattice, "solve_ivp", lambda *a, **k: _Failed())
    times = np.linspace(0.0, 1.0, 3)
    start = np.array([np.sqrt(2.0), 0.4, 0.3, 0.2])
    with pytest.raises(RuntimeError):
        lat2.trajectory(start, times)
    with pytest.raises(RuntimeError):
        lat2.lyapunov(start, times)
    with pytest.raises(RuntimeError):
        lat2.lyapunov(start, times, method="qr")


def test_two_trajectory_collapse_raises(lat2):
    # delta below the float resolution of the separation: ln(0/0) guard
    times = np.linspace(0.0, 1.0, 3)
    with pytest.raises(RuntimeError):
        lat2.lyapunov(np.ones(4), times, method="two_trajectory", delta=1e-300)


def test_fixed_points_empty_search(lat2, monkeypatch):
    monkeypatch.setattr(lat2, "_fp_seeds", lambda n_random: [])
    assert lat2.fixed_points().shape == (0, 4)


def test_fixed_points_sparse_sign_seeds_for_large_l(monkeypatch):
    # exercise the L > log2(cap) branch where sign patterns are sampled
    monkeypatch.setattr(bohnasa.mean_field.lattice, "_MAX_EXACT_SEED_PATTERNS", 8)
    lat = MeanFieldLattice(4, 1.35)
    pts = lat.fixed_points(n_random=10)
    assert len(pts) >= 1
    for z in pts:
        assert lat.particle_number(z) == pytest.approx(4.0, abs=1e-12)
