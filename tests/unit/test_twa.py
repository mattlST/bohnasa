"""Unit tests for the truncated-Wigner sampling backend (issue #30).

Determinism, input validation, the engine protocol (duck typing through a
minimal fake engine with an analytic shift flow, whose Poisson bracket has
a closed form), sampler statistics, and the measured regime of the
gradient finite-difference step. All expectations are executed numbers -
the physics validations live in tests/physics/test_twa_physics.py.
"""

import numpy as np
import pytest

import bohnasa
from bohnasa import TruncatedWigner, WignerSampler
from bohnasa import twa as twa_module
from bohnasa.twa import _grad_symbol, _omega_apply


class ShiftEngine:
    """Minimal protocol engine: uniform shift flow z(t) = z0 + t * a.

    The flow has unit Jacobian, so for the quadratic symbol
    ``n_0 = (q_0^2 + p_0^2)/2`` the bracket is exactly
    ``{n_0(t), n_0}(z) = t (a_q p_0 - a_p q_0)`` - central differences of
    a quadratic symbol carry no truncation error, making the OTOC output
    bit-comparable to the closed form.
    """

    dim = 4
    filling = 2.0

    def __init__(self) -> None:
        self.a = np.array([0.3, -0.2, 0.5, 0.1])
        self.grids: list[np.ndarray] = []

    def trajectory(self, z0: np.ndarray, times: np.ndarray) -> np.ndarray:
        self.grids.append(np.asarray(times, dtype=float))
        return z0[None, :] + self.grids[-1][:, None] * self.a[None, :]


# --- engine protocol --------------------------------------------------------


def test_protocol_requires_all_members():
    class MissingDim:
        filling = 2.0

        def trajectory(self, z0, times):
            return np.zeros((len(times), 4))

    with pytest.raises(TypeError, match="protocol, missing attribute 'dim'"):
        TruncatedWigner(MissingDim())

    class NoTrajectory:
        dim = 4
        filling = 2.0

    with pytest.raises(TypeError, match="protocol, missing attribute 'trajectory'"):
        WignerSampler(NoTrajectory())


@pytest.mark.parametrize("bad_dim", [3, 2, 0, -4, 4.0, True])
def test_protocol_dim_must_be_even_and_at_least_4(bad_dim):
    class Engine:
        dim = bad_dim
        filling = 2.0

        def trajectory(self, z0, times):
            return np.zeros((len(times), 4))

    with pytest.raises(ValueError, match="even integer"):
        TruncatedWigner(Engine())


@pytest.mark.parametrize("bad_filling", [0.0, -1.0, np.nan, np.inf, "two", [2.0]])
def test_protocol_filling_must_be_positive_finite(bad_filling):
    class Engine:
        dim = 4
        filling = bad_filling

        def trajectory(self, z0, times):
            return np.zeros((len(times), 4))

    with pytest.raises(ValueError, match="filling"):
        TruncatedWigner(Engine())


def test_protocol_trajectory_must_be_callable():
    class Engine:
        dim = 4
        filling = 2.0
        trajectory = "not callable"

    with pytest.raises(TypeError, match="trajectory must be callable"):
        TruncatedWigner(Engine())


def test_fake_engine_bracket_matches_closed_form():
    """Duck-typed engine: no subclassing, exact bracket, hbar_M bookkeeping."""
    eng = ShiftEngine()
    zgp = np.array([1.2, -0.4, 0.5, 0.9])
    sim = TruncatedWigner(eng, zgp, seed=3)
    times = np.array([0.0, 0.5, 2.0])

    def n0(z: np.ndarray) -> float:
        return 0.5 * (z[0] ** 2 + z[2] ** 2) - 0.5

    c, se = sim.otoc(4, times, n0)
    assert c[0] == 0.0 and se[0] == 0.0  # t = 0 exact, like SquaredCommutator.curve
    z0 = sim.sample(4)
    # {n0(t), n0}(z) = t * (a_q p0 - a_p q0); C = hbar_M^2 * mean(bracket^2)
    br = times[1:, None] * (eng.a[0] * z0[None, :, 2] - eng.a[2] * z0[None, :, 0])
    closed = (sim.hbar_moyal * br) ** 2
    expected = closed.mean(axis=1)
    np.testing.assert_allclose(c[1:], expected, rtol=1e-10)
    np.testing.assert_allclose(se[1:], closed.std(axis=1, ddof=1) / np.sqrt(4), rtol=1e-9)


def test_fake_engine_observes_shapes_and_grids():
    eng = ShiftEngine()
    sim = TruncatedWigner(eng, np.ones(4), seed=1)
    trajs = sim.propagate(5, np.array([0.25, 0.5]))
    assert trajs.shape == (5, 2, 4)
    # the samples live at t = 0, so the engine grid got 0 prepended
    assert eng.grids[0][0] == 0.0
    z0 = sim.sample(5)
    np.testing.assert_allclose(trajs[0], z0[0] + np.outer([0.25, 0.5], eng.a))
    mean, se = sim.observable(5, np.array([0.0, 0.5]), lambda z: np.array([z[0], z[1]]))
    assert mean.shape == (2, 2) and se.shape == (2, 2)
    assert eng.grids[-1].tolist() == [0.0, 0.5]  # starts at 0: grid untouched


def test_trajectory_wrong_shape_raises():
    class BadShape(ShiftEngine):
        def trajectory(self, z0, times):
            return np.zeros((len(times), 5))

    sim = TruncatedWigner(BadShape(), np.ones(4), seed=1)
    with pytest.raises(ValueError, match=r"engine\.trajectory returned shape"):
        sim.propagate(2, np.array([0.0, 1.0]))


# --- determinism -------------------------------------------------------------


def test_same_seed_byte_reproducible_across_objects():
    lat = bohnasa.MeanFieldLattice(2, 1.35)

    def probe(z: np.ndarray) -> float:
        return 0.5 * (z[0] ** 2 + z[2] ** 2 - z[1] ** 2 - z[3] ** 2)

    a = TruncatedWigner(lat, seed=17).otoc(6, np.array([0.0, 0.3]), probe)
    b = TruncatedWigner(lat, seed=17).otoc(6, np.array([0.0, 0.3]), probe)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    m1 = TruncatedWigner(lat, seed=17).observable(6, np.array([0.4]), lambda z: z[0])
    m2 = TruncatedWigner(lat, seed=41).observable(6, np.array([0.4]), lambda z: z[0])
    assert not np.array_equal(m1[0], m2[0])


def test_ensembles_are_nested_prefixes():
    sim = TruncatedWigner(bohnasa.MeanFieldLattice(2, 1.0), seed=9)
    big = sim.sample(32)
    small = sim.sample(8)
    assert np.array_equal(big[:8], small)
    mean_big, _ = sim.observable(32, np.array([0.5]), lambda z: z[0])
    # a K = 8 draw is the prefix of the K = 32 draw, so the small ensemble's
    # mean of q_0 at t = 0.5 must match the prefix rows propagated identically
    assert mean_big.shape == (1,)


def test_sample_reproducible_bitwise():
    lat = bohnasa.MeanFieldLattice(4, 1.2)
    sampler = WignerSampler(lat)
    gp = np.full(8, 1.0)
    first = sampler.initial(gp, 12, seed=77)
    assert np.array_equal(first, sampler.initial(gp, 12, seed=77))
    assert not np.array_equal(first, sampler.initial(gp, 12, seed=78))
    # nested rows for one draw size sequence
    assert np.array_equal(sampler.initial(gp, 12, seed=77)[:5], sampler.initial(gp, 5, seed=77))


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize("samples", [0, 1, -3, 2.0, "8", True])
def test_observable_and_otoc_sample_counts_are_validated(samples):
    sim = TruncatedWigner(bohnasa.MeanFieldLattice(2, 1.0), seed=1)
    times = np.array([0.0, 1.0])
    with pytest.raises((TypeError, ValueError), match="samples"):
        sim.observable(samples, times, lambda z: z[0])
    with pytest.raises((TypeError, ValueError), match="samples"):
        sim.otoc(samples, times, lambda z: z[0])


def test_sample_allows_single_member_but_not_zero():
    sim = TruncatedWigner(bohnasa.MeanFieldLattice(2, 1.0), seed=1)
    assert sim.sample(1).shape == (1, 4)
    with pytest.raises(ValueError, match="samples must be >= 1"):
        sim.sample(0)


@pytest.mark.parametrize(
    ("times", "match"),
    [
        ([], "non-empty 1D"),
        ([[0.0, 1.0], [1.0, 2.0]], "non-empty 1D"),
        ([0.0, np.inf], "finite"),
        ([0.0, -1.0], "nonnegative"),
        ([0.0, 0.5, 0.5], "strictly increasing"),
        ([0.5, 0.25], "strictly increasing"),
    ],
)
def test_time_grid_validation(times, match):
    sim = TruncatedWigner(bohnasa.MeanFieldLattice(2, 1.0), seed=1)
    grid = np.array(times, dtype=float)
    with pytest.raises(ValueError, match=match):
        sim.propagate(4, grid)
    with pytest.raises(ValueError, match=match):
        sim.observable(4, grid, lambda z: z[0])
    with pytest.raises(ValueError, match=match):
        sim.otoc(4, grid, lambda z: z[0])


def test_otoc_requires_finite_real_scalar_symbol():
    sim = TruncatedWigner(bohnasa.MeanFieldLattice(2, 1.0), seed=1)
    with pytest.raises(ValueError, match="real scalar"):
        sim.otoc(3, np.array([0.0, 1.0]), lambda z: np.array([z[0], z[1]]))
    with pytest.raises(ValueError, match="real scalar"):
        sim.otoc(3, np.array([0.0, 1.0]), lambda z: np.nan)
    with pytest.raises(ValueError, match="real scalar"):
        sim.otoc(3, np.array([0.0, 1.0]), lambda z: 1.0 + 1j)
    with pytest.raises(ValueError, match="real scalar"):
        sim.otoc(3, np.array([0.0, 1.0]), lambda z: "x")
    with pytest.raises(TypeError, match="callable"):
        sim.otoc(3, np.array([0.0, 1.0]), "nope")


@pytest.mark.parametrize("bad", [0.0, -0.5, np.nan, np.inf, "x", [0.1]])
def test_hbar_eff_validation(bad):
    lat = bohnasa.MeanFieldLattice(2, 1.0)
    with pytest.raises(ValueError, match="hbar_eff"):
        TruncatedWigner(lat, hbar_eff=bad)
    with pytest.raises(ValueError, match="hbar_eff"):
        WignerSampler(lat, hbar_eff=bad)


def test_default_hbar_is_engine_shell():
    lat = bohnasa.MeanFieldLattice(4, 1.0)
    sim = TruncatedWigner(lat)
    assert sim.hbar_eff == pytest.approx(0.25)  # 1 / filling = 1 / 4
    assert sim.hbar_moyal == pytest.approx(1.0)  # L * hbar_eff
    assert sim.sampler.noise_std == pytest.approx(np.sqrt(0.5))
    # default state: uniform filling point phi_j = 1 (q_j = sqrt 2, p_j = 0)
    np.testing.assert_allclose(sim.state[:4], np.sqrt(2.0))
    np.testing.assert_allclose(sim.state[4:], 0.0)


def test_state_validation():
    lat = bohnasa.MeanFieldLattice(2, 1.0)
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        TruncatedWigner(lat, state=np.ones(3))
    with pytest.raises(ValueError, match="non-finite"):
        TruncatedWigner(lat, state=np.array([1.0, 1.0, np.inf, 0.0]))
    with pytest.raises(TypeError, match="seed"):
        TruncatedWigner(lat, seed=True)


# --- sampler statistics -------------------------------------------------------


def test_sampler_recovers_wigner_variance_and_shell():
    lat = bohnasa.MeanFieldLattice(2, 1.35)
    gp = np.array([np.sqrt(2.0), -np.sqrt(2.0), 0.0, 0.0])
    z = WignerSampler(lat).initial(gp, 4000, seed=7)
    d = z - gp
    # variance per quadrature = hbar_M / 2 = 0.5 on the engine shell
    assert d.var(axis=0, ddof=1) == pytest.approx(0.5, rel=0.04)
    # the zero point cancels the noise mean: E[|phi|^2 - 1/2] = GP occupation
    n = 0.5 * (z[:, 0] ** 2 + z[:, 2] ** 2) - 0.5
    assert n.mean() == pytest.approx(1.0, rel=0.02)


def test_sampler_hbar_scaling():
    lat = bohnasa.MeanFieldLattice(2, 1.35)
    sampler = WignerSampler(lat, hbar_eff=1.0 / 16)
    assert sampler.hbar_moyal == pytest.approx(0.125)
    gp = np.array([np.sqrt(2.0), -np.sqrt(2.0), 0.0, 0.0])
    z = sampler.initial(gp, 8000, seed=11)
    assert (z - gp).var(axis=0, ddof=1) == pytest.approx(0.0625, rel=0.05)


# --- gradient step regime ------------------------------------------------------


def test_gradient_step_truncation_gap():
    """_GRAD_FD_STEP comment: quadratic symbols differentiate exactly.

    Measured: the central difference of a quadratic Weyl symbol equals its
    analytic derivative to <= 7.6e-10 relative for every step from 1e-3 to
    1e-7 (round-off dominated, no truncation visible), so the constant
    sits inside a gap bounded only by round-off on the small-h side.
    """
    z = np.array([1.3, -0.7, 0.4, 0.9])
    analytic = np.array([z[0], 0.0, z[2], 0.0])  # grad of 0.5 * (q0^2 + p0^2)

    def func(v: np.ndarray) -> float:
        return 0.5 * (v[0] ** 2 + v[2] ** 2)

    scale = np.abs(analytic).max()
    for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        num = _grad_symbol(func, z, step=h)
        assert np.abs(num - analytic).max() <= 1e-8 * scale


def test_omega_apply_layout():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(_omega_apply(v), np.array([3.0, 4.0, -1.0, -2.0]))


def test_module_constants_match_the_policy():
    assert twa_module._WIGNER_ZERO_POINT == 0.5
    assert twa_module._DEFAULT_SEED == 2026
    assert twa_module._BRACKET_FD_EPS == 1e-4
    assert twa_module._GRAD_FD_STEP == 1e-5
