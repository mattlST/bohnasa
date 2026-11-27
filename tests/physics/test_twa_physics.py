"""Physics validations of the truncated-Wigner backend (issue #30).

Validation (1) - TWA -> mean field for many samples, on a REGULAR
(below-pitchfork) orbit: the nested-prefix ensemble mean collapses onto
the deterministic large-sample average at the Monte-Carlo
1/sqrt(samples) rate, and the surviving gap to the mean-field orbit is
the O(hbar_eff) ensemble back-reaction. (A single chaotic trajectory is
NOT the mean-field orbit, so chaotic initial conditions are excluded
here by design.)

Validation (2) - TWA -> exact quantum at short time / small hbar_eff,
against dense-ED `SquaredCommutator` references: the measured relative
deviation is ~ hbar_eff as t -> 0 (the leading TWA truncation error of
the product rule, complementary to the exact hbar -> 0 bracket point
value pinned separately), grows over the intermediate window, and the
sqrt-C growth exponents track the quantum curve through the issue #34
crossover structure (2 lambda_s -> lambda_s) at tiny L, N.

All bands are measured margins (numpy 2.5.5 / scipy 1.18.1, 16-core
host, fixed seeds; sample counts sized so the file stays near two
minutes). Threading note (issue #30 acceptance): sampler draws and the
quantum dense-eigensolve references are bit-stable across
OMP_NUM_THREADS = 1 vs default; the Radau-propagated TWA values drift
by <= 2e-9 relative between thread regimes (measured on the anchor
curves), so the Monte-Carlo bands below sit >= 7 orders above any
threading effect and NO bit-level equality is claimed for propagated
values (only for the RNG streams, in tests/unit/test_twa.py). No
wall-clock values enter any assertion.
"""

import numpy as np
import pytest

import bohnasa
from bohnasa import MeanFieldLattice, StateObservables, TruncatedWigner
from bohnasa import otoc as oc
from bohnasa import twa as twa_module
from bohnasa.mean_field import dimer as mf
from bohnasa.twa import _grad_symbol, _omega_apply

THETA = 1.35  # unstable-saddle dimer regime (issue #24/#28 anchor point)
LAM = mf.dimer_lambda(THETA)

# saddle GP point phi = (1, -1) on the N = L engine shell
ZGP = np.array([np.sqrt(2.0), -np.sqrt(2.0), 0.0, 0.0])


def _dimer_quantum(N):
    """Fixed-N dimer matching the engine's quarter-circle convention."""
    fock = bohnasa.FixedNFockSpace(N, 2)
    c = np.cos(THETA)
    t = np.array([[0.0, -2.0 * c], [-2.0 * c, 0.0]])
    H = bohnasa.Hamiltonian(fock, one_body=t, onsite_u=np.sin(THETA) / N)
    op = (bohnasa.Number(fock, 0) - bohnasa.Number(fock, 1)) * 0.5
    psi = bohnasa.fixed_n_coherent_state(fock, np.array([1.0, -1.0]) / np.sqrt(2.0))
    return fock, H, op, psi


def _probe_symbol(N):
    """Dictionary (b) partner of the quantum probe (n0 - n1)/2, in engine
    variables (the engine's unit-filling gauge rescales it by N_phys / L)."""
    a = N / 2.0
    return lambda z: a * 0.25 * (z[0] ** 2 + z[2] ** 2 - z[1] ** 2 - z[3] ** 2)


# --- validation (2): short time / small hbar_eff ----------------------------


def test_twa_short_time_matches_quantum_with_hbar_eff_deviation():
    """C_TWA / C_quantum = 1 + hbar_eff + O(hbar_eff^2) as t -> 0.

    Measured (K = 1200, seed 11, t = 1e-3 and 5e-3): relative deviations
    0.1244 / 0.0602 / 0.0294 at N_phys = 8 / 16 / 32 - coefficients
    0.995 / 0.963 / 0.942 of hbar_eff = 1/N_phys, with the two probe
    times agreeing to <= 4e-5 (a genuine t -> 0 plateau). Independent-
    seed scan at N_phys = 8 (five seeds): coefficients 0.96-1.17 (std
    ~0.08), so expect +-0.1-class scatter at the smallest N - every
    draw lands well inside the pinned band. The leading
    correction is the dropped product-rule Moyal term averaged over the
    finite-width wave packet; the unsampled bracket is exact to leading
    order (next test).
    """
    lat = MeanFieldLattice(2, THETA)
    tiny = np.array([1e-3, 5e-3])
    rels = []
    for N in (8, 16, 32):
        _, H, op, psi = _dimer_quantum(N)
        cq = oc.SquaredCommutator(H, op).curve(psi, tiny)
        sim = TruncatedWigner(lat, ZGP, seed=11, hbar_eff=1.0 / N)
        ct, _ = sim.otoc(1200, tiny, _probe_symbol(N))
        rel = ct / cq - 1.0
        assert abs(rel[0] - rel[1]) < 4.0e-4  # t -> 0 plateau
        coef = rel[1] * N
        assert 0.6 < coef < 1.5  # measured 0.995 / 0.963 / 0.942
        rels.append(rel[1])
    assert rels[0] > rels[1] > rels[2]  # shrinks linearly with hbar_eff
    assert 0.5 * (1 / 8) < rels[0] < 1.5 * (1 / 8)


def test_twa_point_value_reproduces_quantum_beta2_exactly():
    """The hbar -> 0 bracket dictionary (c) is exact at leading order:
    hbar_M^2 K2^2 / beta2 - 1 measured to 1.34e-07 (N = 16) using the
    module's own two-stage central-difference gradient of the nested
    bracket K2 = {{O, H}, O} at the GP point. The +hbar_eff coefficient
    of the previous test therefore comes from averaging over the finite
    Wigner wave-packet width, not from the commutator map itself.
    """
    lat = MeanFieldLattice(2, THETA)
    N = 16
    _, H, op, psi = _dimer_quantum(N)
    hd, od = H.matrix.toarray(), op.matrix.toarray()
    ho = hd @ od - od @ hd
    v = ho @ (od @ psi) - od @ (ho @ psi)
    beta2 = np.vdot(v, v).real

    f = _probe_symbol(N)

    def nested(z):  # {O, H}(z)
        return float(_grad_symbol(f, z) @ lat.flow(z))

    k2 = _grad_symbol(nested, ZGP) @ _omega_apply(_grad_symbol(f, ZGP))  # {{O,H},O}
    point_coeff = (2.0 / N) ** 2 * k2**2  # hbar_M^2 K2^2, hbar_M = L / N
    assert abs(point_coeff / beta2 - 1.0) < 1e-5  # measured 1.34e-07


def test_twa_deviation_grows_over_the_window():
    """The truncation deviation rises across the intermediate window:
    measured relative deviations (K = 300, seed 13, N = 16) 0.0908 /
    0.0938 / 0.1098 / 0.1575 at t = 0.05 / 0.2 / 0.5 / 1.0 - from 1.45
    to 2.52 times hbar_eff, monotonically increasing (final/initial
    factor 1.74, band > 1.3). The quantum curve bends toward saturation
    at late times while the TWA ensemble keeps the classical growth;
    that departure is the t-dependent part of the deviation.
    """
    lat = MeanFieldLattice(2, THETA)
    N = 16
    times = np.array([0.05, 0.2, 0.5, 1.0])
    _, H, op, psi = _dimer_quantum(N)
    cq = oc.SquaredCommutator(H, op).curve(psi, times)
    sim = TruncatedWigner(lat, ZGP, seed=13, hbar_eff=1.0 / N)
    ct, _ = sim.otoc(300, times, _probe_symbol(N))
    rel = ct / cq - 1.0
    assert np.all(np.diff(rel) > 0)
    assert np.all(0.75 / N < rel) and np.all(rel < 4.5 / N)
    assert rel[-1] > 1.3 * rel[0]


def test_crossover_exponents_2lam_to_lam_twa_tracks_quantum():
    """The issue #34 statement, made testable at tiny L, N: the sqrt-C
    growth exponent runs 2 lambda_s -> lambda_s, and the TWA ensemble
    tracks the quantum curve on both sides.

    Measured (N = 8, K = 300, seed 17, grid 21 over [0, 2]): early
    window (0.3, 0.8) exponents quantum 1.9897 / TWA 2.0025 against
    2 lambda_s = 1.9412 (+2.5 % / +3.2 %); mid window (0.8, 1.5)
    quantum 0.9514 / TWA 0.9515 against lambda_s = 0.9706 (-2.0 % each).
    The TWA-vs-quantum
    agreement is pinned at 5 % per window (measured 0.64 % / 0.01 %), and
    each curve against the closed-form target at 10 % - the quantum side
    of this comparison is the same crossover the #28 anchors pin via
    d log(C)/dt = 2 lambda_s in tests/physics/test_otoc_physics.py; the
    single-trajectory classical point value is NOT asserted here (its
    windows carry a larger +20-26 % transient: documented in the module
    docstring as the classical side, measured in examples/ex14).
    """
    lat = MeanFieldLattice(2, THETA)
    N = 8
    times = np.linspace(0.0, 2.0, 21)
    _, H, op, psi = _dimer_quantum(N)
    cq = oc.SquaredCommutator(H, op).curve(psi, times)
    sim = TruncatedWigner(lat, ZGP, seed=17, hbar_eff=1.0 / N)
    ct, _ = sim.otoc(300, times, _probe_symbol(N))

    def sqrt_slope(curve, w0, w1):
        m = (times >= w0) & (times <= w1) & (curve > 0)
        return float(np.polyfit(times[m], np.log(curve[m]), 1)[0] / 2.0)

    for w0, w1, target in ((0.3, 0.8, 2 * LAM), (0.8, 1.5, LAM)):
        s_q = sqrt_slope(cq, w0, w1)
        s_t = sqrt_slope(ct, w0, w1)
        assert abs(s_q / target - 1.0) < 0.10  # measured +0.025 / -0.020
        assert abs(s_t / target - 1.0) < 0.10  # TWA tracks within the band
        assert abs(s_t / s_q - 1.0) < 0.05  # measured 0.0064 / 0.0001


# --- validation (1): many samples -> mean field (regular orbit) -------------


def test_sample_mean_converges_to_mean_field_at_mc_rate():
    """Validation (1) on the REGULAR dimer orbit (theta = 1.0, below the
    pitchfork tan(theta) = 2).

    Measured (N_phys = 16 gauge, seed 5, times 9 over [0, 2], reference
    ensemble K = 512): max |mean_K - mean_512| = 0.567 / 0.814 / 0.434 /
    0.271 / 0.208 at K = 16 ... 256 - bounded by 6 / sqrt(K) at every K
    (the K = 32 bump is Monte-Carlo scatter of the max statistic on this
    fixed stream, hence the envelope rather than monotonicity is pinned),
    log-log slope -0.45 (band < -0.35): the Monte-Carlo rate onto the
    deterministic ensemble average. The large-sample gap to the scaled
    mean-field orbit is 0.178 occupation units at K = 512, O(hbar_M) on
    the per-site scale - a verification-run ladder (K >= 2048, several
    seeds) puts the asymptotic floor nearer 0.10-0.13, with the K = 512
    figure carrying reference-ensemble noise. The O(hbar_eff) back-
    reaction reading is unchanged. The TWA -> mean-field
    statement is two-stage: samples -> ensemble average, and hbar_eff ->
    0 -> the mean-field orbit.
    """
    lat = MeanFieldLattice(2, 1.0)
    z_reg = np.array([np.sqrt(2.0 * 1.3), np.sqrt(2.0 * 0.7), 0.2, 0.2])
    z_reg *= 2.0 / np.linalg.norm(z_reg)  # engine shell, non-stationary IC
    times = np.linspace(0.0, 2.0, 9)
    N = 16
    scale = N / 2.0

    def phys_occ(z: np.ndarray) -> float:
        return scale * (0.5 * (z[0] ** 2 + z[2] ** 2)) - 0.5

    sim = TruncatedWigner(lat, z_reg, seed=5, hbar_eff=1.0 / N)
    ref, _ = sim.observable(512, times, phys_occ)
    ks = (16, 32, 64, 128, 256)
    devs = []
    for k in ks:
        mean_k, _ = sim.observable(k, times, phys_occ)
        devs.append(float(np.abs(mean_k - ref).max()))
    devs = np.array(devs)
    assert np.all(devs * np.sqrt(ks) < 6.0)  # measured max 4.61
    rate = float(np.polyfit(np.log(ks), np.log(devs), 1)[0])
    assert rate < -0.35  # measured -0.45
    assert devs[-1] < 0.5 * devs[2]  # measured 0.208 < 0.5 x 0.434
    gp = lat.trajectory(z_reg, times)
    gp_n = scale * (0.5 * gp[:, 0] ** 2 + 0.5 * gp[:, 2] ** 2)
    plateau = float(np.abs(ref - gp_n).max())
    assert 0.0 < plateau < 0.5  # measured 0.178 (occupation units at N/L=8)


def test_observable_matches_exact_quantum_within_standard_error():
    """Semiclassical occupations vs exact ED - the quantum reference
    evaluated through the fast StateObservables path (issue #30 asset
    note). Saddle orbit, N = 16, K = 400, seed 23: measured max |TWA -
    quantum| / stderr = 0.59 (band 3 sigma) at every site and time,
    deviations <= 0.11 occupation units while the standard error is
    0.140 - 0.201. The quantum occupations are stationary at the saddle
    (also pinned) so any TWA motion is sampling + truncation, not
    physics.
    """
    lat = MeanFieldLattice(2, THETA)
    N = 16
    fock, H, op, psi = _dimer_quantum(N)
    obs = StateObservables(fock)
    times = np.linspace(0.0, 1.0, 6)
    eigval, eigvec = np.linalg.eigh(H.matrix.toarray())
    coef = eigvec.conj().T @ psi
    n_q = np.array([obs.density_profile(eigvec @ (np.exp(-1j * eigval * t) * coef)) for t in times])
    assert np.abs(n_q - n_q[0]).max() < 1e-9  # saddle stationary under ED
    sim = TruncatedWigner(lat, ZGP, seed=23, hbar_eff=1.0 / N)

    def phys_n(z: np.ndarray) -> np.ndarray:
        return (N / 2.0) * (0.5 * (z[:2] ** 2 + z[2:] ** 2)) - 0.5

    mean, se = sim.observable(400, times, phys_n)
    assert np.all(np.abs(mean - n_q) < 3.0 * se)  # measured max z 0.59
    assert np.all(se > 0.0)


# --- settings-policy regime probes (named in bohnasa/twa.py comments) --------


def test_bracket_fd_step_lands_in_measured_gap():
    """Paired-seed probe of _BRACKET_FD_EPS: same ensemble, eps patched
    (module constants are monkeypatchable per the settings policy).

    Measured (K = 100, seed 29, N = 16, t = 1.5): C(1e-6) vs C(1e-4)
    agree to relative 5.7e-07 (plateau, >= 3 orders below the knee);
    C(3e-2) shifts by 5.0 % and C(1e-1) by 44 % - the curvature
    truncation knee between 1e-2 and 3e-2, no round-off floor reached
    100x below the default. Bands: 1e-3 (plateau), 0.01 and 0.15 (knee).
    """
    lat = MeanFieldLattice(2, THETA)
    N = 16
    times = np.array([1.5])
    sim = TruncatedWigner(lat, ZGP, seed=29, hbar_eff=1.0 / N)
    f = _probe_symbol(N)

    def curve_with_eps(eps: float) -> np.ndarray:
        twa_module._BRACKET_FD_EPS = eps
        try:
            c, _ = sim.otoc(100, times, f)
        finally:
            twa_module._BRACKET_FD_EPS = 1e-4
        return c

    base = curve_with_eps(1e-4)
    small = curve_with_eps(1e-6)
    mid = curve_with_eps(3e-2)
    large = curve_with_eps(1e-1)
    assert np.abs(small / base - 1.0).max() < 1e-3  # measured 5.7e-07
    assert np.abs(mid / base - 1.0).max() > 0.01  # measured 0.050
    assert np.abs(large / base - 1.0).max() > 0.15  # measured 0.44


@pytest.mark.parametrize(("zero_point", "coef"), [(0.2, 0.38), (0.85, 1.73)])
def test_wigner_zero_point_is_a_tested_invariant(monkeypatch, zero_point, coef):
    """_WIGNER_ZERO_POINT comment: moving the half-quantum factor breaks
    the short-time validation band (tested-invariant, class (a)).

    Measured with the zero point patched to 0.2 / 0.85 (N = 16, K = 300,
    t = 5e-3, seed 11; reference 0.5 -> 0.99): the short-time deviation
    coefficient of hbar_eff becomes 0.38 / 1.73, far outside the
    [0.6, 1.5] band that validation (2a) pins - the sampled estimator
    depends directly on the Wigner packet width (bracket^2 averages the
    packet curvature), so the half-quantum is physics, not a tunable.
    """
    lat = MeanFieldLattice(2, THETA)
    N = 16
    monkeypatch.setattr(twa_module, "_WIGNER_ZERO_POINT", zero_point)
    _, H, op, psi = _dimer_quantum(N)
    tiny = np.array([5e-3])
    cq = oc.SquaredCommutator(H, op).curve(psi, tiny)
    sim = TruncatedWigner(lat, ZGP, seed=11, hbar_eff=1.0 / N)
    ct, _ = sim.otoc(300, tiny, _probe_symbol(N))
    measured = (ct[0] / cq[0] - 1.0) * N
    assert abs(measured - coef) < 0.05  # measured 0.3802 / 1.7293
    assert not (0.6 < measured < 1.5)
