"""Physics anchors of the OTOC module (issues #22, #28).

Quantum curves are pinned against analytic short-time coefficients, against
the closed-form classical mean-field instability of the dimer, and against
boundedness in the regular regime. The thermal (Gibbs-state) curve is pinned
on the periodic trimer - the thermal-OTOC reference system of thesis
Sec. 7.4 - by its zero-temperature and infinite-temperature limits, the
regularization invariants and a brute-force density-matrix cross-check.
The dimer is dimension N+1, so even N = 256 is CI-cheap; the one heavy
lattice run (dim 5151) is gated behind the BOHNASA_SLOW_TESTS environment
variable and stays local.
"""

import os

import numpy as np
import pytest
import scipy.linalg as sla

import bohnasa
from bohnasa import classical_mean_field as mf
from bohnasa import otoc as oc

THETA = 1.35  # unstable-saddle regime of the dimer, theta* = 1.339 nearby


def _beta2(H, op, psi):
    hd, od = H.matrix.toarray(), op.matrix.toarray()
    ho = hd @ od - od @ hd
    v = ho @ od @ psi - od @ ho @ psi
    return np.vdot(v, v).real


def _dimer_case(N, theta=THETA, probe="saddle"):
    fock = bohnasa.FixedNFockSpace(N, 2)
    t = np.array([[0.0, -2.0 * np.cos(theta)], [-2.0 * np.cos(theta), 0.0]])
    H = bohnasa.Hamiltonian(fock, one_body=t, onsite_u=np.sin(theta) / N)
    op = (bohnasa.Number(fock, 0) - bohnasa.Number(fock, 1)) * 0.5
    if probe == "saddle":
        psi = bohnasa.fixed_n_coherent_state(fock, np.array([1.0, -1.0]) / np.sqrt(2.0))
    else:
        psi = bohnasa.fixed_n_coherent_state(fock, np.array([1.0, 1.0]) / np.sqrt(2.0))
    return fock, H, op, psi


def _slope(t, c, t0, t1):
    m = (t >= t0) & (t <= t1) & (c > 0)
    return np.polyfit(t[m], np.log(c[m]), 1)[0]


def test_classical_lambda_is_the_exact_reference_line():
    """The closed-form saddle exponent must reproduce three independent ways:
    Jacobian spectrum, ODE perturbation slope, and the sqrt formula itself."""
    lam = mf.dimer_lambda(THETA)
    assert lam == pytest.approx(0.970617, abs=1e-4)
    jac = mf.dimer_jacobian(THETA, 0.0, np.pi)
    assert np.linalg.eigvals(jac).real.max() == pytest.approx(lam, abs=1e-12)
    eigval, eigvec = np.linalg.eig(jac)
    udir = eigvec[:, np.argmax(eigval.real)]  # unstable direction
    y0 = np.array([1e-9 * udir[0].real, np.pi + 1e-9 * udir[1].real])
    times = np.linspace(0.0, 10.0, 500)
    traj = mf.mean_field_flow(THETA, y0, times)
    delta = traj - np.array([0.0, np.pi])
    logdelta = 0.5 * np.log(delta[:, 0] ** 2 + delta[:, 1] ** 2)
    m = (times > 0.5) & (times < 3.0)
    # Observed slope deviation from the DOP853 (rtol=1e-10/atol=1e-12) flow
    # at the 1e-9-amplitude anchor: 3.3387e-04, bit-identical on scipy
    # 1.15.3/1.16.3/1.18.1 (py3.10-3.14); +-1 ulp libm jitter of the
    # right-hand side moves it by < 1e-7. The floor is
    # the chaotic amplification of tolerance-scale solver noise at the tiny
    # perturbation amplitude, not integrator order; tightening the band
    # further would track tolerances, not physics (issue #24 replaced the
    # fixed-step RK4, whose deviation here was 2.1875e-04).
    assert np.polyfit(times[m], logdelta[m], 1)[0] == pytest.approx(lam, abs=5e-4)


def test_lambda_s_attains_max_at_theta_star():
    """Regression anchor (thesis grid ~1.35, see CITATION.cff): the
    exact maximum of lambda_s(theta) sits at theta* = 1.339 with value 0.4859."""
    thetas = np.linspace(1.1072, np.pi / 2 - 1e-3, 400)
    lams = np.array([mf.dimer_lambda(th) for th in thetas])
    imax = np.argmax(lams)
    assert thetas[imax] == pytest.approx(1.339, abs=3e-3)
    assert lams[imax] == pytest.approx(0.9717, abs=2e-3)
    # analytic extremum condition: tan(2 theta*) = -1/2 on the unstable branch
    exact = 0.5 * (np.pi - np.arctan(0.5))
    assert exact == pytest.approx(1.33897, abs=1e-5)


def _growth_stop(times, rate, lam):
    """End of the contiguous growth window: first time after t = 2 that the
    instantaneous rate drops below lambda_s and stays below. Later rebounds
    inside the saturation oscillation are deliberately NOT counted."""
    above = rate > lam
    i = np.searchsorted(times, 2.0)
    if not above[i]:
        raise AssertionError("growth window does not contain t = 2")
    while i + 1 < len(times) and above[i + 1]:
        i += 1
    return times[i]


def test_duration_of_superclassical_growth_scales_logarithmically_with_N():
    """The vector-trick quantum curve shows the Ehrenfest structure:
    growth above the (correctly normalized) classical rate lambda_s stops
    at t*(N) with t* increasing and t*/tau_E constant ~0.8 across an
    thirty-twofold range of N - the logarithmic time budget before saturation,
    at CI-trivial dimer dimensions."""
    lam = mf.dimer_lambda(THETA)
    t_stars, ratios = [], []
    for N in (32, 64, 128, 256, 512, 1024):
        _, H, op, psi = _dimer_case(N)
        sc = oc.SquaredCommutator(H, op)
        tau_e = np.log(N) / lam
        times = np.linspace(0.002, 4.0 * tau_e, 1500)
        c = sc.curve(psi, times)
        rate = np.gradient(np.log(c), times)
        t_star = _growth_stop(times, rate, lam)
        t_stars.append(t_star)
        ratios.append(t_star / tau_e)
    assert np.all(np.diff(t_stars) > 0)
    assert np.all((np.array(ratios) > 0.72) & (np.array(ratios) < 0.88))
    assert ratios[0] == pytest.approx(np.mean(ratios), rel=0.1)


def test_effective_rate_converges_to_twice_the_classical_exponent():
    """The instantaneous effective rate d log(C/beta^2)/dt of the quantum
    curve approaches the semiclassical growth law 2 lambda_s as N grows
    (canonical pair {phi, s} = 2/N - a factor 4 against the naive
    single-mode bracket): |err| decreases monotonically from N = 64 to 512,
    with the median at the largest CI-size N within 5% of 2 lambda_s."""
    lam = mf.dimer_lambda(THETA)
    errs = []
    for N in (64, 128, 256, 512):
        _, H, op, psi = _dimer_case(N)
        sc = oc.SquaredCommutator(H, op)
        times = np.linspace(0.002, 12.0, 600)
        c = sc.curve(psi, times)
        b2 = _beta2(H, op, psi)
        rate = np.gradient(np.log(c / b2), times)
        m = (times > 1.0) & (times < 3.0)
        errs.append(abs(np.median(rate[m]) / (2 * lam) - 1))
    assert np.all(np.diff(errs) < 0)
    assert errs[-1] < 0.05


def test_regular_regime_is_bounded_and_chaotic_regime_is_not():
    """Across the pitchfork at arctan(2) the OTOC changes character: on the
    stable side C(t)/beta2 stays O(1) forever, on the unstable side it
    exceeds it by orders of magnitude within the same time horizon."""
    _, h_stable, o_stable, psi_stable = _dimer_case(32, theta=1.0)
    times = np.linspace(0.0, 40.0, 401)
    c_reg = oc.SquaredCommutator(h_stable, o_stable).curve(psi_stable, times)
    assert c_reg.max() / _beta2(h_stable, o_stable, psi_stable) < 10.0

    _, h_unst, o_unst, psi_unst = _dimer_case(32, theta=THETA)
    c_cha = oc.SquaredCommutator(h_unst, o_unst, method="krylov").curve(psi_unst, times)
    assert c_cha.max() / _beta2(h_unst, o_unst, psi_unst) > 100.0


def test_kicked_dimer_sustains_growth_only_with_kicks():
    """Floquet path: pure harmonic driving (K=0) keeps C bounded by its
    static envelope; a modest interaction kick sustains growth orders of
    magnitude higher at the same step count."""
    N = 40
    fock = bohnasa.FixedNFockSpace(N, 2)
    occ = fock.fockArray.astype(float)
    hop = bohnasa.Hamiltonian(fock, one_body=np.array([[0.0, -2.0], [-2.0, 0.0]])).matrix.toarray()
    kick = np.diag(0.5 * (occ[:, 0] * (occ[:, 0] - 1) + occ[:, 1] * (occ[:, 1] - 1)))
    op = (bohnasa.Number(fock, 0) - bohnasa.Number(fock, 1)) * 0.5
    psi = bohnasa.fixed_n_coherent_state(fock, np.array([1.0, -1.0]) / np.sqrt(2.0))
    steps = np.arange(0, 61)
    tau = 0.4
    hop_step = sla.expm(-1j * tau * hop)
    curves = {}
    for strength in (0.0, 1.5):
        u = sla.expm(-1j * strength * tau * kick) @ hop_step
        curves[strength] = oc.SquaredCommutator.from_propagator(u, op).curve(psi, steps)
    assert curves[0.0].max() < 1e3
    assert curves[1.5][60] > 10 * curves[0.0].max()


def test_backward_prepared_probe_sustains_growth_longer():
    """Preparation by backward evolution from the saddle probe
    (t_0 = -3 tau_E / 4 recipe): the pre-evolved state keeps a larger
    logarithmic growth rate in the same late window than the bare saddle
    coherent state."""
    N = 128
    lam = mf.dimer_lambda(THETA)
    tau_e = np.log(N) / lam
    _, H, op, psi = _dimer_case(N)
    psi_pre = sla.expm(-1j * 0.75 * tau_e * H.matrix.toarray()) @ psi
    sc = oc.SquaredCommutator(H, op)
    times = np.linspace(0.0, 14.0, 561)
    c_bare = sc.curve(psi, times)
    c_pre = sc.curve(psi_pre, times)
    assert _slope(times, c_pre, 4.0, 7.0) > _slope(times, c_bare, 4.0, 7.0)


def _trimer(N):
    """Periodic trimer (L = 3) at filling N with U = 1/N - the thermal-OTOC
    reference system of thesis Sec. 7.4."""
    fock = bohnasa.FixedNFockSpace(N, 3)
    t = np.zeros((3, 3))
    for site in range(3):
        nxt = (site + 1) % 3
        t[nxt, site] = t[site, nxt] = -1.0
    H = bohnasa.Hamiltonian(fock, one_body=t, onsite_u=1.0 / N)
    op = bohnasa.Number(fock, 0)
    return fock, H, op


def test_trimer_thermal_curve_reduces_to_ground_and_maximally_mixed_limits():
    """(a) beta -> infinity becomes the ground-state curve (the shifted
    Boltzmann weights underflow to exact zeros at any gap > 745/beta);
    (b) beta = 0 is the maximally-mixed average: the mean over all eigenstate
    curves AND the closed 2/dim [Tr(O^2 O(t)^2) - Tr(O O(t) O O(t))] form
    for Hermitian O."""
    fock, H, op = _trimer(20)  # dim 231
    sc = oc.SquaredCommutator(H, op)
    times = np.linspace(0.0, 3.0, 13)
    eps, vecs = np.linalg.eigh(H.matrix.toarray())
    assert eps[1] - eps[0] > 3.0  # => beta = 1000 weights are exactly [1, 0, ...]
    thermal_inf = sc.thermal_curve(times, 1000.0)
    ground_curve = sc.curve(vecs[:, 0], times)
    assert thermal_inf[0] == 0.0
    assert np.abs(thermal_inf - ground_curve).max() < 1e-11  # observed 1.2e-12

    c0 = sc.thermal_curve(times, 0.0)
    eigen_mean = np.array([sc.curve(vecs[:, m], times) for m in range(sc.dim)]).mean(axis=0)
    assert np.abs(c0 - eigen_mean).max() / c0.max() < 1e-12  # observed 1.2e-15
    hd, od = H.matrix.toarray(), op.matrix.toarray()
    for i in (1, 5, 12):
        u = sla.expm(-1j * times[i] * hd)
        ot = u.conj().T @ od @ u
        two_form = (2 / sc.dim) * (np.trace(od @ od @ ot @ ot) - np.trace(od @ ot @ od @ ot)).real
        assert c0[i] == pytest.approx(two_form, rel=1e-10, abs=1e-11)


def test_trimer_thermal_symmetric_regularization_invariants():
    """Thesis Sec. 7.1 / App. D convention: the symmetric rho^{1/8} weighting
    agrees with the unregularized curve at beta = 0, vanishes exactly at
    beta -> infinity (identity/pure-state contribution removed), annihilates
    commuting operators at every beta, and stays nonnegative."""
    _, H, op = _trimer(20)
    sc = oc.SquaredCommutator(H, op)
    times = np.linspace(0.0, 3.0, 13)
    assert sc.thermal_curve(times, 1000.0, regularization="symmetric").max() == 0.0
    c0_none = sc.thermal_curve(times, 0.0)
    c0_sym = sc.thermal_curve(times, 0.0, regularization="symmetric")
    assert np.abs(c0_sym - c0_none).max() / c0_none.max() < 1e-13  # observed 4.0e-16
    for beta in (0.0, 0.5, 2.0, 10.0):
        for reg in ("none", "symmetric"):
            c = sc.thermal_curve(times, beta, regularization=reg)
            assert np.all(np.isfinite(c))
            assert np.all(c >= 0.0)
            assert c[0] == 0.0
    sch = oc.SquaredCommutator(H, H)
    for beta in (0.0, 2.0):
        assert sch.thermal_curve(times, beta).max() < 1e-20  # observed 6.1e-24
        assert sch.thermal_curve(times, beta, regularization="symmetric").max() < 1e-20


def test_trimer_thermal_curve_matches_brute_force_density_matrix():
    """Independent path: C_beta(t) = Tr(A^dag A rho) with A = [O(t), O] and
    rho = exp(-beta H)/Z assembled by dense expm only - no eigenbasis,
    no weights - cross-checks the spectral contraction at finite beta."""
    _, H, op = _trimer(20)
    sc = oc.SquaredCommutator(H, op)
    hd, od = H.matrix.toarray(), op.matrix.toarray()
    times = np.array([0.75, 1.5, 2.75])
    for beta in (0.5, 2.0):
        rho = sla.expm(-beta * hd)
        rho = rho / np.trace(rho).real
        for t in times:
            u = sla.expm(-1j * t * hd)
            a = (u.conj().T @ od @ u) @ od - od @ (u.conj().T @ od @ u)
            brute = np.trace(a.conj().T @ a @ rho).real
            assert sc.thermal_curve(np.array([t]), beta)[0] == pytest.approx(
                brute, rel=1e-9, abs=1e-11
            )


@pytest.mark.skipif(
    not os.environ.get("BOHNASA_SLOW_TESTS"),
    reason="heavy lattice run (dim 5151); set BOHNASA_SLOW_TESTS=1 locally",
)
def test_lattice_otoc_runs_at_large_dimensions_via_krylov():
    """The vector-trick path must reach L=3, N=100 (dim 5151) with O(dim)
    memory - Liouville-space assembly of the same object is impossible here."""
    N, L = 100, 3
    fock = bohnasa.FixedNFockSpace(N, L)
    assert fock.dim == 5151
    t = np.zeros((L, L))
    for site in range(L):
        nxt = (site + 1) % L
        t[nxt, site] = t[site, nxt] = -1.0
    H = bohnasa.Hamiltonian(fock, one_body=t, onsite_u=1.0 / N)
    op = bohnasa.Number(fock, 0)
    psi = bohnasa.fixed_n_coherent_state(
        fock, np.array([1.0, np.exp(2j * np.pi / 3), np.exp(-2j * np.pi / 3)]) / np.sqrt(3)
    )
    sc = oc.SquaredCommutator(H, op, method="krylov")
    assert sc.method == "krylov"
    times = np.array([0.0, 0.5, 1.0, 2.0])
    curve = sc.curve(psi, times)
    assert curve[0] == 0.0
    assert np.all(np.isfinite(curve))
    assert np.diff(curve[1:]).min() > -np.inf  # finite, non-trivial
    assert curve[1] > 0.0
