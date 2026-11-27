"""Physics invariants of the general L-site mean-field engine (issue #26).

Ground truth from independent directions: the closed-form dimer (flow,
fixed points, stability spectra), a hand-written ODE reference solved by
solve_ivp at tight tolerance, exact conservation laws (energy, norm), the
decoupled theta = pi/2 limit with a closed-form solution, and agreement of
the independent Lyapunov implementations (Benettin vs QR vs two-trajectory)
with the analytic dimer saddle exponent. Tolerances are calibrated across the
full CI matrix (scipy 1.15-1.18), not one stack, and each band sits above the
relevant noise floor: ~sqrt(eps) for Jordan-degenerate (double-zero) eigenvalues
and finite-time + threading spread for chaotic Lyapunov estimates. Well-
conditioned comparisons (algebraic energies, nonzero eigenvalues, finite-
difference Jacobians) keep tight bounds. See issue #26.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import bohnasa
from bohnasa.mean_field import dimer as mf

THETA = 1.35  # above the pitchfork tan(theta) = 2


def embed_dimer(s: float, phi: float) -> np.ndarray:
    """Dimer (s, phi) -> lattice state z = (q0, q1, p0, p1), N = 2 gauge.

    Convention (module docstring): s = (n0 - n1)/N, phi = arg(phi1) -
    arg(phi0); we place the pair at psi0 = -phi/2, psi1 = +phi/2.
    """
    n0, n1 = 1.0 + s, 1.0 - s
    q = np.sqrt(2.0) * np.array([np.sqrt(n0) * np.cos(-phi / 2.0), np.sqrt(n1) * np.cos(phi / 2.0)])
    p = np.sqrt(2.0) * np.array([np.sqrt(n0) * np.sin(-phi / 2.0), np.sqrt(n1) * np.sin(phi / 2.0)])
    return np.concatenate((q, p))


def project_dimer(z: np.ndarray) -> tuple[float, float]:
    q, p = z[:2], z[2:]
    n = 0.5 * (q * q + p * p)
    s = float((n[0] - n[1]) / 2.0)
    phi = float(np.angle(q[1] + 1j * p[1]) - np.angle(q[0] + 1j * p[0]))
    return s, phi


def dimer_reference_orbit(theta: float, y0: tuple[float, float], times: np.ndarray) -> np.ndarray:
    """The dimer sphere flow of the mean_field.dimer module, integrated
    independently here from the module docstring's h(s, phi) by hand."""
    c, u = np.cos(theta), np.sin(theta)

    def rhs(_t, y):
        s, phi = y
        root = np.sqrt(max(1.0 - s * s, 1e-300))
        return [-4 * c * root * np.sin(phi), 4 * c * s * np.cos(phi) / root + 2 * u * s]

    return solve_ivp(rhs, (times[0], times[-1]), y0, t_eval=times, rtol=1e-12, atol=1e-14).y


def test_l2_flow_reduces_to_dimer_sphere():
    lat = bohnasa.MeanFieldLattice(2, THETA)
    times = np.linspace(0.0, 8.0, 801)
    y0 = (0.3, 2.0)
    ref = dimer_reference_orbit(THETA, y0, times)
    traj = lat.trajectory(embed_dimer(*y0), times)
    proj = np.array([project_dimer(z) for z in traj])
    err_s = np.abs(proj[:, 0] - ref[0]).max()
    err_p = np.abs(np.unwrap(proj[:, 1]) - np.unwrap(ref[1])).max()
    # observed at rtol=1e-9: ~3e-10
    assert err_s < 1e-8
    assert err_p < 1e-8


def test_l2_energy_matches_dimer_hamiltonian():
    lat = bohnasa.MeanFieldLattice(2, THETA)
    c, u = np.cos(THETA), np.sin(THETA)
    rng = np.random.default_rng(11)
    for s, phi in zip(rng.uniform(-0.95, 0.95, 5), rng.uniform(-np.pi, np.pi, 5), strict=True):
        h_closed = -2 * c * np.sqrt(1 - s * s) * np.cos(phi) + 0.5 * u * (1 + s * s)
        assert lat.hamiltonian(embed_dimer(s, phi)) == pytest.approx(2.0 * h_closed, abs=1e-10)


@pytest.mark.parametrize("theta", [1.0, THETA])
def test_l2_fixed_points_match_dimer_table(theta):
    lat = bohnasa.MeanFieldLattice(2, theta)
    pts = lat.fixed_points()
    table = mf.dimer_fixed_points(theta)
    assert len(pts) == len(table)

    # multiset of (|s|, |phi|): the sign partners are symmetric points, and
    # phi enters the closed forms only through cos(phi), so |.| is gauge-free
    def key(s: float, p: float) -> tuple[float, float]:
        return (round(abs(s), 8), round(abs(p), 8))

    got = sorted(key(*project_dimer(z)) for z in pts)
    ref = sorted(key(float(row[0]), float(row[1])) for row in table)
    assert got == ref
    # broken pair sits at sqrt(1 - 4cos^2/sin^2) (0.8935762543 at theta=1.35)
    max_abs_s = max(abs(project_dimer(z)[0]) for z in pts)
    if np.tan(theta) > 2.0:
        s_star = float(np.sqrt(1 - 4 * np.cos(theta) ** 2 / np.sin(theta) ** 2))
        assert max_abs_s == pytest.approx(s_star, abs=1e-9)
    else:
        assert max_abs_s < 1e-9


@pytest.mark.parametrize("theta", [1.0, THETA])
def test_l2_stability_spectrum_matches_dimer_jacobian(theta):
    lat = bohnasa.MeanFieldLattice(2, theta)
    for z in lat.fixed_points():
        s, phi = project_dimer(z)
        lab = np.sort(np.abs(np.linalg.eigvals(lat.stability_matrix(z))))
        red = np.sort(np.abs(np.linalg.eigvals(mf.dimer_jacobian(theta, s, phi))))
        # rotating-frame pair = dimer pair (largest two, well-conditioned -> 1e-7),
        # plus the phase-orbit zero pair: a 2x2 Jordan block whose spurious
        # eigvals split at ~sqrt(eps) and are LAPACK-build dependent, so the
        # band must sit well above that noise floor.
        assert lab[2:] == pytest.approx(red, abs=1e-7)
        assert lab[:2].max() < 1e-6
    # saddle exponent at the analytic value
    lam_lattice = max(
        np.linalg.eigvals(lat.stability_matrix(z)).real.max() for z in lat.fixed_points()
    )
    assert lam_lattice == pytest.approx(mf.dimer_lambda(theta), abs=1e-7)


def test_lyapunov_saddle_reproduces_dimer_lambda():
    lat = bohnasa.MeanFieldLattice(2, THETA)
    saddle = embed_dimer(0.0, np.pi)
    w, v = np.linalg.eig(lat.stability_matrix(saddle))
    uvec = v[:, np.argmax(w.real)].real
    times = np.linspace(0.0, 12.0, 241)
    target = mf.dimer_lambda(THETA)
    lam_b = lat.lyapunov(saddle, times, v0=uvec, delta=1e-9)
    lam_t = lat.lyapunov(saddle, times, v0=uvec, delta=1e-9, method="two_trajectory")
    # observed 0.97061664 (Benettin) / 0.97061662 (two-trajectory) vs 0.9706166375.
    # Hyperbolic (unstable-manifold seed) -> exponential convergence, so 1e-5 is a
    # genuine invariant with margin against integrator/backend drift, not a
    # one-stack snapshot.
    assert lam_b == pytest.approx(target, rel=1e-5)
    assert lam_t == pytest.approx(target, rel=1e-5)


def _chaotic_l4():
    lat = bohnasa.MeanFieldLattice(4, THETA)
    rng = np.random.default_rng(0)
    z0 = rng.normal(size=8)
    z0 *= np.sqrt(8.0) / np.linalg.norm(z0)
    return lat, z0


def test_lyapunov_methods_agree_on_chaos():
    lat, z0 = _chaotic_l4()
    times = np.linspace(0.0, 100.0, 1001)
    lam_b = lat.lyapunov(z0, times)
    lam_q = lat.lyapunov(z0, times, method="qr")
    assert lam_b > 0.2
    assert lam_q > 0.2
    # Benettin is the backend-stable anchor (0.3459 on every scipy); QR over the
    # same finite horizon is a looser estimate that drifts with the integrator
    # backend - measured |lam_b - lam_q| is 0.002 (scipy>=1.18), 0.014 (1.15) and
    # 0.024 (1.17). The invariant is that two independent algorithms return the
    # same positive exponent to finite-time accuracy, NOT a one-environment 2%
    # snapshot (CODING_RULES.md: invariants over value snapshots). QR's slow
    # convergence is tracked separately, not asserted away here.
    assert abs(lam_b - lam_q) < 0.04
    times40 = np.linspace(0.0, 40.0, 401)
    lam_b40 = lat.lyapunov(z0, times40)
    lam_t40 = lat.lyapunov(z0, times40, method="two_trajectory")
    # Tangent (Benettin) and finite-difference flow estimate the same chaotic
    # max exponent; they agree to ~0 on single-thread builds, but a chaotic
    # trajectory amplifies backend threading noise, so the band must cover the
    # finite-time + threading spread, not a one-stack 0.0000 snapshot.
    assert abs(lam_b40 - lam_t40) < 0.01


def test_energy_and_norm_conservation():
    lat, z0 = _chaotic_l4()
    traj = lat.trajectory(z0, np.linspace(0.0, 30.0, 301))
    energies = np.array([lat.hamiltonian(z) for z in traj])
    numbers = np.array([lat.particle_number(z) for z in traj])
    # observed ptp(H)/|H| ~ 1.1e-9 and ptp(N) ~ 1.4e-9 at rtol=1e-9
    assert np.ptp(energies) / np.abs(energies).max() < 1e-7
    assert np.ptp(numbers) < 1e-7


def test_jacobian_is_exact_and_symplectic():
    lat, z0 = _chaotic_l4()
    rng = np.random.default_rng(2)
    z = rng.normal(size=8)
    jac = lat.jacobian(z)
    step = 1e-6
    fd = np.column_stack(
        [(lat.flow(z + step * e) - lat.flow(z - step * e)) / (2 * step) for e in np.eye(8)]
    )
    assert np.abs(jac - fd).max() < 1e-6  # observed 9.6e-10
    assert abs(np.trace(jac)) < 1e-12
    omega = np.block([[np.zeros((4, 4)), np.eye(4)], [-np.eye(4), np.zeros((4, 4))]])
    assert np.abs(jac.T @ omega + omega @ jac).max() < 1e-12  # observed exactly 0
    # differentiated norm conservation: z^T dF = -F(z)^T (observed <= 1e-16)
    assert np.abs(z @ jac + lat.flow(z)).max() < 1e-10


def test_decoupled_limit_has_closed_form():
    lat = bohnasa.MeanFieldLattice(4, np.pi / 2)  # cos(theta) = 0: no hopping
    z0 = np.array([1.0, 1.5, 0.7, 1.3, 0.4, -0.2, 0.9, 0.1])
    z0 *= np.sqrt(8.0) / np.linalg.norm(z0)
    times = np.linspace(0.0, 10.0, 11)
    traj = lat.trajectory(z0, times)
    n0 = 0.5 * (traj[0, :4] ** 2 + traj[0, 4:] ** 2)
    psi0 = np.angle(traj[0, :4] + 1j * traj[0, 4:])
    err = 0.0
    for k, t in enumerate(times):
        n = 0.5 * (traj[k, :4] ** 2 + traj[k, 4:] ** 2)
        psi = np.angle(traj[k, :4] + 1j * traj[k, 4:])
        err = max(err, np.abs(n - n0).max())
        # i dphi/dt = dH/dphi* with H = (sin/2) sum n_j^2: psi_j(t) = psi_j0 - sin n_j0 t
        phase = psi - (psi0 - np.sin(lat.theta) * n0 * t)
        err = max(err, np.abs(np.angle(np.exp(1j * phase))).max())
    assert err < 1e-8  # observed 4.3e-10


def test_lyapunov_vanishes_on_regular_orbits():
    # all orbits of the decoupled (theta = pi/2) system are quasi-periodic:
    # the measured exponent must decay with the horizon
    lat = bohnasa.MeanFieldLattice(2, np.pi / 2)
    z = np.array([1.1, 0.9, 0.3, 0.5])
    z *= 2.0 / np.linalg.norm(z)
    lam_short = lat.lyapunov(z, np.linspace(0.0, 40.0, 81))
    lam_long = lat.lyapunov(z, np.linspace(0.0, 160.0, 321))
    # observed 0.097 -> 0.033 (logarithmic-stretch bias ~ ln(T)/T). The
    # invariant is monotone decay with horizon, not the exact factor, which is
    # backend-sensitive for a quasi-periodic orbit.
    assert lam_short < 0.15
    assert lam_long < 0.75 * lam_short


def test_uniform_mu_only_rotates_the_occupations():
    z0 = np.random.default_rng(0).normal(size=8)
    z0 *= np.sqrt(8.0) / np.linalg.norm(z0)
    times = np.linspace(0.0, 20.0, 201)
    occ = []
    for mu in (0.0, 0.7):
        lat = bohnasa.MeanFieldLattice(4, THETA, mu=mu)
        traj = lat.trajectory(z0, times)
        occ.append(0.5 * (traj[:, :4] ** 2 + traj[:, 4:] ** 2))
    # uniform mu enters the flow as a pure global phase rotation
    assert np.abs(occ[0] - occ[1]).max() < 1e-6  # observed 5.6e-9


def test_quarter_circle_rule_general_l_gp_correspondence():
    """Engine flow vs GP flow of the consistent quantum model (issue #60).

    The general quarter-circle rule is U = sin(theta) * L / N_phys
    (Hamiltonian translation onsite_u = U/2); the dimer formula
    2 sin(theta)/N coincides with it only at L = 2. Independent reference:
    hand-written GP flow in physical amplitudes alpha (sum |alpha|^2 =
    N_phys), Radau at rtol=1e-11 / atol=1e-13, ex14 seed orbit at L = 4.
    """
    L = N_phys = 4
    c, s = np.cos(THETA), np.sin(THETA)
    z4 = np.array([np.sqrt(2.4), np.sqrt(1.8), np.sqrt(2.0), np.sqrt(1.8), 0.4, -0.3, 0.2, -0.1])
    z4 *= np.sqrt(2 * L) / np.linalg.norm(z4)
    times = np.linspace(0.0, 10.0, 41)
    hop = np.zeros((L, L))
    for j in range(L):
        hop[j, (j + 1) % L] = hop[(j + 1) % L, j] = -c

    def gp_flow(u_quantum):
        a0 = (z4[:L] + 1j * z4[L:]) / np.sqrt(2.0)

        def rhs(_t, y):
            alpha = y[:L] + 1j * y[L:]
            d = -1j * (hop @ alpha + (u_quantum / 2) * (2 * np.abs(alpha) ** 2 - 1) * alpha)
            return np.concatenate((d.real, d.imag))

        sol = solve_ivp(
            rhs,
            (0.0, 10.0),
            np.concatenate((a0.real, a0.imag)),
            t_eval=times,
            rtol=1e-11,
            atol=1e-13,
            method="Radau",
        )
        return np.abs(sol.y[:L] + 1j * sol.y[L:]).T ** 2

    traj = bohnasa.MeanFieldLattice(L, THETA).trajectory(z4, times)
    n_eng = 0.5 * (traj[:, :L] ** 2 + traj[:, L:] ** 2)
    # observed 3.37e-10 with the general rule; the dimer-only rule
    # 2 sin(theta)/N runs at half the interaction and deviates by 0.655
    assert np.abs(gp_flow(s * L / N_phys) - n_eng).max() < 1e-6
    assert np.abs(gp_flow(2 * s / N_phys) - n_eng).max() > 0.1
