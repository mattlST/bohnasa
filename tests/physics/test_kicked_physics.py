"""Physics anchors of the kicked Bose-Hubbard module (issue #29).

The four validations the issue demands: (1) unitarity of the step propagator
across regimes including the strong-kick limit, (2) the classical kicked map
reproducing ``mean_field_flow`` as tau -> 0 on the dimer with measured
first-order Trotter convergence (Richardson), (3) the Floquet OTOC round trip
``SquaredCommutator.from_propagator(k.propagator(fock))`` == explicit
finite-step Heisenberg growth, (4) the quasienergy branch convention
deterministic across runs and stable under tiny parameter perturbation for a
gapped spectrum. Plus the derivation pins tying propagator, map and
commutator signs together.
"""

import numpy as np
import pytest

from bohnasa import (
    FixedNFockSpace,
    KickedBoseHubbard,
    KickedMeanFieldMap,
    Number,
    SquaredCommutator,
    fixed_n_coherent_state,
)
from bohnasa import (
    classical_mean_field as mf,
)

THETA = 1.35  # unstable-saddle regime, same quarter-circle angle as ex09/ex10


def _wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- validation 1
@pytest.mark.parametrize(
    ("L", "N", "tau", "J", "U"),
    [
        (2, 40, 0.4, np.cos(THETA), 2 * np.sin(THETA) / 40),  # ex10 kicked dimer
        (3, 6, 1.0, 1.0, 1.0),  # order-1 kicks
        (3, 8, 2.5, 0.7, 1.3),  # strong kick: tau * U/2 * N(N-1) = 91 rad
        (4, 5, 0.9, 1.2, -0.8),  # attractive interaction
    ],
)
def test_propagator_is_unitary_across_regimes(L, N, tau, J, U):
    """|U^dag U - I|max stays at roundoff, including the strong-kick corner
    where the on-site phase winds by ~91 rad across the basis."""
    k = KickedBoseHubbard(L=L, tau=tau, J=J, U=U)
    fock = FixedNFockSpace(N, L)
    u = k.propagator(fock)
    dev = np.abs(u.conj().T @ u - np.eye(fock.dim)).max()
    assert dev < 5e-15  # observed up to 1.11e-15 (L = 3, N = 8, tau = 2.5)
    # the other side too (square matrix: one side implies the other, but the
    # check is free at these sizes and catches shape mistakes)
    assert np.abs(u @ u.conj().T - np.eye(fock.dim)).max() < 5e-15


# ---------------------------------------------------------------- validation 2
def _dimer_kick(N: int, tau: float, t_end: float):
    """Kicked-dimer trajectory in (s, phi) coordinates at period tau."""
    k = KickedBoseHubbard(L=2, tau=tau, J=np.cos(THETA), U=2 * np.sin(THETA) / N)
    phi0 = np.array([0.3, -1.1])
    occ0 = np.array([0.6 * N, 0.4 * N])
    m = k.classical_map()
    phi, occ = m.iterate((phi0, occ0), int(round(t_end / tau)))
    return m.dimer_variables(phi, occ)


def _err_vs_flow(traj, times):
    s, phi = traj
    ref = mf.mean_field_flow(THETA, (float(s[0]), float(phi[0])), times)
    ds = np.abs(s - ref[:, 0]).max()
    dphi = np.abs(_wrap(phi - ref[:, 1])).max()
    return ds, dphi


def test_classical_map_converges_to_mean_field_flow_as_tau_goes_to_zero():
    """Issue #29 validation: the kicked map's tau -> 0 limit IS
    mean_field_flow (quarter-circle coordinates, s = (n0 - n1)/N,
    phi = varphi_1 - varphi_0). The Lie-Trotter splitting error is measured
    first-order in tau via a two-dt Richardson check on one orbit, not
    assumed - and the wrong-sign dictionary fails, pinning the sign."""
    N, t_end = 20, 4.0
    taus = (0.05, 0.025, 0.0125)
    errs = []
    for tau in taus:
        traj = _dimer_kick(N, tau, t_end)
        times = tau * np.arange(traj[0].size)
        ds, dphi = _err_vs_flow(traj, times)
        errs.append((ds, dphi))
    (ds0, dphi0), (ds1, dphi1), (ds2, dphi2) = errs
    # first order: err(tau) / err(tau / 2) -> 2
    ratio_s_1, ratio_s_2 = ds0 / ds1, ds1 / ds2
    ratio_p_1, ratio_p_2 = dphi0 / dphi1, dphi1 / dphi2
    assert all(r > 1.9 and r < 2.1 for r in (ratio_s_1, ratio_s_2, ratio_p_1, ratio_p_2))
    # observed: s 1.9650 / 1.9825, phi 1.9992 / 1.9992 - clean first order
    assert ds2 < 3e-3 and dphi2 < 1e-2  # observed 2.52e-03 / 7.39e-03

    # the pinned sign convention is the convergent one: flipping the relative
    # phase (phi = varphi_0 - varphi_1) must NOT track the flow
    s, phi = _dimer_kick(N, 0.0125, t_end)
    times = 0.0125 * np.arange(s.size)
    ref = mf.mean_field_flow(THETA, (float(s[0]), float(phi[0])), times)
    wrong = np.abs(_wrap(-phi - ref[:, 1])).max()
    assert wrong > 1.0  # observed 2.89


# ---------------------------------------------------------------- validation 3
def test_floquet_otoc_round_trip_matches_explicit_heisenberg():
    """Issue #29 validation (the API's whole point): the library Floquet
    curve from k.propagator equals an independent finite-step Heisenberg
    growth computed with explicit matrix powers,
    C(n) = ||[U^-n O U^n, O]|psi>||^2, step by step."""
    L, N, tau, J, U = 3, 6, 0.6, 1.0, 0.9
    k = KickedBoseHubbard(L=L, tau=tau, J=J, U=U)
    fock = FixedNFockSpace(N, L)
    u = k.propagator(fock)
    op = Number(fock, 0)
    psi = fixed_n_coherent_state(
        fock, np.array([1.0, np.exp(2j * np.pi / 3), np.exp(-2j * np.pi / 3)]) / np.sqrt(3)
    )
    n_max = 12
    lib = SquaredCommutator.from_propagator(u, op).curve(psi, np.arange(n_max + 1))
    od = op.matrix.toarray()
    brute = np.empty(n_max + 1)
    for n in range(n_max + 1):
        un = np.linalg.matrix_power(u, n)
        o_n = un.conj().T @ od @ un
        v = o_n @ (od @ psi) - od @ (o_n @ psi)
        brute[n] = np.vdot(v, v).real
    assert lib[0] == 0.0
    assert np.abs(lib - brute).max() < 1e-12  # observed 2.49e-14 (dim 28)
    # the curve is a genuine Heisenberg growth, not a flat line: on this
    # small basis it rises then oscillates in the bounded saturation band
    assert brute[4] > 3 * brute[1]  # observed C(4) / C(1) = 5.66
    assert lib[-1] == pytest.approx(brute[-1], rel=1e-12)


def test_floquet_curve_tracks_continuous_time_at_small_tau():
    """The kicked dimer at small period approaches the continuous
    SquaredCommutator curve of H_J + H_U - the driving-side companion of
    validation 2, checking the propagator (not only the map) hits the right
    continuous limit."""
    N, tau = 60, 0.01
    k = KickedBoseHubbard(L=2, tau=tau, J=np.cos(THETA), U=2 * np.sin(THETA) / N)
    fock = FixedNFockSpace(N, 2)
    h = k.hopping_hamiltonian(fock) + k.interaction_hamiltonian(fock)
    op = (Number(fock, 0) - Number(fock, 1)) * 0.5
    psi = fixed_n_coherent_state(fock, np.array([1.0, -1.0]) / np.sqrt(2.0))
    n_max = 120  # physical time up to t = 1.2
    kicked = SquaredCommutator.from_propagator(k.propagator(fock), op).curve(
        psi, np.arange(n_max + 1)
    )
    cont = SquaredCommutator(h, op).curve(psi, tau * np.arange(n_max + 1))
    rel = np.abs(kicked[1:] / cont[1:] - 1.0).max()
    assert rel < 5e-3  # observed 8.64e-04 at tau = 0.01, t <= 1.2
    assert rel == rel  # NaN guard


# ---------------------------------------------------------------- validation 4
def test_quasienergy_branch_deterministic_and_stable_on_gapped_spectrum():
    """Issue #29 validation: repeated calls return BITWISE identical
    quasienergies, and for a spectrum with a measured 0.35 gap to the branch
    cut, a relative 1e-9 parameter perturbation of tau shifts every sorted
    quasienergy far less than the gap (branch labels cannot jump)."""
    k = KickedBoseHubbard(L=3, tau=0.6, J=1.0, U=0.9)
    fock = FixedNFockSpace(6, 3)
    eps_a, vecs_a = k.floquet_spectrum(fock)
    eps_b, vecs_b = k.floquet_spectrum(fock)
    assert np.array_equal(eps_a, eps_b)
    assert np.array_equal(vecs_a, vecs_b)

    fold = 2 * np.pi / k.tau
    gap_cut = min(eps_a.min(), fold - eps_a.max())
    assert gap_cut > 0.3  # observed 0.349 (lower edge 0.515, upper 10.12 of 10.47)

    k_pert = KickedBoseHubbard(L=3, tau=0.6 * (1 + 1e-9), J=1.0, U=0.9)
    eps_p, _ = k_pert.floquet_spectrum(fock)
    shift = np.abs(eps_p - eps_a).max()
    assert shift < 1e-6  # observed 9.48e-09, i.e. ~3 % of the cut gap
    assert shift < 0.05 * gap_cut


# ------------------------------------------------------------------ derivation
def test_heisenberg_first_order_pins_commutator_convention():
    """(U^dag n_0 U - n_0) / (-i tau) -> [n_0, H_J] as tau -> 0 - pinning
    the module's commutator sign against the operators module. The residual
    defect is exactly linear in tau (BCH / Trotter term, slope
    ``dev / tau = 16.90`` measured at tau = 1e-4 and 1e-5), i.e. the step's
    first-order generator IS the stated H_J, not its negative."""
    fock = FixedNFockSpace(5, 3)
    devs = {}
    for tau in (1e-4, 1e-5):
        k = KickedBoseHubbard(L=3, tau=tau, J=1.3, U=0.8)
        u = k.propagator(fock)
        n0 = Number(fock, 0).matrix.toarray()
        h_j = k.hopping_hamiltonian(fock).matrix.toarray()
        slope = (u.conj().T @ n0 @ u - n0) / (-1j * tau)
        devs[tau] = np.abs(slope - (n0 @ h_j - h_j @ n0)).max()
        # the OPPOSITE sign convention cannot be within 1e3 of the defect:
        assert np.abs(slope + (n0 @ h_j - h_j @ n0)).max() > 10 * devs[tau]
    assert devs[1e-4] < 2e-3  # observed 1.69e-03
    assert devs[1e-5] < 2e-4  # observed 1.69e-04
    assert devs[1e-4] / devs[1e-5] == pytest.approx(10.0, rel=1e-3)  # obs 10.0000


def test_map_kick_substep_is_exact_phase_slip():
    """With J = 0 the map is the exact sub-flow of the interaction:
    varphi_j(n) = varphi_j(0) - n tau U n_j (mod 2 pi), occupations fixed
    (up to |exp(i theta)|^2 roundoff) - checking the kick sign and the
    1/2-vs-no-1/2 factor of the convention."""
    N = np.array([3.0, 2.0])
    m = KickedMeanFieldMap(2, tau=0.3, J=0.0, U=1.5)
    assert np.abs(m.hop - np.eye(2)).max() == 0.0  # J = 0: hop is exactly I
    phi, occ = m.iterate((np.array([0.4, -0.2]), N), 5)
    assert np.abs(occ - N[None, :]).max() < 5e-15  # observed 1.33e-15
    expected = np.array([0.4, -0.2])[:, None] - 0.3 * 1.5 * N[:, None] * np.arange(6)[None, :]
    assert np.abs(_wrap(phi.T - expected)).max() < 1e-13  # observed 4.44e-16
    # the wrong sign does not match
    flipped = _wrap(phi.T + 2 * 0.3 * 1.5 * N[:, None] * np.arange(6)[None, :] - expected)
    assert np.abs(flipped).max() > 1.0  # observed 2.70


def test_blocked_and_dense_agree_on_ring_varying_parameters():
    """Symmetry blocking is exact across the drive: the pbc sector path and
    the symmetry-free dense path (explicit bond list) give identical sorted
    quasienergy multisets for several (tau, J, U)."""
    bonds = [[0, 1], [1, 2], [2, 0]]
    fock = FixedNFockSpace(6, 3)
    for tau, J, U in ((0.3, 1.0, 0.5), (0.7, 0.4, 1.1), (1.1, 1.6, -0.7)):
        eps_b, _ = KickedBoseHubbard(3, tau, J, U).floquet_spectrum(fock)
        eps_d, _ = KickedBoseHubbard(3, tau, J, U, connections=bonds).floquet_spectrum(fock)
        assert np.abs(eps_b - eps_d).max() < 1e-9  # observed up to 4.0e-13
