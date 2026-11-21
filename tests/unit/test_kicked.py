"""Unit tests for the kicked Bose-Hubbard module (issue #29).

Validation errors, the pinned factor-order / bond / branch conventions of the
public API, the classical map's exact sub-step algebra, and the consistency
between propagator, single-particle matrix and sector-blocked spectrum.
"""

import numpy as np
import pytest
import scipy.linalg as sla

from bohnasa import (
    FixedNFockSpace,
    Hamiltonian,
    KickedBoseHubbard,
    KickedMeanFieldMap,
)


@pytest.mark.parametrize("cls", [KickedBoseHubbard, KickedMeanFieldMap])
def test_constructor_validation(cls):
    ok = dict(L=3, tau=0.5, J=1.0, U=0.7)
    cls(**ok)  # must not raise
    with pytest.raises(TypeError, match="L must be an int"):
        cls(**{**ok, "L": 3.0})
    with pytest.raises(TypeError, match="L must be an int"):
        cls(**{**ok, "L": True})
    with pytest.raises(ValueError, match="L must be at least 2"):
        cls(**{**ok, "L": 1})
    with pytest.raises(ValueError, match="tau must be positive"):
        cls(**{**ok, "tau": 0.0})
    with pytest.raises(ValueError, match="tau must be positive"):
        cls(**{**ok, "tau": -0.5})
    with pytest.raises(ValueError, match="tau must be positive"):
        cls(**{**ok, "tau": np.nan})
    with pytest.raises(ValueError, match="J and U must be finite"):
        cls(**{**ok, "J": np.inf})
    with pytest.raises(ValueError, match="J and U must be finite"):
        cls(**{**ok, "U": np.nan})
    # malformed explicit bonds (delegated to bond_pairs)
    with pytest.raises(ValueError, match="connections"):
        cls(**{**ok, "connections": [1, 2, 3]})


@pytest.mark.parametrize("caller", ["propagator", "floquet_spectrum"])
def test_fock_validation(caller):
    k = KickedBoseHubbard(L=3, tau=0.5, J=1.0, U=0.7)
    with pytest.raises(TypeError, match="fock must be a FixedNFockSpace"):
        getattr(k, caller)(None)
    with pytest.raises(ValueError, match="fock.L"):
        getattr(k, caller)(FixedNFockSpace(4, 4))


def test_propagator_equals_hand_assembled_step_and_order_matters():
    """The convention U_step = exp(-i tau H_J) exp(-i tau H_U) - the ex10
    hand-assembly - is what propagator returns, and the reversed order is a
    genuinely different operator (so pinning the order is meaningful)."""
    N, theta, tau = 30, 1.35, 0.4
    J, U = np.cos(theta), 2.0 * np.sin(theta) / N
    k = KickedBoseHubbard(L=2, tau=tau, J=J, U=U)
    fock = FixedNFockSpace(N, 2)
    h_hop = Hamiltonian(fock, one_body=np.array([[0.0, -2 * J], [-2 * J, 0.0]])).matrix.toarray()
    h_int = Hamiltonian(fock, onsite_u=U / 2.0).matrix.toarray()
    u_hand = sla.expm(-1j * tau * h_hop) @ sla.expm(-1j * tau * h_int)
    u_lib = k.propagator(fock)
    assert np.abs(u_lib - u_hand).max() == pytest.approx(0.0, abs=1e-13)
    u_rev = sla.expm(-1j * tau * h_int) @ sla.expm(-1j * tau * h_hop)
    order_gap = np.abs(u_lib - u_rev).max()
    assert order_gap > 0.5  # observed 0.6347; the factors do not commute here


def test_n1_propagator_is_free_single_particle_expm():
    """At N = 1 the on-site kick acts trivially (normal ordering), so the
    step must equal exp(-i tau A) of the single-particle matrix - this pins
    the propagator <-> bond-matrix dictionary including the L = 2 double bond."""
    tau, J, U = 0.55, 1.4, 2.2
    for L, double in ((3, False), (2, True)):
        k = KickedBoseHubbard(L=L, tau=tau, J=J, U=U)
        fock = FixedNFockSpace(1, L)
        A = np.zeros((L, L))
        for i in range(L):
            A[(i + 1) % L, i] = A[i, (i + 1) % L] = -J * (2 if double else 1)
        assert np.abs(k.propagator(fock) - sla.expm(-1j * tau * A)).max() < 1e-13  # obs 0.0


def test_map_hop_matches_fock_sector_at_N1_and_obc_has_no_double_bond():
    """KickedMeanFieldMap.hop is exp(-i tau A) of the SAME bond matrix the
    quantum builder uses: at L = 2 the PBC ring rotates at 2J, the OBC chain
    at J (single counted bond)."""
    tau, J = 0.3, 1.1
    m_pbc = KickedMeanFieldMap(2, tau, J, U=0.4, connections="pbc")
    m_obc = KickedMeanFieldMap(2, tau, J, U=0.4, connections="obc")
    for amp, m in ((2.0 * J, m_pbc), (J, m_obc)):
        c, s = np.cos(amp * tau), np.sin(amp * tau)
        # A = -amp * sigma_x -> exp(-i tau A) = cos + i sin(sigma_x)
        exact = np.array([[c, 1j * s], [1j * s, c]])
        assert np.abs(m.hop - exact).max() < 1e-13  # observed 1.2e-16


def test_map_hop_unitary_equals_n1_propagator():
    """The map's hopping sub-step is the SAME unitary as the quantum N = 1
    step (the normal-ordered on-site kick is trivial there, and A is shared
    with the second-quantized builder)."""
    k = KickedBoseHubbard(L=4, tau=0.45, J=1.0, U=0.9)
    fock = FixedNFockSpace(1, 4)
    u = k.propagator(fock)
    perm = np.array([fock.dict[tuple(np.eye(fock.L, dtype=int)[i])] for i in range(fock.L)])
    hop = k.classical_map().hop
    assert np.abs(u[np.ix_(perm, perm)] - hop).max() < 1e-12  # observed 4.4e-16
    assert np.abs(hop.conj().T @ hop - np.eye(fock.L)).max() < 1e-14  # observed 1.1e-15


def test_iterate_validation():
    m = KickedMeanFieldMap(3, tau=0.4, J=1.0, U=0.6)
    with pytest.raises(TypeError, match="n_steps must be an int"):
        m.iterate(np.ones(3), 1.0)
    with pytest.raises(TypeError, match="n_steps must be an int"):
        m.iterate(np.ones(3), True)
    with pytest.raises(ValueError, match="n_steps must be >= 0"):
        m.iterate(np.ones(3), -1)
    with pytest.raises(ValueError, match="alpha must have shape"):
        m.iterate(np.ones(4), 2)
    with pytest.raises(ValueError, match="non-finite"):
        m.iterate(np.array([np.inf, 0.0, 0.0]) + 0j, 2)
    with pytest.raises(ValueError, match="zero state"):
        m.iterate(np.zeros(3) + 0j, 2)
    with pytest.raises(ValueError, match="length 2"):
        m.iterate((np.ones(3), np.ones(3), np.ones(3)), 2)
    with pytest.raises(ValueError, match="shape"):
        m.iterate((np.ones(2), np.ones(2)), 2)
    with pytest.raises(ValueError, match="occ >= 0"):
        m.iterate((np.zeros(3), np.array([1.0, -2.0, 1.0])), 2)
    with pytest.raises(ValueError, match="must be finite, occ >= 0"):
        m.iterate((np.array([np.nan, 0.0, 0.0]), np.ones(3)), 2)


def test_step_validation():
    m = KickedMeanFieldMap(2, tau=0.4, J=1.0, U=0.6)
    with pytest.raises(ValueError, match="alpha must have shape"):
        m.step(np.ones(3))
    with pytest.raises(ValueError, match="non-finite"):
        m.step(np.array([np.nan + 0j, 1.0 + 0j]))


def test_dimer_variables_validation_and_wrap():
    with pytest.raises(ValueError, match="requires L = 2"):
        KickedMeanFieldMap(3, tau=0.1, J=1.0, U=0.5).dimer_variables(
            np.zeros((2, 3)), np.ones((2, 3))
        )
    m = KickedMeanFieldMap(2, tau=0.1, J=1.0, U=0.5)
    with pytest.raises(ValueError, match="shape"):
        m.dimer_variables(np.zeros((2, 2)), np.ones((3, 2)))
    # wrap into (-pi, pi]: phi_rel = -pi lands on +pi
    s, phi = m.dimer_variables(
        np.array([[0.0, 0.0], [0.0, -np.pi]]), np.array([[1.0, 1.0], [1.0, 1.0]])
    )
    assert np.allclose(s, 0.0)
    assert phi.tolist() == [pytest.approx(0.0), pytest.approx(np.pi)]


def test_iterate_initial_conventions_agree():
    """The (phi, occ) pair and the complex-amplitude initial are the same
    state and give bit-identical trajectories."""
    m = KickedMeanFieldMap(3, tau=0.4, J=1.0, U=0.6)
    phi0 = np.array([0.3, -1.1, 2.2])
    occ0 = np.array([3.0, 2.0, 5.0])
    a0 = np.sqrt(occ0) * np.exp(1j * phi0)
    pa, oa = m.iterate((phi0, occ0), 7)
    pb, ob = m.iterate(a0, 7)
    assert np.array_equal(pa, pb) and np.array_equal(oa, ob)  # sqrt(.)**2 == . exact here
    assert oa.shape == (8, 3)


def test_map_global_phase_gauge():
    """Multiplying the initial amplitudes by exp(i delta) shifts every site
    phase identically and leaves all occupations fixed - the gauge freedom
    the module docstring pins by dropping the kick's common exp(+i tau U)."""
    m = KickedMeanFieldMap(3, tau=0.4, J=1.0, U=0.6)
    delta = 0.77
    a0 = np.array([2.0, 1.0 + 1.0j, -1.0])
    p1, o1 = m.iterate(a0, 6)
    p2, o2 = m.iterate(a0 * np.exp(1j * delta), 6)
    assert np.abs(o2 - o1).max() < 1e-13
    shifted = (p1 + delta + np.pi) % (2 * np.pi) - np.pi
    assert np.abs((p2 - shifted + np.pi) % (2 * np.pi) - np.pi).max() < 1e-13


def test_floquet_spectrum_branch_range_and_eigen_relation():
    """All three spectrum paths (pbc blocked, obc blocked, explicit-bonds
    dense fallback) return eps in [0, 2 pi / tau) with matching eigenvectors."""
    for connections in ("pbc", "obc", [[0, 1], [1, 2], [2, 0], [0, 2]]):
        k = KickedBoseHubbard(L=3, tau=0.7, J=1.0, U=0.6, connections=connections)
        fock = FixedNFockSpace(6, 3)
        u = k.propagator(fock)
        eps, vecs = k.floquet_spectrum(fock)
        assert eps.shape == (fock.dim,)
        assert np.all(np.diff(eps) >= 0.0)
        assert eps.min() >= 0.0 and eps.max() < 2 * np.pi / k.tau
        assert np.abs(u @ vecs - vecs * np.exp(-1j * eps * k.tau)).max() < 1e-9
        assert np.abs(vecs.conj().T @ vecs - np.eye(fock.dim)).max() < 1e-9


def test_blocked_and_dense_spectra_agree():
    """The sector path is exact: on a small trimer the sorted quasienergies
    match the dense fallback."""
    k_block = KickedBoseHubbard(L=3, tau=0.7, J=1.0, U=0.6)
    k_dense = KickedBoseHubbard(L=3, tau=0.7, J=1.0, U=0.6, connections=[[0, 1], [1, 2], [2, 0]])
    fock = FixedNFockSpace(8, 3)
    assert fock.dim == 45
    eps_b, _ = k_block.floquet_spectrum(fock)
    eps_d, _ = k_dense.floquet_spectrum(fock)
    assert np.abs(eps_b - eps_d).max() < 1e-10  # observed 2.66e-15 at the same size


def test_zone_edge_labels_as_zero():
    """Regression (caught by the ex13 sweep): U = 0 at tau = 0.7, L = 3,
    N = 6 carries an exactly degenerate eigenvalue cluster at lambda = 1
    (five free states with E = 0); LAPACK scatters them at sub-ulp angles on
    either side of the real axis and ``mod`` folds the negative ones to
    exactly 2 pi - the pre-fix maximum was 8.975979010256552 = 2 pi / tau.
    The half-open convention relabels those to eps = 0, so every
    quasienergy is strictly inside [0, 2 pi / tau)."""
    fock = FixedNFockSpace(6, 3)
    fold = 2 * np.pi / 0.7
    for u_val in (0.0, 0.3):
        eps, _ = KickedBoseHubbard(L=3, tau=0.7, J=1.0, U=u_val).floquet_spectrum(fock)
        assert eps.max() < fold  # the regression: pre-fix, U = 0 hit 2 pi / tau
        assert eps.min() >= 0.0
    eps, _ = KickedBoseHubbard(L=3, tau=0.7, J=1.0, U=0.0).floquet_spectrum(fock)
    # the five free E = 0 states land as zone-edge labels near 0
    assert int(np.sum(eps < 1e-12)) == 5  # observed 4 exact zeros + one at +ulp


def test_free_boson_quasienergy_ladder():
    """U = 0: the fixed-N step is the free one-particle band sum, so the
    quasienergies are the Floquet-zone folds mod(2 pi) of
    ``E = sum_k m_k (-2 J cos(2 pi k / L))`` over the momentum multisets -
    an analytic pin for the branch convention that exercises the cut
    (with J = 3, tau = 0.5 the two-boson ground energy E = -12 sits at
    ``tau E = -6 rad``, outside the principal branch)."""
    from itertools import combinations_with_replacement

    L, N, tau, J = 3, 2, 0.5, 3.0
    k = KickedBoseHubbard(L=L, tau=tau, J=J, U=0.0)
    fock = FixedNFockSpace(N, L)
    band = -2.0 * J * np.cos(2.0 * np.pi * np.arange(L) / L)
    energies = [
        float(np.sum(band[list(multiset)]))
        for multiset in combinations_with_replacement(range(L), N)
    ]
    assert len(energies) == fock.dim
    expected = np.sort(np.mod(np.array(energies) * tau, 2.0 * np.pi) / tau)
    eps, _ = k.floquet_spectrum(fock)
    assert np.abs(eps - expected).max() < 1e-10  # observed 5.33e-15
