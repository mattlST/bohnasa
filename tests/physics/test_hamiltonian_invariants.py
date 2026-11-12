"""Physics invariants of the Hamiltonian builder (issue #17).

Ground truth from three independent directions: analytic single-particle
spectra (flux ring), a hand-assembled few-level matrix (pair-hopping dimer),
and exact structural limits the builder must reproduce (single-particle
reduction at N = 1, Hermiticity transmission).
"""

import numpy as np
import pytest

import bohnasa


def test_one_body_reduces_to_single_particle_matrix():
    """At N = 1 the fixed-N representation of sum t_ij c†_i c_j must be the
    coefficient matrix itself (basis-reordered); every quartic monomial needs
    two annihilations and vanishes identically."""
    rng = np.random.default_rng(11)
    L = 4
    t = rng.normal(size=(L, L)) + 1j * rng.normal(size=(L, L))
    V = rng.normal(size=(L, L, L, L)) + 1j * rng.normal(size=(L, L, L, L))
    fock = bohnasa.FixedNFockSpace(1, L)
    H = bohnasa.Hamiltonian(fock, one_body=t, two_body=V)
    m = H.matrix.toarray()
    for i in range(L):
        for j in range(L):
            row = fock.dict[tuple(int(site == i) for site in range(L))]
            col = fock.dict[tuple(int(site == j) for site in range(L))]
            assert m[row, col] == pytest.approx(t[i, j])


def test_purely_diagonal_coefficients_give_diagonal_matrix():
    """V_ijkl with i == j == k == l (and diagonal one_body) is built purely
    from number operators - no off-diagonal matrix elements may appear."""
    fock = bohnasa.FixedNFockSpace(3, 4)
    rng = np.random.default_rng(5)
    u = rng.normal(size=4)
    mu = rng.normal(size=4)
    H = bohnasa.Hamiltonian(fock, one_body=np.diag(mu), onsite_u=u)
    m = H.matrix.toarray()
    off = m - np.diag(np.diag(m))
    assert np.abs(off).max() == 0.0
    occ = fock.fockArray
    assert np.allclose(np.diag(m), occ @ mu + np.sum(u[None, :] * occ * (occ - 1), axis=1))


def test_hermitian_input_transmits_hermitian_operator():
    L = 5
    fock = bohnasa.FixedNFockSpace(2, L)
    rng = np.random.default_rng(3)
    t = rng.normal(size=(L, L)) + 1j * rng.normal(size=(L, L))
    t = (t + t.conj().T) / 2
    V = rng.normal(size=(L, L, L, L))
    # (c†_i c†_j c_k c_ell)† = c†_ell c†_k c_j c_i, so V + V.transpose(3, 2, 1, 0).conj()
    # is a Hermitian coefficient array
    V = V + V.transpose(3, 2, 1, 0).conj()
    H = bohnasa.Hamiltonian(fock, one_body=t, two_body=V)
    m = H.matrix.toarray()
    assert np.allclose(m, m.conj().T)


def test_non_reciprocal_ring_reproduces_harpe_gauge_spectrum():
    """Peierls-substituted ring at N = 1: eps(k) = -2|t| cos(k + phi/L)."""
    L, phi, t_amp = 6, 0.7, 1.3
    fock = bohnasa.FixedNFockSpace(1, L)
    t = np.zeros((L, L), dtype=complex)
    bond = -t_amp * np.exp(1j * phi / L)
    for site in range(L):
        nxt = (site + 1) % L
        t[nxt, site] = bond
        t[site, nxt] = bond.conj()
    H = bohnasa.Hamiltonian(fock, one_body=t)
    energies = np.sort(np.linalg.eigvalsh(H.matrix.toarray()))
    k = 2 * np.pi * np.arange(L) / L
    analytic = np.sort(-2 * t_amp * np.cos(k + phi / L))
    assert np.abs(energies - analytic).max() < 1e-12


def test_directed_ring_is_the_non_hermitian_passthrough():
    """One-directional hopping: H stays non-Hermitian, eigenvalues are the
    momentum quantized -exp(ik) - unit modulus, traceless."""
    L = 6
    fock = bohnasa.FixedNFockSpace(1, L)
    t = np.zeros((L, L))
    for site in range(L):
        t[(site + 1) % L, site] = -1.0
    H = bohnasa.Hamiltonian(fock, one_body=t)
    m = H.matrix.toarray()
    assert not np.allclose(m, m.conj().T)
    eig = np.linalg.eigvals(m)
    assert np.allclose(np.abs(eig), 1.0, atol=1e-12)
    assert abs(eig.sum()) < 1e-12


def test_pair_hopping_dimer_matches_hand_assembled_matrix():
    """Two-site dimer, N = 2: -t hopping + on-site u + pair hopping g
    (c†_0 c†_0 c_1 c_1 + h.c.). Independent 3x3 assembly in the exact basis
    order - all physics amplitudes (sqrt(2) hop, 2g pair, 2u on-site)
    hand-placed."""
    td, u, g = 1.0, 0.4, 0.25
    fock = bohnasa.FixedNFockSpace(2, 2)
    assert fock.fockSpace == [(0, 2), (1, 1), (2, 0)]
    state_02, state_11, state_20 = range(3)

    V = np.zeros((2, 2, 2, 2))
    V[0, 0, 1, 1] = g
    V[1, 1, 0, 0] = g
    t = np.array([[0.0, -td], [-td, 0.0]])
    H = bohnasa.Hamiltonian(fock, one_body=t, two_body=V) + bohnasa.Hamiltonian(fock, onsite_u=u)
    m = H.matrix.toarray()

    hop = -td * np.sqrt(2.0)
    hand = np.zeros((3, 3))
    hand[state_20, state_11] = hand[state_11, state_20] = hop
    hand[state_11, state_02] = hand[state_02, state_11] = hop
    hand[state_20, state_20] = 2 * u
    hand[state_02, state_02] = 2 * u
    hand[state_20, state_02] = hand[state_02, state_20] = 2 * g
    assert np.allclose(m, hand, atol=1e-12)

    # and the spectrum of the hand matrix is independently checkable
    exact = np.sort(np.linalg.eigvalsh(hand))
    assert np.allclose(np.sort(np.linalg.eigvalsh(m)), exact)
