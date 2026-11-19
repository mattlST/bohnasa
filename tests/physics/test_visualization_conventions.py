"""Physics conventions of the visualization layer (issue #40).

Sign conventions are stated in ``bohnasa/visualization.py`` and pinned here
by derivations and invariants - not value snapshots:

1. uniform (k = 0) plane wave -> zero bond current on every edge;
2. winding plane wave -> the analytic uniform circulation per bond;
3. continuity equation: ``d<n_i>/dt = i <[H, n_i]>`` equals the divergence
   of the bond currents ``J_i->j = t * j_ij`` built from the very bond list
   the Hamiltonian uses (independent Heisenberg-picture cross-check).
"""

import numpy as np
import pytest

import bohnasa
from bohnasa import visualization as viz


def plane_wave_gamma(N: int, L: int, m: int) -> np.ndarray:
    """One-body density matrix of the projected coherent winding state."""
    fock = bohnasa.FixedNFockSpace(N, L)
    phi = np.exp(2j * np.pi * m * np.arange(L) / L) / np.sqrt(L)
    state = bohnasa.fixed_n_coherent_state(fock, phi)
    return bohnasa.one_body_density_matrix(state, fock)


@pytest.mark.parametrize("connections", ["pbc", "obc"])
def test_uniform_plane_wave_has_zero_bond_current_everywhere(connections):
    N, L, m = 3, 6, 0
    gamma = plane_wave_gamma(N, L, m)
    bonds = viz.lattice_bonds(L, connections)
    current = viz.bond_current(gamma, bonds)
    assert np.all(np.abs(current) < 1e-14)  # exactly zero: gamma is real here


@pytest.mark.parametrize("m", [1, 2])
def test_winding_plane_wave_circulates_with_the_analytic_value(m):
    N, L = 3, 6
    gamma = plane_wave_gamma(N, L, m)
    bonds = viz.lattice_bonds(L, "pbc")
    current = viz.bond_current(gamma, bonds)
    # gamma_ij = (N/L) exp(i m (j - i)) -> j_ij = 2 (N/L) sin(m (j - i))
    expected = 2.0 * (N / L) * np.sin(2.0 * np.pi * m * (bonds[:, 1] - bonds[:, 0]) / L)
    assert np.allclose(current, expected, atol=1e-14)
    assert np.allclose(np.abs(current), np.abs(expected).max(), atol=1e-14)  # uniform ring flow


def test_reversed_bond_flips_current_keeps_coherence():
    rng = np.random.default_rng(11)
    gamma = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    gamma = gamma + gamma.conj().T  # Hermitian
    forward = np.array([[1, 3]])
    backward = np.array([[3, 1]])
    assert np.allclose(viz.bond_current(gamma, forward), -viz.bond_current(gamma, backward))
    assert np.allclose(viz.bond_coherence(gamma, forward), viz.bond_coherence(gamma, backward))


def test_continuity_equation_matches_bond_current_divergence():
    """d<n_i>/dt = i<[H, n_i]> must equal sum_j (J_j->i - J_i->j), t = 1."""
    N, L = 3, 4
    fock = bohnasa.FixedNFockSpace(N, L)
    H = bohnasa.Hopping(fock, connections="pbc") * (-1.0)  # t = 1
    rng = np.random.default_rng(3)
    state = rng.normal(size=fock.dim) + 1j * rng.normal(size=fock.dim)
    state /= np.linalg.norm(state)

    h_mat = H.matrix.toarray()
    dn_dt = np.empty(L)
    for i in range(L):
        n_i = np.diag(fock.fockArray[:, i].astype(float))
        dn_dt[i] = (1j * ((h_mat @ n_i - n_i @ h_mat) @ state) @ state.conj()).real

    gamma = bohnasa.one_body_density_matrix(state, fock)
    bonds = H.connections
    current = viz.bond_current(gamma, bonds)  # J_i->j = t * j_ij with t = 1
    divergence = np.zeros(L)
    for (i, j), j_ij in zip(bonds, current, strict=True):
        divergence[i] -= j_ij
        divergence[j] += j_ij

    assert np.abs(dn_dt - divergence).max() < 1e-12


def test_l2_pbc_double_bond_carries_equal_and_opposite_currents():
    """Both entries of the doubled L = 2 ring bond see the same physical bond."""
    fock = bohnasa.FixedNFockSpace(1, 2)
    state = bohnasa.fixed_n_coherent_state(fock, np.array([1.0, 1j]))
    gamma = bohnasa.one_body_density_matrix(state, fock)
    bonds = viz.lattice_bonds(2, "pbc")
    assert bonds.tolist() == [[0, 1], [1, 0]]
    current = viz.bond_current(gamma, bonds)
    # phase drop pi/2 along the bond: j_(0,1) = 2 Im conj(phi_0) phi_1 = 1
    assert current[0] == pytest.approx(1.0, abs=1e-14)
    assert current[0] == pytest.approx(-current[1])
