"""Physics invariants of the symmetry-sector decomposition.

The defining identity: blocks of a symmetry-commuting Hamiltonian reassemble
the full spectrum exactly when concatenated. Includes the composite-``L``
cases where sub-periodic (short-orbit) basis states must be routed to the
momenta of their little group - the bug class that killed the reference
thesis implementation (issue #13).
"""

import numpy as np
import pytest
from scipy.linalg import expm

import bohnasa

PBC_GRID = [(6, 4), (4, 6), (6, 5), (5, 6), (4, 8), (8, 4)]
OBC_GRID = [(2, 3), (3, 5), (4, 5), (3, 6)]


def _hamiltonian(fock, connections):
    return (
        bohnasa.Hopping(fock, connections=connections) * (-1.0) + bohnasa.Interaction(fock) * 0.25
    )


def _sector_spectrum(sectors, ham):
    return np.sort(
        np.concatenate(
            [np.linalg.eigvalsh(sectors.restrict(ham, label).toarray()) for label in sectors.labels]
        )
    )


@pytest.mark.parametrize("N,L", PBC_GRID)
def test_sector_spectra_reassemble_full_spectrum(N, L):
    fock = bohnasa.FixedNFockSpace(N, L)
    ham = _hamiltonian(fock, "pbc")
    sectors = bohnasa.SymmetrySectors(fock, connections="pbc")
    assert np.allclose(
        _sector_spectrum(sectors, ham),
        np.linalg.eigvalsh(ham.matrix.toarray()),
        atol=1e-10,
    )


@pytest.mark.parametrize("N,L", OBC_GRID)
def test_obc_parity_spectra_reassemble_full_spectrum(N, L):
    fock = bohnasa.FixedNFockSpace(N, L)
    ham = _hamiltonian(fock, "obc")
    sectors = bohnasa.SymmetrySectors(fock, connections="obc")
    assert np.allclose(
        _sector_spectrum(sectors, ham),
        np.linalg.eigvalsh(ham.matrix.toarray()),
        atol=1e-10,
    )


def test_short_orbit_little_group_routing_n4_l6():
    """Regression guard: the exact case the thesis implementation failed.

    N=4, L=6 has 20 free orbits plus 2 sub-periodic ones ((2,0,0,2,0,0)-type,
    period 3). Little-group rule: a period-3 orbit supports only the momenta
    k in {0, +-2} - never k=3. The reference implementation dropped them from
    k=+-2 and miscounted them into the k=3 sector (two states lost overall).
    """
    fock = bohnasa.FixedNFockSpace(4, 6)
    sectors = bohnasa.SymmetrySectors(fock, connections="pbc")
    dims = {label: sectors.dim(label) for label in sectors.labels}
    for label, dim in dims.items():
        if abs(label.k) == 2:
            assert dim == 22, f"k={label.k}: short orbits missing from sector"
        if abs(label.k) == 1:
            assert dim == 20, f"k={label.k}: short orbits leaked into free-orbit sector"
    l_half = sum(dim for label, dim in dims.items() if abs(label.k) == 3)
    assert l_half == 20, "short orbits misplaced into the L/2 sector"
    ham = _hamiltonian(fock, "pbc")
    assert np.allclose(
        _sector_spectrum(sectors, ham),
        np.linalg.eigvalsh(ham.matrix.toarray()),
        atol=1e-10,
    )


def test_lanczos_chain_identical_in_sector():
    """A sector-resident start state recurses identically full-space vs block."""
    fock = bohnasa.FixedNFockSpace(6, 4)
    ham = _hamiltonian(fock, "pbc")
    sectors = bohnasa.SymmetrySectors(fock, connections="pbc")

    psi = np.ones(fock.dim, dtype=complex) / np.sqrt(fock.dim)  # translation+mirror fixed
    label = bohnasa.SectorLabel(0, 1)
    block = sectors.restrict(ham, label)
    psi_s = sectors.to_sector(psi, label)
    assert np.allclose(sectors.isometry(label) @ psi_s, psi)  # psi lives in the sector

    v_sector, t_sector = bohnasa.lanczos(psi_s, block.dot, 10)
    v_full, t_full = bohnasa.lanczos(psi, ham, 10)
    np.testing.assert_allclose(t_sector, t_full, atol=1e-12)
    np.testing.assert_allclose(sectors.isometry(label) @ v_sector.T, v_full.T, atol=1e-12)


def test_krylov_complexity_agrees_in_sector():
    """<K(t)> from the (0,+) block equals the full-space run to machine precision."""
    fock = bohnasa.FixedNFockSpace(6, 4)
    ham = _hamiltonian(fock, "pbc")
    sectors = bohnasa.SymmetrySectors(fock, connections="pbc")

    psi = np.ones(fock.dim, dtype=complex) / np.sqrt(fock.dim)
    label = bohnasa.SectorLabel(0, 1)
    block = sectors.restrict(ham, label)
    psi_s = sectors.to_sector(psi, label)

    num_vec = 10
    t = 1.3
    k_full = bohnasa.krylov_complexity(fock, ham, psi, num_vec).matrix.toarray()
    # `fock` only wraps the result of krylov_complexity; the matrix itself is
    # built from the start vector's space, so a sector-space run is consistent
    k_sector = bohnasa.krylov_complexity(fock, block.dot, psi_s, num_vec).matrix.toarray()

    psi_t = expm(-1j * t * ham.matrix.toarray()) @ psi
    psi_s_t = expm(-1j * t * block.toarray()) @ psi_s
    k_t_full = np.vdot(psi_t, k_full @ psi_t)
    k_t_sector = np.vdot(psi_s_t, k_sector @ psi_s_t)
    assert abs(k_t_full) == pytest.approx(abs(k_t_sector), abs=1e-10)
    assert k_t_full == pytest.approx(k_t_sector, abs=1e-10)
