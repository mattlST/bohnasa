"""Physics invariants: spectra against analytically known results."""

import numpy as np
import pytest

import bohnasa


def test_hopping_pbc_one_particle_spectrum():
    N, L = 1, 6
    fock = bohnasa.FixedNFockSpace(N, L)
    t = bohnasa.Hopping(fock, connections="pbc").matrix.toarray()
    # ring: E_m = -2 cos(2 pi m / L), m = 0..L-1
    expected = np.sort(-2 * np.cos(2 * np.pi * np.arange(L) / L))
    assert np.allclose(np.sort(np.linalg.eigvalsh(t)), expected)
    assert np.allclose(t, t.T.conj())


def test_hopping_obc_one_particle_spectrum():
    N, L = 1, 5
    fock = bohnasa.FixedNFockSpace(N, L)
    t = bohnasa.Hopping(fock, connections="obc").matrix.toarray()
    # open chain: E_m = -2 cos(pi m / (L+1)), m = 1..L
    expected = np.sort(-2 * np.cos(np.pi * np.arange(1, L + 1) / (L + 1)))
    assert np.allclose(np.sort(np.linalg.eigvalsh(t)), expected)


def test_hopping_obc_equals_custom_bond_list():
    N, L = 1, 4
    fock = bohnasa.FixedNFockSpace(N, L)
    a = bohnasa.Hopping(fock, connections="obc").matrix.toarray()
    b = bohnasa.Hopping(fock, connections=[(0, 1), (1, 2), (2, 3)]).matrix.toarray()
    assert np.allclose(a, b)


@pytest.mark.parametrize("connections", ["pbc", "obc"])
def test_hopping_with_raw_ndarray_bond_list_is_bit_equal_to_keyword(connections):
    """Regression (issue #40): Hopping(connections=<ndarray from bond_pairs>)
    used to raise on the elementwise ``connections == "pbc"`` comparison; it
    must now assemble the identical matrix as the string keyword."""
    N, L = 2, 4
    fock = bohnasa.FixedNFockSpace(N, L)
    bond_list = bohnasa.bond_pairs(L, connections)
    string = bohnasa.Hopping(fock, connections=connections).matrix
    given = bohnasa.Hopping(fock, connections=bond_list).matrix
    assert np.array_equal(string.toarray(), given.toarray())


def test_hopping_two_particles_bosonic_enhancement():
    # N=2 on a single bond couples (2,0) <-> (1,1) <-> (0,2) with amplitude
    # sqrt(n_i (n_j+1)) = sqrt(2); eigenvalues of that 3x3 chain are {-2,0,2}
    N, L = 2, 2
    fock = bohnasa.FixedNFockSpace(N, L)
    t = bohnasa.Hopping(fock, connections=[(0, 1)]).matrix.toarray()

    i20, i11, i02 = (fock.dict[(2, 0)], fock.dict[(1, 1)], fock.dict[(0, 2)])
    assert np.isclose(t[i11, i20], np.sqrt(2))
    assert np.isclose(t[i02, i11], np.sqrt(2))
    assert np.isclose(t[i20, i11], np.sqrt(2))  # hermitian
    assert np.allclose(np.sort(np.linalg.eigvalsh(t)), [-2.0, 0.0, 2.0])


def test_pbc_L2_double_counts_bonds_known_quirk():
    # For L=2 the 'pbc' bond list zip(range(L), roll(range(L),1)) yields
    # (0,1) AND (1,0) - the same undirected bond twice - so the resulting
    # matrix is 2x the single-bond hopping. Pinned as known behavior:
    # decide separately whether pbc should de-duplicate bonds.
    N, L = 1, 2
    fock = bohnasa.FixedNFockSpace(N, L)
    pbc = bohnasa.Hopping(fock, connections="pbc").matrix.toarray()
    single = bohnasa.Hopping(fock, connections=[(0, 1)]).matrix.toarray()
    assert np.allclose(pbc, 2 * single)
