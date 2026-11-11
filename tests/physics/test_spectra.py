"""Physics invariants: spectra against analytically known results."""

import numpy as np

import bohnasa


def test_hopping_pbc_one_particle_spectrum():
    N, L = 1, 6
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    t = bohnasa.hopping(fock, connections="pbc").matrix.toarray()
    # ring: E_m = -2 cos(2 pi m / L), m = 0..L-1
    expected = np.sort(-2 * np.cos(2 * np.pi * np.arange(L) / L))
    assert np.allclose(np.sort(np.linalg.eigvalsh(t)), expected)
    assert np.allclose(t, t.T.conj())


def test_hopping_obc_one_particle_spectrum():
    N, L = 1, 5
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    t = bohnasa.hopping(fock, connections="obc").matrix.toarray()
    # open chain: E_m = -2 cos(pi m / (L+1)), m = 1..L
    expected = np.sort(-2 * np.cos(np.pi * np.arange(1, L + 1) / (L + 1)))
    assert np.allclose(np.sort(np.linalg.eigvalsh(t)), expected)


def test_hopping_obc_equals_custom_bond_list():
    N, L = 1, 4
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    a = bohnasa.hopping(fock, connections="obc").matrix.toarray()
    b = bohnasa.hopping(fock, connections=[(0, 1), (1, 2), (2, 3)]).matrix.toarray()
    assert np.allclose(a, b)


def test_hopping_two_particles_bosonic_enhancement():
    # N=2 on a single bond couples (2,0) <-> (1,1) <-> (0,2) with amplitude
    # sqrt(n_i (n_j+1)) = sqrt(2); eigenvalues of that 3x3 chain are {-2,0,2}
    N, L = 2, 2
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    t = bohnasa.hopping(fock, connections=[(0, 1)]).matrix.toarray()

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
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    pbc = bohnasa.hopping(fock, connections="pbc").matrix.toarray()
    single = bohnasa.hopping(fock, connections=[(0, 1)]).matrix.toarray()
    assert np.allclose(pbc, 2 * single)
