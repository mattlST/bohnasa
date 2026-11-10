import numpy as np
import pytest

import bohnasa


@pytest.fixture
def fock53():
    return bohnasa.fockStates_fixed_particle_number(5, 3)


def test_number_diagonal_is_occupation(fock53):
    for i in range(3):
        n_i = bohnasa.number(fock53, i)
        assert np.allclose(
            n_i.matrix.diagonal(), fock53.fockArray[:, i]
        )
        assert np.allclose(n_i.matrix.toarray(), n_i.matrix.toarray().T.conj())


def test_interaction_diagonal(fock53):
    u = bohnasa.interaction(fock53)
    expected = np.sum(fock53.fockArray * (fock53.fockArray - 1), axis=1)
    assert np.allclose(u.matrix.diagonal(), expected)


def test_chemical_mu_diagonal(fock53):
    rng = np.random.default_rng(3)
    mu = rng.normal(size=3)
    op = bohnasa.chemical_mu(fock53, mu)
    assert np.allclose(op.matrix.diagonal(), fock53.fockArray @ mu)


def test_correlator_is_annihilate_create(fock53):
    # correlator(i, j) = c_j^dag c_i : moves one particle i -> j
    i, j = 0, 2
    op = bohnasa.correlator(fock53, i, j)
    M = op.matrix.toarray()
    for index, state in enumerate(fock53.fockSpace):
        if state[i] == 0:
            continue
        occ = list(state)
        occ[i] -= 1
        occ[j] += 1
        target = fock53.dict[tuple(occ)]
        expected = np.sqrt(state[i] * (state[j] + 1))
        assert np.isclose(M[target, index], expected)
        assert np.count_nonzero(M[:, index]) == 1


def test_correlator_m_is_adjoint_of_correlator(fock53):
    i, j = 0, 1
    c = bohnasa.correlator(fock53, i, j).matrix.toarray()
    cm = bohnasa.correlator_m(fock53, i, j).matrix.toarray()
    # correlator_m(i, j) = c_i^dag c_j = (c_j^dag c_i)^dag
    assert np.allclose(cm, c.T.conj())
    # and by convention correlator_m(i, j) == correlator(j, i)
    assert np.allclose(cm, bohnasa.correlator(fock53, j, i).matrix.toarray())


@pytest.mark.parametrize("i,j,k,l", [(0, 1, 1, 2), (0, 1, 2, 0), (0, 2, 2, 0)])
def test_bosonic_commutator_of_correlators(fock53, i, j, k, l):
    # [c_j^dag c_i, c_l^dag c_k] = delta_il c_k^dag c_j - delta_jk c_i^dag c_l
    def A(p, q):
        return bohnasa.correlator(fock53, p, q).matrix.toarray()

    lhs = A(i, j) @ A(k, l) - A(k, l) @ A(i, j)
    rhs = (int(i == l)) * A(k, j) - (int(j == k)) * A(i, l)
    assert np.allclose(lhs, rhs)


def test_hopping_pbc_one_particle_spectrum():
    N, L = 1, 6
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    t = bohnasa.hopping(fock, connections="pbc").matrix.toarray()
    # ring: E_m = -2 cos(2 pi m / L), m = 0..L-1
    expected = np.sort(-2 * np.cos(2 * np.pi * np.arange(L) / L))
    assert np.allclose(np.sort(np.linalg.eigvalsh(t)), expected)
    assert np.allclose(t, t.T.conj())


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


def test_correlator_diagonal_equals_number():
    N, L = 3, 3
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    for i in range(L):
        expected = bohnasa.number(fock, i).matrix.toarray()
        assert np.allclose(bohnasa.correlator(fock, i, i).matrix.toarray(), expected)


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


def test_eval_mean_matches_vdot(fock53):
    rng = np.random.default_rng(5)
    psi = rng.normal(size=fock53.dim) + 1j * rng.normal(size=fock53.dim)
    psi /= np.linalg.norm(psi)
    op = bohnasa.number(fock53, 1)
    assert np.isclose(op.eval_mean(psi), np.vdot(psi, op.matrix @ psi))


def test_eval_mean_var_consistency(fock53):
    rng = np.random.default_rng(6)
    psi = rng.normal(size=fock53.dim) + 1j * rng.normal(size=fock53.dim)
    psi /= np.linalg.norm(psi)
    op = bohnasa.hopping(fock53)
    mean, var = op.eval_mean_var(psi)
    a2 = (op.matrix @ op.matrix) @ psi
    assert np.isclose(mean, np.vdot(psi, op.matrix @ psi))
    assert np.isclose(var, np.vdot(psi, a2) - mean**2)


def test_operator_addition(fock53):
    a = bohnasa.number(fock53, 0)
    b = bohnasa.number(fock53, 1)
    c = a + b
    assert np.allclose(
        c.matrix.toarray(),
        a.matrix.toarray() + b.matrix.toarray(),
    )


def test_operator_subtraction(fock53):
    a = bohnasa.number(fock53, 0)
    b = bohnasa.number(fock53, 1)
    c = a - b
    assert np.allclose(
        c.matrix.toarray(),
        a.matrix.toarray() - b.matrix.toarray(),
    )


def test_scalar_multiplication(fock53):
    a = bohnasa.number(fock53, 0)
    c = a * 2.5
    assert np.allclose(c.matrix.toarray(), 2.5 * a.matrix.toarray())
    # original untouched (deepcopy semantics)
    assert np.allclose(a.matrix.toarray(), a.matrix.toarray())
