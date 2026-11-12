"""Unit tests for the generic one-/two-body Hamiltonian builder (issue #17)."""

import numpy as np
import pytest
import scipy.sparse as sp

import bohnasa


def _adjacency(L, connections):
    """Plain site-graph adjacency for the 'pbc'/'obc' bond sets (L >= 3)."""
    A = np.zeros((L, L))
    bonds = range(L) if connections == "pbc" else range(L - 1)
    for i in bonds:
        j = (i + 1) % L if connections == "pbc" else i + 1
        A[i, j] += 1.0
        A[j, i] += 1.0
    return A


@pytest.fixture
def fock33():
    return bohnasa.FixedNFockSpace(2, 3)  # dim 6


def test_sandwich_bridge_pbc_and_obc():
    """The builder must reproduce the hand-assembled -Hopping + U*Interaction."""
    for N, L in [(2, 3), (3, 4), (4, 5), (6, 4)]:
        fock = bohnasa.FixedNFockSpace(N, L)
        for connections in ("pbc", "obc"):
            reference = bohnasa.Hopping(fock, connections=connections) * (-1.0)
            reference = reference + bohnasa.Interaction(fock) * 0.5
            built = bohnasa.Hamiltonian(fock, one_body=-_adjacency(L, connections), onsite_u=0.5)
            assert np.allclose(built.matrix.toarray(), reference.matrix.toarray())


def test_real_inputs_keep_float_dtype_complex_inputs_promote(fock33):
    real = bohnasa.Hamiltonian(fock33, one_body=-_adjacency(3, "pbc"), onsite_u=1.0)
    assert real.matrix.dtype == np.dtype("float64")
    complex_op = bohnasa.Hamiltonian(fock33, one_body=np.eye(3) * 1j)
    assert complex_op.matrix.dtype == np.dtype("complex128")


def test_diagonal_one_body_is_the_chemical_potential(fock33):
    mu = np.array([0.3, -1.2, 0.0])
    built = bohnasa.Hamiltonian(fock33, one_body=np.diag(mu))
    reference = bohnasa.ChemicalPotential(fock33, mu)
    assert np.allclose(built.matrix.toarray(), reference.matrix.toarray())


def test_single_entry_one_body_is_a_correlator(fock33):
    """E_ij feeds the coefficient of c^dag_i c_j - exactly Correlator(fock, j, i)."""
    for i, j in [(0, 2), (1, 0), (2, 1)]:
        E = np.zeros((3, 3))
        E[i, j] = 1.0
        built = bohnasa.Hamiltonian(fock33, one_body=E)
        reference = bohnasa.Correlator(fock33, j, i)
        assert np.allclose(built.matrix.toarray(), reference.matrix.toarray())


def test_onsite_u_matches_four_ladder_product(fock33):
    """V_iiii = 1 is c^dag_i c^dag_i c_i c_i = n_i (n_i - 1)."""
    for site in range(3):
        V = np.zeros((3, 3, 3, 3))
        V[site, site, site, site] = 1.0
        built = bohnasa.Hamiltonian(fock33, two_body=V)
        n_i = bohnasa.Number(fock33, site).matrix.toarray()
        assert np.allclose(built.matrix.toarray(), n_i @ (n_i - np.eye(fock33.dim)))


def test_cross_density_matches_number_product(fock33):
    """V_ijji = 1 is c^dag_i c^dag_j c_j c_i = n_i n_j (i != j)."""
    i, j = 0, 1
    V = np.zeros((3, 3, 3, 3))
    V[i, j, j, i] = 1.0
    built = bohnasa.Hamiltonian(fock33, two_body=V)
    n_i = bohnasa.Number(fock33, i).matrix.toarray()
    n_j = bohnasa.Number(fock33, j).matrix.toarray()
    assert np.allclose(built.matrix.toarray(), n_i @ n_j)


def test_pair_hopping_amplitude(fock33):
    """c^dag_0 c^dag_0 c_1 c_1 maps |0,2> to 2|2,0> - the sqrt(2)*sqrt(2) factor."""
    V = np.zeros((3, 3, 3, 3))
    V[0, 0, 1, 1] = 1.0
    built = bohnasa.Hamiltonian(fock33, two_body=V)
    row = fock33.dict[(2, 0, 0)]
    col = fock33.dict[(0, 2, 0)]
    assert built.matrix[row, col] == pytest.approx(2.0)
    assert abs(built.matrix.toarray()).max() == pytest.approx(2.0)


def test_onsite_u_scalar_vector_and_tensor_agree(fock33):
    u = np.array([1.0, 2.0, 3.0])
    V = np.zeros((3, 3, 3, 3))
    for site in range(3):
        V[site, site, site, site] = u[site]
    tensor_built = bohnasa.Hamiltonian(fock33, two_body=V)
    vector_built = bohnasa.Hamiltonian(fock33, onsite_u=u)
    scalar_built = bohnasa.Hamiltonian(fock33, onsite_u=1.0)
    assert np.allclose(tensor_built.matrix.toarray(), vector_built.matrix.toarray())
    assert np.allclose(scalar_built.matrix.toarray(), bohnasa.Interaction(fock33).matrix.toarray())


def test_empty_input_is_the_zero_operator(fock33):
    built = bohnasa.Hamiltonian(fock33)
    assert built.matrix.nnz == 0
    assert built.matrix.shape == (fock33.dim, fock33.dim)


def test_quartic_that_cannot_act_builds_the_empty_branch():
    """c^dag_0 c^dag_1 c_2 c_2 needs two bosons on site 2 - impossible at N = 1,
    so every input state drops out and the empty-matrix branch is taken."""
    fock1 = bohnasa.FixedNFockSpace(1, 3)
    V = np.zeros((3, 3, 3, 3))
    V[0, 1, 2, 2] = 1.0  # needs two bosons on site 2: impossible at N = 1
    built = bohnasa.Hamiltonian(fock1, two_body=V)
    assert built.matrix.nnz == 0


def test_composes_with_operator_dunders(fock33):
    onsite = bohnasa.Hamiltonian(fock33, onsite_u=1.0)
    diag = bohnasa.Hamiltonian(fock33, one_body=np.eye(3) * 0.5)
    total = 2 * onsite + diag
    assert np.allclose(total.matrix.toarray(), 2 * onsite.matrix.toarray() + diag.matrix.toarray())
    assert isinstance(total, bohnasa.Operator)


def test_wrong_shapes_are_rejected(fock33):
    with pytest.raises(ValueError, match="one_body must have shape"):
        bohnasa.Hamiltonian(fock33, one_body=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="two_body must have shape"):
        bohnasa.Hamiltonian(fock33, two_body=np.zeros((3, 3, 3)))
    with pytest.raises(ValueError, match="onsite_u"):
        bohnasa.Hamiltonian(fock33, onsite_u=np.zeros(4))


def test_two_body_and_onsite_u_are_mutually_exclusive(fock33):
    with pytest.raises(ValueError, match="not both"):
        bohnasa.Hamiltonian(fock33, two_body=np.zeros((3, 3, 3, 3)), onsite_u=1.0)


def test_hamiltonian_feeds_lanczos_and_sectors(fock33):
    """Hamiltonian is a plain Operator: everything downstream accepts it."""
    built = bohnasa.Hamiltonian(fock33, one_body=-_adjacency(3, "pbc"), onsite_u=0.5)
    seed = np.zeros(fock33.dim, dtype=complex)
    seed[0] = 1.0
    v, t_m = bohnasa.lanczos(seed, built, 5)
    assert t_m.shape[0] >= 2
    sectors = bohnasa.SymmetrySectors(fock33, "pbc")
    block = sectors.restrict(built, sectors.labels[0])
    assert block.shape[0] > 0
    assert sp.issparse(block)
