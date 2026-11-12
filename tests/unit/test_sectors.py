"""Unit tests for the symmetry-sector decomposition.

Sector dimensions are checked against irrep-projector traces computed
independently by brute-force fixed-state counting - the definition, not a
second implementation of the same construction.
"""

import numpy as np
import pytest

import bohnasa

GRID = [(6, 4), (4, 6), (6, 5), (5, 6), (4, 8), (8, 4), (2, 3), (1, 2)]


def _translation_fixed_count(fock, tau):
    """#basis states invariant under a translation by ``tau`` sites."""
    return sum(1 for state in fock.fockSpace if tuple(np.roll(state, tau)) == state)


def _reflection_fixed_count(fock, tau):
    """#basis states invariant under translation ``tau`` then reflection."""
    return sum(1 for state in fock.fockSpace if tuple(reversed(np.roll(state, tau))) == state)


def _expected_dim(fock, k, parity):
    """tr P_(k,parity) of the irrep projector over the dihedral group."""
    L = fock.L
    if parity is None:
        trace = sum(
            np.exp(-2j * np.pi * k * tau / L) * _translation_fixed_count(fock, tau)
            for tau in range(L)
        )
        return round((trace / L).real)
    trace = sum(
        np.exp(-2j * np.pi * k * tau / L)
        * (_translation_fixed_count(fock, tau) + parity * _reflection_fixed_count(fock, tau))
        for tau in range(L)
    )
    return round((trace / (2 * L)).real)


@pytest.mark.parametrize("N,L", GRID)
def test_sector_dims_cover_full_space(N, L):
    fock = bohnasa.FixedNFockSpace(N, L)
    sectors = bohnasa.SymmetrySectors(fock, connections="pbc")
    assert sum(sectors.dim(label) for label in sectors.labels) == fock.dim


@pytest.mark.parametrize("N,L", GRID)
def test_sector_dims_match_projector_traces(N, L):
    fock = bohnasa.FixedNFockSpace(N, L)
    sectors = bohnasa.SymmetrySectors(fock, connections="pbc")
    for label in sectors.labels:
        assert sectors.dim(label) == _expected_dim(fock, label.k, label.parity)


@pytest.mark.parametrize("N,L", GRID)
def test_isometries_are_orthonormal_and_complete(N, L):
    fock = bohnasa.FixedNFockSpace(N, L)
    sectors = bohnasa.SymmetrySectors(fock, connections="pbc")
    projector_sum = np.zeros((fock.dim, fock.dim), dtype=complex)
    for label in sectors.labels:
        u = sectors.isometry(label).toarray()
        assert np.allclose(u.conj().T @ u, np.eye(u.shape[1]), atol=1e-12)
        projector_sum += u @ u.conj().T
    assert np.allclose(projector_sum, np.eye(fock.dim), atol=1e-12)


@pytest.mark.parametrize("N,L", GRID)
def test_blocks_are_hermitian_complex(N, L):
    fock = bohnasa.FixedNFockSpace(N, L)
    H = bohnasa.Hopping(fock, connections="pbc") * (-1.0) + bohnasa.Interaction(fock) * 0.25
    sectors = bohnasa.SymmetrySectors(fock, connections="pbc")
    for label in sectors.labels:
        block = sectors.restrict(H, label).toarray()
        assert block.dtype == complex
        assert np.allclose(block, block.conj().T, atol=1e-12)


def test_obc_parity_dims_partition_and_match_projector_traces():
    N, L = 3, 5
    fock = bohnasa.FixedNFockSpace(N, L)
    sectors = bohnasa.SymmetrySectors(fock, connections="obc")
    assert [str(label) for label in sectors.labels] == [
        "(k=0, p=+1)",
        "(k=0, p=-1)",
    ]
    assert sum(sectors.dim(label) for label in sectors.labels) == fock.dim
    # tr (|n><n| +- |r n><r n|)/2 = (dim +- #mirror-fixed states) / 2
    n_fixed = sum(1 for state in fock.fockSpace if tuple(reversed(state)) == state)
    assert sectors.dim(bohnasa.SectorLabel(0, 1)) == (fock.dim + n_fixed) // 2
    assert sectors.dim(bohnasa.SectorLabel(0, -1)) == (fock.dim - n_fixed) // 2


def test_canonical_label_order_and_dims():
    sectors = bohnasa.SymmetrySectors(bohnasa.FixedNFockSpace(6, 4), connections="pbc")
    assert [str(label) for label in sectors.labels] == [
        "(k=0, p=+1)",
        "(k=0, p=-1)",
        "(k=1)",
        "(k=-1)",
        "(k=2, p=+1)",
        "(k=2, p=-1)",
    ]
    assert [sectors.dim(label) for label in sectors.labels] == [16, 6, 20, 20, 8, 14]


@pytest.mark.parametrize("bad", ["period", "xyz", [(0, 1), (1, 2)]])
def test_invalid_connections_raise(bad):
    fock = bohnasa.FixedNFockSpace(2, 3)
    with pytest.raises(ValueError, match="connections"):
        bohnasa.SymmetrySectors(fock, connections=bad)


def test_unknown_label_raises():
    sectors = bohnasa.SymmetrySectors(bohnasa.FixedNFockSpace(2, 3), connections="pbc")
    with pytest.raises(ValueError, match="unknown sector"):
        sectors.dim(bohnasa.SectorLabel(5, None))


def test_restrict_requires_matrix():
    fock = bohnasa.FixedNFockSpace(2, 3)
    sectors = bohnasa.SymmetrySectors(fock, connections="pbc")
    bare = bohnasa.Operator(fock)
    with pytest.raises(ValueError, match="no matrix"):
        sectors.restrict(bare, sectors.labels[0])
