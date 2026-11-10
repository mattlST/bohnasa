import numpy as np
import pytest
from math import comb

import bohnasa


@pytest.mark.parametrize("N,L", [(7, 4), (0, 3), (5, 1), (3, 6), (1, 1)])
def test_dimension_matches_binomial(N, L):
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    assert fock.dim == comb(N + L - 1, L - 1)
    assert len(fock.fockSpace) == fock.dim
    assert fock.fockArray.shape == (fock.dim, L)


def test_states_have_fixed_particle_number_and_are_unique():
    N, L = 6, 4
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    assert np.all(fock.fockArray.sum(axis=1) == N)
    assert len(set(fock.fockSpace)) == fock.dim
    assert all(len(x) == L for x in fock.fockSpace)


def test_dict_is_consistent_inverse_of_fockspace():
    N, L = 5, 3
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    assert len(fock.dict) == fock.dim
    for index, state in enumerate(fock.fockSpace):
        assert fock.dict[state] == index
