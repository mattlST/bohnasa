"""Shared fixtures for the test suite."""

import pytest

import bohnasa


# deliberately legacy constructor names: the whole suite doubles as proof that
# the deprecated alias layer keeps working (switch when aliases drop in Phase 5)
@pytest.fixture
def fock53():
    return bohnasa.fockStates_fixed_particle_number(5, 3)


@pytest.fixture
def fock43():
    return bohnasa.fockStates_fixed_particle_number(4, 3)
