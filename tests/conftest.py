"""Shared fixtures for the test suite."""

import pytest

import bohnasa


@pytest.fixture
def fock53():
    return bohnasa.FixedNFockSpace(5, 3)


@pytest.fixture
def fock43():
    return bohnasa.FixedNFockSpace(4, 3)
