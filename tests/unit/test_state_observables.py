"""Exact parity and bounded-work tests for StateObservables (issue #20).

The reuse path must be a pure refactor: every quantity is asserted
bit-for-bit equal (``assert_array_equal``) to the module-function result on
the same state, across several bases and state types including complex-
valued states, and the reuse regime (build once, evaluate T states) is
pinned by counting matrix constructions and stacked matvecs - no wall-
clock numbers anywhere in this file.
"""

import numpy as np
import pytest
import scipy.sparse as sp

import bohnasa
from bohnasa import observables as obs_mod

_SIX = (
    "one_body_density_matrix",
    "density_profile",
    "orbital_occupations",
    "momentum_distribution",
    "pair_correlation",
    "structure_factor",
)

_BASES = [(1, 3), (2, 3), (3, 5), (2, 6), (4, 3)]


def _states(fock):
    """Named test states for a basis: complex-valued where possible, so a
    dropped conjugation or sign cannot hide in real arithmetic."""
    rng = np.random.default_rng(20)
    out = {}
    s = rng.normal(size=fock.dim) + 1j * rng.normal(size=fock.dim)
    out["complex_random"] = s / np.linalg.norm(s)
    eigen = np.zeros(fock.dim, dtype=complex)
    eigen[fock.dict[(fock.N,) + (0,) * (fock.L - 1)]] = 1.0
    out["eigenstate_real"] = eigen
    a = np.zeros(fock.dim, dtype=complex)
    b = np.zeros(fock.dim, dtype=complex)
    a[fock.dict[fock.fockSpace[0]]] = 1.0
    b[fock.dict[fock.fockSpace[1]]] = 1.0
    out["complex_superposition"] = (a + 1j * b) / np.sqrt(2)
    if fock.N == 1 and fock.L >= 3:
        # directed plane-wave orbital exp(i k_2 j): pins momentum chirality
        phi = np.exp(2j * np.pi * 2 * np.arange(fock.L) / fock.L) / np.sqrt(fock.L)
        wave = np.zeros(fock.dim, dtype=complex)
        for site in range(fock.L):
            occ = tuple(1 if s2 == site else 0 for s2 in range(fock.L))
            wave[fock.dict[occ]] = phi[site]
        out["plane_wave_m2"] = wave
    return out


@pytest.mark.parametrize("N, L", _BASES)
class TestParity:
    """The class results equal the module-function results EXACTLY."""

    @pytest.mark.parametrize("quantity", _SIX)
    def test_bitwise_equal_to_module_function(self, N, L, quantity):
        fock = bohnasa.FixedNFockSpace(N, L)
        bound = bohnasa.StateObservables(fock)
        module_fn = getattr(obs_mod, quantity)
        class_fn = getattr(bound, quantity)
        for name, state in _states(fock).items():
            expected = module_fn(state, fock)
            actual = class_fn(state)
            # bit-for-bit, not just close - the fast path is a pure refactor
            np.testing.assert_array_equal(
                actual, expected, err_msg=f"state type {name}, basis N={N} L={L}"
            )


class _CountingStack(sp.csr_matrix):
    """Copy of the pair stack that counts stacked matvecs (work measurement)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matvecs = 0

    def __matmul__(self, other):
        self.matvecs += 1
        return super().__matmul__(other)


def test_stack_built_once_and_reused(monkeypatch):
    """Bounded work for the reuse regime (issue #20 acceptance).

    Naive (module path) work for T states: T * L**2 matrix constructions.
    Bounded (class path) work: L**2 constructions ONCE plus one stacked
    matvec per state. Both are asserted by counters, deterministically -
    no timing involved.
    """
    fock = bohnasa.FixedNFockSpace(3, 4)
    T = 21

    counts = {"stack_builder": 0, "pair_matrix": 0}
    real_stack = obs_mod._correlator_stack
    real_pair = obs_mod._pair_matrix

    def counting_stack(fock_arg):
        counts["stack_builder"] += 1
        return _CountingStack(real_stack(fock_arg))

    def counting_pair(fock_arg, src, dst):
        counts["pair_matrix"] += 1
        return real_pair(fock_arg, src, dst)

    monkeypatch.setattr(obs_mod, "_correlator_stack", counting_stack)
    monkeypatch.setattr(obs_mod, "_pair_matrix", counting_pair)

    bound = bohnasa.StateObservables(fock)
    assert counts["stack_builder"] == 1
    assert counts["pair_matrix"] == fock.L**2  # every pair exactly once
    assert bound._stack.matvecs == 0  # binding itself applies nothing

    rng = np.random.default_rng(21)
    states = [rng.normal(size=fock.dim) + 1j * rng.normal(size=fock.dim) for _ in range(T)]
    states = [s / np.linalg.norm(s) for s in states]
    gammas = [bound.one_body_density_matrix(s) for s in states]

    # evaluation does NO further construction, no rebuild, and exactly one
    # stacked matvec per state
    assert counts["stack_builder"] == 1
    assert counts["pair_matrix"] == fock.L**2
    assert bound._stack.matvecs == T

    # the reused stack still reproduces the reference path bit-for-bit
    for s, g in zip(states, gammas, strict=True):
        assert np.array_equal(obs_mod.one_body_density_matrix(s, fock), g)


def test_class_validates_state_shape():
    fock = bohnasa.FixedNFockSpace(2, 3)
    bound = bohnasa.StateObservables(fock)
    bad = np.zeros(fock.dim + 1, dtype=complex)
    for name in _SIX:
        with pytest.raises(ValueError, match=r"state must have shape"):
            getattr(bound, name)(bad)


def test_results_are_fresh_arrays_not_shared_state():
    """No hidden caching keyed on object identity: each call returns a new
    array and mutating it cannot affect later evaluations."""
    fock = bohnasa.FixedNFockSpace(2, 3)
    bound = bohnasa.StateObservables(fock)
    rng = np.random.default_rng(22)
    state = rng.normal(size=fock.dim) + 1j * rng.normal(size=fock.dim)
    state /= np.linalg.norm(state)
    first = bound.one_body_density_matrix(state)
    first[...] = 0.0
    second = bound.one_body_density_matrix(state)
    assert second is not first
    np.testing.assert_array_equal(second, obs_mod.one_body_density_matrix(state, fock))
    profile = bound.density_profile(state)
    profile[...] = 0.0
    np.testing.assert_array_equal(
        bound.density_profile(state), obs_mod.density_profile(state, fock)
    )
