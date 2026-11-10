"""Registry of known rule violations in CODING_RULES.md.

Every test here is xfail(strict=True): it fails today and the test *must keep
failing* until the bug is fixed. Fixing a bug (commits B/C/D of the Phase 3
plan) means deleting its xfail marker here - CI then guards against
regressions and silently stale markers.
"""

import numpy as np
import pytest
import scipy.linalg as spla

import bohnasa


@pytest.fixture
def fock43():
    return bohnasa.fockStates_fixed_particle_number(4, 3)


@pytest.mark.xfail(
    strict=True, reason="bug: FockSpace.__str__ returns print(...) -> TypeError"
)
def test_fock_space_str_returns_string(fock43):
    assert isinstance(str(fock43), str)


@pytest.mark.xfail(strict=True, reason="bug: missing __rmul__ -> '2 * op' raises")
def test_scalar_mult_reflected(fock43):
    op = bohnasa.number(fock43, 0)
    doubled = 2 * op
    assert np.allclose(doubled.matrix.toarray(), 2 * op.matrix.toarray())


@pytest.mark.xfail(strict=True, reason="bug: __mul__ returns None for operator rhs")
def test_operator_product_raises(fock43):
    with pytest.raises(TypeError):
        _ = bohnasa.number(fock43, 0) * bohnasa.number(fock43, 1)


@pytest.mark.xfail(
    strict=True, reason="bug: to_vector() returns implicit None when matrix unset"
)
def test_to_vector_raises_without_matrix(fock43):
    op = bohnasa.operator(fock43)
    with pytest.raises(ValueError):
        op.to_vector()


@pytest.mark.xfail(
    strict=True, reason="bug: user input validated via assert, not ValueError"
)
def test_single_fockstate_rejects_wrong_particle_number(fock43):
    with pytest.raises(ValueError):
        bohnasa.single_fockstate(fock43, np.array([1, 1, 1]))  # N=4, sums to 3


@pytest.mark.xfail(
    strict=True, reason="bug: user input validated via assert, not ValueError"
)
def test_coherent_state_rejects_wrong_phi_length(fock43):
    with pytest.raises(ValueError):
        bohnasa.coherent_state_number_projected(fock43, np.array([1.0, 2.0]))


@pytest.mark.xfail(
    strict=True, reason="bug: user input validated via assert, not ValueError"
)
def test_hopping_rejects_malformed_connections(fock43):
    with pytest.raises(ValueError):
        bohnasa.hopping(fock43, connections=[[0, 1, 2]])


@pytest.mark.xfail(
    strict=True,
    reason="bug: time_evolution_exact_ind stores results in dtype=float and "
    "silently discards imaginary parts of complex expectation values",
)
def test_time_evolution_preserves_complex_means(fock43):
    rng = np.random.default_rng(9)
    psi = rng.normal(size=fock43.dim) + 1j * rng.normal(size=fock43.dim)
    psi /= np.linalg.norm(psi)

    H = bohnasa.number(fock43, 0)
    obs = bohnasa.correlator(fock43, 0, 1)  # non-hermitian -> complex expectation
    time = np.linspace(0.0, 1.0, 5)

    res = bohnasa.time_evolution_exact_ind(H, psi, time, [obs])

    # independent reference: direct evolution with the dense propagator
    h = H.matrix.toarray()
    ref = []
    for t in time:
        psi_t = spla.expm(-1j * t * h) @ psi
        ref.append(np.vdot(psi_t, obs.matrix @ psi_t))
    assert np.allclose(res[0, 0, :], ref)
