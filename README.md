# bohnasa

[![Python package](https://github.com/mattlST/bohnasa/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/mattlST/bohnasa/actions/workflows/python-package.yml)

Exact-diagonalization tools for bosonic quantum many-body systems, working in
the fixed particle-number sector of a Fock space. Built for studying operator
growth and Krylov complexity after a quench.

## Features

- `FixedNFockSpace` — enumerate the basis of `N` bosons on `L` sites, with
  occupation-pattern index lookup
- second-quantized operators on sparse matrices: `Number`, `Correlator`
  (`c_j† c_i`), `Hopping`, `Interaction`, `ChemicalPotential`
- `Hamiltonian` — generic normal-ordered 1-/2-body builder
  (`Σ t_ij c†_i c_j + Σ V_ijkl c†_i c†_j c_k c_l`) from coefficient arrays:
  inhomogeneous or non-reciprocal hopping, site-dependent on-site terms,
  full two-body tensors
- `lanczos` — Lanczos recursion with full reorthogonalization
- `krylov_complexity` — the Krylov-level operator, time-evolvable after a
  quench from a projected coherent state (`fixed_n_coherent_state`,
  `time_evolution_exact_ind`)
- `Liouvillian` / `operator_growth` — the operator-growth sibling: Krylov
  complexity of how an operator spreads under the commutator superoperator
- `SymmetrySectors` — crystal-momentum/parity block decomposition of the
  fixed-N space; diagonalization, Lanczos and Krylov chains cost roughly
  `1/L` per block once the state lives in a single sector
- state-resolved observables — `one_body_density_matrix`,
  `orbital_occupations` (Penrose-Onsager condensate fraction),
  `momentum_distribution`, `pair_correlation` (bunching),
  `structure_factor` (`S(0) = 0` on the fixed-N shell, shot-noise `S(k) = 1`
  for the ideal condensate)
- `SquaredCommutator` — OTOC scrambling growth ``C(t) = ‖[O(t), O]‖²`` by
  the vector trick (state propagations only, never Liouville space; dense
  or Krylov propagation, Floquet driving via `from_propagator`), the
  Gibbs-state thermal curve `thermal_curve(times, beta, regularization)`
  (unregularized and symmetric-regularized conventions), and the exact
  mean-field counterpart of the dimer (`dimer_lambda`,
  `dimer_fixed_points`, `mean_field_flow`) as its classical reference line
- `MeanFieldLattice` — the general L-site classical mean-field (DNLS)
  engine: hand-coded flow and analytic Jacobian under `solve_ivp`,
  energy/norm-conserving trajectories, branch search for relative fixed
  points, rotating-frame stability spectra and maximal Lyapunov exponents
  (Benettin / two-trajectory / QR), with the dimer as the exact L = 2
  special case
- runnable [examples](examples/README.md) covering all of the above

## Install

```bash
pip install .                      # core: numpy + scipy only
pip install -e ".[dev,examples]"   # to hack on the package / run examples
```

Requires Python ≥ 3.10.

## Quickstart

Ground state of an interacting open chain by Lanczos, checked against exact
diagonalization:

```python
import numpy as np
import bohnasa

fock = bohnasa.FixedNFockSpace(N=6, L=4)
H = bohnasa.Hopping(fock, connections="obc") * -1.0 + bohnasa.Interaction(fock) * 0.5
# same Hamiltonian from the generic builder:
chain = np.eye(fock.L, k=1) + np.eye(fock.L, k=-1)
assert np.allclose(
    bohnasa.Hamiltonian(fock, one_body=-chain, onsite_u=0.5).matrix.toarray(),
    H.matrix.toarray(),
)

v, T = bohnasa.lanczos(np.ones(fock.dim, dtype=complex) / np.sqrt(fock.dim), H, 40)
m = np.count_nonzero(np.linalg.norm(v, axis=1) > 1e-12)
print(f"ground state energy: {np.linalg.eigvalsh(T[:m, :m].real)[0]:.6f}")
```

Krylov complexity operator for a staggered coherent-state quench:

```python
phi = 1j * np.array([(-1) ** j for j in range(fock.L)], dtype=complex)
sta = bohnasa.fixed_n_coherent_state(fock, phi / np.linalg.norm(phi))
K = bohnasa.krylov_complexity(fock, H, sta, fock.dim - 1)
```

`examples/ex02_operators.py` and `examples/ex03_krylov_complexity.py` carry
both stories to completion, plots included.

## Testing

```bash
pytest --doctest-modules bohnasa/ tests/ -q --cov=bohnasa --cov-fail-under=90
```

## Status

Research code. The public API froze at `v0.1.0`; expect additions, not
breakage.

## License & citation

MIT — see [LICENSE](LICENSE). If you use this in published work, please cite
via `CITATION.cff`. The methods behind the OTOC and mean-field modules
follow the author's doctoral thesis, *Signatures of Instability in Bosonic
Many-Body Systems* (Universität Regensburg, 2025), publicly cited in
`CITATION.cff`.
