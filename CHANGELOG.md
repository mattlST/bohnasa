# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Generic one-/two-body Hamiltonian builder (`Hamiltonian`): normal-ordered
  `Σ t_ij c†_i c_j + Σ V_ijkl c†_i c†_j c_k c_l` from coefficient arrays
  (`one_body`, `two_body`/`onsite_u`), literal coefficients, only nonzero
  entries materialized; reproduces the hand-assembled
  `Hopping * (-1) + Interaction * U` sandwiches exactly (issue #17).
- Symmetry sectors (`SymmetrySectors`, `SectorLabel`): crystal-momentum +
  reflection-parity blocks for periodic chains, mirror-parity blocks for open
  ones; `restrict`/`to_sector` feed Lanczos and Krylov runs in sector space
  (issue #13).
- Operator-growth Krylov complexity: `Liouvillian` (commutator superoperator
  acting on flattened operators, never assembled as a matrix) and
  `operator_growth` (chain + <K̂>(t) of the Heisenberg-evolved operator), the
  sibling convention of `krylov_complexity` (issue #15).
- State-resolved observables (`observables`): `one_body_density_matrix`,
  `density_profile`, `orbital_occupations` (Penrose-Onsager condensate
  fraction), `momentum_distribution`, `pair_correlation` (normal-ordered
  density noise, basis-diagonal by construction), `structure_factor`
  (issue #19).
- OTOC scrambling: `SquaredCommutator` evaluates the squared commutator
  ``C(t) = ||[O(t), O]|psi>||^2`` via the vector trick - state propagations
  only, never Liouville-space assembly - with dense-eig and sparse-Krylov
  paths and a Floquet entry point (`from_propagator`); the exact classical
  half for the dimer (`dimer_fixed_points`, `dimer_jacobian`, `dimer_lambda`,
  `mean_field_flow`) lives in a new module `classical_mean_field`, in the
  quarter-circle convention J = cos(theta), U = 2 sin(theta)/N (issue #22).
- Runnable examples `ex05_symmetry_sectors.py`, `ex06_momentum_spectra.py`,
  `ex07_operator_growth.py`, `ex08_observables.py`, `ex09_otoc.py` and
  `ex10_kicked_dimer.py`
  (block reassembly, sector structure of a coherent quench, Krylov chain
  computed inside one sector, per-sector spectra vs analytic free bosons,
  operator growth, state-resolved observables, OTOC Ehrenfest window,
  Trotterized kicked dimer).
- Security policy (`SECURITY.md`, private vulnerability reporting) and issue
  templates (bug report, feature request).
- Coding rules (`CODING_RULES.md`), contribution process (`CONTRIBUTING.md`), changelog.
- `pyproject.toml` with `dev` extra (`pytest`, `pytest-cov`, `ruff`, `mypy`, `scipy-stubs`);
  minimum supported Python is 3.10.
- PascalCase public API: `Operator`, `Number`, `Correlator`, `Hopping`, `Interaction`,
  `ChemicalPotential`, `FixedNFockSpace`, `fixed_n_coherent_state`, `krylov_complexity`.
  Old lowercase names remain as legacy aliases (to be dropped before 0.1.0).
- `Operator.adjoint()`; legacy `correlator_m` kept as a deprecating shim.
- Module split: `bohnasa.krylov` and `bohnasa.time_evolution` (`bohnasa.util` is a shim).
- numpydoc docstrings with doctested examples across the public API.
- Physics test suite (57 tests + doctests): Fock-space invariants, operator algebra,
  bosonic commutators, analytic hopping spectra, Lanczos orthonormality/isospectrality,
  exact-vs-Krylov ground states.
- CI gates on `ruff check`, `ruff format --check`, `mypy`, pytest with coverage floor (90%)
  across Python 3.10-3.13.
- Runnable `examples/` (basis, operators + ground state, Krylov complexity quench,
  attractive disordered chain), `examples` extra (matplotlib), smoke-tested headless in CI.

### Changed
- `hopping`, `correlator`, `correlator_m` share one transition-matrix builder with a
  unified row/output storage convention.
- `time_evolution_exact_ind` returns complex results; imaginary parts of non-Hermitian
  expectation values are no longer silently discarded.
- Invalid inputs raise `ValueError` instead of `assert`; library code no longer mutates
  caller arrays (`fixed_n_coherent_state`).
- Lanczos reorthogonalization threshold is norm-relative; the loop is bounded (20 passes,
  then `RuntimeError`) instead of infinite; start vectors with zero/non-finite norm raise.

### Fixed
- `correlator_m(i, i)` returned `n_i + 1` (`c c†`) instead of `n_i` (`c† c`).
- `fockStates.__str__` returned `print(...)`; `to_vector` returned `None`; missing
  `__rmul__`/`__sub__`; `__mul__` silently returned `None` for unsupported operands.
- `ChemicalPotential` crashed on list input for `mu`.
- Debug prints removed from `lanczos` and `time_evolution_exact_ind`.

### Removed
- **Breaking (API freeze for 0.1.0):** legacy lowercase aliases
  (`fockStates_fixed_particle_number`, `operator`, `number`, `correlator`, `hopping`,
  `interaction`, `chemical_mu`, `coherent_state_number_projected`, `Krylov_complexity`),
  the deprecated `correlator_m` shim, and the `bohnasa.util` module.
- `setup.py`, `requirements.txt`, tracked `bohnasa.egg-info/`, flake8 from CI,
  dead code and unreachable statements throughout.
- Development notebooks `t02`-`t05`: physics content converted to permanent tests
  and runnable examples (originals recoverable from git history).

## [0.0.0] - 2024-09-13
- Initial private development history: Fock-space enumeration, operators, coherent
  states, Lanczos, Krylov complexity.
