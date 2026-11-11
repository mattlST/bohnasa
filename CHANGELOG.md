# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
- `setup.py`, `requirements.txt`, tracked `bohnasa.egg-info/`, flake8 from CI,
  dead code and unreachable statements throughout.
- Development notebooks `t02`-`t05`: physics content converted to permanent tests
  and runnable examples (originals recoverable from git history).

## [0.0.0] - 2024-09-13
- Initial private development history: Fock-space enumeration, operators, coherent
  states, Lanczos, Krylov complexity.
