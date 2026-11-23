# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Truncated-Wigner semiclassical dynamics (issue #30): `bohnasa/twa.py`
  with `TruncatedWigner` and `WignerSampler` — Wigner-representation
  sampling around a Gross-Pitaevskii point of any engine following the
  documented `dim`/`filling`/`trajectory` protocol (`MeanFieldLattice`
  primary; the merged kicked-ring map of `bohnasa.kicked` matches the map
  structure and awaits a three-method protocol shim), per-sample
  propagation through the engine's own integrator, fixed-seed
  byte-reproducible ensembles with nested prefixes, sample means with
  standard errors, and a TWA squared commutator built from
  Poisson-bracket shadow trajectories. Conventions derived from the
  engine's relation to the fixed-N quantum model: `hbar_eff = 1/N_phys`,
  half-quantum (`hbar_eff * L / 2`) engine-quadrature noise, symbol
  dictionary `n_j <-> (N/L)|phi_j|^2 - 1/2`, commutator rule
  `C = hbar_M^2 <|{O(t), O}|^2>`. Validated against the exact
  `SquaredCommutator` (measured short-time deviation `~1.0 hbar_eff`,
  coefficient `0.94 - 1.00` over `N_phys = 8..32`, rising to `~2.5
  hbar_eff` across the window) and against classical Lyapunov growth
  (the sqrt-C exponents run `2 lambda_s -> lambda_s`, TWA tracking the
  quantum curve within 0.7% - measured 0.64%); the sample mean converges to the
  mean-field orbit at the Monte-Carlo `1/sqrt(samples)` rate on a
  regular orbit. Threading (acceptance): sampler streams and quantum
  eigensolve references are bit-stable across `OMP_NUM_THREADS`, the
  Radau-propagated TWA values are not (<= 2e-9 relative drift) — only
  RNG streams are pinned bitwise, physics bands sit >= 7 orders above
  threading noise. Example `examples/ex14_twa_semiclassical.py`,
  settings-policy constants in CONTRIBUTING.md.
- Kicked (Floquet) Bose-Hubbard ring (`bohnasa.kicked`, issue #29):
  `KickedBoseHubbard(L, tau, J, U)` builds the two-factor drive step
  `U_step = exp(-i tau H_J) exp(-i tau H_U)` from `Hopping`/`Interaction`
  (kick-then-hop order and bond conventions stated once in the module
  docstring, factor order pinned against the ex10-style hand assembly), feeds
  `SquaredCommutator.from_propagator` for Floquet OTOC curves, exposes
  `floquet_spectrum(fock)` returning quasienergies in the stated half-open
  band `[0, 2 pi / tau)` with Floquet modes - symmetry-sector blocked via
  `SymmetrySectors` for named boundary conditions (measured 4.6-5.5 s vs
  39-200 s for the dense fallback at dim 3003) - and `classical_map()`
  returning the classical kicked map `KickedMeanFieldMap` (kick-the-phases /
  linear-hop Gross-Pitaevskii map with `iterate` and a pinned dimer
  `(s, phi)` dictionary). Validated: unitarity at roundoff across regimes
  (strong-kick corner included, max |U^dag U - I| = 1.11e-15), the tau -> 0
  limit of the map reproducing `mean_field_flow` with measured first-order
  Trotter convergence (Richardson ratios 1.965/1.982 in `s`, 1.9992/1.9992
  in `phi`), the `from_propagator` round trip against explicit Heisenberg
  matrix powers (max dev 2.5e-14), and branch-convention determinism
  (bitwise reproducible spectra; tau-perturbation shift 9.5e-9 vs a measured
  0.35 gap to the zone edge). Illustrated by
  `examples/ex13_kicked_ring.py` (quasienergy branches vs U, Floquet OTOC
  vs continuous time, kicked density propagation, classical map orbits).
- Numerical settings policy (issue #25): every numeric knob in `bohnasa/*.py`
  is a named module constant whose comment states the measured regime it
  protects (Lanczos breakdown and reorthogonalization tolerances, OTOC
  dispatch threshold and spectral block size, mean-field solver defaults,
  fixed-point search gates, determinism seeds, symmetry phase tolerance);
  CONTRIBUTING.md separates tested invariants from performance hints, and
  `benchmarks/propagation_scaling.py` records the propagation-mode cost
  curves the threshold comments cite. Value-identical naming only - no
  behavior changes.
- Time-resolved lattice visualization (`bohnasa.visualization`): ring/chain
  geometry (`lattice_geometry`, `lattice_bonds`) sharing one bond-list builder
  with `Hopping` (`operators.bond_pairs`, including the deliberate `L = 2` PBC
  double counting), static snapshots (`draw_lattice`), headless deterministic
  trajectory replays (`animate_lattice`) and 2-D fallback views
  (`occupation_heatmap`, `momentum_heatmap`). Visual grammar (issue #40,
  amended): node circle area linear in on-site occupation; the edge base
  (width ∝ |e|, diverging `coolwarm` color, symmetric trajectory-wide norm
  so zero is neutral) encodes the symmetric bond energy
  `e_ij = 2 Re gamma_ij`, whose sum over the bond list is exactly
  `<Hopping>` (sum-rule test); arrowheads encode the directed bond current
  `j_ij = 2 Im gamma_ij` - pointing first -> second for positive current,
  head size ∝ |j|, hidden at or below a stated threshold (5% of the
  trajectory max |j| by default) - pinned by a continuity-equation
  derivation test. Coherence `|gamma_ij|` remains available as an
  alternative edge mode (`edge_mode="coherence"`); optional doublon
  overlay from `pair_correlation`. All normalizations trajectory-wide
  with an absolute round-off noise floor (1e-14) so cancellation noise
  cannot self-normalize to extremes, colormaps colorblind-safe, colorbars
  accompanying any color-encoded data - one shared pair per figure in
  multi-panel views (`draw_lattice(colorbar=False)` plus
  `draw_lattice(edge_scale=..., current_scale=...)` to replay a frame
  with exactly the animation's norms). Deterministic annotated captions:
  static monospace `header` and auto/`legend` blocks plus a live
  per-frame readout of caller-supplied `(T,)` value arrays
  (`live=[("t", times, "5.2f"), ...]`); `add_captions` places the static
  blocks on hand-built figures, and the arrowheads were made more
  prominent (`(10, 40)` pt head ramp over a shortened `0.35-0.65` shaft,
  linewidth 2.0). matplotlib is imported lazily (`examples` extra;
  runtime floor stays numpy + scipy). Illustrated by
  `examples/ex12_lattice_visualization.py`,
  a current-carrying Gaussian-packet quench (the parity-symmetric Fock
  k=0 protocol of the first revision has vanishing current and would
  render no arrows), replayable as ring or open chain (`CONN`) at a
  cheap 13-frame pipeline default or the full 49-frame replay
  (`BOHNASA_EXAMPLES_SLOW=1`).
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
- Reusable observable evaluation (`StateObservables`, issue #20): binds a
  `FixedNFockSpace` once and pre-assembles the `L²` `Correlator` pair
  matrices into a CSR stack, so `one_body_density_matrix` - and everything
  routed through it, `orbital_occupations` and `momentum_distribution` -
  costs one stacked matvec plus `L²` BLAS dots per state instead of `L²`
  Python-loop matrix constructions per call. Bit-for-bit the module-function
  results (pinned across several bases including complex-valued states; the
  #19 physics anchors re-run through the stack path; bounded work pinned by
  construction/matvec counters, not timing). Module functions keep their
  one-shot behavior.
- OTOC scrambling: `SquaredCommutator` evaluates the squared commutator
  ``C(t) = ||[O(t), O]|psi>||^2`` via the vector trick - state propagations
  only, never Liouville-space assembly - with dense-eig and sparse-Krylov
  paths and a Floquet entry point (`from_propagator`); the exact classical
  half for the dimer (`dimer_fixed_points`, `dimer_jacobian`, `dimer_lambda`,
  `mean_field_flow`) lives in a new module `classical_mean_field`, in the
  quarter-circle convention J = cos(theta), U = 2 sin(theta)/N (issue #22).
- Thermal (density-matrix) OTOC: `SquaredCommutator.thermal_curve(times, beta,
  regularization)` evaluates ``C_beta(t) = Tr([O(t),O]^dag [O(t),O] rho_beta)``
  for the Gibbs state ``rho_beta = exp(-beta H)/Z(beta)`` as a
  Boltzmann-weighted spectral contraction over the energy eigenbasis (still
  no Liouville assembly). `regularization="symmetric"` implements the
  regularized convention of thesis Sec. 7.1 / App. D: the Gibbs weights enter
  as symmetric `rho^{1/8}` insertions on both operator vertices, removing the
  trivial identity/pure-state part (it vanishes at `beta -> infinity` while
  the unregularized curve tends to the ground-state curve; both coincide at
  `beta = 0`). Pinned by those limit invariants, the commuting-operator
  invariance and brute-force `expm` density-matrix cross-checks on dimer and
  trimer systems (issue #28).
- Runnable examples `ex05_symmetry_sectors.py`, `ex06_momentum_spectra.py`,
  `ex07_operator_growth.py`, `ex08_observables.py`, `ex09_otoc.py` and
  `ex10_kicked_dimer.py`
  (block reassembly, sector structure of a coherent quench, Krylov chain
  computed inside one sector, per-sector spectra vs analytic free bosons,
  operator growth, state-resolved observables, OTOC Ehrenfest window,
  Trotterized kicked dimer).
- General L-site classical mean-field engine `MeanFieldLattice`
  (`mean_field_lattice`): the discrete-nonlinear-Schrodinger / mean-field
  Bose-Hubbard lattice in real `(q, p)` phase space with hand-coded numpy
  flow and analytic Jacobian feeding `solve_ivp` (no symbolic gradient
  generation), conserved energy and particle number, rotating-frame fixed
  points with a deterministic branch search, stability spectra and maximal
  Lyapunov exponents by three independent methods (Benettin
  renormalization, two-trajectory separation, QR flow). The closed-form
  dimer is the exact `L = 2` special case (quarter-circle convention, the
  PBC two-site ring counting its single bond twice); reduction,
  conservation, symplecticity and Lyapunov cross-checks are pinned as
  physics invariants. Example `ex11_mean_field_lattice.py` accompanies it
  (issue #26).
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
- `Hopping` and the visualization layer now share one bond-list builder
  (`operators.bond_pairs`); as a side effect `Hopping(connections=<ndarray>)`
  works instead of tripping the elementwise string comparison (pinned by a
  bit-equality regression test, issue #40).
- `mean_field_flow` (`classical_mean_field`) integrates with
  `scipy.integrate.solve_ivp` (adaptive DOP853 by default; caller-adjustable
  keyword-only `method`/`rtol`/`atol`, defaults `DOP853`/`1e-10`/`1e-12`)
  instead of a hand-rolled fixed-step RK4 loop; saddle anchor trajectories
  agree with the replaced reference to 9.7e-9, and the ODE-perturbation-slope
  physics anchor tightened from `abs=5e-3` to `abs=5e-4` (measured floor
  3.34e-4, stable across the CI scipy matrix) (issue #24).
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
- **Breaking (pre-1.0 API):** the `n_steps` parameter of `mean_field_flow`.
  Rationale: the adaptive DOP853 has no step count to configure; the
  parameter only ever controlled the fixed-step RK4 grid, and passing it is
  now a `TypeError` (pinned by a regression test).
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
