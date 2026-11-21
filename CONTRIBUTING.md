# Contributing

Process rules for working on `bohnasa`. Code rules live in [CODING_RULES.md](CODING_RULES.md).

## Development setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"          # numpy, scipy + pytest, ruff, mypy
    pytest tests/                    # the whole suite runs in seconds

## Workflow: branch → PR → rebase-merge (GitHub flow-lite)

`master` is always green. Every change lands through a pull request:

1. **Branch** off `master` as `<type>/<kebab-description>`, where `<type>` is the
   Conventional Commit prefix: `feat/`, `fix/`, `test/`, `refactor/`, `docs/`,
   `ci/`, `chore/`, `perf/`, `build/`.
   Example: `refactor/transition-builder`
2. **Commit** lean, one-line Conventional Commits — one commit per concern.
   Details go in the PR description, not the message.
3. **Open a draft PR early**, push the branch, let CI run. Mark "ready for review"
   only when checks pass.
4. **Merge with rebase-merge** (fast-forward style). Never squash — the individual
   commits are the story of the change. No merge commits — `master` stays a linear
   history, like a good `git log --oneline` should be.
5. **Delete the branch** after merge.

Rules of the road:

- Branches are short-lived: days, not weeks. Rebase onto `master` freely while the
  PR is open; never rebase or force-push `master`.
- No Gitflow: no `develop`, no long-lived release branches. Tag releases on `master`
  (`v0.1.0`, ...); hotfixes are just branches like any other change.
- Size: if a change crosses more than one subsystem, split it into stacked PRs.

## Definition of done (every PR)

- The local gate stack is green — CI runs exactly this, on every push:

      ruff check . && ruff format --check . && mypy bohnasa
      pytest --doctest-modules bohnasa/ tests/ -q --cov=bohnasa --cov-fail-under=90

- Every new feature ships with tests; every fixed bug ships with a regression
  test in `tests/regression/` — permanent guards, not markers to chase.
- The change satisfies CODING_RULES.md; if it deliberately breaks a rule, say why in
  the PR body (rule changes are discussed as openly as code).

## Numerical settings policy

Every numeric *setting* in `bohnasa/*.py` - tolerance, threshold, block size,
iteration cap, solver default, determinism seed - is a named module-level
constant whose comment states the measured regime it protects (issue #25).
Bare literals remain only for structural and mathematical quantities (indices,
shape algebra, exact coefficients of documented closed forms like the dimer
`mu^2 = 8 cos^2(theta) (tan(theta) - 2)`). Retuning a constant never needs a
code-structure change, but it must cite a measurement: rerun
`benchmarks/propagation_scaling.py` (or the probe named in the comment) and
update the comment in the same commit. The constants fall into two classes:

**(a) Tested invariants** - changing the value flips a pinned test or doctest;
a change is only acceptable together with the corresponding test update:

- `classical_mean_field._DEFAULT_RTOL` / `._DEFAULT_ATOL` - the issue #24
  anchor point (`tests/unit/test_mean_field_flow.py` method-agreement and
  rtol-ladder margins; `tests/physics/test_otoc_physics.py` 3.34e-4 slope floor).
- `classical_mean_field._REAL_INPUT_TOL`, `mean_field_lattice._STATE_IMAG_TOL`
  - the `1e-3j` rejection in `tests/unit/test_otoc.py::test_mean_field_flow_rejects_complex_initial`
  and the state-validation tests bracket this value.
- `classical_mean_field._ROOT_FLOOR` - wall guard, provably inactive away
  from |s| = 1 exactly; measured bounded-orbit margin to the wall is
  >= 5.5e-2, so the physics anchors are bit-identical with and without it.
- `mean_field_lattice._DEFAULT_SOLVER_RTOL` / `._DEFAULT_SOLVER_ATOL` -
  trajectory energy-drift doctest bound and the four-digit Lyapunov doctests.
- `mean_field_lattice._SEED_FP_SHELL` / `._SEED_LYAPUNOV_V0` - the
  `fixed_points` doctest shape (L = 2) and the four-digit `lyapunov` doctests
  track these exact streams; `._SEED_LARGE_L_SIGNS` deterministically drives
  the sparse-sign branch exercised by
  `tests/unit/test_mean_field_lattice.py::test_fixed_points_sparse_sign_seeds_for_large_l`.
- `mean_field_lattice._DEFAULT_FP_TOL` / `._ROOT_XTOL` / `._FP_DEDUP_TOL` /
  `._FP_SHELL_TOL` - they gate which roots enter the found-point set pinned
  by the `fixed_points` doctests and `tests/unit/test_mean_field_lattice.py`;
  each sits in a measured gap between accepted and rejected populations.
- `krylov._LANCZOS_BETA_TOL`, `krylov.REORTH_REL_TOL`, `krylov.REORTH_ABS_TOL`
  - inside measured population gaps (breakdown <= 3.8e-13 < tol < true
  couplings >= 7.1e-4; reorth plateau ~1.3e-13 * ||op||); moving them outside
  a gap changes chain depths or trips the staggered-coherent regression case.
- `otoc._DENSE_EIG_LIMIT` - also class (b) by intent, but
  `tests/unit/test_otoc.py` pins a dim-3003 non-tridiagonal Hamiltonian to
  the krylov path (limit < 3003) and the dim-2001 tridiagonal STEMR
  exemption, so retuning requires updating that test in the same commit.
- `sectors._MIRROR_PHASE_TOL` - sits between measured round-off (<= 1.4e-13)
  and the smallest true phase mismatch (2.8e-1); the isometry and
  projector-completeness tests in `tests/unit/test_sectors.py` pin sectors.

**(b) Performance hints** - free to retune per host with a fresh benchmark
citation in the constant's comment, no test update needed:

- `otoc._SPECTRAL_BLOCK` - measured flat 8.0-8.6 s for blocks 128..4096
  (dim 2925, 4096-time curve): a memory ceiling, not a throughput knob.
- `krylov.MAX_REORTH_ITERATIONS` - pure safety cap: measured convergence is
  the 1st pass, so 20 is headroom, not a tuned boundary.
- `mean_field_lattice._DEFAULT_LYAPUNOV_DELTA` - exponent invariant over
  delta in [1e-4, 1e-12] (measured identical to 8 digits).
- `mean_field_lattice._DEFAULT_FP_N_RANDOM`,
  `mean_field_lattice._MAX_EXACT_SEED_PATTERNS`, `mean_field_lattice._ROOT_MAXFEV`
  - coverage/runtime trade-offs of the heuristic fixed-point search (the found
  set grows with them; no doctest at L >= 3 pins completeness).

No environment-variable overrides (issue #25 decision): no tuning use case
has appeared, module constants are already importable and monkeypatchable in
tests, and an override channel would invite values outside the measured
regimes above. Revisit when a host is reported where the default regimes do
not transfer.

## CI gates

Blocking on Python 3.10–3.13: `ruff` lint + format, `mypy` on the public API,
pytest with doctests and a 90% coverage floor, and a headless smoke run of
every script in `examples/`.

### Reproducing CI locally

The py3.10 arm is the one people reproduce by hand. Use the checked-in
recipe instead of ad-hoc container commands:

    ci/repro_arm310.sh              # or: ci/repro_arm310.sh <ref>

It exports the ref with `git archive` (never mount a working tree — untracked
files, a local venv, or a leftover `build/` change what the gates see) and
builds `ci/arm310.Dockerfile` from that clean export, which installs the
package **editable**: `pip install -e ".[dev,examples]"`.

Two failure signatures mean the reproduction went wrong, not the code (a
plain, non-editable `pip install .` produces exactly these 11+11 errors —
see `ci/arm310.Dockerfile`):

- `ruff check .`: 11 F403 star-import errors on `build/lib/bohnasa/__init__.py`
  (the per-file-ignore covers only the real `bohnasa/__init__.py`);
- `pytest --doctest-modules bohnasa/ tests/`: 11 "import file mismatch"
  collection errors (site-packages shadows the source tree).

## Larger ideas first

New physics modules (symmetry sectors, new algorithms) start with an issue for
discussion before a branch exists — decide as late as possible, build as lean as
possible.
