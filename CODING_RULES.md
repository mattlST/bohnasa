# Coding Rules

Rules for `bohnasa`, based on clean code, SOLID, and Lean. Physics correctness comes first:
a rule that hurts numerical correctness or measurable performance is wrong — change the rule.

## Basics

- Python >= 3.10 (`requires-python` in `pyproject.toml`).
- Runtime dependencies: numpy and scipy only. Any new dependency needs a written
  justification in the PR.
- ruff, mypy, and pytest run in CI. Green CI is required before merging anything.

## Style

- PEP 8, enforced by ruff (line length 100, rules E/F/I/UP/B).
- Classes `PascalCase`; functions, methods, variables `snake_case`. No camelCase.
- Never shadow stdlib names — our base class is `Operator`, not `operator`.
- Abbreviations only where physics precedent exists (`pbc`, `obc`, `fock`).

## Git

- Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `chore:`, `refactor:`, `perf:`.
- One lean line per message. Details belong in the code, the changelog, or the PR.
- History rewrites only with a backup bundle and prior discussion.

## Correctness

- No `print()` in library code. Use `logging` or raise.
- User-facing validation raises `ValueError`/`TypeError`, never `assert` (asserts vanish
  under `python -O`). `assert` is only for internal invariants that "cannot" hold.
- Every code path returns a value or raises. No implicit `None`.
- Dunder operators come in complete families (`__add__`/`__sub__`, `__mul__`/`__rmul__`);
  unsupported operand types return `NotImplemented`.
- Inner products of quantum states use `np.vdot` semantics. Never silently drop imaginary
  parts: return complex, or take `.real` with a comment justifying why.

## API & docs

- Every public module, class, and function has a numpydoc-style docstring with at least one
  runnable example.
- Type hints on all public signatures (mypy covers these); internals typed where cheap,
  not enforced.

## Design

- One concept per module: basis (`FockSpace`), operators, states, algorithms.
- A new observable is a new subclass or a call into a shared builder — never an edit to core classes.
- All operators satisfy the same contract: CSR `matrix`, callable on a state vector, consistent
  index/transpose conventions across every builder.
- Algorithms (Lanczos, time evolution) depend on operator callables, not on sparse matrices directly.
- Where measured vectorized numpy beats cleaner abstraction, take the vectorization and mark
  the hot path with a comment.

## Duplication & dead code

- No commented-out code, no unreachable statements. Delete; git remembers.
- The same non-trivial logic appearing twice becomes a helper.

## Testing

- Every merged feature ships with a test. Every fixed bug ships with a regression test
  that references the bug.
- Prefer physics invariants over value snapshots: commutators `[b, b†] = δ`, known spectra
  (1-particle ring: `E_k = -2t cos k`), norm and energy conservation.

## Numerics

- Sparse by default; dense conversion (`todense`, `eigh`) requires a comment saying why.
- Benchmark before optimizing. No performance code without a measurement (Lean: decide late).
