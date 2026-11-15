# Agent Rules

Working agreement for AI coding agents in this repository. Humans: these rules
are what you should expect from an agent session here.

## Authority

- Agents propose, execute only what was individually asked, and never
  self-escalate. An agent that gains new tooling (API access, credentials,
  better shells) does NOT gain new permissions.
- **Pushes, merges, PR creation/closing, force-pushes, and branch deletion are
  human actions.** An agent may prepare the exact command or body text; a
  person fires it.
- "Everything is green" is never sufficient authorization for any of the above.

## Workflow

- Feature work is issue-first: nothing gets implemented before its GitHub
  issue exists. Implementation branches off `master` (`feat/*`, `fix/*`).
- PR bodies link the issue (`Fixes #NN`). One lean Conventional-Commit line
  per commit; details belong in code, changelog, and PR description.
- History stays linear (rebase-merge). No stacked branches.
- The PR diff is the review unit; examples are reviewed by executing them.

## Quality gates (all required before proposing a push)

- `ruff check`, `ruff format --check`, `mypy bohnasa`
- `pytest --doctest-modules bohnasa/ tests/` with coverage floor (90%)
- every example runs headless and is deterministic across repeated runs
  (no wall-clock or timing values in printed output or tests)
- **Every number written in a docstring, test or example must have been
  executed and observed.** If a measurement contradicts a claim - including a
  claim from an issue, PR body or this file - the claim changes, and the
  correction is stated openly in the PR.
- Python 3.10 is the floor and its CI arm resolves older scipy-stubs;
  version-specific failures are reproduced with a `python:3.10-slim` docker
  mirror built from `git archive` (never mount the working tree into the
  container).

## Code and physics conventions

- `CODING_RULES.md` is canonical for style and design; this file only adds the
  agent-process layer.
- Public API: PascalCase classes, snake_case functions, numpydoc docstrings
  with runnable examples, type hints on public signatures.
- Runtime dependencies: numpy + scipy only; matplotlib belongs to the
  `examples` extra.
- Physics conventions (commutator signs, canonical brackets, coupling
  parameterizations) are stated once in a module docstring and pinned by
  derivation or invariance tests, never implied.
- Performance paths carry a measured justification comment (what they bound,
  on what hardware); correctness paths carry an independent cross-check test.

## Content safety

- Never commit local file paths, machine or account names, institution
  identifiers, or references to unpublished private repositories.
- Diffs are scanned for these before push; model names like "Bose-Hubbard"
  are public vocabulary, private repo/account identifiers are not.
- Long-term planning and operational history live OUTSIDE the repository.

## Scope creep

- Review feedback is answered in the open: show the diff before pushing it,
  ask before merging anything, and file a follow-up issue instead of silently
  expanding a PR's scope.
