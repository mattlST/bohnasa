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
