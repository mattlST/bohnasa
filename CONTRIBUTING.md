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

- `pytest tests/` green, **no new warnings** in the output.
- Every new feature ships with tests; every fixed bug ships with a regression test.
- Known-bug markers (`xfail(strict=True)` in `tests/test_known_bugs.py`) are removed
  in the PR that fixes them — a stale marker is a CI failure by design.
- The change satisfies CODING_RULES.md; if it deliberately breaks a rule, say why in
  the PR body (rule changes are discussed as openly as code).

## CI gates

Current: pytest + flake8-critical on Python 3.10 and 3.11.
Planned: ruff (format + lint) and mypy (public API signatures) as blocking checks.

## Larger ideas first

New physics modules (symmetry sectors, new algorithms) start with an issue for
discussion before a branch exists — decide as late as possible, build as lean as
possible.
