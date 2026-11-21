# Reproduction of the py3.10 CI arm for bohnasa (issue #49).
#
# The build context MUST be a clean `git archive <ref>` export - never a
# mounted working tree or a copy of a dirty checkout. Untracked files, local
# venvs, and leftover build/ directories change what the gates lint and
# collect, so a green/red result says nothing about the ref. Use
# ci/repro_arm310.sh, which chains export -> build -> gates in one command.
#
# The install is deliberately EDITABLE. A plain `pip install
# ".[dev,examples]"` builds the wheel in-tree: it leaves a copy of the
# package under build/lib/ AND puts the installed package in site-packages.
# Both shadow the /src tree the gates run on, with two instantly
# recognizable failure signatures (observed with this image and a
# non-editable install):
#   - `ruff check .` reports 11 F403 star-import errors on
#     build/lib/bohnasa/__init__.py - pyproject's per-file-ignore covers only
#     the real bohnasa/__init__.py, not the build/lib copy;
#   - `pytest --doctest-modules bohnasa/ tests/` reports 11 "import file
#     mismatch" collection errors - two same-named module trees compete.
# These 11+11 errors are artifacts of the install mode, not of the code.
FROM python:3.10-slim

WORKDIR /src
COPY . /src/

# Same steps as .github/workflows/python-package.yml for one matrix entry.
RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev,examples]"

CMD ["bash"]
