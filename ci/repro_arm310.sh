#!/usr/bin/env bash
# One-command reproduction of the py3.10 CI arm (issue #49).
#
# Usage: ci/repro_arm310.sh [REF]     (REF defaults to HEAD)
#
# Exports REF with `git archive` (a clean tree: no untracked files, no venv,
# no build/ leftovers), builds ci/arm310.Dockerfile from that export alone,
# and runs the four definition-of-done gates inside the image. Do not replace
# the export with a mounted working tree and do not switch the Dockerfile to
# a non-editable install - see ci/arm310.Dockerfile for the two failure
# signatures those produce.
set -euo pipefail

ref="${1:-HEAD}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
short="$(git -C "$repo_root" rev-parse --short "$ref")"
image="bohnasa-arm310:${short}"

export_dir="$(mktemp -d)"
trap 'rm -rf "$export_dir"' EXIT

echo "== git archive ${ref} (${short}) -> clean export =="
git -C "$repo_root" archive "$ref" | tar -x -C "$export_dir"

echo "== docker build ${image} from the export =="
docker build -f "$export_dir/ci/arm310.Dockerfile" -t "$image" "$export_dir"

echo "== running the CI gates inside ${image} =="
docker run --rm "$image" bash -c '
set -u
gates=(
  "ruff check ."
  "ruff format --check ."
  "mypy bohnasa"
  "pytest --doctest-modules bohnasa/ tests/ --cov=bohnasa --cov-fail-under=90"
)
results=()
overall=0
for gate in "${gates[@]}"; do
  printf "\n== %s\n" "$gate"
  if eval "$gate"; then
    results+=("OK    ${gate}")
  else
    results+=("FAIL  ${gate}")
    overall=1
  fi
done
printf "\n== summary (%s) ==\n" "$(python -V 2>&1)"
for result in "${results[@]}"; do
  printf "%s\n" "$result"
done
exit "$overall"
'
