# Security Policy

## Supported versions

`bohnasa` is pre-1.0 research software maintained by a single author.
Only the current `master` is supported; there are no backport releases.

## Scope

The library performs exact diagonalization on numerical input produced by the
user themself: no network service, no untrusted input handling, no file or
object deserialization. A realistic vulnerability here would be a memory-safety
or input-validation flaw reachable through documented API calls, or a
vulnerability in the `numpy`/`scipy` dependencies.

## Reporting a vulnerability

Please use
[GitHub private vulnerability reporting](https://github.com/mattlST/bohnasa/security/advisories/new)
— the report is private to you and the maintainer, and a fix or mitigation is
discussed there before anything becomes public.

I aim to acknowledge within a week. Given the project's size, responses may
sometimes amount to documenting a limitation rather than patching it; that
still counts as an answer.
