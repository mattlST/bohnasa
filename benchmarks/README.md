# Benchmarks

Scripts here record measurements backing the numeric settings documented in
CONTRIBUTING.md ("Numerical settings policy"). They are not tests and not
part of the package; wall-clock and memory numbers are host-specific and
pinned by nothing.

## propagation_scaling.py

Curve cost vs Hilbert-space dimension for the four `SquaredCommutator`
propagation modes. Run from the repository root:

    python benchmarks/propagation_scaling.py                  # ~1 min sweep
    python benchmarks/propagation_scaling.py --ntimes 51 --reps 3
    python benchmarks/propagation_scaling.py --dims 66,231,861,2080,3300 \
        --stemr-dims 512,2048,8192,32768 --floquet-dims 66,231,861,2080
    python benchmarks/propagation_scaling.py --block-sweep    # _SPECTRAL_BLOCK probe

Peak RSS per configuration: launch under `/usr/bin/time -v`.

Observed on a 16-core x86-64 host (numpy 2.5.5, scipy 1.18.1, CPython 3.14,
2026-11-22, performance governor, idle), invocation `--reps 3` with default
dims (ntimes=11, ~3 min):

    mode           dim     n  median_ms         C_end
    eig-dense       66    11        0.9  4.325532e-02
    eig-dense      231    11        9.8  2.432166e-02
    eig-dense      861    11      143.0  6.178374e-03
    eig-dense     2080    11     6963.3  3.616296e-03
    eig-stemr      512    11      139.2  1.078639e+03
    eig-stemr     2048    11      410.6  6.191773e+00
    eig-stemr     8192    11    13340.6  5.627080e-01
    krylov          66    11       40.0  4.325532e-02
    krylov         231    11       43.0  2.432166e-02
    krylov         861    11       54.5  6.178374e-03
    krylov        2080    11       80.7  3.616296e-03
    floquet         66    16       11.0  7.157485e-02
    floquet       231    16       40.3  3.245741e-02
    floquet       861    16      618.0  1.349593e-02

The `C_end` values are bit-stable across invocations, hosts and BLAS
threading - they are the physics; the times are not and are recorded only
to document the regimes the constants sit in.

The `eig-dense` O(dim^3) growth (0.14 s at 861 vs 7.0 s at 2080) versus the
`krylov` O(dim)-ish growth (0.055 s vs 0.081 s) is the curve that motivates
`_DENSE_EIG_LIMIT`. Fresh-process single-eigensolve timings on this rerun:
0.16 s at 861, 2.53 s at 1770, 5.65 s at 2080, 17.7 s at 2926 (peak RSS
776 MB at 2926) - the numbers quoted in `bohnasa/otoc.py` (0.18/2.5/5.0/
16.6 s) reproduce within ~15 % under these conditions; a single-threaded
run measures several times lower, hence the load/threading caveat there.
