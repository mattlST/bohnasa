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
2026-11-21), default invocation (reps=1, ntimes=11, ~50 s):

    mode           dim     n  median_ms         C_end
    eig-dense       66    11        9.5  4.325532e-02
    eig-dense      231    11       26.3  2.432166e-02
    eig-dense      861    11      329.2  6.178374e-03
    eig-dense     2080    11     9948.4  3.616296e-03
    eig-stemr      512    11      147.2  1.078639e+03
    eig-stemr     2048    11      491.8  6.191773e+00
    eig-stemr     8192    11    16085.9  5.627080e-01
    krylov          66    11       56.4  4.325532e-02
    krylov         231    11       53.0  2.432166e-02
    krylov         861    11       73.4  6.178374e-03
    krylov        2080    11      103.8  3.616296e-03
    floquet         66    16       13.4  7.157485e-02
    floquet        231    16       64.9  3.245741e-02
    floquet        861    16      900.3  1.349593e-02

The `eig-dense` O(dim^3) growth (0.33 s at 861 vs 9.9 s at 2080) versus the
`krylov` O(dim)-ish growth (0.073 s vs 0.104 s) is the curve that motivates
`_DENSE_EIG_LIMIT`; per-configuration fresh-process eigensolve timings
(0.18 s @ 861, 2.5 s @ 1770, 5.0 s @ 2080, 16.6 s @ 2926; peak RSS 340-780 MB
vs under 150 MB for krylov) are quoted in `bohnasa/otoc.py`.
