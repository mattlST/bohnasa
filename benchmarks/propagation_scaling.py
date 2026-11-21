"""Propagation-cost scaling for the four SquaredCommutator paths (issue #25).

Records curve cost vs Hilbert-space dimension for every propagation mode in
the library, so future retunes of the dispatch thresholds documented under
"Numerical settings policy" in CONTRIBUTING.md cite a measurement instead of
a guess (CODING_RULES: benchmark before optimizing):

- ``eig-dense``  : full LAPACK eigh + blocked spectral curve (the path the
  ``_DENSE_EIG_LIMIT`` threshold guards), non-tridiagonal interacting chain.
- ``eig-stemr``  : tridiagonal STEMR eigensolve, O(dim^2) - the reason
  tridiagonal Hamiltonians bypass the threshold.
- ``krylov``     : expm_multiply per propagation, O(dim) memory.
- ``floquet``    : integer powers of a given step propagator (kick counts).

Wall-clock values in the output are measurements, not assertions: this script
is not a test and no number it prints is pinned anywhere. Inputs (states,
Hamiltonians, time grids) are fully deterministic.

Run from the repository root (bohnasa importable, e.g. ``pip install -e .``):

    python benchmarks/propagation_scaling.py                     # small sweep
    python benchmarks/propagation_scaling.py --ntimes 51 --reps 3
    python benchmarks/propagation_scaling.py --dims 66,231,861,2080,3300 \
        --stemr-dims 512,2048,8192,32768                          # bigger sweep
    python benchmarks/propagation_scaling.py --block-sweep        # _SPECTRAL_BLOCK

Peak-RSS per configuration additionally requires launching under
``/usr/bin/time -v``; the script itself reports median wall time only.
"""

import argparse
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import scipy.linalg as sla

import bohnasa
import bohnasa.otoc as oc

Row = tuple[str, int, int, float, float, int | None]


def _lattice_for(dim: int) -> int:
    """Smallest L with FixedNFockSpace(2, L).dim = C(L+1, 2) >= dim."""
    return int(np.ceil((-1 + np.sqrt(1.0 + 8.0 * dim)) / 2))


def _chain_system(L: int, ntimes: int):
    """Two bosons on an open chain with onsite U: banded, NOT tridiagonal in
    the dumped Fock order - the dense-eig regime the threshold guards."""
    fock = bohnasa.FixedNFockSpace(2, L)
    t = np.zeros((L, L))
    for j in range(L - 1):
        t[j, j + 1] = t[j + 1, j] = -1.0
    H = bohnasa.Hamiltonian(fock, one_body=t, onsite_u=0.8)
    probe = (bohnasa.Number(fock, 0) - bohnasa.Number(fock, 1)) * 0.5
    # seeded random probe state: generic overlap with the eigenbasis, so the
    # commutator actually grows (edge Fock states are near-dark here and
    # would make expm_multiply unrealistically cheap)
    rng = np.random.default_rng(7)
    state = rng.normal(size=fock.dim) + 1j * rng.normal(size=fock.dim)
    state /= np.linalg.norm(state)
    times = np.linspace(0.0, 2.0, ntimes)
    return H, probe, state, times


def _tridiagonal_system(dim: int, ntimes: int):
    """Dimer-form chain: N = dim - 1 bosons on L = 2 sites, offdiag +-2 -
    width-1 tridiagonal of exactly `dim` (the shape pinned by
    tests/unit/test_otoc.py), and a cheap O(L) Fock basis build."""
    fock = bohnasa.FixedNFockSpace(dim - 1, 2)
    t = np.array([[0.0, -2.0], [-2.0, 0.0]])
    H = bohnasa.Hamiltonian(fock, one_body=t, onsite_u=0.01)
    probe = (bohnasa.Number(fock, 0) - bohnasa.Number(fock, 1)) * 0.5
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    times = np.linspace(0.0, 2.0, ntimes)
    return H, probe, state, times


def _median_ms(fn: Callable[[], Any], reps: int) -> tuple[float, Any]:
    runs = []
    out = None
    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        out = fn()
        runs.append(1e3 * (time.perf_counter() - t0))
    runs.sort()
    return runs[len(runs) // 2], out


def sweep_eig(dims: list[int], ntimes: int, reps: int) -> Iterator[Row]:
    for dim in dims:
        H, probe, state, times = _chain_system(_lattice_for(dim), ntimes)
        if oc._is_tridiagonal(H.matrix):
            raise AssertionError(f"dim {dim}: chain is unexpectedly tridiagonal")
        ms, curve = _median_ms(
            lambda H=H, probe=probe, state=state, times=times: bohnasa.SquaredCommutator(
                H, probe, method="eig"
            ).curve(state, times),
            reps,
        )
        yield ("eig-dense", len(state), ntimes, ms, float(curve[-1]), None)


def sweep_krylov(dims: list[int], ntimes: int, reps: int) -> Iterator[Row]:
    for dim in dims:
        H, probe, state, times = _chain_system(_lattice_for(dim), ntimes)
        ms, curve = _median_ms(
            lambda H=H, probe=probe, state=state, times=times: bohnasa.SquaredCommutator(
                H, probe, method="krylov"
            ).curve(state, times),
            reps,
        )
        yield ("krylov", len(state), ntimes, ms, float(curve[-1]), None)


def sweep_stemr(stemr_dims: list[int], ntimes: int, reps: int) -> Iterator[Row]:
    for dim in stemr_dims:
        H, probe, state, times = _tridiagonal_system(dim, ntimes)
        sc = bohnasa.SquaredCommutator(H, probe)
        if sc.method != "eig" or not oc._is_tridiagonal(H.matrix):
            raise AssertionError(f"dim {dim}: expected the auto-dispatched STEMR path")
        # build (auto-dispatch + STEMR eigensolve) timed together with the
        # curve, as in the other modes
        ms, curve = _median_ms(
            lambda H=H, probe=probe, state=state, times=times: bohnasa.SquaredCommutator(
                H, probe
            ).curve(state, times),
            reps,
        )
        yield ("eig-stemr", dim, ntimes, ms, float(curve[-1]), None)


def sweep_floquet(dims: list[int], nkicks: int, reps: int) -> Iterator[Row]:
    # dense expm builds the step propagator once per dim: keep floquet-dims
    # small (O(dim^3), not part of the timed curve)
    for dim in dims:
        L = _lattice_for(dim)
        fock = bohnasa.FixedNFockSpace(2, L)
        t = np.zeros((L, L))
        for j in range(L - 1):
            t[j, j + 1] = t[j + 1, j] = -1.0
        dense = bohnasa.Hamiltonian(fock, one_body=t, onsite_u=0.8).matrix.toarray()
        u = sla.expm(-1j * 0.05 * dense)  # fixed-step propagator; build not timed
        probe = (bohnasa.Number(fock, 0) - bohnasa.Number(fock, 1)) * 0.5
        rng = np.random.default_rng(7)  # active probe state, see _chain_system
        state = rng.normal(size=fock.dim) + 1j * rng.normal(size=fock.dim)
        state /= np.linalg.norm(state)
        kicks = np.arange(nkicks + 1, dtype=float)
        sc = bohnasa.SquaredCommutator.from_propagator(u, probe)
        ms, curve = _median_ms(lambda sc=sc, state=state, kicks=kicks: sc.curve(state, kicks), reps)
        yield ("floquet", fock.dim, nkicks, ms, float(curve[-1]), None)


def sweep_block(dims: list[int], ntimes: int, reps: int, blocks: list[int]) -> Iterator[Row]:
    """_SPECTRAL_BLOCK plateau probe for the blocked spectral curve."""
    for dim in dims:
        H, probe, state, times = _chain_system(_lattice_for(dim), ntimes)
        for blk in blocks:
            original = oc._SPECTRAL_BLOCK
            oc._SPECTRAL_BLOCK = blk
            try:
                sc = bohnasa.SquaredCommutator(H, probe, method="eig")
                ms, curve = _median_ms(
                    lambda sc=sc, state=state, times=times: sc.curve(state, times), reps
                )
            finally:
                oc._SPECTRAL_BLOCK = original
            yield ("eig-dense", len(state), ntimes, ms, float(curve[-1]), blk)


def main() -> None:
    parser = argparse.ArgumentParser(description="OTOC propagation-mode scaling curves")
    parser.add_argument(
        "--dims", default="66,231,861,2080", help="comma list: dense-eig and krylov target dims"
    )
    parser.add_argument(
        "--stemr-dims", default="512,2048,8192", help="comma list: tridiagonal (STEMR) dims"
    )
    parser.add_argument(
        "--floquet-dims", default="66,231,861", help="comma list: Floquet dims (dense expm build)"
    )
    parser.add_argument("--ntimes", type=int, default=11, help="curve points per mode")
    parser.add_argument("--nkicks", type=int, default=16, help="max Floquet kick count")
    parser.add_argument("--reps", type=int, default=1, help="median over this many timed runs")
    parser.add_argument(
        "--modes",
        default="eig-dense,eig-stemr,krylov,floquet",
        help="comma list subset of modes to sweep",
    )
    parser.add_argument(
        "--block-sweep", action="store_true", help="also probe _SPECTRAL_BLOCK plateau"
    )
    parser.add_argument("--block-dims", default="861", help="dims for --block-sweep")
    parser.add_argument(
        "--blocks", default="128,256,1024,4096", help="block sizes for --block-sweep"
    )
    args = parser.parse_args()

    dims = [int(x) for x in args.dims.split(",") if x]
    stemr_dims = [int(x) for x in args.stemr_dims.split(",") if x]
    modes = {m.strip() for m in args.modes.split(",") if m.strip()}

    rows: list[Row] = []
    if "eig-dense" in modes:
        rows += list(sweep_eig(dims, args.ntimes, args.reps))
    if "eig-stemr" in modes:
        rows += list(sweep_stemr(stemr_dims, args.ntimes, args.reps))
    if "krylov" in modes:
        rows += list(sweep_krylov(dims, args.ntimes, args.reps))
    if "floquet" in modes:
        rows += list(
            sweep_floquet(
                [int(x) for x in args.floquet_dims.split(",") if x], args.nkicks, args.reps
            )
        )
    if args.block_sweep:
        bdims = [int(x) for x in args.block_dims.split(",") if x]
        blocks = [int(x) for x in args.blocks.split(",") if x]
        rows += list(sweep_block(bdims, max(args.ntimes, 1024), args.reps, blocks))

    print(f"# _DENSE_EIG_LIMIT={oc._DENSE_EIG_LIMIT}  _SPECTRAL_BLOCK={oc._SPECTRAL_BLOCK}")
    print(f"{'mode':10s} {'dim':>7s} {'n':>5s} {'median_ms':>10s} {'C_end':>13s}")
    for mode, dim, n, ms, c_end, blk in rows:
        tail = f"  block={blk:5d}" if blk is not None else ""
        print(f"{mode:10s} {dim:7d} {n:5d} {ms:10.1f} {c_end:13.6e}{tail}")


if __name__ == "__main__":
    main()
