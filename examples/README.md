# Examples

Runnable scripts showing what `bohnasa` does. Each one answers a physics
question, ends with self-check assertions, and is smoke-tested headless in CI
(`MPLBACKEND=Agg`).

| file | topic |
|------|-------|
| `ex01_fock_space.py` | the fixed-N Fock basis: enumeration, dimension, index lookup |
| `ex02_operators.py` | operator assembly, hopping structure, ground state via Lanczos vs exact |
| `ex03_krylov_complexity.py` | Krylov spread ⟨K̂⟩(t) after a staggered coherent quench |
| `ex04_transmon_boson.py` | attractive bosons with disorder: eigenstate participation (IPR) in Fock space |
| `ex05_symmetry_sectors.py` | momentum/parity blocks: spectrum reassembly and a Krylov quench computed inside one sector |
| `ex06_momentum_spectra.py` | per-sector eigenvalues vs the analytic free-boson solution, and interacting ladders with the k/−k degeneracy |

Run:

    pip install -e ".[examples]"
    python examples/ex01_fock_space.py   # ... or any other

Plot windows open by default; set `MPLBACKEND=Agg` to run headless.
