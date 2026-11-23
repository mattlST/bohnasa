# Examples

Runnable scripts showing what `bohnasa` does. Each one answers a physics
question, ends with self-check assertions, and is smoke-tested headless in CI
(`MPLBACKEND=Agg`).

| file | topic |
|------|-------|
| `ex01_fock_space.py` | the fixed-N Fock basis: enumeration, dimension, index lookup |
| `ex02_operators.py` | operator assembly by hand and via the generic `Hamiltonian` builder, ground state via Lanczos vs exact |
| `ex03_krylov_complexity.py` | Krylov spread ⟨K̂⟩(t) after a staggered coherent quench |
| `ex04_transmon_boson.py` | attractive bosons with disorder: eigenstate participation (IPR) in Fock space |
| `ex05_symmetry_sectors.py` | momentum/parity blocks: spectrum reassembly and a Krylov quench computed inside one sector |
| `ex06_momentum_spectra.py` | per-sector eigenvalues vs the analytic free-boson solution, and interacting ladders with the k/−k degeneracy |
| `ex07_operator_growth.py` | Liouvillian operator growth: Krylov spread of a local density, free closure vs interacting climb |
| `ex08_observables.py` | state diagnostics: occupations, momentum distribution, bunching and S(k), pinned on exact plane-wave anchors |
| `ex09_otoc.py` | squared-commutator scrambling growth and its Ehrenfest window across particle numbers |
| `ex10_kicked_dimer.py` | Trotterized dimer driving: small periods regain the continuous curve, larger ones weaken the growth rate |
| `ex11_mean_field_lattice.py` | general L-site mean-field DNLS engine: exact L=2 dimer reduction, L=4 fixed-point branches, three-way Lyapunov cross-check |
| `ex12_lattice_visualization.py` | the geometric view: L=6 ring quench of a current-carrying winding Gaussian packet — pulsing node sizes, bond-energy-colored edges with prominent current arrowheads, header/legend/live captions, one shared colorbar pair (headless GIF + frame strip, byte-deterministic), occupation/momentum heatmaps |
| `ex13_kicked_ring.py` | kicked Bose-Hubbard ring: blocked Floquet quasienergy branches vs U, Floquet OTOC against the continuous-time curve, kicked density propagation, classical kicked-map orbits (all issue #29 validations run as guards) |
| `ex14_twa_semiclassical.py` | truncated-Wigner bridge: quantum vs TWA vs classical OTOC curves through the `2 lambda_s -> lambda_s` crossover, Monte-Carlo sample convergence on a regular orbit, occupation heatmaps of the sample mean vs the mean-field orbit (byte-deterministic) |

Run:

    pip install -e ".[examples]"
    python examples/ex01_fock_space.py   # ... or any other

Plot windows open by default; set `MPLBACKEND=Agg` to run headless.
