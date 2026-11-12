# bohnasa

[![Python package](https://github.com/mattlST/bohnasa/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/mattlST/bohnasa/actions/workflows/python-package.yml)

Exact-diagonalization tools for bosonic quantum many-body systems, working in
the fixed particle-number sector of a Fock space. Built for studying operator
growth and Krylov complexity after a quench.

## Features

- `FixedNFockSpace` — enumerate the basis of `N` bosons on `L` sites, with
  occupation-pattern index lookup
- second-quantized operators on sparse matrices: `Number`, `Correlator`
  (`c_j† c_i`), `Hopping`, `Interaction`, `ChemicalPotential`
- `lanczos` — Lanczos recursion with full reorthogonalization
- `krylov_complexity` — the Krylov-level operator, time-evolvable after a
  quench from a projected coherent state (`fixed_n_coherent_state`,
  `time_evolution_exact_ind`)
- runnable [examples](examples/README.md) covering all of the above

## Install

```bash
pip install .                      # core: numpy + scipy only
pip install -e ".[dev,examples]"   # to hack on the package / run examples
```

Requires Python ≥ 3.10.

## Quickstart

Ground state of an interacting open chain by Lanczos, checked against exact
diagonalization:

```python
import numpy as np
import bohnasa

fock = bohnasa.FixedNFockSpace(N=6, L=4)
H = bohnasa.Hopping(fock, connections="obc") * -1.0 + bohnasa.Interaction(fock) * 0.5

v, T = bohnasa.lanczos(np.ones(fock.dim, dtype=complex) / np.sqrt(fock.dim), H, 40)
m = np.count_nonzero(np.linalg.norm(v, axis=1) > 1e-12)
print(f"ground state energy: {np.linalg.eigvalsh(T[:m, :m].real)[0]:.6f}")
```

Krylov complexity operator for a staggered coherent-state quench:

```python
phi = 1j * np.array([(-1) ** j for j in range(fock.L)], dtype=complex)
sta = bohnasa.fixed_n_coherent_state(fock, phi / np.linalg.norm(phi))
K = bohnasa.krylov_complexity(fock, H, sta, fock.dim - 1)
```

`examples/ex02_operators.py` and `examples/ex03_krylov_complexity.py` carry
both stories to completion, plots included.

## Testing

```bash
pytest --doctest-modules bohnasa/ tests/ -q --cov=bohnasa --cov-fail-under=90
```

## Status

Research code. The public API froze at `v0.1.0`; expect additions, not
breakage.

## License & citation

MIT — see [LICENSE](LICENSE). If you use this in published work, please cite
via `CITATION.cff`.
