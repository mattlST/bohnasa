"""Symmetry-sector decomposition of the fixed-N Fock space.

Models built from ``Hopping`` + ``Interaction`` (+ uniform ``ChemicalPotential``)
commute with the spatial symmetries of the chain: the dihedral group ``D_L``
(translations and reflection) for periodic boundary conditions, reflection only
for open ones. This class splits the fixed-N basis into the irreducible sectors
of that group and provides the change-of-basis isometries, so any symmetric
operator can be restricted to a block of size ``~dim/L`` and solved (or Lanczos-
recursed) independently. See https://github.com/mattlST/bohnasa/issues/13.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .fock_space import FixedNFockSpace
from .operators import Operator

# Numerical settings policy (issue #25): see CONTRIBUTING.md.
#
# Acceptance band for the reflection eigenvalue of a self-inverse momentum
# sector (k = 0, L/2): for mirror-invariant orbits the phase
# exp(-2 j pi k b / L) is exactly +-1 up to round-off, measured <= 1.4e-13
# over all k = L/2 phases with L < 200, while the smallest genuine mismatch
# (non-invariant orbit phase) over 4 <= L < 12 is 2.8e-1. The band separates
# noise from signal by >= 4 orders on either side.
_MIRROR_PHASE_TOL = 1e-12


@dataclass(frozen=True)
class SectorLabel:
    """Identifier of one symmetry sector.

    Attributes
    ----------
    k : int
        Crystal momentum quantum number, sector ``exp(2 j pi k / L)`` under a
        one-site translation. Always ``0`` for open boundaries, where ``k`` is
        meaningless and only ``parity`` labels the sector.
    parity : int or None
        ``+1`` / ``-1`` eigenvalue of the reflection; ``None`` for momentum
        sectors whose partner ``-k`` is distinct (``k != 0`` and, for even ``L``,
        ``k != L/2``), because the reflection maps those sectors into each other
        instead of onto itself.
    """

    k: int
    parity: int | None

    def __str__(self) -> str:
        if self.parity is None:
            return f"(k={self.k})"
        return f"(k={self.k}, p={'+' if self.parity > 0 else '-'}1)"


class SymmetrySectors:
    """The symmetry sectors of a fixed-N Fock space.

    Builds a symmetry-adapted basis per sector from translation/reflection
    orbits of the occupation patterns: a momentum-``k`` orbit vector is
    ``|n; k> = period^-0.5 * sum_a exp(-2 j pi k a / L) |roll(n, a)>``, defined
    only when ``k * period`` is a multiple of ``L`` (the orbit's little-group
    rule - short, sub-periodic orbits have momentum support on a subgroup of
    the Brillouin zone only). At the self-inverse momenta ``k = 0`` (and
    ``k = L/2`` for even ``L``) the reflection is combined into symmetric and
    antisymmetric partners, giving the two parity sectors.

    Parameters
    ----------
    fock : FixedNFockSpace
        Basis to decompose.
    connections : {'pbc', 'periodic', 'obc', 'open'}
        Boundary conditions defining the symmetry group: ``pbc`` gives
        dihedral ``(k, parity)`` sectors, ``obc`` reflection-parity sectors
        only.

    Attributes
    ----------
    labels : list of SectorLabel
        Sectors with nonzero dimension, ordered ``(0, +1), (0, -1), (1), (-1),
        (2), (-2), ...`` and ending at ``(L/2, +1), (L/2, -1)`` for even ``L``.
        ``obc`` yields ``(0, +1), (0, -1)``.

    Raises
    ------
    ValueError
        If ``connections`` is not a recognized boundary-condition string.

    Examples
    --------
    >>> from bohnasa import FixedNFockSpace
    >>> fock = FixedNFockSpace(2, 3)
    >>> sectors = SymmetrySectors(fock, connections="pbc")
    >>> [str(label) for label in sectors.labels]
    ['(k=0, p=+1)', '(k=1)', '(k=-1)']
    >>> [sectors.dim(label) for label in sectors.labels]
    [2, 2, 2]
    """

    def __init__(self, fock: FixedNFockSpace, connections: str = "pbc") -> None:
        if connections in ("pbc", "periodic"):
            periodic = True
        elif connections in ("obc", "open"):
            periodic = False
        else:
            raise ValueError(f"connections must be 'pbc' or 'obc', got {connections!r}")

        self._fock = fock
        self._periodic = periodic
        self._isometries: dict[SectorLabel, sp.csc_matrix] = {}
        self._labels: list[SectorLabel] = []

        if periodic:
            self._build_pbc()
        else:
            self._build_obc()

    # -- public API ---------------------------------------------------------

    @property
    def labels(self) -> list[SectorLabel]:
        """The sectors of this decomposition, in canonical order."""
        return list(self._labels)

    def dim(self, label: SectorLabel) -> int:
        """Dimension of one sector."""
        return self._get(label).shape[1]

    def isometry(self, label: SectorLabel) -> sp.csc_matrix:
        """Change-of-basis matrix ``U`` of one sector, shape ``(dim, dim_s)``.

        Columns are the orthonormal symmetry-adapted basis vectors in the full
        Fock basis; ``U @ U.conj().T`` is the sector's projector and
        ``U.conj().T @ A @ U`` its block.
        """
        return self._get(label)

    def restrict(self, op: Operator, label: SectorLabel) -> sp.csr_matrix:
        """Block of operator ``op`` acting inside one sector.

        Exact as a restriction of ``op`` only when ``op`` commutes with the
        symmetry group (e.g. ``Hopping``, ``Interaction``, uniform
        ``ChemicalPotential``); single-site operators such as ``Number`` map
        between sectors and are not covered here.
        """
        if op.matrix is None:
            raise ValueError("restrict: operator has no matrix representation")
        u = self._get(label)
        return (u.conj().T @ op.matrix @ u).tocsr()

    def to_sector(self, state: np.ndarray, label: SectorLabel) -> np.ndarray:
        """Project a full-space state vector into one sector's basis.

        States that live entirely in one sector (symmetry eigenstates, or
        symmetry-symmetrized initial conditions) are represented exactly, so
        Lanczos/Krylov runs started from ``to_sector(...)`` under ``restrict``
        blocks agree with full-space runs.
        """
        return self._get(label).conj().T @ np.asarray(state, dtype=complex)

    # -- construction -------------------------------------------------------

    def _get(self, label: SectorLabel) -> sp.csc_matrix:
        if label not in self._isometries:
            raise ValueError(f"unknown sector {label}; labels are {self._labels}")
        return self._isometries[label]

    def _add_sector(self, label: SectorLabel, columns: list[dict[int, complex]]) -> None:
        if not columns:
            return  # empty parity sector, e.g. all orbits self-mirror with p=+1
        rows, cols, vals = [], [], []
        for j, col in enumerate(columns):
            for row, val in col.items():
                rows.append(row)
                cols.append(j)
                vals.append(val)
        # built via COO triplets + conversion: scipy-stubs 1.15 (pinned by the
        # Python 3.10 CI arm) mis-resolves the COO-tuple overload of csc_matrix
        coo_data = (
            np.asarray(vals, dtype=complex),
            (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)),
        )
        u = sp.coo_matrix(coo_data, shape=(self._fock.dim, len(columns))).tocsc()
        self._isometries[label] = u
        self._labels.append(label)

    def _orbit_columns(
        self, orbit: list[tuple[int, ...]], k: int, period: int
    ) -> dict[int, complex]:
        """Symmetry-adapted orbit vector ``|n; k>`` as a row-index/value dict."""
        theta = -2j * np.pi * k / self._fock.L
        return {
            self._fock.dict[roll]: np.exp(theta * a) / np.sqrt(period)
            for a, roll in enumerate(orbit)
        }

    def _build_pbc(self) -> None:
        fock = self._fock
        L = fock.L

        # Translation orbits with their period (the length of the orbit) and
        # the location of the mirrored representative inside the orbit table.
        orbits: list[list[tuple[int, ...]]] = []
        periods: list[int] = []
        index_of: dict[tuple[int, ...], tuple[int, int]] = {}
        for state in fock.fockSpace:
            if state in index_of:
                continue
            rolls = [tuple(np.roll(state, a)) for a in range(L)]
            period = next(a for a in range(1, L + 1) if rolls[a % L] == state)
            orbit = list(rolls[:period])
            for a, roll in enumerate(orbit):
                index_of[roll] = (len(orbits), a)
            orbits.append(orbit)
            periods.append(period)

        mirror_of = [index_of[tuple(reversed(orbit[0]))] for orbit in orbits]

        def compatible(k: int, period: int) -> bool:
            return (k * period) % L == 0

        for k in _momenta(L):
            if not _is_self_inverse(k, L):
                # Generic sectors: every compatible orbit is one basis vector.
                # The reflection maps this sector onto its -k partner instead
                # of onto itself, so parity is not a quantum number here.
                columns = [
                    self._orbit_columns(orbit, k, period)
                    for orbit, period in zip(orbits, periods, strict=True)
                    if compatible(k, period)
                ]
                self._add_sector(SectorLabel(k, None), columns)
                continue

            # Self-inverse momenta (k = 0, and k = L/2 for even L) split
            # further by reflection parity.
            for parity in (1, -1):
                parity_columns: list[dict[int, complex]] = []
                paired: set[int] = set()
                for i, (orbit, period) in enumerate(zip(orbits, periods, strict=True)):
                    if i in paired or not compatible(k, period):
                        continue
                    j, b = mirror_of[i]
                    # U(refl) |i; k> = exp(-2 j pi k b / L) |j; k>
                    phase = np.exp(-2j * np.pi * k * b / L)
                    if j == i:
                        if abs(phase - parity) < _MIRROR_PHASE_TOL:
                            parity_columns.append(self._orbit_columns(orbit, k, period))
                    else:
                        parity_columns.append(
                            _combine(
                                self._orbit_columns(orbit, k, period),
                                self._orbit_columns(orbits[j], k, periods[j]),
                                parity * phase,
                            )
                        )
                        paired.update({i, j})
                self._add_sector(SectorLabel(k, parity), parity_columns)

    def _build_obc(self) -> None:
        fock = self._fock
        seen: set[tuple[int, ...]] = set()
        columns: dict[int, list[dict[int, complex]]] = {1: [], -1: []}
        for state in fock.fockSpace:
            if state in seen:
                continue
            mirrored = tuple(reversed(state))
            seen.update({state, mirrored})
            e_state = {fock.dict[state]: 1.0 + 0j}
            if mirrored == state:
                columns[1].append(e_state)  # antisymmetric partner vanishes
            else:
                e_mirror = {fock.dict[mirrored]: 1.0 + 0j}
                for parity in (1, -1):
                    columns[parity].append(_combine(e_state, e_mirror, parity))
        for parity in (1, -1):
            self._add_sector(SectorLabel(0, parity), columns[parity])


def _momenta(L: int) -> list[int]:
    """Signed momentum labels of all translation sectors, canonical order."""
    labels = [0]
    labels += [m for k in range(1, (L + 1) // 2) for m in (k, -k)]
    if L % 2 == 0:
        labels.append(L // 2)
    return labels


def _is_self_inverse(k: int, L: int) -> bool:
    """True where exp(2 j pi k / L) is real, i.e. reflection preserves sector k."""
    return k % L == 0 or (L % 2 == 0 and k % L == L // 2)


def _combine(a: dict[int, complex], b: dict[int, complex], weight: complex) -> dict[int, complex]:
    """(a + weight * b) / sqrt(2) over disjoint-support orthonormal vectors."""
    merged = dict(a)
    for row, val in b.items():
        merged[row] = merged.get(row, 0.0) + weight * val
    return {row: val / np.sqrt(2.0) for row, val in merged.items() if val != 0.0}
