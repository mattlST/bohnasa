from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ..operators import bond_pairs


def _validate_xy(xy: ArrayLike) -> np.ndarray:
    """Coerce site positions to a float array of shape (L, 2)."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be a 2d array with shape (L, 2), got {xy.shape}")
    return xy


def _validate_bonds(bonds: ArrayLike, n_sites: int) -> np.ndarray:
    """Coerce a bond list to an array of shape (E, 2) inside the site range."""
    bonds = np.asarray(bonds)
    if bonds.ndim != 2 or bonds.shape[1] != 2:
        raise ValueError(f"bonds must be a 2d array with shape (E, 2), got {bonds.shape}")
    if bonds.size:
        if bonds.min() < 0 or bonds.max() >= n_sites:
            raise ValueError(
                f"bond indices must lie in [0, {n_sites}), got extremes "
                f"{bonds.min()} and {bonds.max()}"
            )
    return bonds


def lattice_geometry(L: int, connections: str = "pbc") -> np.ndarray:
    r"""Site positions ``(x, y)`` for an ``L``-site ring or open chain.

    Parameters
    ----------
    L : int
        Number of lattice sites, ``L >= 1``.
    connections : {"pbc", "obc"}, default "pbc"
        ``"pbc"`` places the sites counterclockwise on the unit circle
        (site 0 at ``(1, 0)``), matching the ring bond list of
        :func:`lattice_bonds`; ``"obc"`` places them on the x-axis at
        integer positions.

    Returns
    -------
    numpy.ndarray of float, shape (L, 2)
        Site coordinates, one row per site index.

    Raises
    ------
    TypeError
        If ``L`` is not an integer.
    ValueError
        If ``L < 1`` or ``connections`` is neither ``"pbc"`` nor ``"obc"``.

    Notes
    -----
    The geometry only fixes *where* sites are drawn; which site pairs are
    bonded is decided by :func:`lattice_bonds`, i.e. by the very list
    :class:`bohnasa.Hopping` uses (including the deliberate ``L = 2`` PBC
    double-counting of the single ring bond).

    Examples
    --------
    >>> lattice_geometry(2, "obc")
    array([[0., 0.],
           [1., 0.]])
    >>> xy = lattice_geometry(6)
    >>> xy.shape
    (6, 2)
    >>> bool(np.allclose(np.hypot(xy[:, 0], xy[:, 1]), 1.0))
    True
    >>> float(np.round(lattice_geometry(4)[1, 1], 12))
    1.0
    """
    if isinstance(L, bool) or not isinstance(L, (int, np.integer)):
        raise TypeError(f"lattice_geometry: L must be an int, got {type(L).__name__}")
    if L < 1:
        raise ValueError(f"lattice_geometry: L must be >= 1, got {L}")
    if connections in ("pbc", "periodic"):
        angles = 2.0 * np.pi * np.arange(L) / L
        return np.column_stack((np.cos(angles), np.sin(angles)))
    if connections in ("obc", "open"):
        return np.column_stack((np.arange(L, dtype=float), np.zeros(L)))
    raise ValueError(f'lattice_geometry: connections must be "pbc" or "obc", got {connections!r}')


def lattice_bonds(
    L: int, connections: str | np.ndarray | Sequence[Sequence[int]] = "pbc"
) -> np.ndarray:
    r"""Directed bond list ``(i, j)`` matching the drawn geometry.

    Thin delegation to :func:`bohnasa.operators.bond_pairs` - the single
    builder behind :class:`bohnasa.Hopping` - so edge arrays always align
    with the Hamiltonian's bond convention, including the ``L = 2`` PBC
    ring whose single bond appears twice (once per direction).

    Parameters
    ----------
    L : int
        Number of lattice sites.
    connections : {"pbc", "obc"} or array-like of shape (*, 2), default "pbc"
        Ring, open chain, or an explicit bond list passed through.

    Returns
    -------
    numpy.ndarray of int, shape (*, 2)
        Ordered bond list; ``bonds[e] = (i, j)``.

    Examples
    --------
    >>> lattice_bonds(4)
    array([[0, 3],
           [1, 0],
           [2, 1],
           [3, 2]])
    >>> lattice_bonds(2)
    array([[0, 1],
           [1, 0]])
    >>> lattice_bonds(3, "obc")
    array([[0, 1],
           [1, 2]])
    """
    return bond_pairs(L, connections)
