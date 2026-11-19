r"""Time-resolved lattice visualization for evolved Bose-Hubbard data.

This module renders site positions and observable time series of ring
(``connections="pbc"``) and open-chain (``connections="obc"``) lattices as
matplotlib graphics: one-time-step snapshots (:func:`draw_lattice`),
trajectory replays (:func:`animate_lattice`) and 2-D site-vs-time fallback
views (:func:`occupation_heatmap`, :func:`momentum_heatmap`). It pairs with
:mod:`bohnasa.observables`, whose arrays - ``density_profile``,
``one_body_density_matrix``, ``pair_correlation``, ``momentum_distribution`` -
are exactly the inputs consumed here, for both the quantum
(``time_evolution_exact_ind``) and the mean-field
(:class:`bohnasa.MeanFieldLattice`) engines.

matplotlib is imported lazily inside the drawing functions: importing this
module (and the package as a whole) only needs numpy + scipy, and the first
plotting call raises an ``ImportError`` naming the ``examples`` extra when
matplotlib is missing.

Conventions are stated once here and pinned by the tests of issue #40
(amended 2026-11-19 grammar; see the issue's Amendments section):

Geometry.
    ``lattice_geometry`` places site ``i`` of a ring at angle ``2 pi i / L``
    on the unit circle (counterclockwise, site 0 at ``(1, 0)``), and site
    ``i`` of an open chain at ``(i, 0)``. The bond list - including the
    ``L = 2`` PBC convention that the ring's single bond is counted from
    both ends - is produced by :func:`bohnasa.operators.bond_pairs`, the
    same builder :class:`bohnasa.Hopping` uses, so the drawn lattice can
    never drift from the Hamiltonian.

Bond energy and bond current.
    With ``gamma_ij = <c_i^dag c_j>`` from
    :func:`bohnasa.one_body_density_matrix`, the symmetric bond energy is
    ``e_ij = 2 Re gamma_ij`` - invariant under ``i <-> j``, well defined on
    the undirected edge, and equal to the kinetic-energy contribution of
    that bond (in units of the hopping amplitude ``t = 1``): summed over
    the ring bond list it reproduces ``<Hopping>`` exactly. The directed
    bond current is ``j_ij = i (gamma_ji - gamma_ij) = 2 Im gamma_ij``;
    positive ``j_ij`` means particles flow from site ``i`` to site ``j``.
    The raw difference ``gamma_ji - gamma_ij`` of the original issue text
    is purely imaginary, so the factor ``i`` makes the encoded quantity
    real. For the ring Hamiltonian ``H = -t sum (c_j^dag c_i + h.c.)``
    over the same bond list, the continuity equation ``dn_i/dt =
    sum_j (J_j->i - J_i->j)`` with ``J_i->j = t j_ij`` holds to machine
    precision (derivation test).

Visual grammar.
    Node circle *area* is linear in on-site occupation ``<n_i>``
    (radius ∝ sqrt), colored on ``viridis``. The edge BASE (width +
    diverging ``coolwarm`` color) encodes the symmetric bond energy
    ``e_ij``: width ∝ ``|e|`` anchored at zero width for zero energy,
    color ∝ signed ``e`` under a trajectory-wide symmetric norm
    ``±max|e|`` (so zero is always the map center). The edge DIRECTION is
    an arrowhead along the bond pointing first -> second when ``j_ij > 0``
    (flipped when negative), with head size ∝ ``|j|`` under the
    trajectory-wide symmetric current norm ``max|j|``; currents below a
    stated threshold (5% of ``max|j|`` by default) draw no arrowhead, and
    the arrowhead color encodes nothing (size and direction carry the
    full signed information, so no third colorbar is needed). ``edge_mode
    = "coherence"`` replaces the base channel by ``|gamma_ij|`` on
    ``viridis`` while the arrows keep encoding the current. All
    normalizations are computed over the whole trajectory, never per
    frame. Colormaps are colorblind-safe, and whenever color encodes a
    quantity a colorbar is added.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .operators import bond_pairs

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.animation import FuncAnimation
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

__all__ = [
    "lattice_bonds",
    "lattice_geometry",
    "bond_energy",
    "bond_current",
    "bond_coherence",
    "draw_lattice",
    "animate_lattice",
    "occupation_heatmap",
    "momentum_heatmap",
]

_NODE_SIZE_RANGE = (4.0, 400.0)
_EDGE_WIDTH_RANGE = (0.6, 4.6)
_DOUBLON_SIZE_RANGE = (0.0, 150.0)
_ARROW_SCALE_RANGE = (6.0, 24.0)
# arrowhead cutoff as a fraction of the trajectory-wide max |current|
_ARROW_THRESHOLD = 0.05

_IMPORT_ERROR = (
    "bohnasa.visualization requires matplotlib, which is only needed by the "
    "'examples' extra: pip install 'bohnasa[examples]' (source checkouts: "
    "pip install -e '.[examples]')"
)


def _pyplot():
    """Import matplotlib.pyplot lazily, raising a friendly error if absent."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(_IMPORT_ERROR) from exc
    return plt


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


def _scale_linear(
    values: np.ndarray, vmin: float, vmax: float, out_min: float, out_max: float
) -> np.ndarray:
    """Linear map ``[vmin, vmax] -> [out_min, out_max]`` with clipping.

    A degenerate data range (``vmax == vmin``) maps to the midpoint of the
    output range: constant data draws a constant, visible encoding.
    """
    values = np.asarray(values, dtype=float)
    if vmax == vmin:
        return np.full(values.shape, 0.5 * (out_min + out_max), dtype=float)
    scaled = out_min + (out_max - out_min) * (values - vmin) / (vmax - vmin)
    return np.clip(scaled, out_min, out_max)


def _widths_from_magnitude(values: np.ndarray, scale: float, width_range: tuple[float, float]):
    """Edge widths linear in ``|value|``, zero-anchored, ``scale`` = trajectory max."""
    return width_range[0] + (width_range[1] - width_range[0]) * np.clip(
        np.abs(values) / scale, 0.0, 1.0
    )


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


def bond_energy(gamma: ArrayLike, bonds: ArrayLike) -> np.ndarray:
    r"""Symmetric bond energies ``e_ij = 2 Re gamma_ij = gamma_ij + gamma_ji``.

    One real value per bond, in units of the hopping amplitude: the
    kinetic-energy contribution of that bond to ``H = -t sum (c_j^dag c_i
    + h.c.)`` (with ``t = 1`` the sum over the ring or chain bond list
    equals ``<Hopping>`` - pinned in the tests). Being invariant under
    ``i <-> j`` it is well defined on the undirected edge, so it carries
    no orientation-dependent meaning and serves as the orientation-free
    base channel (edge width and color) of the visual grammar; the
    directed flow is carried separately by :func:`bond_current`.

    Parameters
    ----------
    gamma : array-like of complex, shape (L, L)
        One-body density matrix ``gamma_ij = <c_i^dag c_j>`` from
        :func:`bohnasa.one_body_density_matrix` (assumed normalized).
    bonds : array-like of int, shape (E, 2)
        Directed bond list, e.g. from :func:`lattice_bonds`.

    Returns
    -------
    numpy.ndarray of float, shape (E,)
        Bond energy per bond (same value for ``(i, j)`` and ``(j, i)``).

    Examples
    --------
    >>> gamma = np.array([[0.5, 0.25 + 0.25j], [0.25 - 0.25j, 0.5]])
    >>> bond_energy(gamma, [[0, 1], [1, 0]])
    array([0.5, 0.5])
    """
    gamma = np.asarray(gamma, dtype=complex)
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
        raise ValueError(f"bond_energy: gamma must be square 2d, got shape {gamma.shape}")
    bonds = _validate_bonds(bonds, gamma.shape[0])
    return 2.0 * gamma[bonds[:, 0], bonds[:, 1]].real


def bond_current(gamma: ArrayLike, bonds: ArrayLike) -> np.ndarray:
    r"""Bond currents ``j_ij = i (gamma_ji - gamma_ij) = 2 Im gamma_ij``.

    One real value per directed bond of ``bonds``, in units of the hopping
    amplitude; positive means flow from the bond's first site to its
    second (see the module docstring for the convention and the continuity
    derivation test that pins it). In the visual grammar this quantity
    drives the edge arrowhead (direction and size), not the edge base
    width or color.

    Parameters
    ----------
    gamma : array-like of complex, shape (L, L)
        One-body density matrix ``gamma_ij = <c_i^dag c_j>`` from
        :func:`bohnasa.one_body_density_matrix` (assumed normalized).
    bonds : array-like of int, shape (E, 2)
        Directed bond list, e.g. from :func:`lattice_bonds`.

    Returns
    -------
    numpy.ndarray of float, shape (E,)
        Signed current per bond.

    Examples
    --------
    >>> gamma = np.array([[0.5, 0.25j], [-0.25j, 0.5]])
    >>> bond_current(gamma, [[0, 1], [1, 0]])
    array([ 0.5, -0.5])
    """
    gamma = np.asarray(gamma, dtype=complex)
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
        raise ValueError(f"bond_current: gamma must be square 2d, got shape {gamma.shape}")
    bonds = _validate_bonds(bonds, gamma.shape[0])
    return 2.0 * gamma[bonds[:, 0], bonds[:, 1]].imag


def bond_coherence(gamma: ArrayLike, bonds: ArrayLike) -> np.ndarray:
    r"""Bond coherence ``|gamma_ij|`` per directed bond.

    The alternative edge encoding of :func:`animate_lattice`
    (``edge_mode="coherence"``): the magnitude of the one-body density
    matrix across each bond, a non-negative quantity.

    Parameters
    ----------
    gamma : array-like of complex, shape (L, L)
        One-body density matrix ``gamma_ij = <c_i^dag c_j>``.
    bonds : array-like of int, shape (E, 2)
        Directed bond list, e.g. from :func:`lattice_bonds`.

    Returns
    -------
    numpy.ndarray of float, shape (E,)
        ``|gamma_ij|`` per bond.

    Examples
    --------
    >>> gamma = np.array([[0.5, 0.25j], [-0.25j, 0.5]])
    >>> bond_coherence(gamma, [[0, 1], [1, 0]])
    array([0.25, 0.25])
    """
    gamma = np.asarray(gamma, dtype=complex)
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
        raise ValueError(f"bond_coherence: gamma must be square 2d, got shape {gamma.shape}")
    bonds = _validate_bonds(bonds, gamma.shape[0])
    return np.abs(gamma[bonds[:, 0], bonds[:, 1]])


class _LatticeArtists:
    """Matplotlib handles of a lattice view and their per-frame update (internal).

    Edges are ``Line2D`` objects (width and color of the base channel),
    arrowheads are one ``FancyArrowPatch`` per bond (hidden by default; the
    current channel sets direction, size and visibility), nodes are one
    ``PathCollection`` whose ``sizes`` are areas in points^2 - the visual
    grammar's "area linear in occupation" - and the optional doublon
    overlay a second, square-marked ``PathCollection``. When nodes carry
    data color the collection must be created with a scalar-mapped array:
    a collection created with an explicit ``color=`` silently ignores later
    ``set_array`` calls, which would freeze animation frames at one color.
    """

    def __init__(
        self,
        ax,
        xy: np.ndarray,
        bonds: np.ndarray,
        *,
        node_colors: np.ndarray | None = None,
        node_cmap=None,
        node_norm=None,
        edge_cmap=None,
        edge_norm=None,
        node_size_range: tuple[float, float] = _NODE_SIZE_RANGE,
        edge_width_range: tuple[float, float] = _EDGE_WIDTH_RANGE,
        arrow_scale_range: tuple[float, float] = _ARROW_SCALE_RANGE,
        arrow_color: str = "0.15",
    ):
        # function-level import: this class only ever runs after _pyplot()
        # has verified that matplotlib is present (lazy-import contract)
        from matplotlib.patches import FancyArrowPatch

        xy = _validate_xy(xy)
        bonds = _validate_bonds(bonds, len(xy))
        self._xy = xy
        self.bonds = bonds
        self.lines = [
            ax.plot(
                [xy[i, 0], xy[j, 0]],
                [xy[i, 1], xy[j, 1]],
                color="0.65",
                linewidth=edge_width_range[0],
                zorder=1,
                solid_capstyle="round",
            )[0]
            for i, j in bonds
        ]
        # arrowheads span the middle 40% of the bond (30% -> 70%) so they
        # never collide with the node circles; forward/backward position
        # pairs are precomputed, per frame only the pair selection changes
        self.arrows = []
        self._arrow_endpoints = []
        for i, j in bonds:
            delta = xy[j] - xy[i]
            tail = xy[i] + 0.30 * delta
            head = xy[i] + 0.70 * delta
            patch = ax.add_patch(
                FancyArrowPatch(
                    tail,
                    head,
                    arrowstyle="-|>",
                    color=arrow_color,
                    linewidth=1.2,
                    mutation_scale=arrow_scale_range[0],
                    zorder=2,
                )
            )
            patch.set_visible(False)
            self.arrows.append(patch)
            self._arrow_endpoints.append(((tail, head), (head, tail)))
        plain = node_colors is None
        self.nodes = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=np.full(len(xy), 0.5 * sum(node_size_range)),
            c=None if plain else np.asarray(node_colors, dtype=float),
            cmap=None if plain else node_cmap,
            norm=None if plain else node_norm,
            color="0.75" if plain else None,
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
        )
        self.edge_cmap = edge_cmap
        self.edge_norm = edge_norm
        self.node_size_range = node_size_range
        self.edge_width_range = edge_width_range
        self.arrow_scale_range = arrow_scale_range
        self.doublons = None
        ax.set_aspect("equal")
        ax.set_axis_off()

    def set_arrows(self, currents: np.ndarray, j_scale: float, threshold: float) -> None:
        """Drive the arrowheads from the signed bond currents.

        ``j_scale`` is the trajectory-wide max |current| and ``threshold``
        the no-arrow cutoff as a fraction of it; head size is linear in
        ``|j| / j_scale`` over ``arrow_scale_range`` and the arrow points
        first -> second for positive current (reversed endpoints otherwise).
        """
        lo, hi = self.arrow_scale_range
        for patch, (forward, backward), value in zip(
            self.arrows, self._arrow_endpoints, currents, strict=True
        ):
            magnitude = abs(float(value))
            if j_scale <= 0.0 or magnitude <= threshold * j_scale:
                patch.set_visible(False)
                continue
            pos_a, pos_b = forward if value > 0.0 else backward
            patch.set_positions(pos_a, pos_b)
            patch.set_mutation_scale(lo + (hi - lo) * min(magnitude / j_scale, 1.0))
            patch.set_visible(True)

    def set_frame(
        self,
        node_sizes: np.ndarray | None = None,
        node_colors: np.ndarray | None = None,
        edge_widths: np.ndarray | None = None,
        edge_colors: np.ndarray | None = None,
    ) -> None:
        """Apply one frame's prepared arrays to the artists."""
        if node_sizes is not None:
            self.nodes.set_sizes(np.asarray(node_sizes, dtype=float))
        if node_colors is not None:
            self.nodes.set_array(np.asarray(node_colors, dtype=float))
        if edge_widths is not None:
            for line, width in zip(self.lines, edge_widths, strict=True):
                line.set_linewidth(float(width))
        if edge_colors is not None:
            for line, value in zip(self.lines, edge_colors, strict=True):
                line.set_color(self.edge_cmap(self.edge_norm(value)))

    def add_doublons(self, sizes: np.ndarray):
        """Overlay square outlines (area encodes the value, color does not)."""
        self.doublons = self.nodes.axes.scatter(
            self._xy[:, 0],
            self._xy[:, 1],
            s=np.asarray(sizes, dtype=float),
            marker="s",
            facecolors="none",
            edgecolors="k",
            linewidths=1.0,
            zorder=4,
        )
        return self.doublons


def _check_data_length(name: str, arr: ArrayLike, expected: int) -> None:
    """Validate a data array's element count against its artist count."""
    size = np.asarray(arr).size
    if size != expected:
        raise ValueError(f"visualization: {name} must have {expected} entries, got {size}")


def draw_lattice(
    ax: Axes,
    xy: ArrayLike,
    *,
    connections: str | np.ndarray | Sequence[Sequence[int]] = "pbc",
    bonds: ArrayLike | None = None,
    node_sizes: ArrayLike | None = None,
    node_colors: ArrayLike | None = None,
    edge_widths: ArrayLike | None = None,
    edge_colors: ArrayLike | None = None,
    edge_currents: ArrayLike | None = None,
    node_vmin: float | None = None,
    node_vmax: float | None = None,
    edge_vmin: float | None = None,
    edge_vmax: float | None = None,
    node_size_range: tuple[float, float] = _NODE_SIZE_RANGE,
    edge_width_range: tuple[float, float] = _EDGE_WIDTH_RANGE,
    node_cmap: str = "viridis",
    edge_cmap: str = "coolwarm",
    node_label: str = r"$\langle \hat{n}_i \rangle$",
    edge_label: str = r"$e_{ij}$",
    arrow_threshold: float = _ARROW_THRESHOLD,
    arrow_scale_range: tuple[float, float] = _ARROW_SCALE_RANGE,
    arrow_color: str = "0.15",
) -> None:
    r"""Draw one static lattice snapshot on ``ax``.

    ``node_sizes`` and ``node_colors`` are length-``L`` data arrays (e.g.
    ``density_profile`` at one time): areas map linearly onto
    ``node_size_range`` (points^2, so radius ∝ sqrt of the value) and
    colors through ``node_cmap``. ``edge_widths`` and ``edge_colors`` are
    length-``E`` base-channel arrays aligned with ``bonds`` (default: the
    bond list of ``len(xy)`` sites under ``connections``): widths map
    linearly in absolute value onto ``edge_width_range`` (zero value
    draws the minimum width) and colors through ``edge_cmap``.
    ``edge_currents`` (the signed bond current of :func:`bond_current`)
    drives the arrowheads: size ∝ ``|j|`` over
    ``arrow_scale_range``, direction first -> second for ``j > 0``
    (flipped otherwise), no arrowhead at or below ``arrow_threshold``
    times the array's maximum ``|j|``. Normalization bounds default to
    the min/max of the passed node array and, for signed edge data, to
    the symmetric ``±max|edge_colors|`` so zero sits at the diverging
    colormap center.

    Callers replaying a trajectory should pass the trajectory-wide
    bounds explicitly, as :func:`animate_lattice` does, never per-frame
    ones. A colorbar is added whenever color encodes data.

    Warning
    -------
    Explicit ``edge_vmin``/``edge_vmax`` override the symmetric default;
    an asymmetric pair (e.g. a data minimum far from its maximum
    magnitude) makes zero current *not* the colormap center and can
    misread as "no bond energy". Keep signed diverging data symmetric,
    ``edge_vmin = -edge_vmax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    xy : array-like of float, shape (L, 2)
        Site positions, e.g. from :func:`lattice_geometry`.
    connections : {"pbc", "obc"} or array-like of shape (*, 2), default "pbc"
        Bond convention used when ``bonds`` is not given explicitly.
    bonds : array-like of int, shape (E, 2), optional
        Directed bond list the edge arrays are aligned with.
    node_sizes, node_colors : array-like of float, shape (L,), optional
        Per-site data; ``None`` draws plain gray nodes.
    edge_widths, edge_colors : array-like of float, shape (E,), optional
        Base-channel per-bond data (bond energy by convention); ``None``
        draws plain gray edges.
    edge_currents : array-like of float, shape (E,), optional
        Signed bond currents driving the arrowheads; ``None`` hides all
        arrowheads.
    node_vmin, node_vmax, edge_vmin, edge_vmax : float, optional
        Normalization bounds; node defaults are the data minima/maxima,
        edge defaults the symmetric ``±max|edge_colors|`` (see Warning).
    node_size_range : (float, float), default (4.0, 400.0)
        Node area (points^2) at the minimum/maximum value.
    edge_width_range : (float, float), default (0.6, 4.6)
        Edge width at value 0 / at the maximum absolute value.
    node_cmap, edge_cmap : str, default "viridis"/"coolwarm"
        Colorblind-safe colormaps for sites and bonds.
    node_label, edge_label : str
        Colorbar labels for the node/edge colorbars.
    arrow_threshold : float, default 0.05
        No-arrowhead cutoff, as a fraction of ``max|edge_currents|``.
    arrow_scale_range : (float, float), default (6.0, 24.0)
        Arrowhead ``mutation_scale`` (points) at threshold / at the
        maximum absolute current.
    arrow_color : str, default "0.15"
        Arrowhead color; encodes no data (direction and size carry the
        current), hence no colorbar.

    Returns
    -------
    None
        The artists are added to ``ax``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> draw_lattice(ax, lattice_geometry(4),
    ...              node_sizes=[1.0, 2.0, 3.0, 4.0],
    ...              node_colors=[1.0, 2.0, 3.0, 4.0])
    >>> nodes = ax.collections[0]
    >>> nodes.get_sizes()
    array([  4., 136., 268., 400.])
    >>> nodes.get_clim()
    (1.0, 4.0)
    >>> len(ax.lines)
    4
    >>> int(sum(p.get_visible() for p in ax.patches))
    0
    >>> draw_lattice(ax, lattice_geometry(4), edge_currents=[0.5, -0.5, 0.1, 0.0])
    >>> int(sum(p.get_visible() for p in ax.patches))
    3
    >>> plt.close(fig)
    """
    plt = _pyplot()
    xy = _validate_xy(xy)
    if bonds is None:
        bonds = lattice_bonds(len(xy), connections)
    bonds = _validate_bonds(bonds, len(xy))
    for name, arr in (("node_sizes", node_sizes), ("node_colors", node_colors)):
        if arr is not None:
            _check_data_length(name, arr, len(xy))
    for name, arr in (
        ("edge_widths", edge_widths),
        ("edge_colors", edge_colors),
        ("edge_currents", edge_currents),
    ):
        if arr is not None:
            _check_data_length(name, arr, len(bonds))

    node_norm = None
    sizes_arr = None if node_sizes is None else np.asarray(node_sizes, dtype=float)
    colors_arr = None if node_colors is None else np.asarray(node_colors, dtype=float)
    if colors_arr is not None:
        vmin = float(colors_arr.min()) if node_vmin is None else float(node_vmin)
        vmax = float(colors_arr.max()) if node_vmax is None else float(node_vmax)
        node_norm = plt.Normalize(vmin=vmin, vmax=vmax)

    artists = _LatticeArtists(
        ax,
        xy,
        bonds,
        node_colors=colors_arr,
        node_cmap=plt.get_cmap(node_cmap),
        node_norm=node_norm,
        edge_cmap=plt.get_cmap(edge_cmap),
        edge_norm=None,
        node_size_range=node_size_range,
        edge_width_range=edge_width_range,
        arrow_scale_range=arrow_scale_range,
        arrow_color=arrow_color,
    )

    if sizes_arr is not None:
        svmin = float(sizes_arr.min()) if node_vmin is None else float(node_vmin)
        svmax = float(sizes_arr.max()) if node_vmax is None else float(node_vmax)
        artists.set_frame(node_sizes=_scale_linear(sizes_arr, svmin, svmax, *node_size_range))
    if node_colors is not None:
        ax.figure.colorbar(artists.nodes, ax=ax, label=node_label, shrink=0.8, pad=0.02)

    edge_arr = None if edge_colors is None else np.asarray(edge_colors, dtype=float)
    if edge_arr is not None:
        if edge_vmin is None and edge_vmax is None:
            # signed diverging base channel defaults to a symmetric norm so
            # zero (no bond energy) is exactly the colormap center
            amax = float(np.max(np.abs(edge_arr))) or 1.0
            vmin, vmax = -amax, amax
        else:
            vmin = float(edge_arr.min()) if edge_vmin is None else float(edge_vmin)
            vmax = float(edge_arr.max()) if edge_vmax is None else float(edge_vmax)
        artists.edge_norm = plt.Normalize(vmin=vmin, vmax=vmax)
        mappable = plt.cm.ScalarMappable(norm=artists.edge_norm, cmap=plt.get_cmap(edge_cmap))
        mappable.set_array([])
        ax.figure.colorbar(mappable, ax=ax, label=edge_label, shrink=0.8, pad=0.02)
    if edge_widths is not None:
        width_arr = np.asarray(edge_widths, dtype=float)
        scale = float(np.max(np.abs(width_arr))) or 1.0
        artists.set_frame(edge_widths=_widths_from_magnitude(width_arr, scale, edge_width_range))
    if edge_arr is not None:
        artists.set_frame(edge_colors=edge_arr)
    if edge_currents is not None:
        current_arr = np.asarray(edge_currents, dtype=float)
        artists.set_arrows(current_arr, float(np.max(np.abs(current_arr))), arrow_threshold)


def animate_lattice(
    times: ArrayLike,
    occupations: ArrayLike,
    gamma: ArrayLike | None = None,
    *,
    connections: str = "pbc",
    edge_mode: str = "energy",
    doublons: ArrayLike | None = None,
    node_size_range: tuple[float, float] = _NODE_SIZE_RANGE,
    edge_width_range: tuple[float, float] = _EDGE_WIDTH_RANGE,
    node_cmap: str = "viridis",
    edge_cmap: str | None = None,
    arrow_threshold: float = _ARROW_THRESHOLD,
    arrow_scale_range: tuple[float, float] = _ARROW_SCALE_RANGE,
    arrow_color: str = "0.15",
    interval: float = 50,
    ax: Axes | None = None,
) -> FuncAnimation:
    r"""Replay a time-evolved trajectory as a lattice animation.

    ``occupations`` holds ``<n_i(t)>`` per site (rows = time index, e.g.
    stacked ``density_profile``), and ``gamma`` the one-body density
    matrices per time (shape ``(T, L, L)``, e.g. stacked
    ``one_body_density_matrix``). Per grammar (issue #40, amended): the
    edge BASE - width ∝ ``|e|`` and diverging ``coolwarm`` color of
    :func:`bond_energy` ``e_ij = 2 Re gamma_ij`` (``edge_mode="energy"``,
    default; :func:`bond_coherence` ``|gamma_ij|`` on ``viridis`` for
    ``edge_mode="coherence"``) - sums to ``<Hopping>`` over the bonds,
    while an ARROWHEAD per bond shows the directed current
    :func:`bond_current` ``j_ij = 2 Im gamma_ij``: pointing first ->
    second when positive (flipped otherwise), head size ∝ ``|j|``, hidden
    at or below ``arrow_threshold`` times the trajectory-wide max
    ``|j|``. ``doublons`` (shape ``(T, L)``, e.g. the diagonal of
    ``pair_correlation``) optionally overlays square outlines whose area
    is linear in the doublon number.

    Every normalization is trajectory-wide (min/max over all frames of
    ``occupations``; symmetric ``±max|e|`` for the energy base, ``0`` to
    ``max`` for coherence; ``max|j|`` for the arrow scale; the doublon
    maximum), never per frame. The animation is purely data-driven and
    headless-safe: saving it twice yields byte-identical files, and no
    measured wall-clock quantity enters any output.

    Parameters
    ----------
    times : array-like of float, shape (T,)
        Sample times, one per frame (used for titles and frame rate only).
    occupations : array-like of float, shape (T, L)
        On-site occupations per frame.
    gamma : array-like of complex, shape (T, L, L), optional
        One-body density matrices per frame; without it edges stay gray
        and no arrowheads are drawn.
    connections : {"pbc", "obc"}, default "pbc"
        Lattice geometry.
    edge_mode : {"energy", "coherence"}, default "energy"
        Base-channel bond quantity drawn as edge width and color; the
        current always drives the arrowheads (a ``"current"`` request is
        rejected - the amended grammar moved it off the edge base).
    doublons : array-like of float, shape (T, L), optional
        On-site doublon numbers for the square overlay.
    node_size_range : (float, float), default (4.0, 400.0)
        Node area range (points^2) over the trajectory-wide extrema.
    edge_width_range : (float, float), default (0.6, 4.6)
        Edge width range over ``|base value| in [0, trajectory maximum]``.
    node_cmap : str, default "viridis"
        Site colormap (colorblind-safe).
    edge_cmap : str, optional
        Base-channel colormap; defaults to ``"coolwarm"`` for
        ``"energy"`` and ``"viridis"`` for ``"coherence"``.
    arrow_threshold : float, default 0.05
        No-arrowhead cutoff as a fraction of the trajectory-wide max
        ``|current|`` (the stated zero-current threshold of the grammar).
    arrow_scale_range : (float, float), default (6.0, 24.0)
        Arrowhead ``mutation_scale`` (points) at the threshold and at the
        trajectory-wide max ``|current|``.
    arrow_color : str, default "0.15"
        Arrowhead color; encodes no data, so it adds no colorbar.
    interval : float, default 50
        Milliseconds between frames in a live replay - a nominal
        parameter, not a measured time; saving derives its fps from it.
    ax : matplotlib.axes.Axes, optional
        Target axes; a new figure is created when omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object - keep a reference to it while showing or
        saving. Frame ``k`` corresponds to ``times[k]``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> times = np.array([0.0, 1.0, 2.0])
    >>> occ = np.array([[1.0, 1.0], [1.5, 0.5], [2.0, 0.0]])
    >>> fig, ax = plt.subplots()
    >>> anim = animate_lattice(times, occ, ax=ax)
    >>> nodes = ax.collections[0]
    >>> nodes.get_clim()
    (0.0, 2.0)
    >>> nodes.get_sizes()
    array([202., 202.])
    >>> len(ax.patches), int(sum(p.get_visible() for p in ax.patches))
    (2, 0)
    >>> ax.get_title()
    't = 0'
    >>> ax.figure.canvas.draw()  # one draw event keeps matplotlib quiet on GC
    >>> plt.close(fig)
    """
    plt = _pyplot()
    from matplotlib import animation

    times = np.asarray(times, dtype=float)
    occ = np.asarray(occupations, dtype=float)
    if times.ndim != 1 or occ.ndim != 2 or occ.shape[0] != times.size:
        raise ValueError(
            f"animate_lattice: need times (T,) and occupations (T, L), got {times.shape} "
            f"and {occ.shape}"
        )
    if edge_mode == "current":
        raise ValueError(
            'animate_lattice: edge_mode="current" was replaced by the arrowhead channel '
            '(issue #40 amendment); use "energy" or "coherence" for the edge base'
        )
    if edge_mode not in ("energy", "coherence"):
        raise ValueError('animate_lattice: edge_mode must be "energy" or "coherence"')
    n_time, n_sites = occ.shape
    if n_time == 0:
        raise ValueError("animate_lattice: empty trajectory")
    xy = lattice_geometry(n_sites, connections)
    bonds = lattice_bonds(n_sites, connections)

    base_series = None
    current_series = None
    if gamma is not None:
        gamma = np.asarray(gamma, dtype=complex)
        if gamma.shape != (n_time, n_sites, n_sites):
            raise ValueError(
                f"animate_lattice: gamma must have shape {(n_time, n_sites, n_sites)}, "
                f"got {gamma.shape}"
            )
        getter = bond_energy if edge_mode == "energy" else bond_coherence
        base_series = np.array([getter(g, bonds) for g in gamma])
        current_series = np.array([bond_current(g, bonds) for g in gamma])
    if doublons is not None:
        dbl = np.asarray(doublons, dtype=float)
        if dbl.shape != occ.shape:
            raise ValueError(f"animate_lattice: doublons must have shape {occ.shape}")
    else:
        dbl = None

    node_vmin, node_vmax = float(occ.min()), float(occ.max())
    base_scale = 0.0
    if base_series is not None and base_series.size:
        base_scale = (
            float(np.max(base_series))
            if edge_mode == "coherence"
            else float(np.max(np.abs(base_series)))
        )
    if base_scale <= 0.0:
        base_scale = 1.0  # all-zero base: finite normalization, center color, min width
    base_vmin = 0.0 if edge_mode == "coherence" else -base_scale
    # trajectory-wide symmetric current norm for the arrows (0 allowed: all hidden)
    current_scale = 0.0 if current_series is None else float(np.max(np.abs(current_series)))

    cmap_nodes = plt.get_cmap(node_cmap)
    norm_nodes = plt.Normalize(vmin=node_vmin, vmax=node_vmax)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    cmap_edges = (
        None
        if base_series is None
        else plt.get_cmap(edge_cmap or ("coolwarm" if edge_mode == "energy" else "viridis"))
    )
    norm_edges = None if base_series is None else plt.Normalize(vmin=base_vmin, vmax=base_scale)
    artists = _LatticeArtists(
        ax,
        xy,
        bonds,
        node_colors=occ[0],
        node_cmap=cmap_nodes,
        node_norm=norm_nodes,
        edge_cmap=cmap_edges,
        edge_norm=norm_edges,
        node_size_range=node_size_range,
        edge_width_range=edge_width_range,
        arrow_scale_range=arrow_scale_range,
        arrow_color=arrow_color,
    )
    fig.colorbar(artists.nodes, ax=ax, label=r"$\langle \hat{n}_i \rangle$", shrink=0.8, pad=0.02)
    if base_series is not None:
        mappable = plt.cm.ScalarMappable(norm=norm_edges, cmap=cmap_edges)
        mappable.set_array([])
        label = r"$e_{ij}$" if edge_mode == "energy" else r"$|\gamma_{ij}|$"
        fig.colorbar(mappable, ax=ax, label=label, shrink=0.8, pad=0.02)
    dbl_max = 0.0 if dbl is None else float(dbl.max())
    dbl_scale = dbl_max if dbl_max > 0.0 else 1.0  # trajectory-wide, zero-safe

    def doublon_areas(values: np.ndarray) -> np.ndarray:
        return _scale_linear(values, 0.0, dbl_scale, *_DOUBLON_SIZE_RANGE)

    doublon_layer = None if dbl is None else artists.add_doublons(doublon_areas(dbl[0]))
    title = ax.set_title("")

    def frame(k: int) -> Iterable[Artist]:
        """Apply data frame k - a pure function of k, hence deterministic.

        Returns the updated artists (the ``FuncAnimation`` callback
        contract; harmless with ``blit=False``, forward-compatible).
        """
        sizes = _scale_linear(occ[k], node_vmin, node_vmax, *node_size_range)
        widths = None
        if base_series is not None:
            widths = _widths_from_magnitude(base_series[k], base_scale, edge_width_range)
        if doublon_layer is not None and dbl is not None:
            doublon_layer.set_sizes(doublon_areas(dbl[k]))
        artists.set_frame(
            node_sizes=sizes,
            node_colors=occ[k],
            edge_widths=widths,
            edge_colors=None if base_series is None else base_series[k],
        )
        if current_series is not None:
            artists.set_arrows(current_series[k], current_scale, arrow_threshold)
        title.set_text(f"t = {times[k]:g}")
        updated: list[Artist] = [artists.nodes, title, *artists.lines, *artists.arrows]
        if doublon_layer is not None:
            updated.append(doublon_layer)
        return updated

    frame(0)
    return animation.FuncAnimation(
        fig, frame, frames=n_time, interval=interval, blit=False, repeat=True
    )


def _time_heatmap(times, values, ax, vmin, vmax, cmap, ylabel, cbar_label) -> Axes:
    """Shared site/mode-vs-time heatmap (internal).

    Normalization defaults to the full-array min/max, which for a 2-D
    heatmap argument is the whole trajectory by construction - never a
    per-frame window.
    """
    plt = _pyplot()
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("heatmap: times must be a 1d array of at least two samples")
    if values.ndim != 2 or values.shape[0] != times.size:
        raise ValueError(
            f"heatmap: values must have shape (len(times), n), got {values.shape} "
            f"for {times.size} times"
        )
    if vmin is None:
        vmin = float(values.min())
    if vmax is None:
        vmax = float(values.max())
    if vmax == vmin:
        vmax = vmin + 1.0  # constant data: any unit interval renders it flat
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    n_cols = values.shape[1]
    mesh = ax.pcolormesh(
        times,
        np.arange(n_cols, dtype=float),
        values.T,
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        shading="nearest",
    )
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    fig.colorbar(mesh, ax=ax, label=cbar_label)
    return ax


def occupation_heatmap(
    times: ArrayLike,
    n_i_t: ArrayLike,
    *,
    ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> Axes:
    r"""Site-resolved occupation ``n_i(t)`` as a site-vs-time heatmap.

    The 2-D fallback view for trajectories that are hard to read as an
    animation. ``n_i_t`` holds stacked ``density_profile`` rows (time along
    axis 0); the color normalization defaults to the trajectory-wide
    min/max and a colorbar is always added.

    Parameters
    ----------
    times : array-like of float, shape (T,)
        Sample times, at least two.
    n_i_t : array-like of float, shape (T, L)
        ``density_profile`` per time (rows = times).
    ax : matplotlib.axes.Axes, optional
        Target axes; a new figure is created when omitted.
    vmin, vmax : float, optional
        Color normalization; defaults are the array min/max.
    cmap : str, default "viridis"
        Colorblind-safe colormap.

    Returns
    -------
    matplotlib.axes.Axes
        The axes carrying the mesh (a ``QuadMesh`` in ``ax.collections``).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> ax = occupation_heatmap([0.0, 1.0], np.array([[1.0, 2.0], [3.0, 1.0]]))
    >>> ax.collections[0].get_clim()
    (1.0, 3.0)
    >>> ax.get_ylabel()
    'site $i$'
    >>> plt.close("all")
    """
    return _time_heatmap(
        times, n_i_t, ax, vmin, vmax, cmap, "site $i$", r"$\langle \hat{n}_i \rangle$"
    )


def momentum_heatmap(
    times: ArrayLike,
    n_k_t: ArrayLike,
    *,
    ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> Axes:
    r"""Mode-resolved occupation ``n_k(t)`` as a mode-vs-time heatmap.

    The same view as :func:`occupation_heatmap` for stacked
    ``momentum_distribution`` rows; the y axis is the ring momentum mode
    index (``k_n = 2 pi n / L``, the convention of
    :func:`bohnasa.momentum_distribution`).

    Parameters
    ----------
    times : array-like of float, shape (T,)
        Sample times, at least two.
    n_k_t : array-like of float, shape (T, L)
        ``momentum_distribution`` per time (rows = times).
    ax : matplotlib.axes.Axes, optional
        Target axes; a new figure is created when omitted.
    vmin, vmax : float, optional
        Color normalization; defaults are the array min/max.
    cmap : str, default "viridis"
        Colorblind-safe colormap.

    Returns
    -------
    matplotlib.axes.Axes
        The axes carrying the mesh.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> ax = momentum_heatmap([0.0, 1.0], np.array([[2.0, 0.5], [1.0, 1.5]]))
    >>> ax.collections[0].get_clim()
    (0.5, 2.0)
    >>> ax.get_ylabel()
    'mode $k$'
    >>> plt.close("all")
    """
    return _time_heatmap(times, n_k_t, ax, vmin, vmax, cmap, "mode $k$", r"$n_k$")
