from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ._plot import _pyplot
from .artists import _LatticeArtists
from .captions import _legend_lines, _live_text, _place_header, _validate_live
from .channels import bond_coherence, bond_current, bond_energy
from .encoding import (
    _ARROW_LINEWIDTH,
    _ARROW_SCALE_RANGE,
    _ARROW_SHAFT,
    _ARROW_THRESHOLD,
    _DOUBLON_SIZE_RANGE,
    _EDGE_WIDTH_RANGE,
    _NODE_SIZE_RANGE,
    _guard_scale,
    _scale_linear,
    _widths_from_magnitude,
)
from .geometry import _validate_bonds, _validate_xy, lattice_bonds, lattice_geometry

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.animation import FuncAnimation
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes


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
    edge_scale: float | None = None,
    current_scale: float | None = None,
    node_size_range: tuple[float, float] = _NODE_SIZE_RANGE,
    edge_width_range: tuple[float, float] = _EDGE_WIDTH_RANGE,
    node_cmap: str = "viridis",
    edge_cmap: str = "coolwarm",
    node_label: str = r"$\langle \hat{n}_i \rangle$",
    edge_label: str = r"$e_{ij}$",
    arrow_threshold: float = _ARROW_THRESHOLD,
    arrow_scale_range: tuple[float, float] = _ARROW_SCALE_RANGE,
    arrow_shaft: tuple[float, float] = _ARROW_SHAFT,
    arrow_linewidth: float = _ARROW_LINEWIDTH,
    arrow_color: str = "0.15",
    colorbar: bool = True,
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
    edge_scale, current_scale : float, optional
        Trajectory-wide maxima of the base value and of the current. When
        given, edge widths, the symmetric edge-norm default and the
        arrowhead size/cutoff normalize by these instead of the passed
        arrays' own maxima - the way to replay one frame of a trajectory
        with exactly the animation's scaling.
    node_size_range : (float, float), default (4.0, 400.0)
        Node area (points^2) at the minimum/maximum value.
    edge_width_range : (float, float), default (0.6, 4.6)
        Edge width at value 0 / at the maximum absolute value.
    node_cmap, edge_cmap : str, default "viridis"/"coolwarm"
        Colorblind-safe colormaps for sites and bonds.
    node_label, edge_label : str
        Colorbar labels for the node/edge colorbars.
    arrow_threshold : float, default 0.05
        No-arrowhead cutoff, as a fraction of ``max|edge_currents|`` (or
        the absolute cutoff itself when ``max|edge_currents|`` is
        round-off noise, which then draws nothing at all).
    arrow_scale_range : (float, float), default (10.0, 40.0)
        Arrowhead ``mutation_scale`` (points) at threshold / at the
        maximum absolute current.
    arrow_shaft : (float, float), default (0.35, 0.65)
        Fractions of the bond the arrow spans (shortened middle section -
        compact, prominent markers that clear the node circles).
    arrow_linewidth : float, default 2.0
        Arrow shaft line width in points.
    arrow_color : str, default "0.15"
        Arrowhead color; encodes no data (direction and size carry the
        current), hence no colorbar.
    colorbar : bool, default True
        Draw the node/edge colorbars on ``ax``'s figure. Set ``False``
        for multi-panel strips that add ONE shared colorbar pair for the
        whole figure (per-panel bars crowd the neighboring lattices).

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
        arrow_shaft=arrow_shaft,
        arrow_linewidth=arrow_linewidth,
        arrow_color=arrow_color,
    )

    if sizes_arr is not None:
        svmin = float(sizes_arr.min()) if node_vmin is None else float(node_vmin)
        svmax = float(sizes_arr.max()) if node_vmax is None else float(node_vmax)
        artists.set_frame(node_sizes=_scale_linear(sizes_arr, svmin, svmax, *node_size_range))
    if node_colors is not None and colorbar:
        ax.figure.colorbar(artists.nodes, ax=ax, label=node_label, shrink=0.8, pad=0.02)

    edge_arr = None if edge_colors is None else np.asarray(edge_colors, dtype=float)
    if edge_arr is not None:
        if edge_vmin is None and edge_vmax is None:
            # signed diverging base channel defaults to a symmetric norm so
            # zero (no bond energy) is exactly the colormap center; an
            # explicit trajectory-wide edge_scale pins that norm across panels
            amax = _guard_scale(
                float(np.max(np.abs(edge_arr))) if edge_scale is None else float(edge_scale)
            )
            vmin, vmax = -amax, amax
        else:
            vmin = float(edge_arr.min()) if edge_vmin is None else float(edge_vmin)
            vmax = float(edge_arr.max()) if edge_vmax is None else float(edge_vmax)
        artists.edge_norm = plt.Normalize(vmin=vmin, vmax=vmax)
        if colorbar:
            mappable = plt.cm.ScalarMappable(norm=artists.edge_norm, cmap=plt.get_cmap(edge_cmap))
            mappable.set_array([])
            ax.figure.colorbar(mappable, ax=ax, label=edge_label, shrink=0.8, pad=0.02)
    if edge_widths is not None:
        width_arr = np.asarray(edge_widths, dtype=float)
        raw = float(np.max(np.abs(width_arr))) if edge_scale is None else float(edge_scale)
        scale = _guard_scale(raw)
        artists.set_frame(edge_widths=_widths_from_magnitude(width_arr, scale, edge_width_range))
    if edge_arr is not None:
        artists.set_frame(edge_colors=edge_arr)
    if edge_currents is not None:
        current_arr = np.asarray(edge_currents, dtype=float)
        raw_j = (
            float(np.max(np.abs(current_arr))) if current_scale is None else float(current_scale)
        )
        artists.set_arrows(current_arr, raw_j, arrow_threshold)


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
    arrow_shaft: tuple[float, float] = _ARROW_SHAFT,
    arrow_linewidth: float = _ARROW_LINEWIDTH,
    arrow_color: str = "0.15",
    header: str | Sequence[str] | None = None,
    legend: str | Sequence[str] | bool | None = None,
    live: Sequence[tuple[str, ArrayLike, str]] | None = None,
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
        ``|current|`` (the stated zero-current threshold of the grammar);
        a trajectory whose max ``|current|`` is round-off noise (below
        ``1e-14``) draws no arrows at all.
    arrow_scale_range : (float, float), default (10.0, 40.0)
        Arrowhead ``mutation_scale`` (points) at the threshold and at the
        trajectory-wide max ``|current|``.
    arrow_shaft : (float, float), default (0.35, 0.65)
        Fractions of the bond the arrow spans (shortened middle section).
    arrow_linewidth : float, default 2.0
        Arrow shaft line width in points.
    arrow_color : str, default "0.15"
        Arrowhead color; encodes no data, so it adds no colorbar.
    header : str or sequence of str, optional
        Static system-parameter text block, monospace, top-left over the
        axes (figure coordinates). ``None`` draws nothing.
    legend : str, sequence of str, True, or None, default None
        Static encoding legend below the header. ``True`` (or the
        equivalent string ``"auto"``) auto-generates the lines from the
        trajectory-wide normalization bounds actually used by this view
        (node range, edge norms, current norm, arrow cutoff); any other
        string/sequence is drawn verbatim; ``None``/False draws nothing.
    live : sequence of (label, values, format-spec), optional
        Per-frame readout line, bottom-center, monospace. ``values`` is a
        precomputed ``(T,)`` array (one value per frame; a scalar fills
        all frames) and ``format-spec`` a Python format spec applied to
        ``float(values[k])`` - e.g. ``("t", times, "5.2f")`` or
        ``("<H_hop>", hop_means, "+.3f")``. The view never computes
        physics: the caller supplies every number to be displayed.
    interval : float, default 50
        Milliseconds between frames in a live replay - a nominal
        parameter, not a measured time; saving derives its fps from it.
    ax : matplotlib.axes.Axes, optional
        Target axes; a new figure is created when omitted.

    Notes
    -----
    Text blocks (header/legend/live) are deterministic figure artists:
    they participate in every rendered frame and in saved files, and the
    live readout changes only with the frame index - never with wall
    clock. Adding any of them reserves figure margins via
    ``fig.subplots_adjust``.

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
    base_raw_max = 0.0
    if base_series is not None and base_series.size:
        base_raw_max = (
            float(np.max(base_series))
            if edge_mode == "coherence"
            else float(np.max(np.abs(base_series)))
        )
    # round-off noise below the floor must not self-normalize: the guarded
    # scale renders noise as minimum width / center color
    base_scale = _guard_scale(base_raw_max)
    base_vmin = 0.0 if edge_mode == "coherence" else -base_scale
    # raw trajectory-wide current norm: set_arrows hides everything below
    # the noise floor; the legend states the true (unguarded) norm
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
        arrow_shaft=arrow_shaft,
        arrow_linewidth=arrow_linewidth,
        arrow_color=arrow_color,
    )
    fig.colorbar(artists.nodes, ax=ax, label=r"$\langle \hat{n}_i \rangle$", shrink=0.8, pad=0.02)
    if base_series is not None:
        mappable = plt.cm.ScalarMappable(norm=norm_edges, cmap=cmap_edges)
        mappable.set_array([])
        label = r"$e_{ij}$" if edge_mode == "energy" else r"$|\gamma_{ij}|$"
        fig.colorbar(mappable, ax=ax, label=label, shrink=0.8, pad=0.02)
    dbl_max = 0.0 if dbl is None else float(dbl.max())
    dbl_scale = _guard_scale(dbl_max)  # trajectory-wide, noise-floor-safe

    def doublon_areas(values: np.ndarray) -> np.ndarray:
        return _scale_linear(values, 0.0, dbl_scale, *_DOUBLON_SIZE_RANGE)

    doublon_layer = None if dbl is None else artists.add_doublons(doublon_areas(dbl[0]))

    # --- annotation blocks (issue #40 round 3) -----------------------------
    legend_block: str | Sequence[str] | None
    if legend is True or legend == "auto":
        legend_block = _legend_lines(
            (node_vmin, node_vmax),
            edge_mode,
            None if base_series is None else (base_vmin, base_scale),
            None if current_series is None else current_scale,
            arrow_threshold * current_scale,
            dbl is not None,
        )
    elif legend is False:
        legend_block = None
    else:
        legend_block = legend
    caption_artists = _place_header(fig, header, legend_block)
    live_fields = _validate_live(live, n_time) if live is not None else []
    if caption_artists or live_fields:
        fig.subplots_adjust(top=0.80, bottom=0.10 if live_fields else 0.04, left=0.02, right=0.97)
    live_text = None
    if live_fields:
        live_text = fig.text(0.5, 0.045, "", ha="center", family="monospace", size=8.0)
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
        if live_text is not None:
            live_text.set_text(_live_text(live_fields, k))
        updated: list[Artist] = [artists.nodes, title, *artists.lines, *artists.arrows]
        if doublon_layer is not None:
            updated.append(doublon_layer)
        updated.extend(caption_artists)
        if live_text is not None:
            updated.append(live_text)
        return updated

    frame(0)
    return animation.FuncAnimation(
        fig, frame, frames=n_time, interval=interval, blit=False, repeat=True
    )
