from __future__ import annotations

import numpy as np

from .encoding import (
    _ARROW_LINEWIDTH,
    _ARROW_SCALE_RANGE,
    _ARROW_SHAFT,
    _EDGE_WIDTH_RANGE,
    _NODE_SIZE_RANGE,
    _NOISE_FLOOR,
)
from .geometry import _validate_bonds, _validate_xy


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
        arrow_shaft: tuple[float, float] = _ARROW_SHAFT,
        arrow_linewidth: float = _ARROW_LINEWIDTH,
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
        # arrowheads span a shortened middle fraction of the bond (default
        # 35% -> 65%) so they read as compact, prominent markers and never
        # collide with the node circles; forward/backward position pairs are
        # precomputed, per frame only the pair selection changes
        self.arrows = []
        self._arrow_endpoints = []
        for i, j in bonds:
            delta = xy[j] - xy[i]
            tail = xy[i] + arrow_shaft[0] * delta
            head = xy[i] + arrow_shaft[1] * delta
            patch = ax.add_patch(
                FancyArrowPatch(
                    tail,
                    head,
                    arrowstyle="-|>",
                    color=arrow_color,
                    linewidth=arrow_linewidth,
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
        # Explicit view limits: autoscale collapses the axes box for
        # collinear geometries (the open chain has zero y-extent), so pad
        # both directions from the LARGER coordinate span and every
        # geometry keeps a proportional frame around its markers.
        span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-9)
        pad = 0.12 * span
        ax.set_xlim(float(xy[:, 0].min()) - pad, float(xy[:, 0].max()) + pad)
        ax.set_ylim(float(xy[:, 1].min()) - pad, float(xy[:, 1].max()) + pad)
        ax.set_aspect("equal")
        ax.set_axis_off()

    def set_arrows(self, currents: np.ndarray, j_scale: float, threshold: float) -> None:
        """Drive the arrowheads from the signed bond currents.

        ``j_scale`` is the trajectory-wide max |current| and ``threshold``
        the no-arrow cutoff as a fraction of it; head size is linear in
        ``|j| / j_scale`` over ``arrow_scale_range`` and the arrow points
        first -> second for positive current (reversed endpoints otherwise).
        Currents at or below the cutoff - and every bond when ``j_scale``
        is round-off noise (below the module noise floor) - draw nothing.
        """
        lo, hi = self.arrow_scale_range
        noisy = j_scale < _NOISE_FLOOR
        for patch, (forward, backward), value in zip(
            self.arrows, self._arrow_endpoints, currents, strict=True
        ):
            magnitude = abs(float(value))
            if noisy or magnitude <= threshold * j_scale:
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
