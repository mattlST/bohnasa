from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_lines(block: str | Sequence[str] | None) -> list[str]:
    """Normalize a None / str / sequence-of-str text block to a line list."""
    if block is None:
        return []
    if isinstance(block, str):
        return [block]
    return [str(line) for line in block]


def _legend_lines(
    node_bounds: tuple[float, float],
    edge_mode: str,
    edge_bounds: tuple[float, float] | None,
    current_scale: float | None,
    cutoff: float,
    has_doublons: bool,
) -> list[str]:
    r"""Encoding-legend lines from the normalization bounds actually used.

    ASCII grammar shorthand (these lines are figure captions, not mathtext
    renderers): ``g`` is ``gamma_ij = <c_i^dag c_j>``, ``propto`` marks a
    linear encoding. The bounds are the trajectory-wide values the view
    normalizes by, so the legend states the norms of the rendered picture
    rather than of the raw arrays; with no ``gamma`` the edge and arrow
    clauses are omitted entirely.
    """
    squares = "   squares: doublons n_i(n_i-1)" if has_doublons else ""
    if edge_bounds is not None and current_scale is not None:
        lo, hi = edge_bounds
        edge_desc = (
            f"edge width+color propto 2Re g_ij ({lo:+.2f}..{hi:+.2f})"
            if edge_mode == "energy"
            else f"edge width+color propto |g_ij| [0, {hi:.2f}]"
        )
        return [
            f"node area propto <n_i> in [{node_bounds[0]:.2f}, {node_bounds[1]:.2f}]   {edge_desc}",
            f"arrows propto j_ij = 2Im g_ij (+-{current_scale:.2f})   "
            f"no arrow |j| <= {cutoff:.3f}{squares}",
        ]
    return [f"node area propto <n_i> in [{node_bounds[0]:.2f}, {node_bounds[1]:.2f}]{squares}"]


def _place_header(fig, header: str | Sequence[str] | None, legend: str | Sequence[str] | None):
    """Place the static header/legend monospace block at the figure top-left.

    Returns the list of ``Text`` artists created. Line spacing is computed
    from the figure height and point size (deterministic in the inputs).
    """
    artists = []
    height = float(fig.get_size_inches()[1])
    y = 0.985
    for block, size in ((header, 7.5), (legend, 7.0)):
        lines = _as_lines(block)
        if not lines:
            continue
        artists.append(
            fig.text(0.015, y, "\n".join(lines), va="top", ha="left", family="monospace", size=size)
        )
        y -= (len(lines) * size / 72.0 / height) * 1.35 + 0.01
    return artists


def add_captions(
    fig,
    *,
    header: str | Sequence[str] | None = None,
    legend: str | Sequence[str] | None = None,
) -> list:
    """Draw static monospace captions (top-left) on a figure.

    Companion of :func:`animate_lattice`'s ``header``/``legend`` parameters
    for hand-built layouts such as the multi-panel frame strip: the header
    states the system parameters, the legend the encoding norms actually
    used. Captions are figure-level ``fig.text`` artists - deterministic,
    monospace, and part of every rendered frame / saved page.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure.
    header, legend : str or sequence of str, optional
        Text block(s); the legend block is placed below the header.
        Pass preformatted lines (see :func:`_legend_lines` usage in the
        examples for the encoding shorthand).

    Returns
    -------
    list of matplotlib.text.Text
        The created text artists (header first, then legend).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> texts = add_captions(fig, header=["L=6 ring  U/t = 1.0"], legend=["node area propto n_i"])
    >>> len(texts)
    2
    >>> texts[0].get_text()
    'L=6 ring  U/t = 1.0'
    >>> texts2 = add_captions(fig, header="one line", legend=["a", "b"])
    >>> [len(t.get_text().splitlines()) for t in texts2]
    [1, 2]
    >>> plt.close(fig)
    """
    return _place_header(fig, header, legend)


def _validate_live(live, n_time: int) -> list[tuple[str, np.ndarray, str]]:
    """Coerce and check the live-readout fields to (label, (T,) values, spec)."""
    fields: list[tuple[str, np.ndarray, str]] = []
    for k, entry in enumerate(live):
        label, values, spec = entry
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = np.full(n_time, float(arr))
        if arr.shape != (n_time,):
            raise ValueError(
                f"animate_lattice: live[{k}] ({label!r}) must have shape ({n_time},) "
                f"= one value per frame, got {arr.shape}"
            )
        fields.append((str(label), arr, str(spec)))
    return fields


def _live_text(fields: list[tuple[str, np.ndarray, str]], k: int) -> str:
    """Render frame k of the live readout: ``label = value`` monospace line."""
    return "   ".join(
        f"{label} = {format(float(values[k]), spec)}" for label, values, spec in fields
    )
