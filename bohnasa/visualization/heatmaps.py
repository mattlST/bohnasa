from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ._plot import _pyplot

if TYPE_CHECKING:
    from matplotlib.axes import Axes


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
