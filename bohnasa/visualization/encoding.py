from __future__ import annotations

import numpy as np

_NODE_SIZE_RANGE = (4.0, 400.0)
_EDGE_WIDTH_RANGE = (0.6, 4.6)
_DOUBLON_SIZE_RANGE = (0.0, 150.0)
_ARROW_SCALE_RANGE = (10.0, 40.0)
_ARROW_SHAFT = (0.35, 0.65)
_ARROW_LINEWIDTH = 2.0
# arrowhead cutoff as a fraction of the trajectory-wide max |current|
_ARROW_THRESHOLD = 0.05
# absolute round-off noise floor: a trajectory maximum below this is numeric
# noise, not signal; truthiness guards (`max or 1.0`) miss it because
# float64 noise like 6.12e-17 is truthy and would self-normalize to full
# widths, extreme colors and full-size arrows
_NOISE_FLOOR = 1e-14


def _scale_linear(
    values: np.ndarray, vmin: float, vmax: float, out_min: float, out_max: float
) -> np.ndarray:
    """Linear map ``[vmin, vmax] -> [out_min, out_max]`` with clipping.

    A degenerate data range (``vmax - vmin`` below the round-off noise
    floor) maps to the midpoint of the output range: constant data - or
    pure numerical noise - draws a constant, visible encoding instead of
    self-normalized extremes.
    """
    values = np.asarray(values, dtype=float)
    if vmax - vmin < _NOISE_FLOOR:
        return np.full(values.shape, 0.5 * (out_min + out_max), dtype=float)
    scaled = out_min + (out_max - out_min) * (values - vmin) / (vmax - vmin)
    return np.clip(scaled, out_min, out_max)


def _widths_from_magnitude(values: np.ndarray, scale: float, width_range: tuple[float, float]):
    """Edge widths linear in ``|value|``, zero-anchored, ``scale`` = trajectory max."""
    return width_range[0] + (width_range[1] - width_range[0]) * np.clip(
        np.abs(values) / scale, 0.0, 1.0
    )


def _guard_scale(max_abs: float) -> float:
    """Round-off noise floor for a normalization scale (issue #40, B2).

    Truthiness guards (``max or 1.0``) only catch an *exactly* zero maximum;
    float64 cancellation noise like 6.12e-17 is truthy and would normalize
    itself to full width / full arrow / colormap extreme. Treating any
    maximum below ``_NOISE_FLOOR`` as degenerate (returns 1.0, so the true
    values all map to ~zero: minimum encoding) makes noise render as noise.
    """
    return max_abs if max_abs >= _NOISE_FLOOR else 1.0
