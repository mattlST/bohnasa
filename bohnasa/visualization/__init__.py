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
    ``e_ij = 2 Re gamma_ij`` - invariant under ``i <-> j``, well defined
    on the undirected edge. Under the ring convention ``H = -t sum
    (c_j^dag c_i + h.c.)`` (units ``t = 1``) the *kinetic*-energy
    contribution of that bond is ``-e_ij``: summed over the bond list,
    ``e`` reproduces ``<Hopping> = -<H_kin>`` exactly. The directed bond
    current is ``j_ij = i (gamma_ji - gamma_ij) = 2 Im gamma_ij``;
    positive ``j_ij`` means particles flow from site ``i`` to site ``j``.
    The raw difference ``gamma_ji - gamma_ij`` of the original issue text
    is purely imaginary, so the factor ``i`` makes the encoded quantity
    real. For the ring Hamiltonian above over the same bond list, the
    continuity equation ``dn_i/dt = sum_j (J_j->i - J_i->j)`` with
    ``J_i->j = t j_ij`` holds to machine precision (derivation test).

Visual grammar.
    Node circle *area* is linear in on-site occupation ``<n_i>``
    (radius ∝ sqrt), colored on ``viridis``. The edge BASE (width +
    diverging ``coolwarm`` color) encodes the symmetric bond energy
    ``e_ij``: width ∝ ``|e|`` anchored at zero width for zero energy,
    color ∝ signed ``e`` under a trajectory-wide symmetric norm
    ``±max|e|`` (so zero is always the map center). The edge DIRECTION
    is an arrowhead over a shortened middle section of the bond,
    pointing first -> second when ``j_ij > 0`` (flipped when negative),
    with head size linear in ``|j|`` under the trajectory-wide current
    norm ``max|j|``; currents at or below a stated threshold (5% of
    ``max|j|`` by default) draw no arrowhead, and the arrowhead color
    encodes nothing (size and direction carry the full signed
    information, so no third colorbar is needed). ``edge_mode =
    "coherence"`` replaces the base channel by ``|gamma_ij|`` on
    ``viridis`` while the arrows keep encoding the current. All
    normalizations are computed over the whole trajectory, never per
    frame; trajectory maxima below an absolute round-off floor (1e-14)
    are noise, not signal, and render as minimum width, center color,
    no arrows. Colormaps are colorblind-safe, and whenever color
    encodes a quantity a colorbar is added - one shared pair at the
    figure right in multi-panel views, never per-panel bars.

Annotations.
    Views may carry deterministic monospace text: a static
    system-parameter ``header`` block (top-left, figure coordinates), a
    static encoding ``legend`` below it (auto-generated from the
    trajectory-wide normalization bounds actually used, or supplied),
    and a live per-frame readout of caller-provided ``(T,)`` value
    arrays with Python format specs (bottom-center). The visualization
    computes no physics for the captions - it renders exactly the
    numbers handed to it - and no wall-clock quantity enters any of
    them.
"""

from .captions import _legend_lines as _legend_lines
from .captions import add_captions
from .channels import bond_coherence, bond_current, bond_energy
from .encoding import (
    _ARROW_SCALE_RANGE as _ARROW_SCALE_RANGE,
)
from .encoding import (
    _ARROW_THRESHOLD as _ARROW_THRESHOLD,
)
from .encoding import (
    _EDGE_WIDTH_RANGE as _EDGE_WIDTH_RANGE,
)
from .encoding import (
    _NODE_SIZE_RANGE as _NODE_SIZE_RANGE,
)
from .encoding import (
    _NOISE_FLOOR as _NOISE_FLOOR,
)
from .geometry import lattice_bonds, lattice_geometry
from .heatmaps import momentum_heatmap, occupation_heatmap
from .views import animate_lattice, draw_lattice

__all__ = [
    "lattice_bonds",
    "lattice_geometry",
    "bond_energy",
    "bond_current",
    "bond_coherence",
    "draw_lattice",
    "animate_lattice",
    "add_captions",
    "occupation_heatmap",
    "momentum_heatmap",
]
