"""Unit tests for bohnasa.visualization (issue #40).

Pixel-free by design: only artist properties (sizes, widths, clim, colors),
axes bookkeeping (colorbars), byte-deterministic file output and error
paths are inspected - never rasterized pixels.
"""

import builtins
import subprocess
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pytest

import bohnasa
from bohnasa import visualization as viz


@pytest.fixture
def figax():
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


# ---------------------------------------------------------------------------
# geometry & bonds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", [2, 3, 4, 6])
def test_pbc_geometry_point_count_and_closure(L):
    xy = viz.lattice_geometry(L)
    assert xy.shape == (L, 2)
    assert np.allclose(np.hypot(xy[:, 0], xy[:, 1]), 1.0)  # every site on the unit circle
    bonds = viz.lattice_bonds(L)
    assert bonds.shape == (L, 2)  # L bonds in the ring, closure included
    chord = 2.0 * np.sin(np.pi / L)
    for i, j in bonds:
        assert np.linalg.norm(xy[i] - xy[j]) == pytest.approx(chord)
    # the wrap-around bond (0, L-1) is present - the ring closes
    assert [0, L - 1] in bonds.tolist()


@pytest.mark.parametrize("L", [2, 3, 5])
def test_obc_geometry_is_a_line(L):
    xy = viz.lattice_geometry(L, "obc")
    assert np.allclose(xy[:, 1], 0.0)
    assert np.allclose(xy[:, 0], np.arange(L))
    bonds = viz.lattice_bonds(L, "obc")
    assert bonds.shape == (L - 1, 2)  # no closure bond on the open chain


def test_l2_pbc_double_bond_convention_is_the_hopping_one():
    fock = bohnasa.FixedNFockSpace(1, 2)
    hop = bohnasa.Hopping(fock, connections="pbc")
    bonds = viz.lattice_bonds(2)
    np.testing.assert_array_equal(bonds, hop.connections)
    assert bonds.tolist() == [[0, 1], [1, 0]]  # the single ring bond, counted twice


@pytest.mark.parametrize("L", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("connections", ["pbc", "obc"])
def test_bonds_are_the_hopping_bond_list(L, connections):
    fock = bohnasa.FixedNFockSpace(1, L)
    hop = bohnasa.Hopping(fock, connections=connections)
    np.testing.assert_array_equal(viz.lattice_bonds(L, connections), hop.connections)


def test_geometry_validation():
    with pytest.raises(ValueError, match=r"L must be >= 1"):
        viz.lattice_geometry(0)
    with pytest.raises(TypeError, match="must be an int"):
        viz.lattice_geometry(4.0)
    with pytest.raises(ValueError, match="connections"):
        viz.lattice_geometry(3, "twisted")


# ---------------------------------------------------------------------------
# bond current / coherence helpers (validation; physics pinned separately)
# ---------------------------------------------------------------------------


def test_bond_helpers_validate_shapes():
    gamma = np.eye(3, dtype=complex)
    for helper in (viz.bond_energy, viz.bond_current, viz.bond_coherence):
        with pytest.raises(ValueError, match="square"):
            helper(np.ones((2, 3)), [[0, 1]])
    with pytest.raises(ValueError, match="square"):
        viz.bond_coherence(np.ones(3), [[0, 1]])
    with pytest.raises(ValueError, match=r"\(E, 2\)"):
        viz.bond_current(gamma, np.ones((3, 3)))
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        viz.bond_energy(gamma, [[0, 3]])


# ---------------------------------------------------------------------------
# draw_lattice
# ---------------------------------------------------------------------------


def test_draw_lattice_node_area_monotonic_linear(figax):
    fig, ax = figax
    occ = [0.0, 0.5, 1.0]
    viz.draw_lattice(ax, viz.lattice_geometry(3, "obc"), node_sizes=occ, node_colors=occ)
    nodes = ax.collections[0]
    sizes = nodes.get_sizes()
    # area is linear in value: equal spacing between the mapped areas
    assert np.allclose(np.diff(sizes), sizes[1] - sizes[0])
    assert np.all(sizes[1:] > sizes[:-1])
    lo, hi = viz._NODE_SIZE_RANGE
    assert sizes[0] == pytest.approx(lo)
    assert sizes[-1] == pytest.approx(hi)
    assert nodes.get_clim() == (0.0, 1.0)
    plt.close(fig)


def test_draw_lattice_edge_width_monotonic_zero_anchored_and_sign_symmetric(figax):
    fig, ax = figax
    currents = np.array([0.0, 1.0, -2.0, 2.0])
    viz.draw_lattice(ax, viz.lattice_geometry(4), edge_widths=currents)
    widths = np.array([line.get_linewidth() for line in ax.lines])
    w0, w1 = viz._EDGE_WIDTH_RANGE
    assert widths[0] == pytest.approx(w0)  # zero base value draws minimum width
    assert widths[1] == pytest.approx(w0 + (w1 - w0) * 0.5)
    assert widths[2] == pytest.approx(w1)
    assert widths[3] == pytest.approx(widths[2])  # |j| encodes, sign does not (width)
    plt.close(fig)


def test_draw_lattice_explicit_norm_bounds(figax):
    fig, ax = figax
    occ = [1.0, 2.0, 3.0]
    viz.draw_lattice(
        ax,
        viz.lattice_geometry(3, "obc"),
        node_sizes=occ,
        node_colors=occ,
        node_vmin=0.0,
        node_vmax=6.0,
    )
    nodes = ax.collections[0]
    assert nodes.get_clim() == (0.0, 6.0)
    lo, hi = viz._NODE_SIZE_RANGE
    assert np.allclose(nodes.get_sizes(), [lo + (hi - lo) * v / 6.0 for v in occ])
    plt.close(fig)


def test_draw_lattice_colorbars_only_when_color_encodes(figax):
    fig, ax = figax
    viz.draw_lattice(ax, viz.lattice_geometry(4), node_sizes=[1.0, 2.0, 3.0, 4.0])
    assert len(fig.axes) == 1  # gray nodes, no color data -> no colorbar
    plt.close(fig)

    fig, ax = plt.subplots()
    data = [1.0, 2.0, 3.0, 4.0]
    viz.draw_lattice(
        ax, viz.lattice_geometry(4), node_colors=data, edge_colors=data, edge_widths=data
    )
    assert len(fig.axes) == 3  # main axes + node colorbar + edge mappable colorbar
    assert ax.lines[0].get_color() != "0.65"  # edges re-colored from data
    plt.close(fig)


def test_draw_lattice_signed_edge_base_defaults_symmetric_zero_neutral(figax):
    fig, ax = figax
    # default edge normalization is the symmetric +-max|e| interval, so
    # zero bond energy sits exactly at the diverging colormap center
    viz.draw_lattice(ax, viz.lattice_geometry(4), edge_colors=[-4.0, 0.0, 2.0, 4.0])
    assert matplotlib_color(ax.lines[1].get_color()) == pytest.approx(
        matplotlib_color(plt.cm.coolwarm(0.5)), abs=1e-12
    )
    assert matplotlib_color(ax.lines[0].get_color()) == pytest.approx(
        matplotlib_color(plt.cm.coolwarm(0.0)), abs=1e-12
    )
    plt.close(fig)


def test_draw_lattice_arrowhead_threshold_size_and_direction(figax):
    fig, ax = figax
    xy = viz.lattice_geometry(4)
    # bonds [(0,3), (1,0), (2,1), (3,2)]; j_scale 0.5, cutoff 0.05 * 0.5 = 0.025
    viz.draw_lattice(ax, xy, edge_currents=[-0.5, 0.5, 0.01, 0.0])
    assert [patch.get_visible() for patch in ax.patches] == [True, True, False, False]
    lo, hi = viz._ARROW_SCALE_RANGE
    assert ax.patches[0].get_mutation_scale() == pytest.approx(hi)
    tail0, head0 = ax.patches[0]._posA_posB
    # negative j on (0, 3): reversed arrow, tail nearer the second site
    assert np.linalg.norm(tail0 - xy[3]) < np.linalg.norm(tail0 - xy[0])
    assert np.linalg.norm(head0 - xy[0]) < np.linalg.norm(head0 - xy[3])
    tail1, _ = ax.patches[1]._posA_posB
    # positive j on (1, 0): forward arrow, tail nearer the first site
    assert np.linalg.norm(tail1 - xy[1]) < np.linalg.norm(tail1 - xy[0])
    # explicit arrow_threshold moves the cutoff: with 0.8 the third bond
    # (|j| = 0.3 <= 0.8 * 0.5 = 0.4) is hidden although it draws above the
    # default 0.05 cutoff
    ax2 = plt.subplots(figsize=(3.0, 3.0), dpi=50)[1]
    viz.draw_lattice(ax2, xy, edge_currents=[-0.5, 0.5, 0.3, 0.0], arrow_threshold=0.8)
    assert [patch.get_visible() for patch in ax2.patches] == [True, True, False, False]
    ax3 = plt.subplots(figsize=(3.0, 3.0), dpi=50)[1]
    viz.draw_lattice(ax3, xy, edge_currents=[-0.5, 0.5, 0.3, 0.0])  # default cutoff
    assert [patch.get_visible() for patch in ax3.patches] == [True, True, True, False]
    plt.close(fig)
    plt.close(ax2.figure)
    plt.close(ax3.figure)


def test_draw_lattice_validates_data_lengths(figax):
    fig, ax = figax
    with pytest.raises(ValueError, match="node_sizes must have 4 entries"):
        viz.draw_lattice(ax, viz.lattice_geometry(4), node_sizes=[1.0, 2.0])
    with pytest.raises(ValueError, match="edge_widths must have 4 entries"):
        viz.draw_lattice(ax, viz.lattice_geometry(4), edge_widths=[1.0, 2.0])
    with pytest.raises(ValueError, match="edge_currents must have 4 entries"):
        viz.draw_lattice(ax, viz.lattice_geometry(4), edge_currents=[1.0])
    plt.close(fig)


def test_draw_lattice_accepts_explicit_bond_list(figax):
    fig, ax = figax
    xy = viz.lattice_geometry(4, "obc")
    viz.draw_lattice(ax, xy, bonds=[[0, 1], [2, 3]], edge_widths=[0.0, 1.0])
    assert len(ax.lines) == 2
    widths = [line.get_linewidth() for line in ax.lines]
    assert widths[0] == pytest.approx(viz._EDGE_WIDTH_RANGE[0])
    assert widths[1] == pytest.approx(viz._EDGE_WIDTH_RANGE[1])
    plt.close(fig)


# ---------------------------------------------------------------------------
# animate_lattice
# ---------------------------------------------------------------------------


def _tiny_axis():
    return plt.subplots(figsize=(3.2, 3.2), dpi=50)[1]


def _animate(*args, **kwargs):
    """animate_lattice plus one canvas draw event.

    matplotlib warns when a FuncAnimation is garbage-collected without ever
    having rendered; a single headless draw marks the animation as started.
    Returns (ax, anim).
    """
    ax = kwargs.pop("ax", None) or _tiny_axis()
    anim = viz.animate_lattice(*args, ax=ax, **kwargs)
    ax.figure.canvas.draw()
    return ax, anim


def test_animate_normalizes_nodes_over_the_whole_trajectory():
    times = np.array([0.0, 1.0, 2.0])
    occ = np.array([[1.0, 1.0], [0.0, 2.0], [1.5, 0.5]])  # extremes in frame 1
    ax, anim = _animate(times, occ)
    nodes = ax.collections[0]
    assert nodes.get_clim() == (0.0, 2.0)  # trajectory-wide, not frame-0 (1.0, 1.0)
    lo, hi = viz._NODE_SIZE_RANGE
    assert np.allclose(nodes.get_sizes(), [lo + (hi - lo) * 0.5, lo + (hi - lo) * 0.5])
    from matplotlib.animation import FuncAnimation

    assert isinstance(anim, FuncAnimation)
    plt.close("all")


def _gamma_from_bond_values(L, bond_values):
    """gamma with gamma[i, j] = value and gamma[j, i] = conj(value); real diagonal."""
    gamma = np.eye(L, dtype=complex)
    for (i, j), value in bond_values.items():
        gamma[i, j] = value
        gamma[j, i] = np.conj(value)
    return gamma


def test_animate_edge_base_is_energy_and_arrows_are_current():
    """Width/color carry 2 Re gamma (symmetric +-max over the trajectory);

    arrowheads carry 2 Im gamma: direction by sign, size proportional to
    |j| under the trajectory-wide current norm, hidden below threshold.
    """
    times = np.array([0.0, 1.0, 2.0])
    occ = np.full((3, 4), 0.5)
    gamma = np.array(
        [
            _gamma_from_bond_values(4, {(1, 0): 0.25 + 0.25j}),
            _gamma_from_bond_values(4, {}),
            _gamma_from_bond_values(4, {(0, 3): 0.5, (1, 0): 0.5, (2, 1): 0.5, (3, 2): 0.5}),
        ]
    )
    ax, anim = _animate(times, occ, gamma)
    w0, w1 = viz._EDGE_WIDTH_RANGE
    lo, hi = viz._ARROW_SCALE_RANGE
    # frame 0: bond (1, 0) [list index 1] has e = 0.5, j = 0.5. The
    # trajectory-wide energy max is 1.0 (frame 2), so its width sits at the
    # midpoint; a per-frame normalization would put it at full width w1.
    assert ax.lines[1].get_linewidth() == pytest.approx(w0 + (w1 - w0) * 0.5)
    for idx in (0, 2, 3):  # zero-energy bonds: min width, colormap center
        assert ax.lines[idx].get_linewidth() == pytest.approx(w0)
        assert matplotlib_color(ax.lines[idx].get_color()) == pytest.approx(
            matplotlib_color(plt.cm.coolwarm(0.5)), abs=1e-12
        )
    # arrowheads: the trajectory current max is 0.5 (frame 0 only), so
    # bond 1 shows a full-size forward arrow and the rest stay hidden
    assert [patch.get_visible() for patch in ax.patches] == [False, True, False, False]
    assert ax.patches[1].get_mutation_scale() == pytest.approx(hi)
    xy = viz.lattice_geometry(4)
    tail, head = ax.patches[1]._posA_posB  # matplotlib-internal, test-only probe
    # positive j on the directed bond (1, 0): tail nearer site 1, head nearer 0
    assert np.linalg.norm(tail - xy[1]) < np.linalg.norm(tail - xy[0])
    assert np.linalg.norm(head - xy[0]) < np.linalg.norm(head - xy[1])
    # step to the last frame through the real animation callback
    anim._func(2)
    assert [line.get_linewidth() for line in ax.lines][0] == pytest.approx(w1)
    assert ax.lines[0].get_color() != "0.65"  # signed e = +1.0 colors warm
    assert not any(patch.get_visible() for patch in ax.patches)  # frame 2: j = 0
    plt.close("all")


def matplotlib_color(color):
    """Normalize a matplotlib color spec to an RGBA tuple."""
    from matplotlib.colors import to_rgba

    return np.array(to_rgba(color))


def test_animate_coherence_mode_encodes_magnitude():
    ax = _tiny_axis()
    times = np.array([0.0, 1.0])
    occ = np.full((2, 4), 0.5)
    gamma = np.array(
        [
            _gamma_from_bond_values(4, {(0, 1): 0.4, (1, 2): 0.1, (2, 3): 0.2, (3, 0): -0.4}),
            _gamma_from_bond_values(4, {(0, 1): 0.4, (1, 2): 0.1, (2, 3): 0.2, (3, 0): -0.4}),
        ]
    )
    ax, anim = _animate(times, occ, gamma, edge_mode="coherence")
    widths = np.array([line.get_linewidth() for line in ax.lines])
    # bond order is [(0,3), (1,0), (2,1), (3,2)] -> |gamma| = [0.4, 0.4, 0.1, 0.2]
    assert widths[2] < widths[3] < widths[0]
    assert widths[1] == pytest.approx(widths[0])
    # real-valued gamma entries: coherence shows, arrows must stay hidden
    assert not any(patch.get_visible() for patch in ax.patches)
    plt.close("all")


def test_animate_doublon_overlay_area_monotonic():
    times = np.array([0.0, 1.0])
    occ = np.full((2, 4), 0.5)
    doublons = np.array([[0.0, 0.5, 1.0, 2.0], [2.0, 1.0, 0.5, 0.0]])
    ax, anim = _animate(times, occ, doublons=doublons)
    assert len(ax.collections) == 2  # nodes + doublon layer
    squares = ax.collections[1]
    assert squares.get_sizes()[0] == pytest.approx(0.0)  # zero doublons -> invisible
    assert np.all(np.diff(squares.get_sizes()) > 0)
    ax2, anim2 = _animate(times, occ)  # no doublons requested
    assert len(ax2.collections) == 1
    plt.close("all")


def test_animate_saved_output_is_deterministic(tmp_path):
    times = np.linspace(0.0, 1.0, 4)
    occ = np.array([[1.0, 0.0, 0.5, 0.5]] * 4).T * np.array([1.0, 0.8, 0.6, 0.4])[:, None]
    gamma0 = _gamma_from_bond_values(4, {(0, 1): 0.3, (1, 2): -0.1, (2, 3): 0.2, (3, 0): 0.0})
    gamma = np.stack([gamma0] * 4)

    def save(path):
        ax = _tiny_axis()
        anim = viz.animate_lattice(times, occ, gamma, ax=ax)
        anim.save(path, writer="pillow", fps=10)
        plt.close("all")

    first, second = tmp_path / "a.gif", tmp_path / "b.gif"
    save(first)
    save(second)
    assert first.read_bytes() == second.read_bytes()
    assert len(first.read_bytes()) > 0


def test_animate_zero_current_gives_uniform_center_color_everywhere():
    times = np.array([0.0, 1.0])
    occ = np.full((2, 6), 1.0)
    gamma = np.stack([np.eye(6, dtype=complex)] * 2)  # real gamma -> j = 0
    ax, anim = _animate(times, occ, gamma)
    center = matplotlib_color(plt.cm.coolwarm(0.5))
    for line in ax.lines:
        assert matplotlib_color(line.get_color()) == pytest.approx(center, abs=1e-12)
        assert line.get_linewidth() == pytest.approx(viz._EDGE_WIDTH_RANGE[0])
    assert not any(patch.get_visible() for patch in ax.patches)  # no arrows at j = 0
    plt.close("all")


def test_animate_interior_frame_sizes_use_trajectory_bounds():
    """Node SIZE normalization at a strictly interior, non-constant frame.

    Stepping the real animation callback (``anim._func`` is the stored
    frame function, a matplotlib-internal but stable attribute) at frame
    index 1 and reading the artist sizes: a per-frame normalization
    mutant would map the frame's own [1, 3] range onto the full size
    range and fail both assertions below.
    """
    times = np.array([0.0, 1.0, 2.0])
    occ = np.array([[4.0, 0.0], [1.0, 3.0], [2.0, 2.0]])  # trajectory range 0..4
    ax, anim = _animate(times, occ)
    anim._func(1)
    sizes = ax.collections[0].get_sizes()
    lo, hi = viz._NODE_SIZE_RANGE
    expected = [lo + (hi - lo) * 1.0 / 4.0, lo + (hi - lo) * 3.0 / 4.0]
    assert np.allclose(sizes, expected)  # trajectory-wide bounds
    assert not np.allclose(sizes, [lo, hi])  # the per-frame-normalization mutant
    assert ax.collections[0].get_clim() == (0.0, 4.0)
    assert ax.get_title() == "t = 1"
    plt.close("all")


def test_animate_validation():
    times = np.array([0.0, 1.0])
    occ = np.ones((2, 3))
    with pytest.raises(ValueError, match=r"times \(T,\) and occupations \(T, L\)"):
        viz.animate_lattice(times, np.ones((3, 2)))
    with pytest.raises(ValueError, match="edge_mode"):
        viz.animate_lattice(times, occ, edge_mode="vibes")
    with pytest.raises(ValueError, match="arrowhead channel"):
        viz.animate_lattice(times, occ, edge_mode="current")
    with pytest.raises(ValueError, match="gamma must have shape"):
        viz.animate_lattice(times, occ, np.ones((2, 3, 3, 3), dtype=complex))
    with pytest.raises(ValueError, match="doublons"):
        viz.animate_lattice(times, occ, doublons=np.ones((2, 4)))
    plt.close("all")


# ---------------------------------------------------------------------------
# heatmaps
# ---------------------------------------------------------------------------


def test_heatmap_defaults_use_trajectory_wide_bounds():
    values = np.array([[1.0, 2.0], [5.0, 0.0]])  # (T, L): min 0 max 5 over full array
    ax = viz.occupation_heatmap([0.0, 1.0], values)
    mesh = ax.collections[0]
    assert mesh.get_clim() == (0.0, 5.0)
    assert ax.get_ylabel() == "site $i$"
    assert len(ax.figure.axes) == 2  # main + colorbar
    plt.close("all")


def test_heatmap_explicit_bounds_and_momentum_view():
    ax = viz.momentum_heatmap([0.0, 1.0], np.array([[2.0, 0.5], [1.0, 1.5]]), vmin=-1.0, vmax=9.0)
    assert ax.collections[0].get_clim() == (-1.0, 9.0)
    assert ax.get_ylabel() == "mode $k$"
    plt.close("all")


def test_heatmap_validation():
    with pytest.raises(ValueError, match="at least two"):
        viz.occupation_heatmap([0.0], np.ones((1, 2)))
    with pytest.raises(ValueError, match=r"\(len\(times\), n\)"):
        viz.occupation_heatmap([0.0, 1.0], np.ones((3, 2)))
    plt.close("all")


def test_heatmap_constant_data_renders_flat():
    ax = viz.occupation_heatmap([0.0, 1.0], np.full((2, 3), 2.0))
    assert ax.collections[0].get_clim() == (2.0, 3.0)  # degenerate range widened by 1
    plt.close("all")


# ---------------------------------------------------------------------------
# lazy matplotlib / ImportError path
# ---------------------------------------------------------------------------


def test_plotting_without_matplotlib_raises_import_error(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError(f"blocked: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="examples"):
        viz.occupation_heatmap([0.0, 1.0], np.ones((2, 2)))
    with pytest.raises(ImportError, match="examples"):
        viz.animate_lattice([0.0, 1.0], np.ones((2, 2)))
    with pytest.raises(ImportError, match="examples"):
        viz.draw_lattice(None, [[0.0, 0.0], [1.0, 0.0]])
    # pure data functions keep working without matplotlib
    assert viz.lattice_geometry(3).shape == (3, 2)
    assert viz.lattice_bonds(3).shape == (3, 2)
    assert np.allclose(viz.bond_current(np.eye(2, dtype=complex), [[0, 1]]), 0.0)


def test_module_imports_without_matplotlib():
    """`import bohnasa.visualization` must not pull matplotlib (runtime floor)."""
    code = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.split(".")[0] == "matplotlib":
                raise ImportError(f"blocked: {name}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked
        import bohnasa
        import bohnasa.visualization
        print("imported")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout
