"""Kicked (Floquet) Bose-Hubbard ring: propagator, quasienergies, classical map.

One drive period of the model alternates two delta-kicks of duration
``tau``: first the on-site interaction, then the hopping,

    U_step = exp(-i tau H_J) exp(-i tau H_U),
    H_J = -J sum_(i,j) in bonds (c^dag_j c_i + h.c.) = -J * Hopping(fock, connections),
    H_U = (U / 2) sum_i n_i (n_i - 1)           = (U / 2) * Interaction(fock),

the normal-ordered Bose-Hubbard split with the bond convention of
:func:`bohnasa.operators.bond_pairs` (including its deliberate ``L = 2`` PBC
double counting, which makes the dimer hopping ``2J`` - the ex10 /
``mean_field.dimer`` convention). The rightmost factor acts first: every
basis state first accumulates its on-site phases ``exp(-i tau (U/2) n(n-1))``,
then hops. This is the ordering :class:`bohnasa.SquaredCommutator` uses through
:meth:`~bohnasa.SquaredCommutator.from_propagator`: the Floquet Heisenberg
operator after ``n`` periods is ``O(n) = U_step^dag n O U_step^n``, so
``times`` passed to the curve are kick counts. The split is a Lie-Trotter
product, not a Strang symmetrization: the O(tau^2) per-step commutator
defect (O(tau) accumulated) is physical - it is part of the model's
definition - and the tests pin the factor order explicitly.

Quasienergies follow ``U_step |v> = exp(-i eps tau) |v>`` with the branch
convention ``eps = mod(-arg(lambda), 2 pi) / tau``, i.e. ``eps`` in
``[0, 2 pi / tau)``: the principal argument of the eigenvalue ``lambda`` is
folded into ``[0, 2 pi)`` BEFORE dividing by ``tau`` (the fold and the division
do not commute; folding the angle first is what makes the list exactly the
band ``[0, 2 pi / tau)``), and the zone edge reached through ``mod``
roundoff (``lambda`` within float noise of ``+1``) is relabelled to ``eps = 0``
so the half-open band is exact. Quasienergies within roundoff of the zone edge
``pi / tau`` can land at either band end depending on the sign of a 1e-16
imaginary part - physical quasienergy, unphysical label; the branch choice is
otherwise deterministic across runs (single-threaded LAPACK ``geev``, stable
sort).

Classical limit. :meth:`KickedBoseHubbard.classical_map` represents the same
drive on Gross-Pitaevskii amplitudes ``alpha_j = <c_j> = sqrt(n_j) exp(i
varphi_j)`` (``i dalpha_j/dt = dH_MF/dalpha*_j``, canonical bracket
``{varphi_j, n_k} = delta_jk``) with the c-number energy

    H_MF = sum_jk A_jk alpha*_j alpha_k + (U / 2) sum_j n_j^2 ,
    A = -J * (M + M^T),  M[j, i] = 1 for each directed bond (i, j),

the normal-ordered quantum energy read out literally except for the ``-1`` in
``n (n - 1)``, which multiplies every configuration amplitude by the same
``exp(+i tau U / 2)`` per boson - ``exp(+i tau U N / 2)`` per kick for the
whole state, and exactly ``exp(+i tau U)`` only at N = 2 - and is invisible
in all relative phases and occupations (the gauge is
pinned by dropping it; ``iterate`` reports the resulting phase convention).
The map is the same kick-then-hop Lie-Trotter split as the quantum step: kick
``alpha_j -> exp(-i tau U n_j) alpha_j``, then hop ``alpha -> exp(-i tau A)
alpha`` - both sub-flows are exact for their factor, so the local error is the
O(tau^2) commutator defect of Trotterization, not of the representation.

Dimer dictionary (``L = 2``). The family conventions are stated once in
:mod:`bohnasa.mean_field`; in this ``L = 2`` special case, with
``s = (n_0 - n_1) / N``, ``N = n_0 + n_1`` and
``phi = varphi_1 - varphi_0``, the tau -> 0 limit of the map is the flow
of :func:`bohnasa.mean_field.dimer.mean_field_flow` in its quarter-circle
coordinates ``J = cos(theta)``, ``U = 2 sin(theta) / N`` (canonical bracket
``{phi, s} = 2 / N``). The sign of the relative phase is forced by the
Josephson current: ``ds/dt = -4 J sqrt(1 - s^2) sin(phi)`` requires
``phi = varphi_1 - varphi_0`` (site 0 minus site 1 gives the wrong sign).
The physics tests measure the first-order Trotter convergence instead of
assuming it (``tests/physics/test_kicked_physics.py``).

Floquet spectra: :meth:`KickedBoseHubbard.floquet_spectrum` block-diagonalizes
the step by :class:`bohnasa.SymmetrySectors` whenever ``connections`` is a
named boundary condition (both kicks are symmetric), and falls back to a dense
``dim x dim`` eigendecomposition otherwise.

See :mod:`bohnasa.otoc` for the OTOC machinery the propagator feeds, and
:mod:`bohnasa.mean_field.dimer` / :mod:`bohnasa.mean_field.lattice` for
the continuous-time classical references.
"""

from collections.abc import Sequence

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from .fock_space import FixedNFockSpace
from .operators import Hopping, Interaction, Operator, bond_pairs
from .sectors import SymmetrySectors

__all__ = ["KickedBoseHubbard", "KickedMeanFieldMap"]


def _check_geometry(L: int, connections: str | np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    """Validate ``L`` / ``connections`` and return the directed bond list."""
    if isinstance(L, bool) or not isinstance(L, (int, np.integer)):
        raise TypeError(f"kicked: L must be an int, got {type(L).__name__}")
    if L < 2:
        raise ValueError(f"kicked: L must be at least 2, got {L}")
    return bond_pairs(L, connections)


def _check_dynamics(tau: float, J: float, U: float) -> tuple[float, float, float]:
    """Validate the drive parameters, returning them as floats."""
    tau, J, U = float(tau), float(J), float(U)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"kicked: tau must be positive and finite, got {tau}")
    if not np.isfinite(J) or not np.isfinite(U):
        raise ValueError(f"kicked: J and U must be finite, got J={J}, U={U}")
    return tau, J, U


def _bond_matrix(L: int, connections: str | np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    """Hopping matrix of ``-J * Hopping`` per unit ``J``: entry ``-1`` at
    ``(j, i)`` and ``(i, j)`` for every directed bond ``(i, j)``, so a
    ``L = 2`` PBC ring carries ``+-2`` off the diagonal (``bond_pairs``
    double counting, matching the second-quantized builder)."""
    A = np.zeros((L, L), dtype=float)
    for i, j in _check_geometry(L, connections):
        A[int(j), int(i)] -= 1.0
        A[int(i), int(j)] -= 1.0
    return A


def _fold_quasienergies(lam: np.ndarray, tau: float) -> np.ndarray:
    """Branch map ``eps = mod(-arg(lam), 2 pi) / tau`` into ``[0, 2 pi / tau)``.

    ``np.mod`` can return exactly ``2 pi`` when the argument is a sub-ulp
    negative number (eigenvalue within float roundoff of ``+1``); that zone
    edge is the same eigenvalue as ``eps = 0``, so it is relabelled to 0 to
    keep the documented half-open band exact.
    """
    eps = np.mod(-np.angle(lam), 2.0 * np.pi) / tau
    return np.where(eps >= 2.0 * np.pi / tau, 0.0, eps)


class KickedMeanFieldMap:
    r"""Classical kicked Bose-Hubbard map: kick phases, then linear hop.

    State: complex amplitudes ``alpha`` of shape ``(L,)`` with occupations
    ``n_j = |alpha_j|^2`` and phases ``varphi_j = arg(alpha_j)``. One step is
    the interaction kick ``alpha_j -> exp(-i tau U n_j) alpha_j`` followed by
    the hopping kick ``alpha -> exp(-i tau A) alpha`` with ``A`` the
    single-particle matrix of ``-J * Hopping(fock, connections)`` (module
    docstring; the ``L = 2`` ring counts its single bond twice). The total
    norm ``sum_j n_j`` is exactly conserved; site occupations are not (that
    is the hop).

    Parameters
    ----------
    L : int
        Number of sites (>= 2).
    tau : float
        Kick period (drive units); positive.
    J : float
        Hopping amplitude of ``-J * Hopping``.
    U : float
        On-site coupling of ``(U / 2) sum n (n - 1)``.
    connections : {'pbc', 'periodic', 'obc', 'open'} or array-like of shape (*, 2)
        Bond geometry, shared with :class:`bohnasa.Hopping`.

    Examples
    --------
    >>> import numpy as np
    >>> m = KickedMeanFieldMap(3, tau=0.5, J=1.0, U=0.8)
    >>> alpha = np.array([2.0, 1.0 + 1.0j, -1.0 + 2.0j])
    >>> float(np.round(np.sum(np.abs(m.step(alpha)) ** 2) / np.sum(np.abs(alpha) ** 2), 12))
    1.0

    A kicked-free sequence (``J = 0``) rotates each site phase by exactly
    ``-tau U n_j`` per step and leaves the occupations fixed:

    >>> m0 = KickedMeanFieldMap(2, tau=0.3, J=0.0, U=1.5)
    >>> phi, occ = m0.iterate((np.array([0.4, -0.2]), np.array([4.0, 1.0])), 3)
    >>> np.round(occ, 12).tolist()
    [[4.0, 1.0], [4.0, 1.0], [4.0, 1.0], [4.0, 1.0]]
    >>> np.round((phi[3] - phi[0] + np.pi) % (2 * np.pi) - np.pi, 10).tolist()
    [0.8831853072, -1.35]
    """

    def __init__(
        self,
        L: int,
        tau: float,
        J: float,
        U: float,
        connections: str | np.ndarray | Sequence[Sequence[int]] = "pbc",
    ) -> None:
        bonds = _check_geometry(L, connections)
        self.L = int(L)
        self.tau, self.J, self.U = _check_dynamics(tau, J, U)
        self.connections = connections
        self.bonds = bonds
        A = self.J * _bond_matrix(self.L, bonds)
        # eigh of the real symmetric single-particle matrix: the exponential
        # is unitary to roundoff by construction and deterministic (the
        # single-threaded LAPACK path; the hop matrix is L x L, small)
        w, V = np.linalg.eigh(A)
        self.hop = (V * np.exp(-1j * self.tau * w)) @ V.T

    def step(self, alpha: np.ndarray) -> np.ndarray:
        """Apply one full kick period to the amplitude vector (new array)."""
        a = np.asarray(alpha, dtype=complex)
        if a.shape != (self.L,):
            raise ValueError(
                f"KickedMeanFieldMap: alpha must have shape ({self.L},), got {a.shape}"
            )
        if not np.all(np.isfinite(a)):
            raise ValueError("KickedMeanFieldMap: alpha has non-finite entries")
        kicked = a * np.exp(-1j * self.tau * self.U * np.abs(a) ** 2)
        return self.hop @ kicked

    def iterate(
        self,
        initial: np.ndarray | tuple[np.ndarray, np.ndarray],
        n_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate the map for ``n_steps`` periods from ``initial``.

        Parameters
        ----------
        initial : numpy.ndarray or tuple of numpy.ndarray
            Either complex amplitudes ``alpha_0`` of shape ``(L,)`` or the
            canonical pair ``(phi, occ)`` (phases and occupations, each
            ``(L,)``); the pair is read as ``alpha = sqrt(occ) exp(i phi)``.
        n_steps : int
            Number of periods (>= 0); the returned trajectory includes the
            initial point.

        Returns
        -------
        phi : numpy.ndarray of float, shape (n_steps + 1, L)
            Site phases ``arg(alpha_j)`` in ``(-pi, pi]`` per period (the
            map's global-phase gauge, see module docstring).
        occ : numpy.ndarray of float, shape (n_steps + 1, L)
            Site occupations ``|alpha_j|^2``.

        Raises
        ------
        TypeError
            If ``n_steps`` is not an integer.
        ValueError
            If ``n_steps`` is negative or the initial state has the wrong
            shape, non-finite entries, negative occupations, or vanishes.

        Examples
        --------
        >>> import numpy as np
        >>> m = KickedMeanFieldMap(3, tau=0.4, J=1.0, U=0.6)
        >>> phi, occ = m.iterate(np.array([2.0, 1.0, 1.0]) + 0j, 10)
        >>> phi.shape, occ.shape
        ((11, 3), (11, 3))
        >>> float(np.round(occ.sum(axis=1).min(), 12))
        6.0
        """
        if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
            raise TypeError(
                f"KickedMeanFieldMap.iterate: n_steps must be an int, got {type(n_steps).__name__}"
            )
        if n_steps < 0:
            raise ValueError(f"KickedMeanFieldMap.iterate: n_steps must be >= 0, got {n_steps}")
        if isinstance(initial, tuple):
            if len(initial) != 2:
                raise ValueError(
                    "KickedMeanFieldMap.iterate: (phi, occ) initial must have length 2, "
                    f"got {len(initial)}"
                )
            phi0 = np.asarray(initial[0], dtype=float)
            occ0 = np.asarray(initial[1], dtype=float)
            if phi0.shape != (self.L,) or occ0.shape != (self.L,):
                raise ValueError(
                    f"KickedMeanFieldMap.iterate: phi and occ must have shape ({self.L},), "
                    f"got {phi0.shape} and {occ0.shape}"
                )
            if np.any(occ0 < 0.0) or not np.all(np.isfinite(phi0)) or not np.all(np.isfinite(occ0)):
                raise ValueError("KickedMeanFieldMap.iterate: initial must be finite, occ >= 0")
            a = np.sqrt(occ0) * np.exp(1j * phi0)
        else:
            a = np.asarray(initial, dtype=complex)
            if a.shape != (self.L,):
                raise ValueError(
                    f"KickedMeanFieldMap.iterate: alpha must have shape ({self.L},), got {a.shape}"
                )
            if not np.all(np.isfinite(a)):
                raise ValueError("KickedMeanFieldMap.iterate: alpha has non-finite entries")
        if np.sum(np.abs(a) ** 2) == 0.0:
            raise ValueError("KickedMeanFieldMap.iterate: the zero state has no phase")
        out_phi = np.empty((n_steps + 1, self.L))
        out_occ = np.empty((n_steps + 1, self.L))
        for k in range(n_steps + 1):
            out_phi[k] = np.angle(a)
            out_occ[k] = np.abs(a) ** 2
            if k < n_steps:
                a = self.step(a)
        return out_phi, out_occ

    def dimer_variables(self, phi: np.ndarray, occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Trajectory in ``(s, phi_rel)`` coordinates of :func:`bohnasa.mean_field_flow`.

        ``s = (n_0 - n_1) / (n_0 + n_1)`` and ``phi_rel = varphi_1 - varphi_0``
        wrapped to ``(-pi, pi]``, the dictionary pinned in the module
        docstring (the sign of the relative phase is the one that reproduces
        the Josephson current ``ds/dt = -4 J sqrt(1 - s^2) sin(phi)``).

        Parameters
        ----------
        phi, occ : numpy.ndarray
            Trajectory arrays of shape ``(..., 2)`` (e.g. the output of
            :meth:`iterate` for ``L = 2``).

        Returns
        -------
        s : numpy.ndarray of float
            Population imbalance, shape ``phi.shape[:-1]``.
        phi_rel : numpy.ndarray of float
            Relative phase ``varphi_1 - varphi_0`` in ``(-pi, pi]``.

        Raises
        ------
        ValueError
            If the map is not a dimer (``L != 2``) or the arrays do not have
            a shared trailing axis of length 2.

        Examples
        --------
        >>> import numpy as np
        >>> m = KickedMeanFieldMap(2, tau=0.1, J=1.0, U=0.5)
        >>> phi, occ = m.iterate(np.array([2.0, 1.0]) + 0j, 1)
        >>> s, phi_rel = m.dimer_variables(phi, occ)
        >>> float(np.round(s[0], 12))
        0.6
        >>> float(phi_rel[0])
        0.0
        """
        if self.L != 2:
            raise ValueError(f"dimer_variables: requires L = 2, this map has L = {self.L}")
        p = np.asarray(phi, dtype=float)
        n = np.asarray(occ, dtype=float)
        if p.shape[-1:] != (2,) or n.shape != p.shape:
            raise ValueError(
                f"dimer_variables: phi and occ must share shape (..., 2), "
                f"got {p.shape} and {n.shape}"
            )
        s = (n[..., 0] - n[..., 1]) / (n[..., 0] + n[..., 1])
        phi_rel = (p[..., 1] - p[..., 0] + np.pi) % (2 * np.pi) - np.pi
        phi_rel = np.where(phi_rel == -np.pi, np.pi, phi_rel)  # wrap to (-pi, pi]
        return s, phi_rel


class KickedBoseHubbard:
    r"""Floquet kicked Bose-Hubbard ring: step propagator, spectrum, map.

    One period is ``U_step = exp(-i tau H_J) exp(-i tau H_U)`` with
    ``H_J = -J * Hopping(fock, connections)`` and
    ``H_U = (U / 2) * Interaction(fock)`` - see the module docstring for the
    conventions (bond list, factor order, quasienergy branch). The class is
    basis-free: pass a :class:`bohnasa.FixedNFockSpace` with ``fock.L == L``
    to :meth:`propagator` and :meth:`floquet_spectrum`.

    Parameters
    ----------
    L : int
        Number of sites (>= 2).
    tau : float
        Kick period (drive units); positive.
    J : float
        Hopping amplitude of ``-J * Hopping``.
    U : float
        On-site coupling of ``(U / 2) sum n (n - 1)``.
    connections : {'pbc', 'periodic', 'obc', 'open'} or array-like of shape (*, 2)
        Bond geometry. Named boundary conditions additionally enable the
        symmetry-sector blocking of :meth:`floquet_spectrum`; for an explicit
        bond list the spectrum falls back to the full dense problem.

    Raises
    ------
    TypeError
        If ``L`` is not an integer.
    ValueError
        If ``L < 2``, ``tau <= 0``, ``J``/``U`` non-finite, or ``connections``
        is malformed (validation delegated to :func:`bohnasa.bond_pairs`).

    See Also
    --------
    bohnasa.SquaredCommutator.from_propagator :
        Floquet OTOC curves from :meth:`propagator` (``times`` are kick
        counts).
    KickedMeanFieldMap :
        the classical counterpart, returned by :meth:`classical_map`.

    Examples
    --------
    >>> import numpy as np
    >>> from bohnasa import FixedNFockSpace
    >>> k = KickedBoseHubbard(L=3, tau=0.5, J=1.0, U=0.8)
    >>> fock = FixedNFockSpace(4, 3)
    >>> U_step = k.propagator(fock)
    >>> float(np.abs(U_step.conj().T @ U_step - np.eye(fock.dim)).max()) < 1e-13
    True

    The step feeds the Floquet OTOC directly (issue #29's motivating API):

    >>> from bohnasa import Number, SquaredCommutator, fixed_n_coherent_state
    >>> op = Number(fock, 0)
    >>> psi = fixed_n_coherent_state(fock, np.ones(3) / np.sqrt(3.0))
    >>> c = SquaredCommutator.from_propagator(U_step, op).curve(psi, np.arange(6))
    >>> float(c[0]), bool(np.all(c[1:] > 0.0))
    (0.0, True)

    Quasienergies come out of the stated branch, ``[0, 2 pi / tau)``:

    >>> eps, vecs = k.floquet_spectrum(fock)
    >>> bool(eps.min() >= 0.0 and eps.max() < 2 * np.pi / k.tau)
    True
    >>> float(np.abs(U_step @ vecs - vecs * np.exp(-1j * eps * k.tau)).max()) < 1e-9
    True
    """

    def __init__(
        self,
        L: int,
        tau: float,
        J: float,
        U: float,
        connections: str | np.ndarray | Sequence[Sequence[int]] = "pbc",
    ) -> None:
        bonds = _check_geometry(L, connections)
        self.L = int(L)
        self.tau, self.J, self.U = _check_dynamics(tau, J, U)
        self.connections = connections
        self.bonds = bonds

    def _check_fock(self, fock: FixedNFockSpace, caller: str) -> None:
        """Reject bases that are not ``FixedNFockSpace`` or mismatched in ``L``."""
        if not isinstance(fock, FixedNFockSpace):
            raise TypeError(f"{caller}: fock must be a FixedNFockSpace, got {type(fock).__name__}")
        if fock.L != self.L:
            raise ValueError(f"{caller}: fock.L = {fock.L} does not match L = {self.L}")

    def hopping_hamiltonian(self, fock: FixedNFockSpace) -> Operator:
        """Generator of the hopping kick: ``-J * Hopping(fock, connections)``.

        Examples
        --------
        >>> import numpy as np
        >>> from bohnasa import FixedNFockSpace
        >>> k = KickedBoseHubbard(L=2, tau=0.5, J=1.3, U=0.7)
        >>> fock = FixedNFockSpace(3, 2)
        >>> hop = k.hopping_hamiltonian(fock).matrix
        >>> float(np.round(hop[fock.dict[(2, 1)], fock.dict[(3, 0)]], 10))
        -4.5033320997

        (the L = 2 ring counts its single bond twice, so one hop carries
        ``-2 J sqrt(n_0 (n_1 + 1)) = -2.6 sqrt(3)``)
        """
        self._check_fock(fock, "hopping_hamiltonian")
        return Hopping(fock, connections=self.connections) * (-self.J)

    def interaction_hamiltonian(self, fock: FixedNFockSpace) -> Operator:
        """Generator of the on-site kick: ``(U / 2) * Interaction(fock)``.

        Examples
        --------
        >>> import numpy as np
        >>> from bohnasa import FixedNFockSpace
        >>> k = KickedBoseHubbard(L=2, tau=0.5, J=1.3, U=0.7)
        >>> fock = FixedNFockSpace(3, 2)
        >>> h = k.interaction_hamiltonian(fock).matrix.diagonal()
        >>> np.round(np.sort(np.asarray(h)), 10).tolist()
        [0.7, 0.7, 2.1, 2.1]
        """
        self._check_fock(fock, "interaction_hamiltonian")
        return Interaction(fock) * (self.U / 2.0)

    def propagator(self, fock: FixedNFockSpace) -> np.ndarray:
        r"""Dense Floquet step ``U_step = exp(-i tau H_J) exp(-i tau H_U)``.

        Feed the result to :meth:`bohnasa.SquaredCommutator.from_propagator`
        for Floquet OTOC curves (``times`` are kick counts).

        Parameters
        ----------
        fock : FixedNFockSpace
            Fixed-N basis with ``fock.L == L``.

        Returns
        -------
        numpy.ndarray of complex, shape (dim, dim)
            Unitary step matrix. Dense by contract: every documented consumer
            (``from_propagator``, :meth:`floquet_spectrum` fallback) either
            stores ``dim x dim`` densely itself or eigensolves it, so a
            sparse return would only defer the conversion. The dense ``expm``
            of the hopping generator costs ``O(dim^3)`` - measured 15.0 s at
            ``dim = 3003`` on a 16-core host (20.7 s in a repeat run; dense
            ``expm`` wall time tracks BLAS thread state), which is why
            :meth:`floquet_spectrum` exponentiates inside the symmetry
            blocks instead (and why an OTOC run on huge bases batches the
            two factors' actions, as ``examples/ex10_kicked_dimer.py`` does).

        Raises
        ------
        TypeError
            If ``fock`` is not a ``FixedNFockSpace``.
        ValueError
            If ``fock.L != L``.
        """
        self._check_fock(fock, "propagator")
        h_hop = self.hopping_hamiltonian(fock).matrix.toarray()
        u_hop = sla.expm(-1j * self.tau * h_hop)
        kick_phase = self._kick_phases(fock)
        return u_hop * kick_phase[None, :]  # right-multiply by the diagonal kick

    def _kick_phases(self, fock: FixedNFockSpace) -> np.ndarray:
        """Diagonal of ``exp(-i tau (U / 2) sum n (n - 1))`` in the Fock basis."""
        occ = fock.fockArray.astype(float)
        energy = (self.U / 2.0) * np.sum(occ * (occ - 1.0), axis=1)
        return np.exp(-1j * self.tau * energy)

    def floquet_spectrum(self, fock: FixedNFockSpace) -> tuple[np.ndarray, np.ndarray]:
        r"""Quasienergies and Floquet modes of :meth:`propagator`.

        Branch convention (module docstring): ``eps = mod(-arg(lambda), 2 pi)
        / tau`` in ``[0, 2 pi / tau)``, ascending; ties between equal
        quasienergies keep the collection order (sector label order, then
        LAPACK order within the block), and the sort is ``kind="stable"``, so
        the list is identical across runs.

        For named ``connections`` both kicks commute with the lattice
        symmetry group, so the step is block-diagonal in the
        :class:`bohnasa.SymmetrySectors` basis and every block is
        exponentiated and eigensolved separately at ``O((dim / L)^3)``
        instead of ``O(dim^3)``: measured on an ``L = 6``, ``N = 10`` ring
        (dim 3003, 8 blocks up to dim 504, 16-core host), the dense fallback
        took 39-200 s across repeated runs (LAPACK ``geev`` wall time on
        near-unitary spectra varies strongly between runs) versus 4.6-5.5 s
        for the blocked pipeline - an order of magnitude and more, the
        number that makes quasienergy sweeps affordable. The two paths'
        spectra agree to 3.6e-14 there. Explicit bond-list geometries keep
        no symmetry and use the dense fallback.

        Parameters
        ----------
        fock : FixedNFockSpace
            Fixed-N basis with ``fock.L == L``.

        Returns
        -------
        eps : numpy.ndarray of float, shape (dim,)
            Quasienergies in ``[0, 2 pi / tau)``, ascending.
        vecs : numpy.ndarray of complex, shape (dim, dim)
            Floquet modes as columns in the full Fock basis, ``vecs[:, m]``
            matching ``eps[m]``; orthonormal (sector isometries lift
            eigenvectors unitarily).
        """
        self._check_fock(fock, "floquet_spectrum")
        if isinstance(self.connections, str):
            sectors = SymmetrySectors(fock, connections=self.connections)
            h_hop = self.hopping_hamiltonian(fock)
            kick_op = Operator(fock)
            kick_op.matrix = sp.diags(self._kick_phases(fock))
            eps_parts, vec_parts = [], []
            for label in sectors.labels:
                w = sectors.isometry(label)
                block_h = sectors.restrict(h_hop, label)
                block_k = sectors.restrict(kick_op, label)
                u_hop = sla.expm(-1j * self.tau * block_h.toarray().astype(complex))
                lam, vv = np.linalg.eig(u_hop @ block_k.toarray())
                eps_parts.append(_fold_quasienergies(lam, self.tau))
                vec_parts.append(np.asarray(w @ vv))  # (dim, dim_s) lift
            eps = np.concatenate(eps_parts)
            vecs = np.hstack(vec_parts)
        else:
            u = self.propagator(fock)
            lam, vecs = np.linalg.eig(u)
            eps = _fold_quasienergies(lam, self.tau)
        order = np.argsort(eps, kind="stable")
        return eps[order], vecs[:, order]

    def classical_map(self) -> KickedMeanFieldMap:
        """The classical kicked map of the same drive (module docstring).

        Examples
        --------
        >>> import numpy as np
        >>> k = KickedBoseHubbard(L=2, tau=0.02, J=np.cos(1.35), U=2 * np.sin(1.35) / 20.0)
        >>> phi, occ = k.classical_map().iterate((np.array([0.3, -1.1]), np.array([12.0, 8.0])), 1)
        >>> bool(np.all(np.abs(phi) <= np.pi))
        True
        >>> float(np.round(np.sum(occ[1]) / 20.0, 12))
        1.0
        """
        return KickedMeanFieldMap(self.L, self.tau, self.J, self.U, self.connections)
