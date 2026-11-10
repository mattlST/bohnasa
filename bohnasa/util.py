"""Deprecated shim: contents moved to ``bohnasa.krylov`` and ``bohnasa.time_evolution``.

Kept so that ``from bohnasa.util import lanczos`` keeps working for existing
scripts; to be dropped in Phase 5.
"""

from .krylov import (  # noqa: F401
    MAX_REORTH_ITERATIONS,
    krylov_complexity,
    Krylov_complexity,
    lanczos,
)
from .time_evolution import time_evolution_exact_ind  # noqa: F401
