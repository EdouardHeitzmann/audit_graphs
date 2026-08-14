from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

try:
    from ..election_graphs.datatypes import (
        EdgeAction,
        EdgeRef,
        EdgeStatus,
        ElectionEdge,
        ElectionState,
        ElectionStatus,
        VertexRef,
    )
except ImportError:
    from election_graphs.datatypes import (
        EdgeAction,
        EdgeRef,
        EdgeStatus,
        ElectionEdge,
        ElectionState,
        ElectionStatus,
        VertexRef,
    )


@dataclass(frozen=True, slots=True)
class StateKey:
    seated_at: tuple[Optional[EdgeRef], ...]
    hopefuls: frozenset[int]


@dataclass(slots=True)
class WIGMRuntimeCache:
    """
    Expansion-only cache. This should not be considered durable graph data.
    """
    bool_ballot_matrix: NDArray[np.bool_]
    pos_vec: NDArray[np.integer]
    fpv_vec: NDArray[np.integer] | None = None


__all__ = [
    "EdgeAction",
    "EdgeRef",
    "EdgeStatus",
    "ElectionEdge",
    "ElectionState",
    "ElectionStatus",
    "StateKey",
    "VertexRef",
    "WIGMRuntimeCache",
]
