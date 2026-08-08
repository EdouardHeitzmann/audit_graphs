"""Escape-edge critical-margin selection shared by the audit drivers."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from ..election_graphs.datatypes import EdgeAction
except ImportError:
    from election_graphs.datatypes import EdgeAction
    from test_processes.cobra import CriticalMarginType
else:
    from .cobra import CriticalMarginType


def critical_margin_for_escape(
    vertex: Any,
    candidate: int,
    action: EdgeAction,
    *,
    quota: float,
    MoI: float,
    simultaneous: bool,
) -> dict[str, Any] | None:
    """
    Identify the critical margin justifying the non-inclusion of an escape edge.

    Elimination escapes are justified by a forced winner above quota, or else
    by the candidate-to-candidate margin against the lowest hopeful. Election
    escapes are justified by a stronger challenger (sequential semantics only),
    or else by the candidate sitting below quota. Returns None when no margin
    separates the escape edge; callers decide whether that is an error.
    """
    tallies = np.asarray(vertex.tallies, dtype=np.float64)

    if action == EdgeAction.ELIMINATE:
        forced = np.where(tallies >= quota + MoI)[0]
        if len(forced) > 0:
            winner = int(forced[np.argmax(tallies[forced])])
            return {
                "type": CriticalMarginType.CANDIDATE_ABOVE_QUOTA,
                "candidate": winner,
                "margin": float(tallies[winner] - quota),
            }

        hopefuls = np.asarray(sorted(vertex.key.hopefuls), dtype=int)
        lowest = int(hopefuls[np.argmin(tallies[hopefuls])])
        if lowest == candidate:
            return None
        return {
            "type": CriticalMarginType.CANDIDATE_TO_CANDIDATE,
            "c": int(candidate),
            "l": lowest,
            "margin": float(tallies[candidate] - tallies[lowest]),
        }

    c_tally = float(tallies[candidate])
    if not simultaneous:
        challengers = np.asarray(
            [
                idx
                for idx in np.where(tallies > c_tally + MoI)[0]
                if idx != candidate
            ],
            dtype=int,
        )
        if len(challengers) > 0:
            winner = int(challengers[np.argmax(tallies[challengers])])
            return {
                "type": CriticalMarginType.CANDIDATE_TO_CANDIDATE,
                "c": winner,
                "l": int(candidate),
                "margin": float(tallies[winner] - c_tally),
            }

    if c_tally + MoI < quota:
        return {
            "type": CriticalMarginType.CANDIDATE_BELOW_QUOTA,
            "candidate": int(candidate),
            "margin": float(quota - c_tally),
        }
    return None
