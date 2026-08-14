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


def seat_scarce(vertex: Any, *, quota: float, MoI: float, m: int) -> bool:
    """More hopefuls within MoI of quota than seats remain at this vertex."""
    tallies = np.asarray(vertex.tallies, dtype=np.float64)
    hopefuls = np.asarray(sorted(vertex.key.hopefuls), dtype=int)
    window_count = int(np.sum(tallies[hopefuls] >= quota - MoI))
    return window_count > int(m) - int(vertex.degree)


def critical_margin_for_escape(
    vertex: Any,
    candidate: int,
    action: EdgeAction,
    *,
    quota: float,
    MoI: float,
    simultaneous: bool,
    m: int | None = None,
) -> dict[str, Any] | None:
    """
    Identify the critical margin justifying the non-inclusion of an escape edge.

    Elimination escapes are justified by a forced winner above quota, or else
    by the candidate-to-candidate margin against the lowest hopeful. Election
    escapes are justified by a stronger challenger, or else by the candidate
    sitting below quota. Under simultaneous semantics the challenger
    justification applies only under seat scarcity (``m`` given and more
    quota-window hopefuls than remaining seats), where the plausible winners
    are decided head-to-head; otherwise every candidate within MoI of quota
    is includable, so only the below-quota margin can exclude one. Returns
    None when no margin separates the escape edge; callers decide whether
    that is an error.
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
    elif (
        m is not None
        and c_tally >= quota - MoI
        and seat_scarce(vertex, quota=quota, MoI=MoI, m=m)
    ):
        # Head-to-head justification under seat scarcity: the candidate's
        # most favorable group seats it alongside the strongest other window
        # candidates, so the binding rival is the r-th strongest plausible
        # other (r = remaining seats minus mandatory definite winners).
        window_others = sorted(
            (
                int(h)
                for h in vertex.key.hopefuls
                if int(h) != int(candidate) and tallies[h] >= quota - MoI
            ),
            key=lambda h: float(tallies[h]),
            reverse=True,
        )
        definite_count = sum(
            1 for h in window_others if tallies[h] > quota + MoI
        )
        remaining_plausible_seats = (
            int(m) - int(vertex.degree) - definite_count
        )
        rival_index = definite_count + remaining_plausible_seats - 1
        if remaining_plausible_seats >= 1 and rival_index < len(window_others):
            rival = window_others[rival_index]
            rival_tally = float(tallies[rival])
            if rival_tally > c_tally + MoI:
                return {
                    "type": CriticalMarginType.CANDIDATE_TO_CANDIDATE,
                    "c": rival,
                    "l": int(candidate),
                    "margin": float(rival_tally - c_tally),
                }

    if c_tally + MoI < quota:
        return {
            "type": CriticalMarginType.CANDIDATE_BELOW_QUOTA,
            "candidate": int(candidate),
            "margin": float(quota - c_tally),
        }
    return None
