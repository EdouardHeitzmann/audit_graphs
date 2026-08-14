from __future__ import annotations

from collections.abc import Iterable
from math import comb

import numpy as np
from numpy.typing import NDArray

from .datatypes import SENTINEL


def _normalize_candidates(candidates: Iterable[int] | None) -> frozenset[int]:
    if candidates is None:
        return frozenset()
    return frozenset(int(candidate) for candidate in candidates)


def fpv_tallies_from_matrix(
    ballot_matrix: NDArray[np.integer],
    wt_vec: NDArray[np.float64],
    n_candidates: int,
    *,
    allowed_candidates: Iterable[int] | None = None,
    masked_candidates: Iterable[int] | None = None,
) -> NDArray[np.float64]:
    """
    Compute first-preference tallies after removing masked candidates.

    If allowed_candidates is provided, only those candidates can receive first
    preferences; this is useful for checking tallies after condensing to a
    proposed strong set.
    """
    allowed = None
    if allowed_candidates is not None:
        allowed = _normalize_candidates(allowed_candidates)

    masked = _normalize_candidates(masked_candidates)
    tallies = np.zeros(n_candidates, dtype=np.float64)

    for row, weight in zip(ballot_matrix, wt_vec):
        for raw_candidate in row:
            candidate = int(raw_candidate)
            if candidate == SENTINEL:
                break
            if candidate in masked:
                continue
            if allowed is not None and candidate not in allowed:
                continue

            tallies[candidate] += float(weight)
            break

    return tallies


def maximum_possible_tallies_from_matrix(
    ballot_matrix: NDArray[np.integer],
    wt_vec: NDArray[np.float64],
    n_candidates: int,
    strong_candidates: Iterable[int],
    *,
    masked_candidates: Iterable[int] | None = None,
) -> NDArray[np.float64]:
    """
    Count mentions before the first strong candidate in each condensed row.
    """
    strong = _normalize_candidates(strong_candidates)
    masked = _normalize_candidates(masked_candidates)
    mentions = np.zeros(n_candidates, dtype=np.float64)

    for row, weight in zip(ballot_matrix, wt_vec):
        for raw_candidate in row:
            candidate = int(raw_candidate)
            if candidate == SENTINEL:
                break
            if candidate in masked:
                continue
            if candidate in strong:
                break

            mentions[candidate] += float(weight)

    return mentions


def weak_candidates_from_strong(
    ballot_matrix: NDArray[np.integer],
    wt_vec: NDArray[np.float64],
    n_candidates: int,
    strong_candidates: Iterable[int],
    *,
    MoI: float,
    quota: float,
    verify_strong: bool = False,
    masked_candidates: Iterable[int] | None = None,
) -> tuple[frozenset[int], NDArray[np.float64]]:
    """
    Determine the weak set induced by a prescribed strong set.

    A non-strong candidate is weak when its maximum possible tallies are at least MoI
    below the smallest current first-preference tally among strong candidates.
    """
    strong = _normalize_candidates(strong_candidates)
    masked = _normalize_candidates(masked_candidates)
    active_strong = strong - masked
    if not active_strong:
        raise ValueError("Cannot derive weak candidates from an empty strong set.")

    if verify_strong:
        strong_only_tallies = fpv_tallies_from_matrix(
            ballot_matrix,
            wt_vec,
            n_candidates,
            allowed_candidates=active_strong,
            masked_candidates=masked,
        )
        bad_strong = [
            candidate
            for candidate in active_strong
            if strong_only_tallies[candidate] + MoI >= quota
        ]
        if bad_strong:
            details = ", ".join(
                f"{candidate}: {strong_only_tallies[candidate]}"
                for candidate in sorted(bad_strong)
            )
            raise ValueError(
                "Strong candidates must remain below quota minus MoI when "
                f"standing alone; violating tallies are {details}."
            )

    current_fpv = fpv_tallies_from_matrix(
        ballot_matrix,
        wt_vec,
        n_candidates,
        masked_candidates=masked,
    )
    smallest_strong_fpv = min(float(current_fpv[candidate]) for candidate in active_strong)
    maximum_possible_tallies = maximum_possible_tallies_from_matrix(
        ballot_matrix,
        wt_vec,
        n_candidates,
        active_strong,
        masked_candidates=masked,
    )

    weak = frozenset(
        candidate
        for candidate in range(n_candidates)
        if (
            candidate not in active_strong
            and candidate not in masked
            and maximum_possible_tallies[candidate] + MoI <= smallest_strong_fpv
        )
    )

    return weak, maximum_possible_tallies


def search_strong_weak_candidates(
    ballot_matrix: NDArray[np.integer],
    wt_vec: NDArray[np.float64],
    n_candidates: int,
    *,
    remaining_seats: int,
    MoI: float,
    quota: float,
    masked_candidates: Iterable[int] | None = None,
) -> tuple[frozenset[int], frozenset[int], NDArray[np.float64]]:
    """
    Greedily search for a strong set inducing the largest weak set.

    Candidates are added in current FPV order, beginning with enough candidates
    to fill the remaining seats. The search stops once a valid expansion no
    longer grows the induced weak set.
    """
    masked = _normalize_candidates(masked_candidates)
    if remaining_seats <= 0:
        return frozenset(), frozenset(), np.zeros(n_candidates, dtype=np.float64)

    current_fpv = fpv_tallies_from_matrix(
        ballot_matrix,
        wt_vec,
        n_candidates,
        masked_candidates=masked,
    )
    candidate_order = [
        candidate
        for candidate in np.argsort(-current_fpv)
        if int(candidate) not in masked
    ]
    if len(candidate_order) < remaining_seats:
        raise ValueError(
            "Cannot search strong candidates: fewer active candidates than "
            "remaining seats."
        )

    best_strong: frozenset[int] | None = None
    best_weak: frozenset[int] = frozenset()
    best_mentions = np.zeros(n_candidates, dtype=np.float64)

    for size in range(remaining_seats, len(candidate_order) + 1):
        strong = frozenset(int(candidate) for candidate in candidate_order[:size])
        strong_only_tallies = fpv_tallies_from_matrix(
            ballot_matrix,
            wt_vec,
            n_candidates,
            allowed_candidates=strong,
            masked_candidates=masked,
        )
        if any(strong_only_tallies[candidate] + MoI >= quota for candidate in strong):
            continue

        weak, mentions = weak_candidates_from_strong(
            ballot_matrix,
            wt_vec,
            n_candidates,
            strong,
            MoI=MoI,
            quota=quota,
            verify_strong=False,
            masked_candidates=masked,
        )

        if best_strong is None or len(weak) > len(best_weak):
            best_strong = strong
            best_weak = weak
            best_mentions = mentions
            continue

        break

    if best_strong is None:
        raise ValueError(
            "Could not find a valid strong set whose condensed tallies stay "
            "below quota minus MoI."
        )

    return best_strong, best_weak, best_mentions
