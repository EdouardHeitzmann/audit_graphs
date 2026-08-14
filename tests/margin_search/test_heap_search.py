from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import io

import numpy as np

from src.election_graphs.datatypes import EdgeAction
from src.margin_search import heap_based_search
from src.wigm_graphs.graph_wigm import WIGMGraphConstructor


@dataclass
class MatrixProfile:
    candidates: list[str]
    ballot_matrix: np.ndarray
    wt_vec: np.ndarray

    @property
    def total_ballot_wt(self) -> float:
        return float(self.wt_vec.sum())


def first_preference_profile(tallies: list[int]) -> MatrixProfile:
    return MatrixProfile(
        candidates=[f"C{candidate}" for candidate in range(len(tallies))],
        ballot_matrix=np.array(
            [[candidate, -127] for candidate in range(len(tallies))],
            dtype=np.int8,
        ),
        wt_vec=np.asarray(tallies, dtype=np.float64),
    )


def test_expand_margin_admits_optional_election_at_exact_boundary():
    profile = first_preference_profile([41, 40, 19])
    incremental = WIGMGraphConstructor(
        profile,
        m=1,
        MoI=1,
        memory_lite=True,
    )
    incremental.build()
    incremental.expand_margin(10)

    direct = WIGMGraphConstructor(
        profile,
        m=1,
        MoI=10,
        memory_lite=True,
    )
    direct.build()

    def elected_candidates(graph):
        return {
            edge.candidate
            for edge in graph.outgoing_edges(graph.root_ref)
            if EdgeAction(edge.action).is_election
        }

    assert elected_candidates(incremental) == {0}
    assert elected_candidates(incremental) == elected_candidates(direct)


def test_heap_search_can_verify_fresh_final_graph():
    profile = first_preference_profile([60, 30, 10])

    with redirect_stdout(io.StringIO()):
        graph = heap_based_search(
            profile=profile,
            m=1,
            constructor_cls=WIGMGraphConstructor,
            memory_lite=True,
            verify_output=True,
        )

    assert graph.quota == 51.0
    assert graph.MoI == 25.0
    assert graph.MoI < graph.quota / 2.0
    assert not graph.used_seeded_build


def test_heap_search_half_quota_cap_is_strict_for_even_quota():
    profile = first_preference_profile([59, 29, 10])

    with redirect_stdout(io.StringIO()):
        graph = heap_based_search(
            profile=profile,
            m=1,
            constructor_cls=WIGMGraphConstructor,
            memory_lite=True,
        )

    assert graph.quota == 50.0
    assert graph.MoI == 24.0
    assert graph.MoI < graph.quota / 2.0


def test_heap_search_can_override_half_quota_cap():
    profile = first_preference_profile([60, 30, 10])

    with redirect_stdout(io.StringIO()):
        graph = heap_based_search(
            profile=profile,
            m=1,
            constructor_cls=WIGMGraphConstructor,
            memory_lite=True,
            allow_moi_at_or_above_half_quota=True,
        )

    assert graph.quota == 51.0
    # The runner-up's seating edge appears inclusively at its threshold
    # max(60, 51) - 30 = 30, so the maximal coherent MoI is 29.
    assert graph.MoI == 29.0
    assert graph.MoI >= graph.quota / 2.0
