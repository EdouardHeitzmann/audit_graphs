from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.election_graphs.datatypes import EdgeAction
import src.test_processes.delta_method as delta_method
from src.test_processes import (
    CobraMentionsNoiseFilterCompiler,
    CobraNoiseFilterCompiler,
    CobraQuotaNoiseFilterCompiler,
    CriticalMarginType,
    DeltaMethodCompiler,
    GlobalAuditDriver,
    VertexInterpreter,
)
from src.wigm_graphs.graph_wigm import WIGMGraphConstructor
from src.wigm_graphs.seeded import SeededWIGMGraphConstructor


@dataclass
class MatrixProfile:
    candidates: list[str]
    ballot_matrix: np.ndarray
    wt_vec: np.ndarray

    @property
    def total_ballot_wt(self) -> float:
        return float(self.wt_vec.sum())


def synthetic_profile() -> MatrixProfile:
    candidates = ["W1", "S2", "W3", "S3", "L1", "W2", "L2", "S1"]
    ballots = [
        (25_000, ["W1"]),
        (15_000, ["W1", "S2", "W3"]),
        (15_000, ["W1", "S2", "S3"]),
        (5_000, ["W1", "L1"]),
        (30_000, ["W1", "S3"]),
        (60_000, ["W1", "W3"]),
        (40_000, ["W2", "W3"]),
        (40_000, ["W2", "S3"]),
        (50_000, ["W3"]),
        (70_000, ["S3"]),
        (5_000, ["L1", "W3"]),
        (4_999, ["L2", "W3"]),
        (20_000, ["S1", "W2"]),
        (10_000, ["S2", "W3"]),
        (10_000, ["S2", "W2"]),
    ]
    ballot_matrix = np.full((len(ballots), 3), -127, dtype=np.int8)
    weights = []
    for row_idx, (weight, ranking) in enumerate(ballots):
        ballot_matrix[row_idx, : len(ranking)] = [
            candidates.index(candidate) for candidate in ranking
        ]
        weights.append(weight)

    return MatrixProfile(
        candidates=candidates,
        ballot_matrix=ballot_matrix,
        wt_vec=np.array(weights, dtype=np.float64),
    )


def build_synthetic_graph() -> WIGMGraphConstructor:
    graph = WIGMGraphConstructor(
        synthetic_profile(),
        m=3,
        MoI=9_000,
        memory_lite=True,
        simultaneous=True,
    )
    graph.build()
    return graph


def _basepoint_transfers(basepoint: np.ndarray, quota: float) -> np.ndarray:
    row_totals = np.asarray(basepoint, dtype=np.float64).sum(axis=1)
    degree = int(np.log2(len(row_totals)))
    transfers = np.empty(degree, dtype=np.float64)

    for winner in range(degree):
        tally = 0.0
        for row_mask, row_total in enumerate(row_totals):
            if not ((row_mask >> winner) & 1):
                continue
            weight = 1.0
            for earlier in range(winner):
                if (row_mask >> earlier) & 1:
                    weight *= transfers[earlier]
            tally += weight * row_total
        transfers[winner] = 1.0 - float(quota) / tally

    return transfers


def test_vertex_interpreter_degree_one_basepoint_and_margin():
    graph = build_synthetic_graph()
    l1 = graph.candidate_names.index("L1")
    vertex = next(v for v in graph.layers[1] if l1 in v.key.hopefuls)

    interpreter = VertexInterpreter(graph, vertex)
    base_point = interpreter.base_point("S2", "L1")

    np.testing.assert_array_equal(
        base_point,
        np.array(
            [
                [20_000, 5_000, 224_999],
                [30_000, 5_000, 115_000],
            ],
            dtype=np.float64,
        ),
    )

    margin, _ = delta_method._numerical_recursive_margin_and_gradient(
        base_point,
        graph.quota,
    )
    assert margin == pytest.approx(
        vertex.tallies[graph.candidate_names.index("S2")] - vertex.tallies[l1]
    )


def test_vertex_interpreter_degree_two_basepoint_and_margin():
    graph = build_synthetic_graph()
    vertex = graph.layers[6][0]

    assert [
        graph.candidate_names[graph.edge(edge_ref).candidate]
        for edge_ref in vertex.key.seated_at
        if edge_ref is not None
    ] == ["W1", "W2"]

    interpreter = VertexInterpreter(graph, vertex)
    base_point = interpreter.base_point("W3", "S3")

    np.testing.assert_array_equal(
        base_point,
        np.array(
            [
                [69_999, 70_000, 10_000],
                [75_000, 45_000, 30_000],
                [40_000, 40_000, 20_000],
                [0, 0, 0],
            ],
            dtype=np.float64,
        ),
    )

    margin, _ = delta_method._numerical_recursive_margin_and_gradient(
        base_point,
        graph.quota,
    )
    assert margin == pytest.approx(
        vertex.tallies[graph.candidate_names.index("W3")]
        - vertex.tallies[graph.candidate_names.index("S3")]
    )


def test_delta_method_symbolic_margins_match_degree_one_and_two_graph_tallies():
    graph = build_synthetic_graph()
    l1 = graph.candidate_names.index("L1")
    cases = [
        (next(v for v in graph.layers[1] if l1 in v.key.hopefuls), "S2", "L1"),
        (graph.layers[6][0], "W3", "S3"),
    ]

    for vertex, c, l in cases:
        interpreter = VertexInterpreter(graph, vertex)
        compiler = DeltaMethodCompiler(
            interpreter,
            margin_type=CriticalMarginType.CANDIDATE_TO_CANDIDATE,
            c=c,
            l=l,
        )
        unchanged_sample = graph.ballot_matrix[:2]
        result = compiler.evaluate(unchanged_sample, unchanged_sample)
        c_idx = graph.candidate_names.index(c)
        l_idx = graph.candidate_names.index(l)

        assert result.estimated_margin == pytest.approx(
            vertex.tallies[c_idx] - vertex.tallies[l_idx]
        )
        assert result.gradient.shape == (3 * (1 << vertex.degree),)


def test_memory_lite_wigm_edges_store_source_fpv_vectors():
    graph = build_synthetic_graph()
    seating_edges = [
        edge
        for layer in graph.edge_layers
        for edge in layer
        if EdgeAction(edge.action).is_seating
    ]

    assert seating_edges
    assert all(edge.fpv_vec is not None for edge in seating_edges)
    assert all(edge.fpv_vec.shape == graph.root_wt_vec.shape for edge in seating_edges)


def test_winner_prefix_indices_match_edgewise_accumulation():
    graph = build_synthetic_graph()
    vertex = next(
        vertex
        for layer in graph.layers
        for vertex in layer
        if vertex.degree >= 2
    )
    interpreter = VertexInterpreter(graph, vertex)

    expected = np.zeros(graph.ballot_matrix.shape[0], dtype=np.int64)
    for bit, edge in enumerate(interpreter.seating_edges):
        expected |= (edge.fpv_vec == edge.candidate).astype(np.int64) << bit

    first = interpreter.winner_prefix_indices()
    cached = interpreter.winner_prefix_indices(copy=False)

    assert np.array_equal(first, expected)
    assert np.array_equal(cached, expected)
    assert cached is interpreter._prefix_cache
    assert cached.dtype.itemsize == 1


def test_cached_profile_masses_reproduce_rowwise_base_points():
    graph = build_synthetic_graph()
    vertex = next(
        vertex
        for layer in graph.layers
        for vertex in layer
        if vertex.degree >= 2
    )
    interpreter = VertexInterpreter(graph, vertex)
    weights = interpreter.profile_wt_vec()

    cached_point = interpreter.base_point(0, 1)
    rowwise_point = interpreter.base_point(0, 1, wt_vec=weights)
    assert np.allclose(cached_point, rowwise_point)

    candidate_point = interpreter.candidate_base_point(0, 1)
    expected_candidate_point = np.zeros(interpreter.shape, dtype=np.float64)
    prefixes = interpreter.winner_prefix_indices(copy=False)
    fpv = interpreter.current_fpv_vec(copy=False)
    columns = np.where(fpv == 0, 1, 2)
    np.add.at(expected_candidate_point, (prefixes, columns), weights)
    assert np.allclose(candidate_point, expected_candidate_point)

    mass, prefix_mass = interpreter.profile_mass_by_prefix_candidate(copy=False)
    cached_mass, cached_prefix_mass = interpreter.profile_mass_by_prefix_candidate(
        copy=False
    )
    assert mass is cached_mass
    assert prefix_mass is cached_prefix_mass


def test_vertex_interpreter_projects_sampled_rows_to_theta_keys():
    graph = build_synthetic_graph()
    vertex = graph.layers[1][0]
    interpreter = VertexInterpreter(graph, vertex)

    theta = interpreter.theta(
        graph.ballot_matrix[0],
        graph.ballot_matrix[10],
        "S2",
        "L1",
    )
    assert theta.shape == (2, 3)
    assert theta.sum() == 0
    assert interpreter.theta_from_key((4, 1)).tolist() == [
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    theta_key = interpreter.theta_key(
        graph.ballot_matrix[0],
        graph.ballot_matrix[10],
        "S2",
        "L1",
    )
    assert interpreter.theta_from_key(theta_key).shape == theta.shape
    np.testing.assert_array_equal(interpreter.theta_from_key(theta_key), theta)


def test_seeded_basepoint_uses_unseeded_weights_for_transfer_recursion():
    class Profile:
        candidates = ["Very0", "Very1", "Strong", "Weak"]
        ballot_matrix = np.array(
            [
                [0, -127, -127, -127],
                [1, -127, -127, -127],
                [2, 3, -127, -127],
                [3, -127, -127, -127],
            ],
            dtype=np.int8,
        )
        wt_vec = np.array([1200, 900, 300, 100], dtype=np.float64)
        total_ballot_wt = float(wt_vec.sum())

    graph = SeededWIGMGraphConstructor(
        Profile(),
        m=3,
        MoI=50,
        simultaneous=True,
        memory_lite=True,
    )
    graph.seeded_build(
        very_strong_candidates={0, 1},
        strong_candidates={2},
        weak_candidates={3},
    )
    assert not np.array_equal(graph.root_wt_vec, graph._unseeded_root_wt_vec)

    seed_vertex = next(
        vertex
        for layer in graph.layers
        for vertex in layer
        if frozenset(vertex.key.hopefuls) == {2}
    )
    interpreter = VertexInterpreter(graph, seed_vertex)
    basepoint = interpreter.base_point(2, 3)
    transfers = _basepoint_transfers(basepoint, graph.quota)

    assert np.all(np.isfinite(transfers))
    assert np.all(transfers >= -1.0)
    assert np.all(transfers <= 1.0)


def test_cobra_noise_filter_compiler_uses_half_radius_dilution():
    graph = build_synthetic_graph()
    interpreter = VertexInterpreter(graph, graph.layers[1][0])
    compiler = CobraNoiseFilterCompiler(
        interpreter,
        "S2",
        "L1",
        radius=4_000,
        profile=True,
    )

    assert compiler.critical_margin == 2_000
    assert compiler.v == pytest.approx(4_000 / (2 * graph.profile.total_ballot_wt))

    unchanged = compiler.update(graph.ballot_matrix[0], graph.ballot_matrix[0])
    noisy = compiler.update(graph.ballot_matrix[0], graph.ballot_matrix[10])

    assert compiler.num_updates == 2
    assert compiler.w_history == [0.0, 1.0]
    assert compiler.theta_key_history[0] is None
    assert compiler.theta_key_history[1] is not None
    assert noisy != unchanged


def test_cobra_mentions_noise_filter_compiler_uses_mentions_coordinates():
    graph = build_synthetic_graph()
    interpreter = VertexInterpreter(graph, graph.layers[1][0])
    strong = {
        graph.candidate_names.index("S2"),
        graph.candidate_names.index("W3"),
    }
    weak = graph.candidate_names.index("L1")
    maximum_possible_tallies = np.zeros(graph.n_candidates, dtype=np.float64)
    maximum_possible_tallies[weak] = 5_000.0

    compiler = CobraMentionsNoiseFilterCompiler(
        interpreter,
        "L1",
        strong_candidates=strong,
        maximum_possible_tallies=maximum_possible_tallies,
        lowest_strong_candidate=graph.candidate_names.index("S2"),
        lowest_strong_tally=30_000,
        radius=4_000,
        profile=True,
    )

    assert compiler._mentions_column(np.array([4, 1, -127])) == 1
    assert compiler._mentions_column(np.array([1, 4, -127])) == 0
    assert compiler._mentions_column(np.array([2, 4, -127])) == 2
    theta_key = compiler._mentions_theta_key(
        np.array([4, 1, -127]),
        np.array([1, 4, -127]),
    )
    assert theta_key[0] != theta_key[1]

    unchanged = compiler.update(np.array([1, 4, -127]), np.array([1, 4, -127]))
    noisy = compiler.update(np.array([1, 4, -127]), np.array([4, 1, -127]))

    assert compiler.critical_margin == 2_000
    assert compiler.v == pytest.approx(4_000 / (2 * graph.profile.total_ballot_wt))
    assert compiler.w_history == [0.0, 1.0]
    assert noisy != unchanged


def test_global_audit_driver_noise_filters_use_escape_critical_margins():
    graph = build_synthetic_graph()
    driver = GlobalAuditDriver(
        graph,
        noise_level=0.1,
        sample_size=5,
        seed=23,
        compiler_type="noise",
        print_diagnostics_every=0,
        simultaneous=True,
    )

    assert driver.compiler_type == "noise"
    assert driver.compilers
    assert all(
        isinstance(
            compiler,
            (CobraNoiseFilterCompiler, CobraQuotaNoiseFilterCompiler),
        )
        for compiler in driver.compilers
    )
    assert {compiler.MoI for compiler in driver.compilers} == {float(graph.MoI)}
    assert all(
        compiler.radius == pytest.approx(2.0 * compiler.critical_margin)
        for compiler in driver.compilers
    )
    for compiler, info in zip(driver.compilers, driver.compiler_info):
        vertex = graph.layers[info.base_layer][info.base_local_id]
        escape_margin = driver._noise_filter_margin(
            vertex,
            info.candidate,
            info.action,
            simultaneous=True,
        )["margin"]
        if isinstance(compiler, CobraNoiseFilterCompiler):
            assert compiler.critical_margin == pytest.approx(escape_margin / 2.0)
        else:
            assert isinstance(compiler, CobraQuotaNoiseFilterCompiler)
            assert compiler.critical_margin == pytest.approx(escape_margin)
    assert all(not info.escape_id.endswith("-N") for info in driver.compiler_info)

    driver.run(ballot_matrix=graph.ballot_matrix, num_steps=1)
    assert driver.i == 1
    assert {compiler.num_updates for compiler in driver.compilers} == {1}

