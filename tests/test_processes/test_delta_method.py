from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pytest
import sympy as sp

import src.test_processes.delta_method as delta_method
from src.test_processes import (
    CriticalMarginType,
    DeltaMethodAuditDriver,
    DeltaMethodCompiler,
    DeltaSampleProjection,
    K_upper,
    VertexInterpreter,
    alternative_K_upper,
    precompute_symbolic_derivatives,
    symbolic_derivative_cache_info,
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


@pytest.fixture(scope="module")
def degree_zero_case():
    profile = MatrixProfile(
        candidates=["A", "B", "C"],
        ballot_matrix=np.array(
            [
                [0, 1, -127],
                [1, 0, -127],
                [2, -127, -127],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([60, 30, 10], dtype=np.float64),
    )
    graph = WIGMGraphConstructor(
        profile,
        m=1,
        MoI=5,
        memory_lite=True,
        simultaneous=True,
    )
    graph.build()
    interpreter = VertexInterpreter(graph, graph.layers[0][0])
    census = np.repeat(profile.ballot_matrix, profile.wt_vec.astype(int), axis=0)
    return graph, interpreter, census


def multi_edge_case():
    profile = MatrixProfile(
        candidates=["A", "B", "C"],
        ballot_matrix=np.array(
            [
                [0, 1, -127],
                [1, 0, -127],
                [2, -127, -127],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([50, 30, 20], dtype=np.float64),
    )
    graph = WIGMGraphConstructor(
        profile,
        m=1,
        MoI=5,
        memory_lite=True,
        simultaneous=True,
    )
    graph.build()
    census = np.repeat(profile.ballot_matrix, profile.wt_vec.astype(int), axis=0)
    return graph, census


def seeded_batch_case():
    profile = MatrixProfile(
        candidates=["Very", "StrongA", "StrongB", "Weak"],
        ballot_matrix=np.array(
            [
                [0, -127, -127, -127],
                [1, -127, -127, -127],
                [2, -127, -127, -127],
                [3, 2, -127, -127],
                [-127, -127, -127, -127],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array([900, 300, 250, 100, 500], dtype=np.float64),
    )
    graph = SeededWIGMGraphConstructor(
        profile,
        m=3,
        MoI=50,
        memory_lite=True,
        simultaneous=True,
    )
    graph.seeded_build(
        very_strong_candidates={0},
        strong_candidates={1, 2},
        weak_candidates={3},
        diagnostics=False,
    )
    census = np.repeat(profile.ballot_matrix, profile.wt_vec.astype(int), axis=0)
    return graph, census


def seeded_multiweak_batch_case():
    profile = MatrixProfile(
        candidates=[
            "Very",
            "StrongA",
            "StrongB",
            "Weak",
            "OtherWeak",
            "InBetween",
        ],
        ballot_matrix=np.array(
            [
                [0, -127, -127, -127, -127, -127],
                [1, -127, -127, -127, -127, -127],
                [2, -127, -127, -127, -127, -127],
                [3, 2, -127, -127, -127, -127],
                [4, 2, -127, -127, -127, -127],
                [5, 2, -127, -127, -127, -127],
                [-127, -127, -127, -127, -127, -127],
            ],
            dtype=np.int8,
        ),
        wt_vec=np.array(
            [900, 300, 250, 100, 80, 70, 500], dtype=np.float64
        ),
    )
    graph = SeededWIGMGraphConstructor(
        profile,
        m=3,
        MoI=50,
        memory_lite=True,
        simultaneous=True,
    )
    graph.seeded_build(
        very_strong_candidates={0},
        strong_candidates={1, 2},
        weak_candidates={3, 4},
        diagnostics=False,
    )
    census = np.repeat(profile.ballot_matrix, profile.wt_vec.astype(int), axis=0)
    return graph, census


def test_hypergeometric_upper_bound_matches_direct_small_population():
    N, n, t, alpha = 18, 7, 2, 0.05

    feasible = []
    for K in range(t, N - (n - t) + 1):
        probability = sum(
            math.comb(K, observed) * math.comb(N - K, n - observed)
            for observed in range(
                max(0, n - (N - K)),
                min(t, n, K) + 1,
            )
        ) / math.comb(N, n)
        if probability >= alpha:
            feasible.append(K)

    assert alternative_K_upper(N, n, t, alpha) == max(feasible)
    assert K_upper(N, n, alpha) == alternative_K_upper(N, n, 0, alpha)


def test_candidate_to_candidate_corrects_totals_and_builds_lower_bound(
    degree_zero_case,
):
    _, interpreter, census = degree_zero_case
    compiler = DeltaMethodCompiler(
        interpreter,
        margin_type=CriticalMarginType.CANDIDATE_TO_CANDIDATE,
        c="A",
        l="B",
    )

    cvr = np.repeat(census[:1], 20, axis=0)
    ballots = cvr.copy()
    ballots[0] = np.array([1, 0, -127], dtype=np.int8)
    result = compiler.evaluate(cvr, ballots)

    np.testing.assert_allclose(result.corrected_point, [[55.0, 35.0, 10.0]])
    np.testing.assert_allclose(result.gradient, [1.0, -1.0, 0.0])
    assert result.estimated_margin == pytest.approx(20.0)
    assert result.discrepancy_count == 1
    assert result.population_discrepancy_upper == alternative_K_upper(
        100, 20, 1, alpha=0.005
    )
    assert result.coordinate_variance_upper == pytest.approx(
        result.population_discrepancy_upper / 100
    )
    assert result.finite_population_correction == pytest.approx(80 / 99)
    assert result.lower_bound == pytest.approx(
        result.estimated_margin - result.critical_value * result.standard_error
    )
    assert result.sampled_discrepancies is None
    assert compiler.last_result is result
    compiler.clear_result()
    assert compiler.last_result is None


def test_compiler_tracks_standardized_discrepancies_only_when_requested(
    degree_zero_case,
):
    _, interpreter, census = degree_zero_case
    compiler = DeltaMethodCompiler(
        interpreter,
        margin_type=CriticalMarginType.CANDIDATE_TO_CANDIDATE,
        c="A",
        l="B",
        track_diagnostics=True,
    )
    cvr = np.repeat(census[:1], 5, axis=0)
    ballots = cvr.copy()
    ballots[0] = np.array([1, 0, -127], dtype=np.int8)

    result = compiler.evaluate(cvr, ballots)

    assert result.sampled_discrepancies == ((0, 1),)


def test_transition_sufficient_statistics_match_dense_covariance(degree_zero_case):
    _, interpreter, census = degree_zero_case
    compiler = DeltaMethodCompiler(
        interpreter,
        margin_type=CriticalMarginType.CANDIDATE_TO_CANDIDATE,
        c="A",
        l="B",
    )
    cvr = census[:20]
    ballots = cvr.copy()
    ballots[0] = np.array([1, 0, -127], dtype=np.int8)
    ballots[1] = np.array([2, -127, -127], dtype=np.int8)
    projection = DeltaSampleProjection.from_samples(interpreter, cvr, ballots)
    plus, minus = compiler._flat_sample_coordinates(projection)
    coordinate_sum, covariance, count = compiler._transition_statistics(plus, minus)

    dense = np.zeros((len(cvr), 3), dtype=np.float64)
    np.add.at(dense, (np.arange(len(cvr)), plus), 1.0)
    np.add.at(dense, (np.arange(len(cvr)), minus), -1.0)
    np.testing.assert_allclose(coordinate_sum, dense.sum(axis=0))
    np.testing.assert_allclose(covariance, np.cov(dense, rowvar=False, ddof=1))
    assert count == np.count_nonzero(np.any(dense != 0.0, axis=1))


def test_symbolic_derivatives_are_cached_once_per_degree():
    precompute_symbolic_derivatives((0, 1, 2))
    before = symbolic_derivative_cache_info()
    precompute_symbolic_derivatives((0, 1, 2))
    after = symbolic_derivative_cache_info()

    assert after.misses == before.misses
    assert after.hits >= before.hits + 3


@pytest.mark.parametrize("degree", range(4))
def test_numerical_recursive_gradient_matches_symbolic_gradient(degree):
    rng = np.random.default_rng(100 + degree)
    point = rng.uniform(100.0, 500.0, size=(1 << degree, 3))
    quota = 20.0

    numerical_margin, numerical_gradient = (
        delta_method._numerical_recursive_margin_and_gradient(point, quota)
    )
    _, symbolic_evaluator = delta_method._symbolic_margin_functions(degree)
    symbolic_values = symbolic_evaluator(*point.reshape(-1), quota)

    assert numerical_margin == pytest.approx(float(symbolic_values[0]))
    np.testing.assert_allclose(
        numerical_gradient,
        np.asarray(symbolic_values[1:], dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("degree", (4, 5))
def test_high_degree_margin_evaluator_avoids_symbolic_differentiation(
    monkeypatch,
    degree,
):
    delta_method._symbolic_margin_functions.cache_clear()

    def fail(*args, **kwargs):
        raise AssertionError("high-degree evaluator used symbolic derivatives")

    monkeypatch.setattr(sp, "diff", fail)
    monkeypatch.setattr(sp, "lambdify", fail)
    model, evaluator = delta_method._symbolic_margin_functions(degree)
    point = np.full((1 << degree, 3), 1_000.0, dtype=np.float64)

    values = evaluator(*point.reshape(-1), 100.0)

    assert len(model.variables) == point.size
    assert len(values) == point.size + 1
    assert np.all(np.isfinite(np.asarray(values, dtype=np.float64)))


@pytest.mark.parametrize(
    ("margin_type", "candidate", "expected_margin"),
    [
        (CriticalMarginType.CANDIDATE_ABOVE_QUOTA, "A", 9.0),
        (CriticalMarginType.CANDIDATE_BELOW_QUOTA, "B", 21.0),
    ],
)
def test_quota_compilers_treat_quota_as_a_constant(
    degree_zero_case,
    margin_type,
    candidate,
    expected_margin,
):
    _, interpreter, census = degree_zero_case
    compiler = DeltaMethodCompiler(
        interpreter,
        margin_type=margin_type,
        candidate=candidate,
    )
    result = compiler.evaluate(census, census)

    assert compiler.recorded_margin == pytest.approx(expected_margin)
    assert result.estimated_margin == pytest.approx(expected_margin)
    assert result.standard_error == pytest.approx(0.0)
    assert result.lower_bound == pytest.approx(expected_margin)
    assert result.certified
    assert compiler.certify(census, census)


def test_risk_split_validation(degree_zero_case):
    _, interpreter, _ = degree_zero_case
    with pytest.raises(ValueError, match="alpha_K"):
        DeltaMethodCompiler(
            interpreter,
            margin_type=CriticalMarginType.CANDIDATE_TO_CANDIDATE,
            c=0,
            l=1,
            alpha=0.05,
            alpha_K=0.05,
        )


def test_delta_driver_uses_full_risk_budget_for_every_edge():
    graph, census = multi_edge_case()
    driver = DeltaMethodAuditDriver(
        graph,
        BAL=census,
        CVR=census,
        alpha=0.08,
        alpha_K=0.01,
        verbose=False,
    )

    assert len(driver.compilers) > 1
    assert all(compiler.alpha == pytest.approx(0.08) for compiler in driver.compilers)
    assert all(
        compiler.alpha_K == pytest.approx(0.01) for compiler in driver.compilers
    )
    assert driver.run()
    assert len(driver.outcomes) == len(driver.compilers)
    assert all(outcome.certified for outcome in driver.outcomes)
    assert all(compiler.last_result is None for compiler in driver.compilers)
    unique_vertices = {compiler.base_vertex.ref for compiler in driver.compilers}
    assert driver.projection_build_count == len(unique_vertices)


def test_delta_driver_can_stop_after_first_failed_interval():
    graph, census = multi_edge_case()
    driver = DeltaMethodAuditDriver(
        graph,
        BAL=census[:2],
        CVR=census[:2],
        stop_on_failure=True,
        verbose=False,
    )

    assert not driver.run()
    assert len(driver.outcomes) == 1
    assert not driver.outcomes[0].certified


def test_delta_driver_implicit_sampler_is_without_replacement():
    graph, _ = multi_edge_case()
    driver = DeltaMethodAuditDriver(
        graph,
        sample_size=10,
        noise_level=0.1,
        seed=17,
        verbose=False,
    )

    assert driver.sampler is not None
    assert not driver.sampler.with_replacement
    assert len(np.unique(driver.sampler.sampled_serials)) == driver.sample_size


def test_delta_driver_explicitly_defers_black_box_seeded_graphs():
    class BlackBoxSeededGraph:
        used_seeded_build = True
        vertex_post_seed_tallies = {}

    with pytest.raises(NotImplementedError, match="black-box seeded seatings"):
        DeltaMethodAuditDriver(BlackBoxSeededGraph(), sample_size=2)


def test_delta_driver_initializes_batch_seeded_mentions_compilers():
    graph, census = seeded_batch_case()
    driver = DeltaMethodAuditDriver(
        graph,
        BAL=census,
        CVR=census,
        verbose=False,
    )

    mentions = [
        compiler
        for compiler in driver.compilers
        if compiler.margin_type == CriticalMarginType.CANDIDATE_TO_MENTIONS
    ]
    assert len(mentions) == 1
    compiler = mentions[0]
    assert compiler.weak_candidate == 3
    assert compiler.strong_candidates == frozenset({1, 2})
    assert compiler.strong_candidate == 2
    assert compiler.lowest_strong_tally == pytest.approx(250.0)
    assert compiler.recorded_margin == pytest.approx(150.0)
    assert compiler.label == "C0-M3"
    assert driver.lookup_compiler("c0-m3") is compiler
    np.testing.assert_allclose(
        driver.seed_prebatch_strong_tallies,
        [0.0, 300.0, 250.0, 100.0],
    )
    np.testing.assert_allclose(
        driver.seed_maximum_possible_tallies,
        [0.0, 0.0, 0.0, 100.0],
    )

    rows = np.array(
        [
            [2, -127, -127, -127],
            [3, 2, -127, -127],
            [1, 3, 2, -127],
            [-127, -127, -127, -127],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(compiler._mentions_columns(rows), [0, 1, 2, 2])

    assert driver.run()
    outcome = next(
        outcome for outcome in driver.outcomes if outcome.escape_id == "C0-M3"
    )
    assert outcome.estimated_margin == pytest.approx(150.0)
    assert outcome.lower_bound == pytest.approx(150.0)


def test_seeded_mentions_use_actual_prebatch_fpv_with_other_candidates():
    graph, census = seeded_multiweak_batch_case()
    driver = DeltaMethodAuditDriver(
        graph,
        BAL=census,
        CVR=census,
        verbose=False,
    )
    compiler = driver.lookup_compiler("E0-M3")

    rows = np.array(
        [
            [2, 3, -127, -127, -127, -127],
            [0, 2, -127, -127, -127, -127],
            [4, 2, -127, -127, -127, -127],
            [5, 2, -127, -127, -127, -127],
            [3, 4, 2, -127, -127, -127],
            [4, 3, 2, -127, -127, -127],
            [1, 3, 2, -127, -127, -127],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(
        compiler._mentions_columns(rows),
        [0, 0, 2, 2, 1, 1, 2],
    )
    assert compiler.prebatch_seated_candidates == frozenset({0})
    assert compiler.lowest_strong_tally == pytest.approx(250.0)
    assert compiler.recorded_margin == pytest.approx(150.0)

    assert driver.run()
    outcome = next(
        outcome for outcome in driver.outcomes if outcome.escape_id == "E0-M3"
    )
    assert outcome.estimated_margin == pytest.approx(150.0)


def test_delta_driver_verbose_run_prints_layer_and_interval_progress(capsys):
    graph, census = multi_edge_case()
    driver = DeltaMethodAuditDriver(
        graph,
        BAL=census,
        CVR=census,
        verbose=True,
    )

    assert driver.run()
    output = capsys.readouterr().out
    assert "Starting Delta-Method audit:" in output
    assert "Layer 0" in output
    assert "Vertex A0:" in output
    assert "overall 1/" in output
    assert "escape=" in output
    assert "assert " in output
    assert "CI=[" in output
    assert "PASS" in output
    assert "Delta-Method audit:" in output
    assert "Lowest 3 escape-edge interval endpoints:" in output
    ranked = sorted(driver.outcomes, key=lambda outcome: outcome.lower_bound)[:3]
    summary = output.split("Lowest 3 escape-edge interval endpoints:", 1)[1]
    positions = [summary.index(outcome.escape_id) for outcome in ranked]
    assert positions == sorted(positions)


def test_print_diagnostics_selectively_reruns_with_same_seed_and_caches(capsys):
    graph, _ = multi_edge_case()
    driver = DeltaMethodAuditDriver(
        graph,
        sample_size=40,
        noise_level=0.2,
        seed=29,
        verbose=False,
    )
    driver.run()
    identifier = driver.compiler_info[0].escape_id
    original_outcome = driver.outcomes[0]
    compiler = driver.compilers[0]

    assert not compiler.track_diagnostics
    assert driver.diagnostic_results == {}
    result = driver.print_diagnostics(identifier)
    output = capsys.readouterr().out

    assert driver.diagnostic_rerun_count == 1
    assert not compiler.track_diagnostics
    assert result.estimated_margin == pytest.approx(original_outcome.estimated_margin)
    assert result.discrepancy_count == original_outcome.discrepancy_count
    assert result.sampled_discrepancies is not None
    assert len(result.sampled_discrepancies) == result.discrepancy_count
    assert all(
        isinstance(cvr_index, int) and isinstance(paper_index, int)
        for cvr_index, paper_index in result.sampled_discrepancies
    )
    assert f"Delta-Method diagnostics for {identifier}:" in output
    assert "Symbolic margin function:" in output
    assert "Conservative numerical covariance matrix:" in output
    assert "Numerical gradient array:" in output
    assert "Numerical margin variance:" in output
    assert "Standardized sampled discrepancies" in output
    assert "Sample-corrected margin estimate:" in output

    cached = driver.print_diagnostics(identifier)
    capsys.readouterr()
    assert cached is result
    assert driver.diagnostic_rerun_count == 1
