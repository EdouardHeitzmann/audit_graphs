from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import src.replication.driver_statistics as experiment


def test_default_delta_fraction_grid_reaches_one_tenth_percent():
    fractions = experiment._delta_fraction_grid(
        maximum_fraction=0.5,
        candidate_fractions=experiment.DEFAULT_DELTA_FRACTIONS,
        total_ballot_wt=10_000,
    )

    assert fractions[-1] == pytest.approx(0.001)
    assert all(left > right for left, right in zip(fractions, fractions[1:]))


def test_custom_delta_grid_starts_at_fraction_closest_to_one_tenth(
    monkeypatch,
):
    graph = SimpleNamespace(
        _unseeded_root_wt_vec=np.array([10_000.0]),
    )
    attempted = []

    def fake_delta_trials(graph, *, sample_fraction, **kwargs):
        attempted.append(sample_fraction)
        successes = 9 if sample_fraction >= 0.18 else 8
        return successes, int(10_000 * sample_fraction)

    monkeypatch.setattr(experiment, "run_delta_trials", fake_delta_trials)

    result = experiment.find_min_delta_sample_fraction(
        graph,
        maximum_fraction=0.5,
        candidate_fractions=(0.4, 0.18, 0.11, 0.04),
        include_maximum_fraction=False,
        verbose=False,
    )

    assert attempted == [0.11, 0.18]
    assert result.fraction == pytest.approx(0.18)


def test_collect_driver_statistics_runs_noise_and_delta_trials(
    monkeypatch,
    capsys,
):
    profile = SimpleNamespace(
        total_ballot_wt=100,
        candidates=["A", "B", "C"],
    )
    loader_paths = []
    construction_kwargs = []
    noise_calls = []
    delta_calls = []

    class FakeGraph:
        MoI = 7.0
        ballot_matrix = np.array([[0, 1, 2]], dtype=np.int8)
        _unseeded_root_wt_vec = np.array([100.0])
        used_seeded_build = False
        simultaneous = True

        def __init__(self, supplied_profile, **kwargs):
            assert supplied_profile is profile
            construction_kwargs.append(kwargs)

        def build(self):
            pass

        def add_natural_edges(self):
            pass

        def assign_tightest_margins(self):
            pass

        def coherence_check(self):
            return True

    def fake_load_numpy(path):
        loader_paths.append(path)
        return profile, 2, None, None, None

    class FakeNoiseDriver:
        def __init__(self, audit_graph, **kwargs):
            assert isinstance(audit_graph, FakeGraph)
            self.seed = kwargs["seed"]
            self.i = 0
            noise_calls.append(kwargs)

        def run(self, ballot_matrix):
            assert ballot_matrix is FakeGraph.ballot_matrix
            self.i = 10 + self.seed
            return self.seed != 9

    class FakeDeltaDriver:
        def __init__(self, audit_graph, **kwargs):
            assert isinstance(audit_graph, FakeGraph)
            self.fraction = kwargs["fractional_sample_size"]
            self.seed = kwargs["seed"]
            self.sample_size = int(100 * self.fraction)
            delta_calls.append(kwargs)

        def run(self):
            success_limit = 9 if self.fraction >= 0.2 else 8
            return self.seed < success_limit

    monkeypatch.setattr(experiment, "load_numpy", fake_load_numpy)
    monkeypatch.setattr(experiment, "WIGMGraphConstructor", FakeGraph)
    monkeypatch.setattr(experiment, "GlobalAuditDriver", FakeNoiseDriver)
    monkeypatch.setattr(experiment, "DeltaMethodAuditDriver", FakeDeltaDriver)

    returned_graph, statistics = experiment.collect_driver_statistics(
        "election.csv",
        enforced_MoI=7,
        fractional_sample_size=0.5,
        noise_level=0.02,
        candidate_fractions=(0.5, 0.2, 0.15, 0.1, 0.05),
        verbose=True,
    )

    assert "at enforced MoI 7." in capsys.readouterr().out
    assert isinstance(returned_graph, FakeGraph)
    assert loader_paths[0].name == "election.csv"
    assert construction_kwargs == [
        {
            "m": 2,
            "MoI": 7.0,
            "memory_lite": True,
            "simultaneous": True,
        }
    ]
    assert len(noise_calls) == 10
    assert {call["seed"] for call in noise_calls} == set(range(10))
    assert all(call["compiler_type"] == "noise" for call in noise_calls)
    assert all(call["fractional_sample_size"] == 0.5 for call in noise_calls)
    assert all(call["noise_level"] == 0.02 for call in noise_calls)
    assert statistics.noise_sample_sizes == tuple(range(10, 20))
    assert statistics.average_noise_sample_size == pytest.approx(14.5)
    assert statistics.noise_successes == (True,) * 9 + (False,)

    assert len(delta_calls) == 30
    assert [attempt.fraction for attempt in statistics.delta_search.attempts] == [
        0.1,
        0.15,
        0.2,
    ]
    assert statistics.delta_search.fraction == pytest.approx(0.2)
    assert statistics.delta_search.sample_size == 20
    assert statistics.total_ballot_wt == 100
    assert statistics.moi == pytest.approx(7.0)
    assert statistics.candidate_count == 3
    assert statistics.seats == 2


@pytest.mark.parametrize("batch_elim", [False, True])
def test_collect_driver_statistics_can_use_an_enforced_moi(
    monkeypatch,
    batch_elim,
):
    profile = SimpleNamespace(
        total_ballot_wt=100,
        candidates=["A", "B", "C"],
    )
    construction_calls = []
    noise_calls = []

    class FakeWIGMGraphConstructor:
        def __init__(self, supplied_profile, **kwargs):
            assert supplied_profile is profile
            construction_calls.append(("init", kwargs))
            self.MoI = kwargs["MoI"]
            self.simultaneous = kwargs["simultaneous"]
            self.used_seeded_build = False

        def build(self):
            construction_calls.append(("build", None))

        def seeded_build(self):
            construction_calls.append(("seeded_build", None))
            self.simultaneous = True
            self.used_seeded_build = True

        def add_natural_edges(self):
            construction_calls.append(("add_natural_edges", None))

        def assign_tightest_margins(self):
            construction_calls.append(("assign_tightest_margins", None))

        def coherence_check(self):
            construction_calls.append(("coherence_check", None))
            return True

    monkeypatch.setattr(
        experiment,
        "load_numpy",
        lambda path: (profile, 2, None, None, None),
    )
    monkeypatch.setattr(
        experiment,
        "WIGMGraphConstructor",
        FakeWIGMGraphConstructor,
    )
    monkeypatch.setattr(
        experiment,
        "SeededWIGMGraphConstructor",
        FakeWIGMGraphConstructor,
    )
    monkeypatch.setattr(
        experiment,
        "run_noise_trials",
        lambda *args, **kwargs: (
            noise_calls.append((args, kwargs))
            or ((10,) * 10, (True,) * 10)
        ),
    )
    monkeypatch.setattr(
        experiment,
        "find_min_delta_sample_fraction",
        lambda *args, **kwargs: experiment.DeltaFractionSearch(
            fraction=0.1,
            sample_size=10,
            attempts=(),
        ),
    )

    graph, statistics = experiment.collect_driver_statistics(
        "election.csv",
        enforced_MoI=17,
        batch_elim=batch_elim,
        verbose=False,
    )

    assert graph.used_seeded_build is batch_elim
    assert graph.simultaneous is True
    assert noise_calls[0][1]["simultaneous"] is True
    assert statistics.moi == 17.0
    assert construction_calls == [
        (
            "init",
            {
                "m": 2,
                "MoI": 17.0,
                "memory_lite": True,
                "simultaneous": True,
            },
        ),
        ("seeded_build" if batch_elim else "build", None),
        ("add_natural_edges", None),
        ("assign_tightest_margins", None),
        ("coherence_check", None),
    ]


def test_missing_enforced_moi_is_rejected():
    with pytest.raises(ValueError, match="enforced_MoI is required"):
        experiment.collect_driver_statistics(
            "election.csv",
            verbose=False,
        )


def test_skip_mismatch_runs_only_delta_trials(monkeypatch, capsys):
    profile = SimpleNamespace(
        total_ballot_wt=100,
        candidates=["A", "B", "C"],
    )
    class FakeGraph:
        MoI = 7.0
        used_seeded_build = False
        simultaneous = True

        def __init__(self, supplied_profile, **kwargs):
            assert supplied_profile is profile

        def build(self):
            pass

        def add_natural_edges(self):
            pass

        def assign_tightest_margins(self):
            pass

        def coherence_check(self):
            return True

    delta_search = experiment.DeltaFractionSearch(
        fraction=0.1,
        sample_size=10,
        attempts=(),
    )
    delta_calls = []

    monkeypatch.setattr(
        experiment,
        "load_numpy",
        lambda path: pytest.fail("A loaded profile should bypass load_numpy."),
    )
    monkeypatch.setattr(
        experiment,
        "WIGMGraphConstructor",
        FakeGraph,
    )
    monkeypatch.setattr(
        experiment,
        "run_noise_trials",
        lambda *args, **kwargs: pytest.fail("Noise trials should be skipped."),
    )

    def fake_delta_search(*args, **kwargs):
        delta_calls.append((args, kwargs))
        return delta_search

    monkeypatch.setattr(
        experiment,
        "find_min_delta_sample_fraction",
        fake_delta_search,
    )

    _, statistics = experiment.collect_driver_statistics(
        profile,
        m=2,
        enforced_MoI=7,
        skip_mismatch=True,
        delta_sample_sizes=[0.2, 0.08, 0.01],
    )

    assert len(delta_calls) == 1
    assert delta_calls[0][1]["candidate_fractions"] == (0.2, 0.08, 0.01)
    assert delta_calls[0][1]["include_maximum_fraction"] is False
    assert statistics.noise_sample_sizes == ()
    assert statistics.noise_successes == ()
    assert statistics.average_noise_sample_size is None
    assert statistics.delta_search is delta_search
    assert "Skipping mismatch/noise-driver trials." in capsys.readouterr().out

    experiment.print_driver_statistics(
        statistics,
        trials=10,
        success_cutoff=9,
        maximum_fraction=0.5,
    )
    assert "Average noise-driver sample size = skipped" in capsys.readouterr().out


def test_loaded_profile_requires_number_of_seats():
    profile = SimpleNamespace(
        total_ballot_wt=100,
        candidates=["A", "B", "C"],
    )

    with pytest.raises(ValueError, match="m is required"):
        experiment.collect_driver_statistics(
            profile,
            enforced_MoI=17,
            verbose=False,
        )


def test_end_to_end_statistics_tester_prints_summary_and_returns_graph(
    monkeypatch,
    capsys,
):
    graph = object()
    loaded_profile = object()
    collect_calls = []
    statistics = experiment.DriverStatistics(
        total_ballot_wt=1_000,
        moi=42.0,
        candidate_count=8,
        seats=3,
        noise_sample_sizes=(10,) * 10,
        noise_successes=(True,) * 10,
        average_noise_sample_size=10.0,
        delta_search=experiment.DeltaFractionSearch(
            fraction=0.05,
            sample_size=50,
            attempts=(),
        ),
    )
    def fake_collect(*args, **kwargs):
        collect_calls.append((args, kwargs))
        return graph, statistics

    monkeypatch.setattr(experiment, "collect_driver_statistics", fake_collect)

    returned = experiment.end_to_end_statistics_tester(loaded_profile, m=3)
    output = capsys.readouterr().out

    assert returned is graph
    assert collect_calls[0][0] == (loaded_profile,)
    assert collect_calls[0][1]["m"] == 3
    assert "N = 1000" in output
    assert "M = 42" in output
    assert "C = 8" in output
    assert "m = 3" in output
    assert "Average noise-driver sample size (10 trials) = 10.00" in output
    assert "Minimal Delta sample size (9/10 successes) = 50" in output


def test_main_forwards_enforced_moi_and_batch_elim(monkeypatch):
    calls = []
    monkeypatch.setattr(
        experiment,
        "end_to_end_statistics_tester",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = experiment.main(
        [
            "election.csv",
            "--enforced-moi",
            "23",
            "--batch-elim",
            "--skip-mismatch",
            "--delta-sample-sizes",
            "0.2,0.08,0.01",
        ]
    )

    assert result == 0
    assert calls[0][0][0].name == "election.csv"
    assert calls[0][1]["enforced_MoI"] == 23
    assert calls[0][1]["batch_elim"] is True
    assert calls[0][1]["skip_mismatch"] is True
    assert calls[0][1]["delta_sample_sizes"] == (0.2, 0.08, 0.01)
    assert calls[0][1]["simultaneous"] is True
