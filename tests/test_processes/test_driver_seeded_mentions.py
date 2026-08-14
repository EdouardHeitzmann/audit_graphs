from __future__ import annotations

import warnings

import numpy as np

from src.test_processes.cobra import (
    CobraMentionsNoiseFilterCompiler,
    CobraNoiseFilterCompiler,
    CobraQuotaNoiseFilterCompiler,
)
from src.test_processes.driver import GlobalAuditDriver
from src.wigm_graphs.graph_wigm import WIGMGraphConstructor
from src.wigm_graphs.seeded import SeededWIGMGraphConstructor


def test_seeded_mentions_use_post_very_strong_wt_vec_not_unseeded_root():
    # The very-strong surplus transfers at 3/8 to StrongA, so the prebatch
    # tallies discriminate root_wt_vec (post-seed) from _unseeded_root_wt_vec.
    class Profile:
        candidates = ["StrongA", "StrongB", "Weak", "Very"]
        ballot_matrix = np.array(
            [
                [3, 0, 2, -127],
                [0, 2, -127, -127],
                [2, 1, -127, -127],
                [1, 2, -127, -127],
                [-127, -127, -127, -127],
            ],
            dtype=np.int8,
        )
        wt_vec = np.array([96, 10, 10, 30, 90], dtype=np.float64)
        total_ballot_wt = float(wt_vec.sum())

    graph = SeededWIGMGraphConstructor(
        Profile(),
        m=3,
        MoI=10,
        simultaneous=True,
        memory_lite=True,
    )
    graph.seeded_build(
        very_strong_candidates={3},
        strong_candidates={0, 1},
        weak_candidates={2},
    )
    assert graph.quota == 60
    rows = np.array([[3, 0, 2, -127]], dtype=np.int8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        driver = GlobalAuditDriver(
            graph,
            noise_level=0.0,
            BAL=rows,
            CVR=rows,
            print_diagnostics_every=0,
            simultaneous=True,
        )

    # Ballot 0 reaches StrongA at the transfer value (96 - 60) / 96 = 3/8, so
    # its prebatch weight is 36 rather than the unseeded 96.
    assert driver.seed_prebatch_strong_tallies.tolist() == [46.0, 30.0, 10.0, 0.0]
    assert driver.seed_maximum_possible_tallies.tolist() == [0.0, 0.0, 10.0, 0.0]


def test_driver_initializes_from_seeded_graph_with_very_strong_preseed_path():
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
    rows = np.array([[0, 2, 3, -127]], dtype=np.int8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        driver = GlobalAuditDriver(
            graph,
            noise_level=0.0,
            BAL=rows,
            CVR=rows,
            print_diagnostics_every=0,
            simultaneous=True,
        )

    assert len(driver.compilers) > 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        noise_driver = GlobalAuditDriver(
            graph,
            noise_level=0.0,
            BAL=rows,
            CVR=rows,
            compiler_type="noise",
            print_diagnostics_every=0,
            simultaneous=True,
        )

    assert all(
        isinstance(
            compiler,
            (
                CobraNoiseFilterCompiler,
                CobraQuotaNoiseFilterCompiler,
                CobraMentionsNoiseFilterCompiler,
            ),
        )
        for compiler in noise_driver.compilers
    )
    assert any(
        isinstance(compiler, CobraMentionsNoiseFilterCompiler)
        for compiler in noise_driver.compilers
    )
    assert {compiler.MoI for compiler in noise_driver.compilers} == {
        float(graph.MoI)
    }
    assert all(
        compiler.radius == 2.0 * compiler.critical_margin
        for compiler in noise_driver.compilers
    )
    mentions_compiler = next(
        compiler
        for compiler in noise_driver.compilers
        if isinstance(compiler, CobraMentionsNoiseFilterCompiler)
    )
    assert mentions_compiler.critical_margin == (
        mentions_compiler.lowest_strong_tally
        - mentions_compiler.maximum_possible_tallies[mentions_compiler.weak_candidate]
    )

