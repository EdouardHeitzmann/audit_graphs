"""The Minneapolis rows of the results table remove write-ins but keep their ballots.

The official Minneapolis tabulations compute the election threshold from the
full ballot count, including ballots that only rank write-in candidates (for
Board of Taxation 2025 the official threshold is q = 35,561, i.e.
N = 106,682). The write-in pseudo-candidates ("UWI"/"writein") each condense
many distinct written-in names, so they are removed from the profiles before
graph construction; the emptied ballots stay behind so the total ballot
weight -- and with it the Droop quota -- matches the official tabulation.
"""

from pathlib import Path

import pytest

from src.election_graphs.numpy_profile import (
    load_numpy,
    remove_and_condense_numpy_profile,
)
from src.wigm_graphs.graph_wigm import WIGMGraphConstructor

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "minneapolis"

# (file, write-in pseudo-candidate, C without write-ins, N, m); N is the
# profile weight before removing write-ins, matching the official tabulation.
TABLE_ROWS = [
    ("pb_2025.csv", "UWI", 8, 113_348, 3),
    ("pb_2021.csv", "writein", 7, 106_650, 3),
    ("bot_2025.csv", "UWI", 3, 106_682, 2),
    ("bot_2021.csv", "writein", 4, 95_625, 2),
]


@pytest.mark.parametrize(
    ("filename", "write_in", "candidates", "total_weight", "seats"),
    TABLE_ROWS,
)
def test_removing_write_ins_keeps_the_official_ballot_weight(
    filename,
    write_in,
    candidates,
    total_weight,
    seats,
):
    profile, loaded_seats, *_ = load_numpy(DATA_DIR / filename)

    cleaned = remove_and_condense_numpy_profile(
        profile, [write_in], remove_empty_ballots=False
    )

    assert loaded_seats == seats
    assert write_in not in cleaned.candidates
    assert len(cleaned.candidates) == candidates
    assert round(profile.total_ballot_wt) == total_weight
    assert round(cleaned.total_ballot_wt) == total_weight
    assert len(cleaned.wt_vec) == len(profile.wt_vec)


def test_bot_2025_quota_matches_the_official_threshold():
    profile, m, *_ = load_numpy(DATA_DIR / "bot_2025.csv")

    cleaned = remove_and_condense_numpy_profile(
        profile, ["UWI"], remove_empty_ballots=False
    )
    graph = WIGMGraphConstructor(
        cleaned, m=int(m), MoI=1.0, simultaneous=True, memory_lite=True
    )

    assert graph.quota == 35_561


