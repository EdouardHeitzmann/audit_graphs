"""Re-create the ASNs of the writeup's results table (Section 3.4).

Covers the first six rows: the four Scottish local elections and the first two
NSW local elections. Graph construction is hard-coded to the Margin of
Insecurity listed in the table's ``M`` column, and the audits follow the
protocol of Section 3.4: CVRs are noised with 2% mismatches, each audit runs
at risk level alpha = 5%, the Mismatch ASN is the average sample size of 10
audits provided all 10 certify below N/2, and the Delta ASN is the smallest
sample size certifying at least 9 of 10 seeded trials.

Run from the repository root:

    .venv/bin/python -m src.replication.reproduce_table_asns
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .driver_statistics import DriverStatistics, collect_driver_statistics

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True, slots=True)
class TableRow:
    election: str
    profile_path: Path
    moi: int


# The MoI values are the table's M column, except Cumberland Greystanes, which
# uses the true last-round margin of 22 rather than the tabulated 24.
TABLE_ROWS: tuple[TableRow, ...] = (
    TableRow(
        "Sgire Nan Loch 2022",
        DATA_DIR / "scot-elex" / "eilean_siar_2022_ward5.csv",
        141,
    ),
    TableRow(
        "Bearsden North 2017",
        DATA_DIR / "scot-elex" / "east_dunbartonshire_2017_ward2.csv",
        87,
    ),
    TableRow(
        "Ardrossan and Arran 2012",
        DATA_DIR / "scot-elex" / "north_ayrshire_2012_ward5.csv",
        78,
    ),
    TableRow(
        "Saltcoats and Stevenston 2022",
        DATA_DIR / "scot-elex" / "north_ayrshire_2022_saltcoats_and_stevenston.csv",
        148,
    ),
    TableRow(
        "Shellharbour Ward D 2024",
        DATA_DIR / "nsw_local" / "city_of_shellharbour_ward_d_2024.csv",
        1933,
    ),
    TableRow(
        "Cumberland Greystanes 2017",
        DATA_DIR / "nsw_local" / "cumberland_greystanes_ward_2017.csv",
        22,
    ),
)


def mismatch_asn(statistics: DriverStatistics) -> float | None:
    """Average noise-driver sample size, valid only if all trials certified."""
    if not statistics.noise_successes or not all(statistics.noise_successes):
        return None
    return statistics.average_noise_sample_size


def delta_asn(statistics: DriverStatistics) -> int | None:
    return statistics.delta_search.sample_size


def format_asn(value: float | int | None) -> str:
    if value is None:
        return "X"
    return f"{float(value):g}"


def run_rows(
    rows: Sequence[TableRow],
    *,
    noise_level: float = 0.02,
    alpha: float = 0.05,
    trials: int = 10,
    success_cutoff: int = 9,
    fractional_sample_size: float = 0.5,
    base_seed: int = 0,
    verbose: bool = True,
) -> list[tuple[TableRow, DriverStatistics]]:
    results: list[tuple[TableRow, DriverStatistics]] = []
    for row in rows:
        print(f"=== {row.election} (M = {row.moi}) ===")
        _, statistics = collect_driver_statistics(
            row.profile_path,
            enforced_MoI=row.moi,
            noise_level=noise_level,
            alpha=alpha,
            trials=trials,
            success_cutoff=success_cutoff,
            fractional_sample_size=fractional_sample_size,
            base_seed=base_seed,
            verbose=verbose,
        )
        results.append((row, statistics))
        print(
            f"{row.election}: N={statistics.total_ballot_wt}, "
            f"C={statistics.candidate_count}, m={statistics.seats}, "
            f"Mismatch ASN={format_asn(mismatch_asn(statistics))}, "
            f"Delta ASN={format_asn(delta_asn(statistics))}"
        )
    return results


def print_summary(results: Sequence[tuple[TableRow, DriverStatistics]]) -> None:
    print()
    print(
        f"{'Election':<32}{'C':>4}{'m':>4}{'N':>9}{'M':>7}"
        f"{'Mismatch ASN':>14}{'Delta ASN':>11}"
    )
    for row, statistics in results:
        print(
            f"{row.election:<32}"
            f"{statistics.candidate_count:>4}"
            f"{statistics.seats:>4}"
            f"{statistics.total_ballot_wt:>9}"
            f"{row.moi:>7}"
            f"{format_asn(mismatch_asn(statistics)):>14}"
            f"{format_asn(delta_asn(statistics)):>11}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        choices=range(1, len(TABLE_ROWS) + 1),
        help="1-indexed subset of table rows to run (default: all six)",
    )
    parser.add_argument("--noise-level", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--success-cutoff", type=int, default=9)
    parser.add_argument("--fractional-sample-size", type=float, default=0.5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    rows = (
        TABLE_ROWS
        if args.rows is None
        else tuple(TABLE_ROWS[index - 1] for index in args.rows)
    )
    results = run_rows(
        rows,
        noise_level=args.noise_level,
        alpha=args.alpha,
        trials=args.trials,
        success_cutoff=args.success_cutoff,
        fractional_sample_size=args.fractional_sample_size,
        base_seed=args.base_seed,
        verbose=not args.quiet,
    )
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
