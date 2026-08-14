"""Re-create the ASNs of every row of the writeup's results table (Section 3.4).

Each row runs at the table's ``M`` column value with the table's graph
construction method: rows footnoted \\textsuperscript{1} are built with batch
elimination (``seeded_build``), all others with the plain plausible DFS. The
audits follow the protocol of Section 3.4: CVRs are noised with 2% mismatches,
each audit runs at risk level alpha = 5%, the Mismatch ASN is the average
sample size of 10 audits provided all 10 certify below N/2, and the Delta ASN
is the smallest sample size certifying at least 9 of 10 seeded trials.

Rows whose Mismatch ASN is an X in the table skip the mismatch audit here too
(it is time-consuming and known not to certify). The M values are the maximal
coherent Margins of Insecurity under the seat-scarcity seating rule, as found
by ``heap_based_search`` (seven rows are therefore smaller than the M column
of earlier drafts; Wollongong Ward 3 audits at M = 500 per the table's
footnote 2). The table's empty Queensland and Western Australia rows are
omitted.

The Minneapolis profiles remove their write-in pseudo-candidates before graph
construction while keeping the emptied ballots, so the total ballot weight and
Droop quota match the official tabulations (see
tests/replication/test_minneapolis_preprocessing.py).

Run from the repository root:

    .venv/bin/python -m src.replication.reproduce_full_table_asns
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..election_graphs.numpy_profile import (
    load_numpy,
    remove_and_condense_numpy_profile,
)
from .driver_statistics import DriverStatistics, collect_driver_statistics
from .reproduce_table_asns import delta_asn, format_asn, mismatch_asn

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True, slots=True)
class TableRow:
    election: str
    profile_path: Path
    moi: int
    batch_elim: bool = False
    skip_mismatch: bool = False
    # Write-in pseudo-candidate to remove before graph construction.
    # Minneapolis keeps the emptied ballots, because its official thresholds
    # are computed from the full ballot weight; Albany drops them, because
    # Alameda County pre-exhausts write-ins before round one.
    write_in: str | None = None
    keep_empty_ballots: bool = True

    def load(self) -> tuple[object, int]:
        """Return the (profile, m) pair to audit for this row."""
        profile, m, *_ = load_numpy(self.profile_path)
        if self.write_in is not None:
            profile = remove_and_condense_numpy_profile(
                profile,
                [self.write_in],
                remove_empty_ballots=not self.keep_empty_ballots,
            )
        return profile, int(m)


TABLE_ROWS: tuple[TableRow, ...] = (
    # Scottish Local Elections.
    TableRow(
        "Sgire Nan Loch 2022",
        DATA_DIR / "scot-elex" / "eilean_siar_2022_ward5.csv",
        141,
    ),
    TableRow(
        "Bearsden North 2017",
        DATA_DIR / "scot-elex" / "east_dunbartonshire_2017_ward2.csv",
        87,
        skip_mismatch=True,
    ),
    TableRow(
        "Ardrossan and Arran 2012",
        DATA_DIR / "scot-elex" / "north_ayrshire_2012_ward5.csv",
        75,
        skip_mismatch=True,
    ),
    TableRow(
        "Saltcoats and Stevenston 2022",
        DATA_DIR / "scot-elex" / "north_ayrshire_2022_saltcoats_and_stevenston.csv",
        148,
        skip_mismatch=True,
    ),
    # NSW Local Elections.
    TableRow(
        "Shellharbour Ward D 2024",
        DATA_DIR / "nsw_local" / "city_of_shellharbour_ward_d_2024.csv",
        1933,
    ),
    TableRow(
        "Cumberland Greystanes 2017",
        DATA_DIR / "nsw_local" / "cumberland_greystanes_ward_2017.csv",
        21,
        skip_mismatch=True,
    ),
    TableRow(
        "Wollongong Ward 3 2021",
        DATA_DIR / "nsw_local" / "city_of_wollongong_ward_3_2021.csv",
        500,
        skip_mismatch=True,
    ),
    TableRow(
        "Penrith South 2024",
        DATA_DIR / "nsw_local" / "city_of_penrith_south_ward_2024.csv",
        600,
        skip_mismatch=True,
    ),
    # Albany, CA City Council. Write-ins are removed and their emptied
    # ballots dropped, matching Alameda County's pre-exhaustion of write-in
    # candidates before round one.
    TableRow(
        "Albany City Council 2022",
        DATA_DIR / "albany" / "albany_council_2022.csv",
        129,
        skip_mismatch=True,
        write_in="write-in",
        keep_empty_ballots=False,
    ),
    TableRow(
        "Albany City Council 2024",
        DATA_DIR / "albany" / "albany_council_2024.csv",
        192,
        skip_mismatch=True,
        write_in="write-in",
        keep_empty_ballots=False,
    ),
    # Portland, OR City Council.
    TableRow(
        "Portland District 1 2024",
        DATA_DIR / "portland" / "portland_d1_2024.csv",
        671,
        skip_mismatch=True,
    ),
    TableRow(
        "Portland District 2 2024",
        DATA_DIR / "portland" / "portland_d2_2024.csv",
        2739,
        batch_elim=True,
    ),
    TableRow(
        "Portland District 3 2024",
        DATA_DIR / "portland" / "portland_d3_2024.csv",
        300,
        skip_mismatch=True,
    ),
    TableRow(
        "Portland District 4 2024",
        DATA_DIR / "portland" / "portland_d4_2024.csv",
        200,
        skip_mismatch=True,
    ),
    # Minneapolis, MN. Write-in pseudo-candidates are removed while keeping
    # their ballots, so N and the quota match the official tabulations.
    TableRow(
        "Minneapolis Parks Board 2025",
        DATA_DIR / "minneapolis" / "pb_2025.csv",
        3209,
        skip_mismatch=True,
        write_in="UWI",
    ),
    TableRow(
        "Minneapolis Parks Board 2021",
        DATA_DIR / "minneapolis" / "pb_2021.csv",
        1111,
        skip_mismatch=True,
        write_in="writein",
    ),
    TableRow(
        "Minneapolis Board of Taxation 2025",
        DATA_DIR / "minneapolis" / "bot_2025.csv",
        8467,
        write_in="UWI",
    ),
    TableRow(
        "Minneapolis Board of Taxation 2021",
        DATA_DIR / "minneapolis" / "bot_2021.csv",
        5406,
        write_in="writein",
    ),
    # Australian Senate.
    TableRow(
        "Australian Senate ACT 2025",
        DATA_DIR / "australia_federal" / "act_2025_votekit.csv",
        15000,
    ),
    TableRow(
        "Australian Senate NT 2025",
        DATA_DIR / "australia_federal" / "nt_2025_votekit.csv",
        1500,
    ),
    TableRow(
        "Australian Senate Victoria 2025",
        DATA_DIR / "australia_federal" / "vic_2025_votekit.csv",
        20000,
        batch_elim=True,
        skip_mismatch=True,
    ),
    TableRow(
        "Australian Senate NSW 2025",
        DATA_DIR / "australia_federal" / "nsw_2025_votekit.csv",
        12000,
        batch_elim=True,
        skip_mismatch=True,
    ),
    TableRow(
        "Australian Senate Tasmania 2025",
        DATA_DIR / "australia_federal" / "tas_2025_votekit.csv",
        5000,
        batch_elim=True,
    ),
    TableRow(
        "Australian Senate SA 2025",
        DATA_DIR / "australia_federal" / "sa_2025_votekit.csv",
        7000,
        batch_elim=True,
        skip_mismatch=True,
    ),
    TableRow(
        "Australian Senate QLD 2025",
        DATA_DIR / "australia_federal" / "qld_2025_votekit.csv",
        30000,
        batch_elim=True,
        skip_mismatch=True,
    ),
    TableRow(
        "Australian Senate WA 2025",
        DATA_DIR / "australia_federal" / "wa_2025_votekit.csv",
        2850,
        batch_elim=True,
        skip_mismatch=True,
    ),
)


@dataclass(frozen=True, slots=True)
class RowResult:
    row: TableRow
    statistics: DriverStatistics | None
    error: str | None = None

    @property
    def mismatch_cell(self) -> str:
        if self.statistics is None:
            return "FAILED"
        if self.error is not None:
            return "ERROR"
        if self.row.skip_mismatch:
            return "skipped"
        return format_asn(mismatch_asn(self.statistics))

    @property
    def delta_cell(self) -> str:
        if self.statistics is None:
            return "FAILED"
        return format_asn(delta_asn(self.statistics))


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
) -> list[RowResult]:
    results: list[RowResult] = []
    for row in rows:
        build = "batch elimination" if row.batch_elim else "plain DFS"
        print(f"=== {row.election} (M = {row.moi}, {build}) ===", flush=True)

        try:
            profile, m = row.load()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"{row.election}: FAILED to load ({error})", flush=True)
            results.append(RowResult(row, None, error))
            continue

        def collect(skip_mismatch: bool) -> DriverStatistics:
            _, statistics = collect_driver_statistics(
                profile,
                m=m,
                enforced_MoI=row.moi,
                batch_elim=row.batch_elim,
                skip_mismatch=skip_mismatch,
                noise_level=noise_level,
                alpha=alpha,
                trials=trials,
                success_cutoff=success_cutoff,
                fractional_sample_size=fractional_sample_size,
                base_seed=base_seed,
                verbose=verbose,
            )
            return statistics

        error: str | None = None
        statistics: DriverStatistics | None = None
        try:
            statistics = collect(row.skip_mismatch)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"{row.election}: FAILED ({error})", flush=True)
            if not row.skip_mismatch:
                print(
                    f"{row.election}: retrying with the mismatch audit skipped.",
                    flush=True,
                )
                try:
                    statistics = collect(True)
                except Exception as retry_exc:
                    print(
                        f"{row.election}: retry FAILED "
                        f"({type(retry_exc).__name__}: {retry_exc})",
                        flush=True,
                    )

        results.append(RowResult(row, statistics, error))
        if statistics is not None:
            result = results[-1]
            print(
                f"{row.election}: N={statistics.total_ballot_wt}, "
                f"C={statistics.candidate_count}, m={statistics.seats}, "
                f"Mismatch ASN={result.mismatch_cell}, "
                f"Delta ASN={result.delta_cell}",
                flush=True,
            )
    return results


def print_summary(results: Sequence[RowResult]) -> None:
    print()
    print(
        f"{'Election':<36}{'C':>4}{'m':>4}{'N':>10}{'M':>7}"
        f"{'Mismatch ASN':>14}{'Delta ASN':>11}"
    )
    for result in results:
        statistics = result.statistics
        if statistics is None:
            print(f"{result.row.election:<36}  FAILED: {result.error}")
            continue
        print(
            f"{result.row.election:<36}"
            f"{statistics.candidate_count:>4}"
            f"{statistics.seats:>4}"
            f"{statistics.total_ballot_wt:>10}"
            f"{result.row.moi:>7}"
            f"{result.mismatch_cell:>14}"
            f"{result.delta_cell:>11}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        choices=range(1, len(TABLE_ROWS) + 1),
        help="1-indexed subset of table rows to run (default: all)",
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
