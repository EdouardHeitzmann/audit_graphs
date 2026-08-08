"""End-to-end sample-size statistics for the noise and Delta audit drivers."""

from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
import io
from numbers import Integral
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from ..election_graphs.numpy_profile import load_numpy

from ..margin_search import heap_based_search
from ..test_processes.delta_method import DeltaMethodAuditDriver
from ..test_processes.driver import GlobalAuditDriver
from ..wigm_graphs.graph_wigm import WIGMGraphConstructor
from ..wigm_graphs.seeded import SeededWIGMGraphConstructor


DEFAULT_DELTA_FRACTIONS: tuple[float, ...] = (
    1 / 2,
    1 / 3,
    1 / 5,
    15 / 100,
    1 / 10,
    1 / 15,
    1 / 20,
    1 / 25,
    1 / 30,
    1 / 40,
    1 / 50,
    1 / 60,
    1 / 75,
    1 / 100,
    1 / 125,
    1 / 150,
    1 / 200,
    1 / 250,
    1 / 300,
    1 / 400,
    1 / 500,
    1 / 600,
    1 / 800,
    1 / 1000,
)


@dataclass(frozen=True, slots=True)
class DeltaFractionAttempt:
    fraction: float
    sample_size: int
    successes: int
    trials: int


@dataclass(frozen=True, slots=True)
class DeltaFractionSearch:
    fraction: float | None
    sample_size: int | None
    attempts: tuple[DeltaFractionAttempt, ...]


@dataclass(frozen=True, slots=True)
class DriverStatistics:
    total_ballot_wt: int
    optimal_lam: float
    candidate_count: int
    seats: int
    noise_sample_sizes: tuple[int, ...]
    noise_successes: tuple[bool, ...]
    average_noise_sample_size: float | None
    delta_search: DeltaFractionSearch


def _validate_experiment_parameters(
    *,
    fractional_sample_size: float,
    noise_level: float,
    trials: int,
    success_cutoff: int,
) -> None:
    if not 0.0 < fractional_sample_size < 1.0:
        raise ValueError("fractional_sample_size must lie strictly between 0 and 1.")
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must lie in [0, 1].")
    if trials <= 0:
        raise ValueError("trials must be positive.")
    if not 1 <= success_cutoff <= trials:
        raise ValueError("success_cutoff must lie between 1 and trials.")


def _driver_output_context(suppress_driver_output: bool):
    if not suppress_driver_output:
        return nullcontext()
    return redirect_stdout(io.StringIO())


def run_noise_trials(
    graph: Any,
    *,
    fractional_sample_size: float = 0.5,
    noise_level: float = 0.02,
    seeds: Sequence[int] = tuple(range(10)),
    alpha: float = 0.05,
    simultaneous: bool | None = None,
    suppress_driver_output: bool = True,
    verbose: bool = True,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Run the noise-filter driver and return final sample sizes and outcomes."""
    if not seeds:
        raise ValueError("seeds cannot be empty.")
    if len(set(map(int, seeds))) != len(seeds):
        raise ValueError("Noise trials require distinct seeds.")
    simultaneous = bool(
        getattr(graph, "simultaneous", False)
        if simultaneous is None
        else simultaneous
    )

    sample_sizes: list[int] = []
    successes: list[bool] = []
    for trial, seed in enumerate(seeds, start=1):
        with _driver_output_context(suppress_driver_output):
            driver = GlobalAuditDriver(
                graph,
                noise_level=float(noise_level),
                fractional_sample_size=float(fractional_sample_size),
                seed=int(seed),
                compiler_type="noise",
                print_diagnostics_every=0,
                simultaneous=simultaneous,
                alpha=float(alpha),
            )
            success = bool(driver.run(graph.ballot_matrix))
        sample_sizes.append(int(driver.i))
        successes.append(success)
        if verbose:
            status = "PASS" if success else "CAP REACHED"
            print(
                f"Noise trial {trial}/{len(seeds)} (seed={seed}): "
                f"{status} after {driver.i} samples."
            )

    return tuple(sample_sizes), tuple(successes)


def run_delta_trials(
    graph: Any,
    *,
    sample_fraction: float,
    noise_level: float = 0.02,
    seeds: Sequence[int] = tuple(range(10)),
    alpha: float = 0.05,
    alpha_K: float | None = None,
    suppress_driver_output: bool = True,
) -> tuple[int, int]:
    """Return ``(successes, sample_size)`` for one Delta sample fraction."""
    if not seeds:
        raise ValueError("seeds cannot be empty.")

    successes = 0
    resolved_sample_size: int | None = None
    for seed in seeds:
        with _driver_output_context(suppress_driver_output):
            driver = DeltaMethodAuditDriver(
                graph,
                fractional_sample_size=float(sample_fraction),
                noise_level=float(noise_level),
                seed=int(seed),
                alpha=float(alpha),
                alpha_K=alpha_K,
                verbose=False,
            )
            success = bool(driver.run())
        if resolved_sample_size is None:
            resolved_sample_size = int(driver.sample_size)
        elif resolved_sample_size != int(driver.sample_size):
            raise RuntimeError("Delta trials resolved inconsistent sample sizes.")
        successes += success

    assert resolved_sample_size is not None
    return int(successes), resolved_sample_size


def _delta_fraction_grid(
    maximum_fraction: float,
    candidate_fractions: Sequence[float],
    total_ballot_wt: int,
    *,
    include_maximum_fraction: bool = True,
) -> tuple[float, ...]:
    fractions = [float(maximum_fraction)] if include_maximum_fraction else []
    fractions.extend(
        float(fraction)
        for fraction in candidate_fractions
        if 0.0 < float(fraction) <= maximum_fraction
    )
    unique = sorted(set(fractions), reverse=True)
    usable = tuple(
        fraction
        for fraction in unique
        if fraction < 1.0 and int(total_ballot_wt * fraction) >= 2
    )
    if not usable:
        raise ValueError("No Delta candidate fraction produces at least two samples.")
    return usable


def find_min_delta_sample_fraction(
    graph: Any,
    *,
    maximum_fraction: float = 0.5,
    candidate_fractions: Sequence[float] = DEFAULT_DELTA_FRACTIONS,
    include_maximum_fraction: bool = True,
    noise_level: float = 0.02,
    seeds: Sequence[int] = tuple(range(10)),
    success_cutoff: int = 9,
    alpha: float = 0.05,
    alpha_K: float | None = None,
    suppress_driver_output: bool = True,
    verbose: bool = True,
) -> DeltaFractionSearch:
    """Search the notebook fraction grid for a 9-of-10 Delta success rate."""
    trials = len(seeds)
    if not 1 <= success_cutoff <= trials:
        raise ValueError("success_cutoff must lie between 1 and len(seeds).")
    total_ballot_wt = int(round(float(graph._unseeded_root_wt_vec.sum())))
    fractions = _delta_fraction_grid(
        float(maximum_fraction),
        candidate_fractions,
        total_ballot_wt,
        include_maximum_fraction=include_maximum_fraction,
    )
    start_index = min(
        range(len(fractions)),
        key=lambda index: abs(fractions[index] - 0.10),
    )
    attempts: list[DeltaFractionAttempt] = []

    def audit_at(index: int) -> DeltaFractionAttempt:
        fraction = fractions[index]
        successes, sample_size = run_delta_trials(
            graph,
            sample_fraction=fraction,
            noise_level=noise_level,
            seeds=seeds,
            alpha=alpha,
            alpha_K=alpha_K,
            suppress_driver_output=suppress_driver_output,
        )
        attempt = DeltaFractionAttempt(
            fraction=fraction,
            sample_size=sample_size,
            successes=successes,
            trials=trials,
        )
        attempts.append(attempt)
        if verbose:
            print(
                f"Delta fraction {fraction:.6g} ({sample_size} ballots): "
                f"{successes}/{trials} runs passed."
            )
        return attempt

    first = audit_at(start_index)
    best: DeltaFractionAttempt | None = None
    if first.successes >= success_cutoff:
        best = first
        for index in range(start_index + 1, len(fractions)):
            attempt = audit_at(index)
            if attempt.successes < success_cutoff:
                break
            best = attempt
    else:
        for index in range(start_index - 1, -1, -1):
            attempt = audit_at(index)
            if attempt.successes >= success_cutoff:
                best = attempt
                break

    return DeltaFractionSearch(
        fraction=None if best is None else best.fraction,
        sample_size=None if best is None else best.sample_size,
        attempts=tuple(attempts),
    )


def collect_driver_statistics(
    profile_path: str | Path | Any,
    m: int | None = None,
    *,
    fractional_sample_size: float = 0.5,
    noise_level: float = 0.02,
    trials: int = 10,
    success_cutoff: int = 9,
    base_seed: int = 0,
    candidate_fractions: Sequence[float] = DEFAULT_DELTA_FRACTIONS,
    delta_sample_sizes: Sequence[float] | None = None,
    alpha: float = 0.05,
    alpha_K: float | None = None,
    memory_lite: bool = True,
    simultaneous: bool = True,
    verify_output: bool = False,
    enforced_LAM: int | None = None,
    batch_elim: bool = False,
    skip_mismatch: bool = False,
    suppress_driver_output: bool = True,
    verbose: bool = True,
) -> tuple[WIGMGraphConstructor, DriverStatistics]:
    """Load an election, construct its graph, and run both driver studies."""
    _validate_experiment_parameters(
        fractional_sample_size=fractional_sample_size,
        noise_level=noise_level,
        trials=trials,
        success_cutoff=success_cutoff,
    )
    if enforced_LAM is not None and (
        isinstance(enforced_LAM, bool) or not isinstance(enforced_LAM, int)
    ):
        raise TypeError("enforced_LAM must be an int or None.")
    if enforced_LAM is not None and enforced_LAM < 0:
        raise ValueError("enforced_LAM must be non-negative.")
    if batch_elim and enforced_LAM is None:
        raise ValueError("batch_elim=True requires an enforced_LAM.")
    if delta_sample_sizes is not None:
        delta_sample_sizes = tuple(float(value) for value in delta_sample_sizes)
        if not delta_sample_sizes:
            raise ValueError("delta_sample_sizes cannot be empty.")
        if any(not 0.0 < value < 1.0 for value in delta_sample_sizes):
            raise ValueError("Every delta_sample_sizes value must lie in (0, 1).")
    if m is not None:
        if isinstance(m, bool) or not isinstance(m, Integral):
            raise TypeError("m must be an int.")
        if int(m) <= 0:
            raise ValueError("m must be positive.")

    if isinstance(profile_path, (str, Path)):
        path = Path(profile_path)
        pf, loaded_m, _, _, _ = load_numpy(path)
        loaded_m = int(loaded_m)
        if m is not None and int(m) != loaded_m:
            raise ValueError(
                f"Explicit m={m} does not match the profile file's m={loaded_m}."
            )
        m = loaded_m
        source_description = str(path)
    else:
        pf = profile_path
        if m is None:
            raise ValueError("m is required when passing a loaded profile.")
        source_description = "the supplied loaded profile"

    m = int(m)
    seeds = tuple(range(int(base_seed), int(base_seed) + int(trials)))
    if verbose:
        if enforced_LAM is None:
            print(
                f"Loading {source_description} and searching for the optimal unseeded "
                "WIGM LAM."
            )
        else:
            build_kind = "batch-elimination seeded" if batch_elim else "unseeded"
            print(
                f"Loading {source_description} and constructing a {build_kind} "
                "WIGM graph "
                f"at enforced LAM {enforced_LAM}."
            )
    with _driver_output_context(suppress_driver_output):
        if enforced_LAM is None:
            graph = heap_based_search(
                profile=pf,
                m=m,
                memory_lite=bool(memory_lite),
                constructor_cls=WIGMGraphConstructor,
                simultaneous=bool(simultaneous),
                verify_output=bool(verify_output),
            )
        else:
            constructor_cls = (
                SeededWIGMGraphConstructor if batch_elim else WIGMGraphConstructor
            )
            graph = constructor_cls(
                pf,
                m=m,
                LAM=float(enforced_LAM),
                memory_lite=bool(memory_lite),
                simultaneous=bool(simultaneous),
            )
            if batch_elim:
                graph.seeded_build()
            else:
                graph.build()
            graph.add_natural_edges()
            graph.assign_tightest_margins()
            graph.coherence_check()

    seeded_graph = bool(getattr(graph, "used_seeded_build", False))
    if seeded_graph != bool(batch_elim):
        raise RuntimeError(
            "Statistics experiment graph seeding state does not match batch_elim."
        )
    graph_simultaneous = bool(getattr(graph, "simultaneous", simultaneous))
    if verbose and enforced_LAM is None:
        print(f"Optimal LAM found: M = {float(graph.LAM):g}.")

    if skip_mismatch:
        noise_sample_sizes: tuple[int, ...] = ()
        noise_successes: tuple[bool, ...] = ()
        average_noise_sample_size = None
        if verbose:
            print("Skipping mismatch/noise-driver trials.")
    else:
        noise_sample_sizes, noise_successes = run_noise_trials(
            graph,
            fractional_sample_size=fractional_sample_size,
            noise_level=noise_level,
            seeds=seeds,
            alpha=alpha,
            simultaneous=graph_simultaneous,
            suppress_driver_output=suppress_driver_output,
            verbose=verbose,
        )
        average_noise_sample_size = float(fmean(noise_sample_sizes))
    delta_search = find_min_delta_sample_fraction(
        graph,
        maximum_fraction=fractional_sample_size,
        candidate_fractions=(
            candidate_fractions
            if delta_sample_sizes is None
            else delta_sample_sizes
        ),
        include_maximum_fraction=delta_sample_sizes is None,
        noise_level=noise_level,
        seeds=seeds,
        success_cutoff=success_cutoff,
        alpha=alpha,
        alpha_K=alpha_K,
        suppress_driver_output=suppress_driver_output,
        verbose=verbose,
    )
    statistics = DriverStatistics(
        total_ballot_wt=int(round(float(pf.total_ballot_wt))),
        optimal_lam=float(graph.LAM),
        candidate_count=len(pf.candidates),
        seats=m,
        noise_sample_sizes=noise_sample_sizes,
        noise_successes=noise_successes,
        average_noise_sample_size=average_noise_sample_size,
        delta_search=delta_search,
    )
    return graph, statistics


def print_driver_statistics(
    statistics: DriverStatistics,
    *,
    trials: int,
    success_cutoff: int,
    maximum_fraction: float,
) -> None:
    print("End-to-end audit statistics:")
    print(f"  N = {statistics.total_ballot_wt}")
    print(f"  M = {statistics.optimal_lam:g}")
    print(f"  C = {statistics.candidate_count}")
    print(f"  m = {statistics.seats}")
    if statistics.average_noise_sample_size is None:
        print("  Average noise-driver sample size = skipped")
    else:
        print(
            f"  Average noise-driver sample size ({trials} trials) = "
            f"{statistics.average_noise_sample_size:.2f}"
        )
    if statistics.delta_search.sample_size is None:
        print(
            "  Minimal Delta sample size "
            f"({success_cutoff}/{trials} successes) = not found at or below "
            f"fraction {maximum_fraction:g}"
        )
    else:
        print(
            "  Minimal Delta sample size "
            f"({success_cutoff}/{trials} successes) = "
            f"{statistics.delta_search.sample_size} "
            f"(fraction {statistics.delta_search.fraction:.6g})"
        )


def end_to_end_statistics_tester(
    profile_path: str | Path | Any,
    m: int | None = None,
    *,
    fractional_sample_size: float = 0.5,
    noise_level: float = 0.02,
    trials: int = 10,
    success_cutoff: int = 9,
    base_seed: int = 0,
    candidate_fractions: Sequence[float] = DEFAULT_DELTA_FRACTIONS,
    delta_sample_sizes: Sequence[float] | None = None,
    alpha: float = 0.05,
    alpha_K: float | None = None,
    memory_lite: bool = True,
    simultaneous: bool = True,
    verify_output: bool = False,
    enforced_LAM: int | None = None,
    batch_elim: bool = False,
    skip_mismatch: bool = False,
    suppress_driver_output: bool = True,
    verbose: bool = True,
) -> WIGMGraphConstructor:
    """Run the paper statistics experiment, print its summary, and return the graph."""
    graph, statistics = collect_driver_statistics(
        profile_path,
        m=m,
        fractional_sample_size=fractional_sample_size,
        noise_level=noise_level,
        trials=trials,
        success_cutoff=success_cutoff,
        base_seed=base_seed,
        candidate_fractions=candidate_fractions,
        delta_sample_sizes=delta_sample_sizes,
        alpha=alpha,
        alpha_K=alpha_K,
        memory_lite=memory_lite,
        simultaneous=simultaneous,
        verify_output=verify_output,
        enforced_LAM=enforced_LAM,
        batch_elim=batch_elim,
        skip_mismatch=skip_mismatch,
        suppress_driver_output=suppress_driver_output,
        verbose=verbose,
    )
    printed_maximum_fraction = fractional_sample_size
    if delta_sample_sizes is not None:
        usable_custom_fractions = [
            float(value)
            for value in delta_sample_sizes
            if 0.0 < float(value) <= fractional_sample_size
        ]
        if usable_custom_fractions:
            printed_maximum_fraction = max(usable_custom_fractions)
    print_driver_statistics(
        statistics,
        trials=trials,
        success_cutoff=success_cutoff,
        maximum_fraction=printed_maximum_fraction,
    )
    return graph


def _parse_fraction_grid(value: str) -> tuple[float, ...]:
    try:
        fractions = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Delta fractions must be a comma-separated list of numbers."
        ) from exc
    if not fractions or any(not 0.0 < fraction < 1.0 for fraction in fractions):
        raise argparse.ArgumentTypeError("Every Delta fraction must lie in (0, 1).")
    return fractions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_path", type=Path)
    parser.add_argument("--fractional-sample-size", type=float, default=0.5)
    parser.add_argument("--noise-level", type=float, default=0.02)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--success-cutoff", type=int, default=9)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha-k", type=float)
    parser.add_argument(
        "--delta-fractions",
        type=_parse_fraction_grid,
        default=DEFAULT_DELTA_FRACTIONS,
    )
    parser.add_argument("--delta-sample-sizes", type=_parse_fraction_grid)
    parser.add_argument(
        "--simultaneous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="construct a simultaneous graph (default: enabled)",
    )
    parser.add_argument("--verify-output", action="store_true")
    parser.add_argument("--enforced-lam", type=int)
    parser.add_argument("--batch-elim", action="store_true")
    parser.add_argument("--skip-mismatch", action="store_true")
    parser.add_argument("--show-driver-output", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    end_to_end_statistics_tester(
        args.profile_path,
        fractional_sample_size=args.fractional_sample_size,
        noise_level=args.noise_level,
        trials=args.trials,
        success_cutoff=args.success_cutoff,
        base_seed=args.base_seed,
        candidate_fractions=args.delta_fractions,
        delta_sample_sizes=args.delta_sample_sizes,
        alpha=args.alpha,
        alpha_K=args.alpha_k,
        simultaneous=args.simultaneous,
        verify_output=args.verify_output,
        enforced_LAM=args.enforced_lam,
        batch_elim=args.batch_elim,
        skip_mismatch=args.skip_mismatch,
        suppress_driver_output=not args.show_driver_output,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
