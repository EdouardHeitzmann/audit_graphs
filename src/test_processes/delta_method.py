from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from .cobra import CriticalMarginType
from ..election_graphs.datatypes import EdgeAction, ElectionStatus
from ..election_graphs.utils import (
    fpv_tallies_from_matrix,
    frozen_mentions_from_matrix,
)
from .interpreter import COORDINATE_COLUMNS, VertexInterpreter
from .noise import ImplicitSampler

if TYPE_CHECKING:
    from noise_filtered_linearizer import SymbolicMarginModel


def log_comb(n: int, k: int) -> float:
    """Return ``log(binomial(n, k))``, with impossible choices mapped to -inf."""
    n = int(n)
    k = int(k)
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_log_cdf_leq(N: int, K: int, n: int, t: int) -> float:
    """Compute ``log(P(T <= t))`` for ``T ~ Hypergeom(N, K, n)``.

    This is the stable log-sum-exp version of the legacy implementation in
    ``old_src/edouard/audit_machinery.py``.
    """
    N, K, n, t = map(int, (N, K, n, t))
    if N < 0 or not 0 <= K <= N or not 0 <= n <= N:
        raise ValueError("Need 0 <= K <= N and 0 <= n <= N.")

    t_min = max(0, n - (N - K))
    t_max = min(n, K)
    upper = min(t, t_max)
    if upper < t_min:
        return float("-inf")
    if t >= t_max:
        return 0.0

    log_terms = [
        log_comb(K, j) + log_comb(N - K, n - j)
        for j in range(t_min, upper + 1)
    ]
    maximum = max(log_terms)
    return (
        maximum
        + math.log(math.fsum(math.exp(value - maximum) for value in log_terms))
        - log_comb(N, n)
    )


@lru_cache(maxsize=None)
def alternative_K_upper(
    N: int,
    n: int,
    t: int,
    alpha: float = 0.05,
) -> int:
    """One-sided upper confidence bound on population discrepancy count ``K``.

    ``t`` discrepancies are observed in a simple random sample without
    replacement of size ``n`` from a population of size ``N``. The returned
    value is the largest feasible ``K`` for which ``P(T <= t | K) >= alpha``.
    """
    N, n, t = map(int, (N, n, t))
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    if not 1 <= n <= N:
        raise ValueError("Need 1 <= n <= N.")
    if not 0 <= t <= n:
        raise ValueError(f"Need 0 <= t <= n, got t={t}, n={n}.")

    log_alpha = math.log(alpha)
    lo = t
    hi = N - (n - t)
    best = t
    while lo <= hi:
        midpoint = (lo + hi) // 2
        if hypergeom_log_cdf_leq(N, midpoint, n, t) >= log_alpha:
            best = midpoint
            lo = midpoint + 1
        else:
            hi = midpoint - 1
    return best


def K_upper(N: int, n: int, alpha: float = 0.05) -> int:
    """Special case of :func:`alternative_K_upper` for zero discrepancies."""
    return alternative_K_upper(N, n, 0, alpha=alpha)


@dataclass(frozen=True, slots=True)
class DeltaMethodResult:
    """Diagnostics and certification result for one fixed-size edge audit."""

    certified: bool
    estimated_margin: float
    lower_bound: float
    standard_error: float
    margin_variance: float
    critical_value: float
    alpha: float
    alpha_K: float
    alpha_0: float
    sample_size: int
    population_size: int
    discrepancy_count: int
    population_discrepancy_upper: int
    coordinate_variance_upper: float
    finite_population_correction: float
    corrected_point: NDArray[np.float64]
    gradient: NDArray[np.float64]
    covariance_matrix: NDArray[np.float64]
    sampled_discrepancies: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True, slots=True)
class DeltaSampleProjection:
    """Vertex-local sample information shared by all compilers at a vertex."""

    vertex_ref: Any
    shape: tuple[int, int]
    cvr_prefixes: NDArray[np.int64]
    cvr_fpv: NDArray[np.int64]
    ballot_prefixes: NDArray[np.int64]
    ballot_fpv: NDArray[np.int64]
    cvr_rows: NDArray[np.integer]
    ballot_rows: NDArray[np.integer]

    @property
    def sample_size(self) -> int:
        return int(len(self.cvr_fpv))

    @classmethod
    def from_samples(
        cls,
        interpreter: VertexInterpreter,
        cvr_sample: NDArray[np.integer],
        ballot_sample: NDArray[np.integer],
        *,
        cvr_fpv_cache: dict[frozenset[int], NDArray[np.int64]] | None = None,
        ballot_fpv_cache: dict[frozenset[int], NDArray[np.int64]] | None = None,
    ) -> "DeltaSampleProjection":
        cvr = np.asarray(cvr_sample)
        ballots = np.asarray(ballot_sample)
        if cvr.ndim != 2 or ballots.ndim != 2 or cvr.shape != ballots.shape:
            raise ValueError(
                "cvr_sample and ballot_sample must be same-shaped 2D arrays."
            )
        cvr_prefixes, cvr_fpv = cls._project_matrix(
            interpreter, cvr, cvr_fpv_cache
        )
        ballot_prefixes, ballot_fpv = cls._project_matrix(
            interpreter, ballots, ballot_fpv_cache
        )
        return cls(
            vertex_ref=interpreter.vertex.ref,
            shape=interpreter.shape,
            cvr_prefixes=cvr_prefixes,
            cvr_fpv=cvr_fpv,
            ballot_prefixes=ballot_prefixes,
            ballot_fpv=ballot_fpv,
            cvr_rows=cvr,
            ballot_rows=ballots,
        )

    @classmethod
    def _project_matrix(
        cls,
        interpreter: VertexInterpreter,
        rows: NDArray[np.integer],
        fpv_cache: dict[frozenset[int], NDArray[np.int64]] | None,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        prefix = np.zeros(len(rows), dtype=np.int64)
        for bit, edge in enumerate(interpreter.seating_edges):
            source = interpreter.graph.vertex(edge.src)
            fpv = cls._fpv_for_hopefuls(rows, source.key.hopefuls, fpv_cache)
            prefix |= (fpv == edge.candidate).astype(np.int64) << bit
        current_fpv = cls._fpv_for_hopefuls(
            rows, interpreter.vertex.key.hopefuls, fpv_cache
        )
        return prefix, current_fpv

    @staticmethod
    def _fpv_for_hopefuls(
        rows: NDArray[np.integer],
        hopefuls: Any,
        cache: dict[frozenset[int], NDArray[np.int64]] | None,
    ) -> NDArray[np.int64]:
        key = frozenset(int(candidate) for candidate in hopefuls)
        if cache is not None and key in cache:
            return cache[key]

        eligible = np.isin(rows, tuple(key))
        has_fpv = np.any(eligible, axis=1)
        positions = np.argmax(eligible, axis=1)
        fpv = np.full(len(rows), -127, dtype=np.int64)
        fpv[has_fpv] = rows[np.arange(len(rows)), positions][has_fpv]
        if cache is not None:
            cache[key] = fpv
        return fpv


SYMBOLIC_GRADIENT_MAX_DEGREE = 3


def _numerical_recursive_margin_and_gradient(
    point: NDArray[np.float64],
    quota: float,
) -> tuple[float, NDArray[np.float64]]:
    """Evaluate the recursive WIGM margin and its exact numerical gradient.

    This is forward-mode differentiation of ``build_recursive_margin``. It
    avoids constructing the enormous expanded SymPy derivatives at degrees 4
    and 5 while retaining the same coordinate and constant-quota semantics.
    """
    totals = np.asarray(point, dtype=np.float64)
    if totals.ndim != 2 or totals.shape[1] != 3:
        raise ValueError(
            "Expected a numerical margin point with shape (2**degree, 3)."
        )
    row_count = int(totals.shape[0])
    degree = int(math.log2(row_count)) if row_count else -1
    if degree < 0 or 1 << degree != row_count:
        raise ValueError(
            "The number of numerical margin rows must be a power of 2."
        )

    width = int(totals.size)
    row_totals = totals.sum(axis=1)
    row_total_gradients = np.zeros((row_count, width), dtype=np.float64)
    for mask in range(row_count):
        start = mask * 3
        row_total_gradients[mask, start : start + 3] = 1.0

    transfer_values = np.empty(degree, dtype=np.float64)
    transfer_gradients = np.zeros((degree, width), dtype=np.float64)

    def transfer_product(
        mask: int,
        stop: int,
    ) -> tuple[float, NDArray[np.float64]]:
        product = 1.0
        product_gradient = np.zeros(width, dtype=np.float64)
        for winner in range(stop):
            if not mask & (1 << winner):
                continue
            product_gradient = (
                product_gradient * transfer_values[winner]
                + product * transfer_gradients[winner]
            )
            product *= transfer_values[winner]
        return product, product_gradient

    for winner in range(degree):
        tally = 0.0
        tally_gradient = np.zeros(width, dtype=np.float64)
        for mask in range(row_count):
            if not mask & (1 << winner):
                continue
            product, product_gradient = transfer_product(mask, winner)
            tally += product * row_totals[mask]
            tally_gradient += (
                product * row_total_gradients[mask]
                + row_totals[mask] * product_gradient
            )
        if not np.isfinite(tally) or tally == 0.0:
            raise FloatingPointError(
                f"Winner {winner} has invalid recursive tally {tally}."
            )
        transfer_values[winner] = 1.0 - float(quota) / tally
        transfer_gradients[winner] = (
            float(quota) / (tally * tally)
        ) * tally_gradient

    margin = 0.0
    gradient = np.zeros(width, dtype=np.float64)
    for mask in range(row_count):
        product, product_gradient = transfer_product(mask, degree)
        candidate_difference = totals[mask, 0] - totals[mask, 1]
        margin += product * candidate_difference
        gradient += candidate_difference * product_gradient
        gradient[mask * 3] += product
        gradient[mask * 3 + 1] -= product

    if not np.isfinite(margin) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError(
            "The recursive numerical margin or gradient is non-finite."
        )
    return float(margin), gradient


@lru_cache(maxsize=None)
def _symbolic_margin_model(degree: int) -> "SymbolicMarginModel":
    """Build the symbolic margin itself, without differentiating it."""
    import sympy as sp
    from noise_filtered_linearizer import build_recursive_margin, make_symbolic_array

    symbols = make_symbolic_array(int(degree), prefix=f"delta_d{degree}")
    quota = sp.Symbol(f"delta_quota_d{degree}", real=True, positive=True)
    return build_recursive_margin(symbols, quota)


@lru_cache(maxsize=None)
def _symbolic_margin_functions(
    degree: int,
) -> tuple["SymbolicMarginModel", Callable[..., Any]]:
    """Cache a recursive margin and its degree-appropriate evaluator."""
    degree = int(degree)
    model = _symbolic_margin_model(degree)
    if degree > SYMBOLIC_GRADIENT_MAX_DEGREE:

        def numerical_evaluator(*values: float) -> tuple[float, ...]:
            expected = len(model.variables) + 1
            if len(values) != expected:
                raise ValueError(
                    f"Expected {expected} margin arguments; got {len(values)}."
                )
            margin, gradient = _numerical_recursive_margin_and_gradient(
                np.asarray(values[:-1], dtype=np.float64).reshape(
                    model.symbols.shape
                ),
                float(values[-1]),
            )
            return (margin, *gradient)

        return model, numerical_evaluator

    import sympy as sp

    gradient = tuple(sp.diff(model.margin, variable) for variable in model.variables)
    evaluator = sp.lambdify(
        (*model.variables, model.quota),
        (model.margin, *gradient),
        modules="numpy",
        cse=True,
    )
    return model, evaluator


def precompute_symbolic_derivatives(
    degrees: Iterable[int] = range(6),
) -> Any:
    """Populate the process-wide margin-evaluator cache for selected degrees.

    Degrees through 3 cache symbolic gradients. Higher degrees cache the exact
    numerical fallback to avoid impractically large symbolic derivatives.
    """
    for degree in degrees:
        degree = int(degree)
        if not 0 <= degree <= 5:
            raise ValueError("Delta-Method degrees must lie in [0, 5].")
        _symbolic_margin_functions(degree)
    return _symbolic_margin_functions.cache_info()


def symbolic_derivative_cache_info() -> Any:
    """Return hit/miss statistics for the process-wide evaluator cache."""
    return _symbolic_margin_functions.cache_info()


class DeltaMethodCompiler:
    """Fixed-sample Delta-Method compiler for one WIGM escape-edge margin.

    The sample matrices use the same convention as the existing compiler
    update methods: ``cvr_sample`` first and the corresponding paper
    ``ballot_sample`` second. Sampling must be simple random sampling without
    replacement.
    """

    def __init__(
        self,
        interpreter: VertexInterpreter,
        *,
        margin_type: CriticalMarginType | str,
        c: int | str | None = None,
        l: int | str | None = None,
        candidate: int | str | None = None,
        weak_candidate: int | str | None = None,
        strong_candidates: Iterable[int | str] | None = None,
        frozen_mentions: NDArray[np.float64] | None = None,
        lowest_strong_candidate: int | str | None = None,
        lowest_strong_tally: int | float | None = None,
        quota: int | float | None = None,
        N: int | None = None,
        alpha: float = 0.05,
        alpha_K: float | None = None,
        track_diagnostics: bool = False,
        label: str | None = None,
    ) -> None:
        self.interpreter = interpreter
        self.graph = interpreter.graph
        self.base_vertex = interpreter.vertex
        self.margin_type = CriticalMarginType(margin_type)

        self.alpha = float(alpha)
        self.alpha_K = self.alpha / 10.0 if alpha_K is None else float(alpha_K)
        self.alpha_0 = self.alpha - self.alpha_K
        self.track_diagnostics = bool(track_diagnostics)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if not 0.0 < self.alpha_K < self.alpha:
            raise ValueError("alpha_K must lie strictly between 0 and alpha.")

        inferred_N = float(np.asarray(interpreter.profile_wt_vec()).sum())
        if N is None:
            N = int(round(inferred_N))
        self.N = int(N)
        if self.N <= 1:
            raise ValueError("N must be an integer greater than one.")
        if not math.isclose(inferred_N, self.N, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                f"N={self.N} does not match profile ballot weight {inferred_N}."
            )

        graph_quota = getattr(self.graph, "quota", None)
        if quota is None and graph_quota is None:
            raise ValueError("quota is required when the graph does not expose it.")
        self.quota = float(graph_quota if quota is None else quota)
        self.c: int | None = None
        self.l: int | None = None
        self.candidate: int | None = None
        self.candidate_column: int | None = None
        self.weak_candidate: int | None = None
        self.strong_candidate: int | None = None
        self.strong_candidates: frozenset[int] = frozenset()
        self.prebatch_seated_candidates: frozenset[int] = frozenset()
        self.frozen_mentions: NDArray[np.float64] | None = None
        self.lowest_strong_tally: float | None = None
        expected_recorded_margin: float | None = None

        if self.margin_type == CriticalMarginType.CANDIDATE_TO_CANDIDATE:
            if (
                c is None
                or l is None
                or candidate is not None
                or weak_candidate is not None
            ):
                raise ValueError(
                    "Candidate-to-candidate margins require c and l only."
                )
            self.c = interpreter.candidate_index(c)
            self.l = interpreter.candidate_index(l)
            if self.c == self.l:
                raise ValueError("c and l must be distinct candidates.")
            self.base_point = interpreter.base_point(self.c, self.l)
        elif self.margin_type in {
            CriticalMarginType.CANDIDATE_ABOVE_QUOTA,
            CriticalMarginType.CANDIDATE_BELOW_QUOTA,
        }:
            if (
                candidate is None
                or c is not None
                or l is not None
                or weak_candidate is not None
            ):
                raise ValueError(
                    "Candidate-to-quota margins require candidate only."
                )
            self.candidate = interpreter.candidate_index(candidate)
            self.candidate_column = (
                COORDINATE_COLUMNS["c"]
                if self.margin_type == CriticalMarginType.CANDIDATE_ABOVE_QUOTA
                else COORDINATE_COLUMNS["l"]
            )
            self.base_point = self._quota_base_point()
        elif self.margin_type == CriticalMarginType.CANDIDATE_TO_MENTIONS:
            if (
                weak_candidate is None
                or candidate is not None
                or c is not None
                or l is not None
            ):
                raise ValueError(
                    "Candidate-to-mentions margins require weak_candidate only."
                )
            self.weak_candidate = interpreter.candidate_index(weak_candidate)
            self.strong_candidates = frozenset(
                interpreter.candidate_index(value)
                for value in (() if strong_candidates is None else strong_candidates)
            )
            if not self.strong_candidates:
                raise ValueError("strong_candidates cannot be empty.")
            if self.weak_candidate in self.strong_candidates:
                raise ValueError("weak_candidate cannot also be strong.")
            if frozen_mentions is None:
                raise ValueError("frozen_mentions is required for mentions margins.")
            self.frozen_mentions = np.asarray(frozen_mentions, dtype=np.float64)
            if (
                self.frozen_mentions.ndim != 1
                or len(self.frozen_mentions) <= self.weak_candidate
            ):
                raise ValueError("frozen_mentions must be a candidate-indexed vector.")
            if lowest_strong_candidate is None:
                if self.base_vertex.tallies is None:
                    raise ValueError("lowest_strong_candidate is required.")
                tallies = np.asarray(self.base_vertex.tallies, dtype=np.float64)
                self.strong_candidate = min(
                    self.strong_candidates,
                    key=lambda value: tallies[value],
                )
            else:
                self.strong_candidate = interpreter.candidate_index(
                    lowest_strong_candidate
                )
            if self.strong_candidate not in self.strong_candidates:
                raise ValueError(
                    "lowest_strong_candidate must belong to strong_candidates."
                )
            self.prebatch_seated_candidates = frozenset(
                int(value) for value in interpreter.winners
            )
            if lowest_strong_tally is None:
                if self.base_vertex.tallies is None:
                    raise ValueError("lowest_strong_tally is required.")
                self.lowest_strong_tally = float(
                    np.asarray(self.base_vertex.tallies, dtype=np.float64)[
                        self.strong_candidate
                    ]
                )
            else:
                self.lowest_strong_tally = float(lowest_strong_tally)
            expected_recorded_margin = self.lowest_strong_tally - float(
                self.frozen_mentions[self.weak_candidate]
            )
            self.base_point = self._mentions_base_point()
        else:  # pragma: no cover - enum validation makes this defensive only
            raise ValueError(f"Unsupported margin type: {self.margin_type.value}")

        if not math.isclose(
            float(self.base_point.sum()), self.N, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError("The local base point does not sum to N.")
        self.recorded_margin, self.recorded_gradient = self._margin_and_gradient(
            self.base_point
        )
        if expected_recorded_margin is not None and not math.isclose(
            self.recorded_margin,
            expected_recorded_margin,
            rel_tol=1e-10,
            abs_tol=1e-8,
        ):
            raise ValueError(
                "Candidate-to-mentions symbolic margin does not match the "
                "seeded recorded margin: "
                f"{self.recorded_margin} != {expected_recorded_margin}."
            )
        if self.recorded_margin <= 0.0:
            raise ValueError(
                "The recorded critical margin must be positive; got "
                f"{self.recorded_margin}. Check the candidate orientation."
            )
        self.label = label or self._default_label()
        self.last_result: DeltaMethodResult | None = None

    @property
    def degree(self) -> int:
        return int(self.base_vertex.degree)

    def evaluate(
        self,
        cvr_sample: NDArray[np.integer],
        ballot_sample: NDArray[np.integer],
    ) -> DeltaMethodResult:
        """Compute the one-sided confidence interval and certification decision."""
        projection = DeltaSampleProjection.from_samples(
            self.interpreter, cvr_sample, ballot_sample
        )
        return self.evaluate_projection(projection)

    def evaluate_projection(
        self,
        projection: DeltaSampleProjection,
    ) -> DeltaMethodResult:
        """Evaluate from a vertex projection reusable by sibling compilers."""
        if projection.vertex_ref != self.base_vertex.ref:
            raise ValueError("The sample projection belongs to a different vertex.")
        if projection.shape != self.interpreter.shape:
            raise ValueError("The sample projection has the wrong coordinate shape.")
        n = projection.sample_size
        if not 2 <= n <= self.N:
            raise ValueError(f"Need 2 <= sample size <= N; got n={n}, N={self.N}.")

        plus, minus = self._flat_sample_coordinates(projection)
        sampled_discrepancies = None
        if self.track_diagnostics:
            sampled_discrepancies = tuple(
                (int(cvr_index), int(paper_index))
                for paper_index, cvr_index in zip(plus, minus)
                if paper_index != cvr_index
            )
        coordinate_sum, sample_covariance, discrepancy_count = (
            self._transition_statistics(plus, minus)
        )
        population_discrepancy_upper = alternative_K_upper(
            self.N,
            n,
            discrepancy_count,
            alpha=self.alpha_K,
        )
        variance_upper = population_discrepancy_upper / self.N

        covariance = np.array(sample_covariance, copy=True)
        diagonal = np.diag_indices_from(covariance)
        covariance[diagonal] = np.maximum(covariance[diagonal], variance_upper)

        corrected_point = self.base_point + (self.N / n) * coordinate_sum.reshape(
            self.interpreter.shape
        )
        if not np.any(coordinate_sum):
            estimated_margin = self.recorded_margin
            gradient = self.recorded_gradient
        else:
            estimated_margin, gradient = self._margin_and_gradient(corrected_point)
        fpc = (self.N - n) / (self.N - 1)
        covariance_scale = (self.N**2 / n) * fpc
        margin_variance = float(gradient @ covariance @ gradient * covariance_scale)
        if margin_variance < 0.0 and math.isclose(
            margin_variance, 0.0, rel_tol=0.0, abs_tol=1e-10
        ):
            margin_variance = 0.0
        if not np.isfinite(margin_variance) or margin_variance < 0.0:
            raise FloatingPointError(
                f"Delta-Method margin variance is invalid: {margin_variance}."
            )

        standard_error = math.sqrt(margin_variance)
        critical_value = float(norm.ppf(1.0 - self.alpha_0))
        lower_bound = estimated_margin - critical_value * standard_error
        result = DeltaMethodResult(
            certified=bool(lower_bound > 0.0),
            estimated_margin=estimated_margin,
            lower_bound=lower_bound,
            standard_error=standard_error,
            margin_variance=margin_variance,
            critical_value=critical_value,
            alpha=self.alpha,
            alpha_K=self.alpha_K,
            alpha_0=self.alpha_0,
            sample_size=n,
            population_size=self.N,
            discrepancy_count=discrepancy_count,
            population_discrepancy_upper=population_discrepancy_upper,
            coordinate_variance_upper=variance_upper,
            finite_population_correction=fpc,
            corrected_point=corrected_point,
            gradient=gradient,
            covariance_matrix=covariance,
            sampled_discrepancies=sampled_discrepancies,
        )
        self.last_result = result
        return result

    compile = evaluate

    def certify(
        self,
        cvr_sample: NDArray[np.integer],
        ballot_sample: NDArray[np.integer],
    ) -> bool:
        """Return only the edge-certification decision for the fixed sample."""
        return self.evaluate(cvr_sample, ballot_sample).certified

    def clear_result(self) -> None:
        """Release the numerical diagnostic arrays retained from the last audit."""
        self.last_result = None

    def symbolic_margin_expression(self) -> Any:
        """Return the generic symbolic margin expression for this compiler."""
        model, _ = _symbolic_margin_functions(self.degree)
        if self.margin_type == CriticalMarginType.CANDIDATE_ABOVE_QUOTA:
            return model.margin - model.quota
        if self.margin_type == CriticalMarginType.CANDIDATE_BELOW_QUOTA:
            return model.margin + model.quota
        return model.margin

    def _flat_sample_coordinates(
        self,
        projection: DeltaSampleProjection,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        if self.margin_type == CriticalMarginType.CANDIDATE_TO_CANDIDATE:
            cvr_columns = self._candidate_columns(projection.cvr_fpv)
            ballot_columns = self._candidate_columns(projection.ballot_fpv)
        elif self.margin_type == CriticalMarginType.CANDIDATE_TO_MENTIONS:
            cvr_columns = self._mentions_columns(projection.cvr_rows)
            ballot_columns = self._mentions_columns(projection.ballot_rows)
        else:
            cvr_columns = np.where(
                projection.cvr_fpv == self.candidate,
                self.candidate_column,
                COORDINATE_COLUMNS["o"],
            )
            ballot_columns = np.where(
                projection.ballot_fpv == self.candidate,
                self.candidate_column,
                COORDINATE_COLUMNS["o"],
            )
        plus = projection.ballot_prefixes * 3 + ballot_columns
        minus = projection.cvr_prefixes * 3 + cvr_columns
        return plus.astype(np.int64, copy=False), minus.astype(np.int64, copy=False)

    def _candidate_columns(
        self, fpv: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        columns = np.full(len(fpv), COORDINATE_COLUMNS["o"], dtype=np.int64)
        columns[fpv == self.c] = COORDINATE_COLUMNS["c"]
        columns[fpv == self.l] = COORDINATE_COLUMNS["l"]
        return columns

    def _transition_statistics(
        self,
        plus: NDArray[np.int64],
        minus: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        """Return discrepancy sums/covariance without an n-by-p dense matrix."""
        n = len(plus)
        width = int(np.prod(self.interpreter.shape))
        transitions = np.bincount(
            plus * width + minus,
            minlength=width * width,
        ).reshape(width, width).astype(np.float64, copy=False)
        plus_counts = transitions.sum(axis=1)
        minus_counts = transitions.sum(axis=0)
        coordinate_sum = plus_counts - minus_counts

        sum_outer_products = -(transitions + transitions.T)
        diagonal = np.diag_indices(width)
        sum_outer_products[diagonal] += plus_counts + minus_counts
        covariance = (
            sum_outer_products - np.outer(coordinate_sum, coordinate_sum) / n
        ) / (n - 1)
        discrepancy_count = int(n - np.trace(transitions))
        return coordinate_sum, covariance, discrepancy_count

    def _quota_base_point(self) -> NDArray[np.float64]:
        return self.interpreter.candidate_base_point(
            self.candidate,
            int(self.candidate_column),
        )

    def _mentions_base_point(self) -> NDArray[np.float64]:
        weights = self.interpreter.profile_wt_vec()
        prefixes = self.interpreter.winner_prefix_indices(copy=False)
        columns = self._mentions_columns(self.graph.ballot_matrix)
        point = np.zeros(self.interpreter.shape, dtype=np.float64)
        np.add.at(point, (prefixes, columns), weights)
        return point

    def _mentions_columns(
        self,
        rows: NDArray[np.integer],
    ) -> NDArray[np.int64]:
        """Map rows using pre-batch FPV and frozen-mentions semantics.

        The candidate column requires the selected strong candidate to be the
        ballot's actual FPV after removing candidates seated before the batch.
        The mentions column requires the weak candidate to occur before every
        strong candidate. All remaining ballots are neutral.
        """
        ballots = np.asarray(rows)
        prebatch_eligible = (ballots >= 0) & ~np.isin(
            ballots, tuple(self.prebatch_seated_candidates)
        )
        has_prebatch_fpv = np.any(prebatch_eligible, axis=1)
        prebatch_positions = np.argmax(prebatch_eligible, axis=1)
        prebatch_fpv = np.full(len(ballots), -127, dtype=np.int64)
        prebatch_fpv[has_prebatch_fpv] = ballots[
            np.arange(len(ballots)), prebatch_positions
        ][has_prebatch_fpv]

        strong_mask = np.isin(ballots, tuple(self.strong_candidates))
        has_strong = np.any(strong_mask, axis=1)
        first_strong_positions = np.argmax(strong_mask, axis=1)
        weak_mask = ballots == self.weak_candidate
        has_weak = np.any(weak_mask, axis=1)
        weak_positions = np.argmax(weak_mask, axis=1)
        is_weak_mention = has_weak & (
            ~has_strong | (weak_positions < first_strong_positions)
        )
        is_strong_tally = prebatch_fpv == self.strong_candidate
        if np.any(is_weak_mention & is_strong_tally):
            raise ValueError(
                "A ballot cannot be both a pre-batch strong tally and a "
                "frozen weak mention."
            )

        columns = np.full(len(ballots), COORDINATE_COLUMNS["o"], dtype=np.int64)
        columns[is_strong_tally] = COORDINATE_COLUMNS["c"]
        columns[is_weak_mention] = COORDINATE_COLUMNS["l"]
        return columns

    def _margin_and_gradient(
        self, point: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        _, evaluator = _symbolic_margin_functions(self.degree)
        values = evaluator(*point.reshape(-1), self.quota)
        raw_margin = float(values[0])
        gradient = np.asarray(values[1:], dtype=np.float64)
        if self.margin_type == CriticalMarginType.CANDIDATE_ABOVE_QUOTA:
            margin = raw_margin - self.quota
        elif self.margin_type == CriticalMarginType.CANDIDATE_BELOW_QUOTA:
            margin = raw_margin + self.quota
        else:
            margin = raw_margin
        if not np.isfinite(margin) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError(
                "The corrected profile produced a non-finite WIGM margin or gradient."
            )
        return float(margin), gradient

    def _default_label(self) -> str:
        vertex_label = (
            self.graph.vertex_label(self.base_vertex.ref)
            if hasattr(self.graph, "vertex_label")
            else str(self.base_vertex.ref)
        )
        names = self.graph.candidate_names
        if self.margin_type == CriticalMarginType.CANDIDATE_TO_CANDIDATE:
            return f"{vertex_label}: {names[self.c]}>{names[self.l]} (delta)"
        if self.margin_type == CriticalMarginType.CANDIDATE_TO_MENTIONS:
            return (
                f"{vertex_label}: {names[self.strong_candidate]}>"
                f"{names[self.weak_candidate]} mentions (delta)"
            )
        relation = (
            "above"
            if self.margin_type == CriticalMarginType.CANDIDATE_ABOVE_QUOTA
            else "below"
        )
        return f"{vertex_label}: {names[self.candidate]} {relation} quota (delta)"


@dataclass(frozen=True, slots=True)
class DeltaEscapeCompilerInfo:
    escape_id: str
    base_label: str
    base_layer: int
    base_local_id: int
    action: EdgeAction
    candidate: int
    candidate_name: str


@dataclass(frozen=True, slots=True)
class DeltaMethodOutcome:
    """Memory-light driver record for one evaluated escape-edge compiler."""

    compiler_index: int
    escape_id: str
    certified: bool
    estimated_margin: float
    lower_bound: float
    standard_error: float
    discrepancy_count: int
    population_discrepancy_upper: int


class DeltaMethodAuditDriver:
    """Run fixed-size Delta-Method intervals for all supported escape edges.

    This is an intersection-union test: every local compiler receives the
    driver's full risk budget. The ``alpha`` value is therefore not divided by
    the number of escape edges. Batch-elimination seeded graphs additionally
    receive strong-candidate quota and weak-candidate mentions assertions.
    """

    def __init__(
        self,
        audit_graph: Any,
        *,
        sample_size: int | None = None,
        fractional_sample_size: float | None = None,
        noise_level: float = 0.0,
        seed: int | None = None,
        alpha: float = 0.05,
        alpha_K: float | None = None,
        stop_on_failure: bool = False,
        cache_noised_rows: bool = False,
        simultaneous: bool | None = None,
        retain_diagnostics: bool = False,
        verbose: bool = True,
        BAL: NDArray[np.integer] | None = None,
        CVR: NDArray[np.integer] | None = None,
    ) -> None:
        self.seeded_graph = bool(
            getattr(audit_graph, "used_seeded_build", False)
        )
        if self.seeded_graph and self._is_black_box_seeded_graph(audit_graph):
            raise NotImplementedError(
                "DeltaMethodAuditDriver does not support black-box seeded "
                "seatings yet."
            )
        self.audit_graph = audit_graph
        self.alpha = float(alpha)
        self.alpha_K = self.alpha / 10.0 if alpha_K is None else float(alpha_K)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        if not 0.0 < self.alpha_K < self.alpha:
            raise ValueError("alpha_K must lie strictly between 0 and alpha.")
        self.stop_on_failure = bool(stop_on_failure)
        self.retain_diagnostics = bool(retain_diagnostics)
        self.verbose = bool(verbose)
        self.simultaneous = bool(
            getattr(audit_graph, "simultaneous", False)
            if simultaneous is None
            else simultaneous
        )
        self.seed = seed
        self.noise_level = float(noise_level)
        self.seed_very_strong_candidates = frozenset(
            int(candidate)
            for candidate in getattr(
                audit_graph, "seed_very_strong_candidates", frozenset()
            )
        )
        self.seed_strong_candidates = frozenset(
            int(candidate)
            for candidate in getattr(
                audit_graph, "seed_strong_candidates", frozenset()
            )
        )
        self.seed_weak_candidates = frozenset(
            int(candidate)
            for candidate in getattr(
                audit_graph, "seed_weak_candidates", frozenset()
            )
        )
        self.seed_frozen_mentions: NDArray[np.float64] | None = None
        self.seed_prebatch_strong_tallies: NDArray[np.float64] | None = None
        self.sampler: ImplicitSampler | None = None
        self.BAL: NDArray[np.integer] | None = None
        self.CVR: NDArray[np.integer] | None = None
        self.sample_size = self._initialize_sample_source(
            sample_size=sample_size,
            fractional_sample_size=fractional_sample_size,
            cache_noised_rows=cache_noised_rows,
            BAL=BAL,
            CVR=CVR,
        )
        if self.sample_size < 2:
            raise ValueError("Delta-Method audits require a sample size of at least 2.")

        self.interpreters: dict[Any, VertexInterpreter] = {}
        self.compilers: list[DeltaMethodCompiler] = []
        self.compiler_info: list[DeltaEscapeCompilerInfo] = []
        self.compiler_index_by_escape_id: dict[str, int] = {}
        self.outcomes: list[DeltaMethodOutcome] = []
        self.diagnostic_results: dict[int, DeltaMethodResult] = {}
        self.projection_build_count = 0
        self.diagnostic_rerun_count = 0
        self.last_run_seconds: float | None = None
        self._last_ballot_matrix: NDArray[np.integer] | None = None
        self.canonical_winner_set = self._canonical_winner_set_from_terminal()
        self._initialize_compilers()
        if self.seeded_graph:
            self._initialize_seeded_graph_compilers()

    @property
    def certified(self) -> bool:
        return len(self.outcomes) == len(self.compilers) and all(
            outcome.certified for outcome in self.outcomes
        )

    @property
    def failed_outcomes(self) -> tuple[DeltaMethodOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if not outcome.certified)

    def run(
        self,
        ballot_matrix: NDArray[np.integer] | None = None,
    ) -> bool:
        """Evaluate each edge once, returning whether every edge certified."""
        started = perf_counter()
        if self.CVR is None:
            self._last_ballot_matrix = (
                self.audit_graph.ballot_matrix
                if ballot_matrix is None
                else np.asarray(ballot_matrix)
            )
        cvr_sample, ballot_sample = self._sample_matrices(ballot_matrix)
        self.outcomes = []
        self.diagnostic_results = {}
        self.projection_build_count = 0
        projection_cache: dict[Any, DeltaSampleProjection] = {}
        cvr_fpv_cache: dict[frozenset[int], NDArray[np.int64]] = {}
        ballot_fpv_cache: dict[frozenset[int], NDArray[np.int64]] = {}
        layer_edge_totals, layer_vertex_totals, vertex_edge_totals = (
            self._progress_totals()
        )
        layer_order = sorted(layer_edge_totals)
        layer_positions = {
            layer: position for position, layer in enumerate(layer_order, start=1)
        }
        layer_progress: dict[int, int] = {}
        vertex_progress: dict[tuple[int, int], int] = {}
        active_layer: int | None = None
        active_vertex: tuple[int, int] | None = None
        if self.verbose:
            source = "provided BAL/CVR" if self.CVR is not None else "implicit sampler"
            print(
                "Starting Delta-Method audit: "
                f"{len(self.compilers)} escape edges, "
                f"{len(self.interpreters)} vertices, n={self.sample_size}, "
                f"alpha={self.alpha:g}, alpha_K={self.alpha_K:g}, "
                f"source={source}, seeded={self.seeded_graph}."
            )
        for index, (compiler, info) in enumerate(
            zip(self.compilers, self.compiler_info)
        ):
            if info.base_layer != active_layer:
                active_layer = info.base_layer
                active_vertex = None
                if self.verbose:
                    print(
                        f"Layer {active_layer} "
                        f"({layer_positions[active_layer]}/{len(layer_order)}): "
                        f"{layer_edge_totals[active_layer]} escape edges across "
                        f"{layer_vertex_totals[active_layer]} vertices."
                    )
            vertex_key = (info.base_layer, info.base_local_id)
            if vertex_key != active_vertex:
                active_vertex = vertex_key
                if self.verbose:
                    print(
                        f"  Vertex {info.base_label}: "
                        f"{vertex_edge_totals[vertex_key]} escape edges."
                    )

            projection = projection_cache.get(compiler.base_vertex.ref)
            if projection is None:
                projection = DeltaSampleProjection.from_samples(
                    compiler.interpreter,
                    cvr_sample,
                    ballot_sample,
                    cvr_fpv_cache=cvr_fpv_cache,
                    ballot_fpv_cache=ballot_fpv_cache,
                )
                projection_cache[compiler.base_vertex.ref] = projection
                self.projection_build_count += 1
            result = compiler.evaluate_projection(projection)
            self.outcomes.append(
                DeltaMethodOutcome(
                    compiler_index=index,
                    escape_id=info.escape_id,
                    certified=result.certified,
                    estimated_margin=result.estimated_margin,
                    lower_bound=result.lower_bound,
                    standard_error=result.standard_error,
                    discrepancy_count=result.discrepancy_count,
                    population_discrepancy_upper=(
                        result.population_discrepancy_upper
                    ),
                )
            )
            if self.retain_diagnostics:
                self.diagnostic_results[index] = result
            else:
                compiler.clear_result()
            layer_progress[info.base_layer] = layer_progress.get(info.base_layer, 0) + 1
            vertex_progress[vertex_key] = vertex_progress.get(vertex_key, 0) + 1
            if self.verbose:
                self._print_edge_progress(
                    index,
                    compiler,
                    info,
                    result,
                    layer_progress[info.base_layer],
                    layer_edge_totals[info.base_layer],
                    vertex_progress[vertex_key],
                    vertex_edge_totals[vertex_key],
                )
            if self.stop_on_failure and not result.certified:
                if self.verbose:
                    print("Stopping after the first failed escape-edge interval.")
                break

        self.last_run_seconds = perf_counter() - started
        if self.verbose:
            self.print_summary()
        return self.certified

    def lookup_compiler(self, escape_id: str) -> DeltaMethodCompiler:
        try:
            index = self.compiler_index_by_escape_id[escape_id.strip().upper()]
        except KeyError:
            known = ", ".join(info.escape_id for info in self.compiler_info[:10])
            raise KeyError(
                f"Unknown compiler escape_id {escape_id!r}. Known IDs: {known}"
            ) from None
        return self.compilers[index]

    def print_diagnostics(
        self,
        compiler_identifier: str,
    ) -> DeltaMethodResult:
        """Print full numerical diagnostics for one escape-edge compiler.

        When the ordinary run did not retain diagnostics, only the requested
        compiler is deterministically re-evaluated against the same sample.
        """
        normalized = compiler_identifier.strip().upper()
        try:
            index = self.compiler_index_by_escape_id[normalized]
        except KeyError:
            known = ", ".join(info.escape_id for info in self.compiler_info[:10])
            raise KeyError(
                f"Unknown compiler_identifier {compiler_identifier!r}. "
                f"Known IDs: {known}"
            ) from None

        compiler = self.compilers[index]
        info = self.compiler_info[index]
        result = self.diagnostic_results.get(index)
        if result is None or result.sampled_discrepancies is None:
            cvr_sample, ballot_sample = self._sample_matrices(
                self._last_ballot_matrix
            )
            projection = DeltaSampleProjection.from_samples(
                compiler.interpreter,
                cvr_sample,
                ballot_sample,
            )
            previous_setting = compiler.track_diagnostics
            compiler.track_diagnostics = True
            try:
                result = compiler.evaluate_projection(projection)
            finally:
                compiler.track_diagnostics = previous_setting
            self.diagnostic_results[index] = result
            self.diagnostic_rerun_count += 1

        import sympy as sp

        expression = compiler.symbolic_margin_expression()
        covariance = np.array2string(
            result.covariance_matrix,
            precision=6,
            suppress_small=False,
            threshold=np.inf,
            max_line_width=120,
        )
        gradient = np.array2string(
            result.gradient,
            precision=6,
            suppress_small=False,
            threshold=np.inf,
            max_line_width=120,
        )
        discrepancies = list(result.sampled_discrepancies or ())
        print(f"Delta-Method diagnostics for {info.escape_id}:")
        print(
            f"  Escape edge: {info.action.name} "
            f"{info.candidate_name}({info.candidate}) from {info.base_label}."
        )
        print(f"  Assertion: {self._margin_description(compiler)}.")
        print("  Symbolic margin function:")
        print(sp.pretty(expression, use_unicode=True))
        print("  Conservative numerical covariance matrix:")
        print(covariance)
        print("  Numerical gradient array:")
        print(gradient)
        print(f"  Numerical margin variance: {result.margin_variance:.12g}")
        print(
            "  Standardized sampled discrepancies "
            "(CVR flat index, paper flat index):"
        )
        print(discrepancies)
        print(
            "  Sample-corrected margin estimate: "
            f"{result.estimated_margin:.12g}"
        )
        print(
            "  One-sided confidence interval: "
            f"[{result.lower_bound:.12g}, inf)"
        )
        return result

    def print_summary(self, limit: int = 3) -> None:
        evaluated = len(self.outcomes)
        passed = sum(outcome.certified for outcome in self.outcomes)
        print(
            f"Delta-Method audit: {passed}/{evaluated} evaluated compilers "
            f"certified ({len(self.compilers)} total)"
            + (
                f" in {self.last_run_seconds:.2f}s."
                if self.last_run_seconds is not None
                else "."
            )
        )
        lowest = sorted(self.outcomes, key=lambda item: item.lower_bound)[: int(limit)]
        if lowest:
            print(
                f"Lowest {len(lowest)} escape-edge interval endpoints:"
            )
        for outcome in lowest:
            compiler = self.compilers[outcome.compiler_index]
            info = self.compiler_info[outcome.compiler_index]
            status = "PASS" if outcome.certified else "FAIL"
            print(
                f"  {outcome.escape_id} {status}: "
                f"lower={outcome.lower_bound:,.2f}, "
                f"escape={info.action.name} {info.candidate_name}"
                f"({info.candidate}); assert {self._margin_description(compiler)}; "
                f"estimate={outcome.estimated_margin:,.2f}, "
                f"SE={outcome.standard_error:,.2f}."
            )

    def _progress_totals(
        self,
    ) -> tuple[dict[int, int], dict[int, int], dict[tuple[int, int], int]]:
        layer_edges: dict[int, int] = {}
        layer_vertices: dict[int, set[int]] = {}
        vertex_edges: dict[tuple[int, int], int] = {}
        for info in self.compiler_info:
            layer_edges[info.base_layer] = layer_edges.get(info.base_layer, 0) + 1
            layer_vertices.setdefault(info.base_layer, set()).add(info.base_local_id)
            key = (info.base_layer, info.base_local_id)
            vertex_edges[key] = vertex_edges.get(key, 0) + 1
        return (
            layer_edges,
            {layer: len(vertices) for layer, vertices in layer_vertices.items()},
            vertex_edges,
        )

    def _print_edge_progress(
        self,
        index: int,
        compiler: DeltaMethodCompiler,
        info: DeltaEscapeCompilerInfo,
        result: DeltaMethodResult,
        layer_position: int,
        layer_total: int,
        vertex_position: int,
        vertex_total: int,
    ) -> None:
        status = "PASS" if result.certified else "FAIL"
        print(
            f"    [{layer_position}/{layer_total}; vertex "
            f"{vertex_position}/{vertex_total}; overall "
            f"{index + 1}/{len(self.compilers)}] {info.escape_id} {status}: "
            f"escape={info.action.name} {info.candidate_name}({info.candidate}); "
            f"assert {self._margin_description(compiler)}; "
            f"CI=[{result.lower_bound:,.2f}, inf), "
            f"estimate={result.estimated_margin:,.2f}, "
            f"recorded={compiler.recorded_margin:,.2f}, "
            f"SE={result.standard_error:,.2f}, "
            f"discrepancies={result.discrepancy_count}, "
            f"K<={result.population_discrepancy_upper}."
        )

    def _margin_description(self, compiler: DeltaMethodCompiler) -> str:
        names = self.audit_graph.candidate_names
        if compiler.margin_type == CriticalMarginType.CANDIDATE_TO_CANDIDATE:
            return f"{names[compiler.c]} > {names[compiler.l]}"
        if compiler.margin_type == CriticalMarginType.CANDIDATE_ABOVE_QUOTA:
            return f"{names[compiler.candidate]} > quota"
        if compiler.margin_type == CriticalMarginType.CANDIDATE_TO_MENTIONS:
            return (
                f"{names[compiler.strong_candidate]} > "
                f"{names[compiler.weak_candidate]} mentions"
            )
        return f"quota > {names[compiler.candidate]}"

    def _initialize_sample_source(
        self,
        *,
        sample_size: int | None,
        fractional_sample_size: float | None,
        cache_noised_rows: bool,
        BAL: NDArray[np.integer] | None,
        CVR: NDArray[np.integer] | None,
    ) -> int:
        if (BAL is None) != (CVR is None):
            raise ValueError("BAL and CVR must be provided together.")
        if BAL is not None and CVR is not None:
            if sample_size is not None or fractional_sample_size is not None:
                raise ValueError(
                    "Do not provide sample sizes together with BAL/CVR matrices."
                )
            bal = np.asarray(BAL)
            cvr = np.asarray(CVR)
            if bal.ndim != 2 or cvr.ndim != 2 or bal.shape != cvr.shape:
                raise ValueError("BAL and CVR must be same-shaped 2D arrays.")
            self.BAL = bal
            self.CVR = cvr
            return int(cvr.shape[0])

        weights = self._profile_weights()
        self.sampler = ImplicitSampler(
            noise_level=self.noise_level,
            n_cands=int(self.audit_graph.n_candidates),
            wt_vec=weights,
            sample_size=sample_size,
            fractional_sample_size=fractional_sample_size,
            with_replacement=False,
            seed=self.seed,
            cache_rows=cache_noised_rows,
        )
        return int(self.sampler.sample_size)

    def _sample_matrices(
        self,
        ballot_matrix: NDArray[np.integer] | None,
    ) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
        if self.CVR is not None and self.BAL is not None:
            return self.CVR, self.BAL
        if self.sampler is None:
            raise ValueError("Driver has no sample source.")
        matrix = (
            self.audit_graph.ballot_matrix
            if ballot_matrix is None
            else np.asarray(ballot_matrix)
        )
        cvr = np.empty((self.sample_size, matrix.shape[1]), dtype=matrix.dtype)
        ballots = np.empty_like(cvr)
        for index in range(self.sample_size):
            cvr[index], ballots[index] = self.sampler.sample(index, matrix)
        return cvr, ballots

    def _profile_weights(self) -> NDArray[np.float64]:
        if hasattr(self.audit_graph, "_unseeded_root_wt_vec"):
            return np.asarray(
                self.audit_graph._unseeded_root_wt_vec, dtype=np.float64
            )
        return np.asarray(self.audit_graph.root_wt_vec, dtype=np.float64)

    @staticmethod
    def _is_black_box_seeded_graph(audit_graph: Any) -> bool:
        """Distinguish uncertain black-box seatings from batch elimination."""
        return bool(
            hasattr(audit_graph, "vertex_post_seed_tallies")
            or hasattr(audit_graph, "seed_weight_scenarios")
            or audit_graph.__class__.__module__.endswith(".black_box")
        )

    def _initialize_seeded_graph_compilers(self) -> None:
        """Add the assertions that justify a batch-elimination graph seed."""
        if not self.seed_strong_candidates:
            raise ValueError(
                "Seeded audit graph did not expose seed_strong_candidates."
            )

        strong_vertex = self._find_strong_only_seed_vertex()
        self._validate_no_election_edges_from_strong_vertex(strong_vertex)
        self._initialize_seeded_mentions_data()
        assert self.seed_frozen_mentions is not None
        assert self.seed_prebatch_strong_tallies is not None

        interpreter = self._interpreter_for_vertex(strong_vertex)
        for candidate in sorted(self.seed_strong_candidates):
            info = self._make_compiler_info(
                strong_vertex, int(candidate), EdgeAction.ELECT
            )
            if info.escape_id.upper() not in self.compiler_index_by_escape_id:
                self._add_compiler(
                    strong_vertex,
                    interpreter,
                    int(candidate),
                    EdgeAction.ELECT,
                )

        lowest_strong_candidate = min(
            self.seed_strong_candidates,
            key=lambda candidate: self.seed_prebatch_strong_tallies[candidate],
        )
        lowest_strong_tally = float(
            self.seed_prebatch_strong_tallies[lowest_strong_candidate]
        )
        for weak_candidate in sorted(self.seed_weak_candidates):
            info = self._make_seed_mentions_compiler_info(
                strong_vertex, int(weak_candidate)
            )
            compiler = DeltaMethodCompiler(
                interpreter,
                margin_type=CriticalMarginType.CANDIDATE_TO_MENTIONS,
                weak_candidate=int(weak_candidate),
                strong_candidates=self.seed_strong_candidates,
                frozen_mentions=self.seed_frozen_mentions,
                lowest_strong_candidate=int(lowest_strong_candidate),
                lowest_strong_tally=lowest_strong_tally,
                alpha=self.alpha,
                alpha_K=self.alpha_K,
                track_diagnostics=self.retain_diagnostics,
                label=info.escape_id,
            )
            self.compiler_index_by_escape_id[info.escape_id.upper()] = len(
                self.compilers
            )
            self.compilers.append(compiler)
            self.compiler_info.append(info)

    def _find_strong_only_seed_vertex(self) -> Any:
        matches = [
            vertex
            for layer in self.audit_graph.layers
            for vertex in layer
            if (
                hasattr(vertex.key, "hopefuls")
                and frozenset(vertex.key.hopefuls)
                == self.seed_strong_candidates
            )
        ]
        if not matches:
            raise ValueError(
                "Seeded audit graph does not contain a vertex whose hopefuls "
                "are exactly the seed strong candidates."
            )
        return min(
            matches,
            key=lambda vertex: (vertex.ref.layer, vertex.ref.local_id),
        )

    def _validate_no_election_edges_from_strong_vertex(
        self, strong_vertex: Any
    ) -> None:
        election_edges = [
            edge
            for edge in self.audit_graph.outgoing_edges(strong_vertex.ref)
            if (
                EdgeAction(edge.action).is_election
                and not self._is_forced_fill_edge(strong_vertex, edge)
            )
        ]
        if election_edges:
            label = self.audit_graph.vertex_label(strong_vertex.ref)
            raise ValueError(
                "Strong-only seed vertex has election edges, so strong "
                "candidates are not safely below quota. "
                f"vertex={label}, edges={election_edges}"
            )

    def _is_forced_fill_edge(self, vertex: Any, edge: Any) -> bool:
        return bool(
            EdgeAction(edge.action) == EdgeAction.FORCE_ELECT
            and hasattr(vertex.key, "hopefuls")
            and len(vertex.key.hopefuls) + int(vertex.degree)
            == int(self.audit_graph.m)
        )

    def _initialize_seeded_mentions_data(self) -> None:
        if not hasattr(self.audit_graph, "root_wt_vec"):
            raise ValueError(
                "Seeded audit graph must expose root_wt_vec for seeded "
                "assertions."
            )
        prebatch_weights = np.asarray(
            self.audit_graph.root_wt_vec, dtype=np.float64
        )
        self.seed_prebatch_strong_tallies = fpv_tallies_from_matrix(
            self.audit_graph.ballot_matrix,
            prebatch_weights,
            int(self.audit_graph.n_candidates),
            masked_candidates=self.seed_very_strong_candidates,
        )

        graph_mentions = getattr(
            self.audit_graph, "seed_frozen_mentions", None
        )
        if graph_mentions is None:
            self.seed_frozen_mentions = frozen_mentions_from_matrix(
                self.audit_graph.ballot_matrix,
                prebatch_weights,
                int(self.audit_graph.n_candidates),
                self.seed_strong_candidates,
                masked_candidates=self.seed_very_strong_candidates,
            )
        else:
            self.seed_frozen_mentions = np.asarray(
                graph_mentions, dtype=np.float64
            )

    def _make_seed_mentions_compiler_info(
        self, vertex: Any, candidate: int
    ) -> DeltaEscapeCompilerInfo:
        base_label = self.audit_graph.vertex_label(vertex.ref)
        return DeltaEscapeCompilerInfo(
            escape_id=f"{base_label}-M{candidate}",
            base_label=base_label,
            base_layer=int(vertex.ref.layer),
            base_local_id=int(vertex.ref.local_id),
            action=EdgeAction.ELIMINATE,
            candidate=int(candidate),
            candidate_name=str(self.audit_graph.candidate_names[candidate]),
        )

    def _initialize_compilers(self) -> None:
        for layer in self.audit_graph.layers:
            for vertex in layer:
                if vertex.status == ElectionStatus.TERMINAL:
                    continue
                if len(vertex.key.hopefuls) + int(vertex.degree) == int(
                    self.audit_graph.m
                ):
                    continue
                outgoing = self.audit_graph.outgoing_edges(vertex.ref)
                if not outgoing and getattr(vertex, "path_multiplicity", 1) == 0:
                    continue
                if vertex.tallies is None:
                    raise ValueError(
                        "Cannot initialize a Delta compiler from a vertex without "
                        f"tallies: {self.audit_graph.vertex_label(vertex.ref)}."
                    )

                interpreter = self._interpreter_for_vertex(vertex)
                forced_winner = self._forced_above_quota_candidate(vertex)
                if forced_winner is not None and any(
                    not self._has_outgoing_elimination(outgoing, candidate)
                    for candidate in vertex.key.hopefuls
                ):
                    self._add_compiler(
                        vertex, interpreter, forced_winner, EdgeAction.ELIMINATE
                    )

                for candidate in sorted(vertex.key.hopefuls):
                    candidate = int(candidate)
                    if forced_winner is None and not self._has_outgoing_elimination(
                        outgoing, candidate
                    ):
                        self._add_compiler(
                            vertex, interpreter, candidate, EdgeAction.ELIMINATE
                        )
                    if self._has_outgoing_election(outgoing, candidate):
                        continue
                    if self._skip_final_reported_winner_seating_escape(
                        vertex, candidate
                    ):
                        continue
                    if (
                        forced_winner is not None
                        and self._forced_election_escape_is_redundant(
                            vertex, candidate
                        )
                    ):
                        continue
                    self._add_compiler(
                        vertex, interpreter, candidate, EdgeAction.ELECT
                    )

    def _add_compiler(
        self,
        vertex: Any,
        interpreter: VertexInterpreter,
        candidate: int,
        action: EdgeAction,
    ) -> None:
        margin = self._critical_margin_for_escape(vertex, candidate, action)
        if margin is None:
            return
        info = self._make_compiler_info(vertex, candidate, action)
        margin_type = CriticalMarginType(margin["type"])
        common = {
            "margin_type": margin_type,
            "alpha": self.alpha,
            "alpha_K": self.alpha_K,
            "track_diagnostics": self.retain_diagnostics,
            "label": info.escape_id,
        }
        if margin_type == CriticalMarginType.CANDIDATE_TO_CANDIDATE:
            compiler = DeltaMethodCompiler(
                interpreter,
                c=int(margin["c"]),
                l=int(margin["l"]),
                **common,
            )
        else:
            compiler = DeltaMethodCompiler(
                interpreter,
                candidate=int(margin["candidate"]),
                **common,
            )
        self.compiler_index_by_escape_id[info.escape_id.upper()] = len(
            self.compilers
        )
        self.compilers.append(compiler)
        self.compiler_info.append(info)

    def _critical_margin_for_escape(
        self,
        vertex: Any,
        candidate: int,
        action: EdgeAction,
    ) -> dict[str, Any] | None:
        tallies = np.asarray(vertex.tallies, dtype=np.float64)
        if action == EdgeAction.ELIMINATE:
            forced = np.where(
                tallies >= self.audit_graph.quota + self.audit_graph.LAM
            )[0]
            if len(forced):
                winner = int(forced[np.argmax(tallies[forced])])
                return {
                    "type": CriticalMarginType.CANDIDATE_ABOVE_QUOTA,
                    "candidate": winner,
                }
            hopefuls = np.asarray(sorted(vertex.key.hopefuls), dtype=int)
            lowest = int(hopefuls[np.argmin(tallies[hopefuls])])
            if lowest == candidate:
                return None
            return {
                "type": CriticalMarginType.CANDIDATE_TO_CANDIDATE,
                "c": candidate,
                "l": lowest,
            }

        candidate_tally = float(tallies[candidate])
        if not self.simultaneous:
            challengers = np.asarray(
                [
                    index
                    for index in np.where(
                        tallies > candidate_tally + self.audit_graph.LAM
                    )[0]
                    if index != candidate
                ],
                dtype=int,
            )
            if len(challengers):
                winner = int(challengers[np.argmax(tallies[challengers])])
                return {
                    "type": CriticalMarginType.CANDIDATE_TO_CANDIDATE,
                    "c": winner,
                    "l": candidate,
                }
        if candidate_tally + self.audit_graph.LAM < self.audit_graph.quota:
            return {
                "type": CriticalMarginType.CANDIDATE_BELOW_QUOTA,
                "candidate": candidate,
            }
        return None

    def _interpreter_for_vertex(self, vertex: Any) -> VertexInterpreter:
        if vertex.ref not in self.interpreters:
            self.interpreters[vertex.ref] = VertexInterpreter(
                self.audit_graph, vertex
            )
        return self.interpreters[vertex.ref]

    def _forced_above_quota_candidate(self, vertex: Any) -> int | None:
        tallies = np.asarray(vertex.tallies, dtype=np.float64)
        hopefuls = np.asarray(sorted(vertex.key.hopefuls), dtype=int)
        forced = hopefuls[
            tallies[hopefuls]
            >= self.audit_graph.quota + self.audit_graph.LAM
        ]
        return (
            None
            if len(forced) == 0
            else int(forced[np.argmax(tallies[forced])])
        )

    @staticmethod
    def _has_outgoing_elimination(outgoing: list[Any], candidate: int) -> bool:
        return any(
            edge.candidate == candidate and edge.action == EdgeAction.ELIMINATE
            for edge in outgoing
        )

    @staticmethod
    def _has_outgoing_election(outgoing: list[Any], candidate: int) -> bool:
        return any(
            edge.candidate == candidate and EdgeAction(edge.action).is_election
            for edge in outgoing
        )

    def _forced_election_escape_is_redundant(
        self, vertex: Any, candidate: int
    ) -> bool:
        tallies = np.asarray(vertex.tallies, dtype=np.float64)
        candidate_tally = float(tallies[candidate])
        if not self.simultaneous:
            challengers = [
                index
                for index in np.where(
                    tallies > candidate_tally + self.audit_graph.LAM
                )[0]
                if index != candidate
            ]
            if challengers:
                return False
        return bool(
            candidate_tally + self.audit_graph.LAM < self.audit_graph.quota
        )

    def _canonical_winner_set_from_terminal(self) -> frozenset[int]:
        terminals = [
            vertex
            for layer in self.audit_graph.layers
            for vertex in layer
            if vertex.status == ElectionStatus.TERMINAL
        ]
        if not terminals:
            return frozenset()
        terminal = min(terminals, key=lambda item: (item.ref.layer, item.ref.local_id))
        if hasattr(self.audit_graph, "_winner_set_for_vertex"):
            return frozenset(
                int(candidate)
                for candidate in self.audit_graph._winner_set_for_vertex(terminal)
            )
        return frozenset(
            int(self.audit_graph.edge(edge_ref).candidate)
            for edge_ref in terminal.key.seated_at
            if edge_ref is not None
        )

    def _skip_final_reported_winner_seating_escape(
        self, vertex: Any, candidate: int
    ) -> bool:
        return (
            int(vertex.degree) == int(self.audit_graph.m) - 1
            and candidate in self.canonical_winner_set
        )

    def _make_compiler_info(
        self, vertex: Any, candidate: int, action: EdgeAction
    ) -> DeltaEscapeCompilerInfo:
        base_label = self.audit_graph.vertex_label(vertex.ref)
        action_code = "E" if action == EdgeAction.ELIMINATE else "W"
        return DeltaEscapeCompilerInfo(
            escape_id=f"{base_label}-{action_code}{candidate}",
            base_label=base_label,
            base_layer=int(vertex.ref.layer),
            base_local_id=int(vertex.ref.local_id),
            action=action,
            candidate=candidate,
            candidate_name=str(self.audit_graph.candidate_names[candidate]),
        )


__all__ = [
    "DeltaMethodCompiler",
    "DeltaMethodAuditDriver",
    "DeltaMethodOutcome",
    "DeltaMethodResult",
    "DeltaSampleProjection",
    "K_upper",
    "alternative_K_upper",
    "hypergeom_log_cdf_leq",
    "log_comb",
    "precompute_symbolic_derivatives",
    "symbolic_derivative_cache_info",
]
