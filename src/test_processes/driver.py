from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .cobra import CobraCompiler
from .cobra import (
    CobraMentionsNoiseFilterCompiler,
    CobraNoiseFilterBase,
    CobraNoiseFilterCompiler,
    CobraQuotaNoiseFilterCompiler,
    CriticalMarginType,
)
from .interpreter import VertexInterpreter
from .margins import critical_margin_for_escape
from .noise import ImplicitSampler

try:
    from ..election_graphs.datatypes import EdgeAction, ElectionStatus
    from ..election_graphs.utils import (
        fpv_tallies_from_matrix,
        maximum_possible_tallies_from_matrix,
    )
except ImportError:
    from election_graphs.datatypes import EdgeAction, ElectionStatus
    from election_graphs.utils import (
        fpv_tallies_from_matrix,
        maximum_possible_tallies_from_matrix,
    )


@dataclass(frozen=True, slots=True)
class EscapeCompilerInfo:
    escape_id: str
    base_label: str
    base_layer: int
    base_local_id: int
    action: EdgeAction
    candidate: int
    candidate_name: str


class GlobalAuditDriver:
    """Audit every missing graph edge with COBRA assertions or noise filters.

    Set ``compiler_type="noise"`` to replace each assertion compiler with a
    local noise-filter compiler. Candidate-to-candidate noise budgets are half
    the escape margin; all other margin types use the full escape margin.
    """

    def __init__(
        self,
        audit_graph: Any,
        noise_level: float,
        noise_guess: float | None = None,
        sample_size: int | None = None,
        fractional_sample_size: float | None = None,
        seed: int | None = None,
        compiler_type: str = "cobra",
        print_diagnostics_every: int = 30,
        alpha: float = 0.05,
        keep_certified_compilers: bool = False,
        cache_noised_rows: bool = False,
        remember_discrepancies: bool = True,
        simultaneous: bool = False,
        use_fallback: bool = True,
        BAL: NDArray[np.integer] | None = None,
        CVR: NDArray[np.integer] | None = None,
    ) -> None:
        compiler_type = str(compiler_type).strip().lower()
        if compiler_type not in {"cobra", "noise"}:
            raise ValueError(
                f"Unsupported compiler_type: {compiler_type}. "
                "Expected 'cobra' or 'noise'."
            )
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")

        self.verbose = False
        self.audit_graph = audit_graph
        self.compiler_type = compiler_type
        self.noise_level = float(noise_level)
        self.noise_guess = self.noise_level if noise_guess is None else float(noise_guess)
        self.print_diagnostics_every = int(print_diagnostics_every)
        self.alpha = float(alpha)
        self.threshold = 1.0 / self.alpha
        self.keep_certified_compilers = bool(keep_certified_compilers)
        self.use_fallback = bool(use_fallback)
        self.i = 0
        self.m = int(audit_graph.m)
        self.sampler: ImplicitSampler | None = None
        self.BAL: NDArray[np.integer] | None = None
        self.CVR: NDArray[np.integer] | None = None
        self.seeded_graph = bool(getattr(audit_graph, "used_seeded_build", False))
        self.seed_very_strong_candidates = frozenset(
            getattr(audit_graph, "seed_very_strong_candidates", frozenset())
        )
        self.seed_strong_candidates = frozenset(
            getattr(audit_graph, "seed_strong_candidates", frozenset())
        )
        self.seed_weak_candidates = frozenset(
            getattr(audit_graph, "seed_weak_candidates", frozenset())
        )
        self.seed_maximum_possible_tallies: NDArray[np.float64] | None = None
        self.seed_prebatch_strong_tallies: NDArray[np.float64] | None = None
        self.sample_size = self._initialize_sample_source(
            audit_graph=audit_graph,
            sample_size=sample_size,
            fractional_sample_size=fractional_sample_size,
            seed=seed,
            cache_noised_rows=cache_noised_rows,
            BAL=BAL,
            CVR=CVR,
        )

        self.interpreters: dict[Any, VertexInterpreter] = {}
        self.compilers: list[CobraNoiseFilterBase | CobraCompiler] = []
        self.compiler_info: list[EscapeCompilerInfo] = []
        self.compiler_index_by_escape_id: dict[str, int] = {}
        self._initialize_compilers(
            audit_graph=audit_graph,
            remember_discrepancies=remember_discrepancies,
            simultaneous=simultaneous,
            use_fallback=self.use_fallback,
        )
        if self.seeded_graph:
            self._initialize_seeded_graph_compilers(
                audit_graph=audit_graph,
                remember_discrepancies=remember_discrepancies,
                simultaneous=simultaneous,
                use_fallback=self.use_fallback,
            )
        self.deduplicate_compilers()

        self.pool: list[int] = list(range(len(self.compilers)))
        self.certified: set[int] = set()
        self.certification_order: list[int] = []

    def _log(self, message: str) -> None:
        if getattr(self, "verbose", False):
            print(message)

    def _wt_vec_from_graph(self, audit_graph: Any) -> NDArray[np.float64]:
        if hasattr(audit_graph, "_unseeded_root_wt_vec"):
            return np.asarray(audit_graph._unseeded_root_wt_vec, dtype=np.float64)
        if hasattr(audit_graph, "root_wt_vec"):
            return np.asarray(audit_graph.root_wt_vec, dtype=np.float64)
        if hasattr(audit_graph, "profile"):
            return ImplicitSampler._wt_vec_from_profile(audit_graph.profile)
        raise ValueError("Could not recover wt_vec from audit_graph.")

    def _initialize_sample_source(
        self,
        audit_graph: Any,
        sample_size: int | None,
        fractional_sample_size: float | None,
        seed: int | None,
        cache_noised_rows: bool,
        BAL: NDArray[np.integer] | None,
        CVR: NDArray[np.integer] | None,
    ) -> int:
        if (BAL is None) != (CVR is None):
            raise ValueError("BAL and CVR must be provided together.")

        if BAL is not None and CVR is not None:
            if sample_size is not None or fractional_sample_size is not None:
                raise ValueError(
                    "sample_size and fractional_sample_size should not be provided "
                    "when BAL/CVR sample matrices are provided."
                )
            bal = np.asarray(BAL)
            cvr = np.asarray(CVR)
            if bal.ndim != 2 or cvr.ndim != 2:
                raise ValueError("BAL and CVR must be two-dimensional arrays.")
            if bal.shape != cvr.shape:
                raise ValueError("BAL and CVR must have the same shape.")
            self.BAL = bal
            self.CVR = cvr
            self._log(
                f"  sample source: provided BAL/CVR matrices with {cvr.shape[0]} rows"
            )
            return int(cvr.shape[0])

        wt_vec = self._wt_vec_from_graph(audit_graph)
        self.sampler = ImplicitSampler(
            noise_level=self.noise_level,
            n_cands=int(audit_graph.n_candidates),
            wt_vec=wt_vec,
            sample_size=sample_size,
            fractional_sample_size=fractional_sample_size,
            with_replacement=True,
            seed=seed,
            cache_rows=cache_noised_rows,
        )
        return int(self.sampler.sample_size)

    def _initialize_compilers(
        self,
        audit_graph: Any,
        remember_discrepancies: bool,
        simultaneous: bool,
        use_fallback: bool,
    ) -> None:
        for layer in audit_graph.layers:
            for vertex in layer:
                if vertex.status == ElectionStatus.TERMINAL:
                    continue
                if not hasattr(vertex.key, "hopefuls"):
                    raise ValueError("Audit graph vertex key does not expose hopefuls.")
                if len(vertex.key.hopefuls) + int(vertex.degree) == self.m:
                    continue

                outgoing = audit_graph.outgoing_edges(vertex.ref)
                if not outgoing and getattr(vertex, "path_multiplicity", 1) == 0:
                    continue
                if vertex.tallies is None:
                    label = (
                        audit_graph.vertex_label(vertex.ref)
                        if hasattr(audit_graph, "vertex_label")
                        else str(vertex.ref)
                    )
                    raise ValueError(
                        "Cannot initialize COBRA compilers from vertex "
                        f"{label}: base_vertex.tallies is not computed."
                    )

                for candidate in sorted(vertex.key.hopefuls):
                    if not self._has_outgoing_elimination(outgoing, candidate):
                        self._add_compiler(
                            audit_graph=audit_graph,
                            vertex=vertex,
                            candidate=int(candidate),
                            action=EdgeAction.ELIMINATE,
                            remember_discrepancies=remember_discrepancies,
                            simultaneous=simultaneous,
                            use_fallback=use_fallback,
                        )
                    if not self._has_outgoing_election(outgoing, candidate):
                        self._add_compiler(
                            audit_graph=audit_graph,
                            vertex=vertex,
                            candidate=int(candidate),
                            action=EdgeAction.ELECT,
                            remember_discrepancies=remember_discrepancies,
                            simultaneous=simultaneous,
                            use_fallback=use_fallback,
                        )

    def _has_outgoing_elimination(
        self,
        outgoing: list[Any],
        candidate: int,
    ) -> bool:
        return any(
            edge.candidate == candidate and edge.action == EdgeAction.ELIMINATE
            for edge in outgoing
        )

    def _has_outgoing_election(
        self,
        outgoing: list[Any],
        candidate: int,
    ) -> bool:
        return any(
            edge.candidate == candidate
            and EdgeAction(edge.action).is_election
            for edge in outgoing
        )

    def _add_compiler(
        self,
        audit_graph: Any,
        vertex: Any,
        candidate: int,
        action: EdgeAction,
        remember_discrepancies: bool,
        simultaneous: bool,
        use_fallback: bool,
    ) -> None:
        info = self._make_compiler_info(
            audit_graph=audit_graph,
            vertex=vertex,
            candidate=candidate,
            action=action,
        )
        compiler: CobraNoiseFilterBase | CobraCompiler
        if self.compiler_type == "noise":
            compiler = self._make_noise_filter_compiler(
                audit_graph=audit_graph,
                vertex=vertex,
                candidate=candidate,
                action=action,
                simultaneous=simultaneous,
                label=info.escape_id,
            )
        else:
            compiler = CobraCompiler(
                vertex,
                candidate,
                action,
                audit_graph=audit_graph,
                noise_level_guess=self.noise_guess,
                simultaneous=simultaneous,
                remember_discrepancies=remember_discrepancies,
                use_fallback=use_fallback,
                compiler_label=info.escape_id,
            )
        self.compilers.append(compiler)
        self.compiler_info.append(info)
        self.compiler_index_by_escape_id[info.escape_id.upper()] = (
            len(self.compilers) - 1
        )

    def _make_noise_filter_compiler(
        self,
        audit_graph: Any,
        vertex: Any,
        candidate: int,
        action: EdgeAction,
        simultaneous: bool,
        label: str,
    ) -> CobraNoiseFilterBase:
        interpreter = self._interpreter_for_vertex(audit_graph, vertex)
        margin = self._noise_filter_margin(vertex, candidate, action, simultaneous)
        margin_type = CriticalMarginType(margin["type"])
        escape_margin = float(margin["margin"])
        noise_budget = (
            escape_margin / 2.0
            if margin_type == CriticalMarginType.CANDIDATE_TO_CANDIDATE
            else escape_margin
        )
        common: dict[str, Any] = {
            "MoI": float(audit_graph.MoI),
            "critical_margin": noise_budget,
            "radius": 2.0 * noise_budget,
            "alpha": self.alpha,
            "label": label,
        }

        compiler: CobraNoiseFilterBase
        if margin_type == CriticalMarginType.CANDIDATE_TO_CANDIDATE:
            compiler = CobraNoiseFilterCompiler(
                interpreter,
                int(margin["c"]),
                int(margin["l"]),
                **common,
            )
        else:
            compiler = CobraQuotaNoiseFilterCompiler(
                interpreter,
                int(margin["candidate"]),
                margin_type=margin_type,
                **common,
            )
        return compiler

    def _interpreter_for_vertex(
        self,
        audit_graph: Any,
        vertex: Any,
    ) -> VertexInterpreter:
        existing = self.interpreters.get(vertex.ref)
        if existing is not None:
            return existing
        interpreter = VertexInterpreter(audit_graph, vertex)
        self.interpreters[vertex.ref] = interpreter
        return interpreter

    def _noise_filter_margin(
        self,
        vertex: Any,
        candidate: int,
        action: EdgeAction,
        simultaneous: bool,
    ) -> dict[str, Any]:
        """Identify the local coordinates whose noise can enable an escape edge."""
        graph = self.audit_graph
        result = critical_margin_for_escape(
            vertex,
            candidate,
            action,
            quota=graph.quota,
            MoI=graph.MoI,
            simultaneous=simultaneous,
            m=self.m,
        )
        if result is not None:
            return result

        if action == EdgeAction.ELIMINATE:
            raise ValueError(
                "Elimination escape edge is incoherent: candidate is already "
                f"the lowest hopeful. candidate={candidate}, "
                f"tallies={np.asarray(vertex.tallies, dtype=np.float64)}."
            )
        raise ValueError(
            "Could not identify a noise-filter margin for election escape edge: "
            f"candidate={candidate}, tally={float(vertex.tallies[candidate])}, "
            f"quota={graph.quota}, MoI={graph.MoI}, simultaneous={simultaneous}."
        )

    def add_compiler(
        self,
        compiler: CobraNoiseFilterBase | CobraCompiler,
        info: EscapeCompilerInfo | None = None,
    ) -> None:
        if info is None:
            if not isinstance(compiler, CobraCompiler):
                raise ValueError("info is required when adding a noise-filter compiler.")
            label = getattr(compiler, "compiler_label", None)
            if label is None:
                label = f"MANUAL{len(self.compilers)}"
                compiler.compiler_label = label
            info = EscapeCompilerInfo(
                escape_id=label,
                base_label=compiler.base_vertex_label or "unknown",
                base_layer=int(getattr(compiler.base_vertex.ref, "layer", -1)),
                base_local_id=int(getattr(compiler.base_vertex.ref, "local_id", -1)),
                action=compiler.edge_type,
                candidate=compiler.candidate_index,
                candidate_name=compiler.candidate_strings[compiler.candidate_index],
            )

        self.compilers.append(compiler)
        self.compiler_info.append(info)
        self.compiler_index_by_escape_id[info.escape_id.upper()] = (
            len(self.compilers) - 1
        )

    def _initialize_seeded_graph_compilers(
        self,
        audit_graph: Any,
        remember_discrepancies: bool,
        simultaneous: bool,
        use_fallback: bool,
    ) -> None:
        self._print_seeded_graph_flag(audit_graph)
        if not self.seed_strong_candidates:
            raise ValueError("Seeded audit graph did not expose seed_strong_candidates.")

        strong_vertex = self._find_strong_only_seed_vertex(audit_graph)
        self._validate_no_election_edges_from_strong_vertex(audit_graph, strong_vertex)
        self._initialize_seeded_mentions_data(audit_graph)
        assert self.seed_maximum_possible_tallies is not None
        assert self.seed_prebatch_strong_tallies is not None
        maximum_possible_tallies = self.seed_maximum_possible_tallies
        strong_tallies = self.seed_prebatch_strong_tallies

        for candidate in sorted(self.seed_strong_candidates):
            self._add_compiler(
                audit_graph=audit_graph,
                vertex=strong_vertex,
                candidate=int(candidate),
                action=EdgeAction.ELECT,
                remember_discrepancies=remember_discrepancies,
                simultaneous=simultaneous,
                use_fallback=use_fallback,
            )

        lowest_strong_candidate = min(
            self.seed_strong_candidates,
            key=lambda candidate: strong_tallies[candidate],
        )
        lowest_strong_tally = float(strong_tallies[lowest_strong_candidate])

        for weak_candidate in sorted(self.seed_weak_candidates):
            info = self._make_seed_mentions_compiler_info(
                audit_graph=audit_graph,
                vertex=strong_vertex,
                candidate=int(weak_candidate),
            )
            compiler: CobraNoiseFilterBase | CobraCompiler
            if self.compiler_type == "noise":
                interpreter = self._interpreter_for_vertex(audit_graph, strong_vertex)
                critical_margin = (
                    lowest_strong_tally - float(maximum_possible_tallies[weak_candidate])
                )
                compiler = CobraMentionsNoiseFilterCompiler(
                    interpreter,
                    int(weak_candidate),
                    strong_candidates=self.seed_strong_candidates,
                    maximum_possible_tallies=maximum_possible_tallies,
                    lowest_strong_candidate=int(lowest_strong_candidate),
                    lowest_strong_tally=lowest_strong_tally,
                    MoI=float(audit_graph.MoI),
                    critical_margin=critical_margin,
                    radius=2.0 * critical_margin,
                    alpha=self.alpha,
                    label=info.escape_id,
                )
            else:
                compiler = CobraCompiler(
                    strong_vertex,
                    int(weak_candidate),
                    EdgeAction.ELIMINATE,
                    audit_graph=audit_graph,
                    noise_level_guess=self.noise_guess,
                    simultaneous=simultaneous,
                    remember_discrepancies=remember_discrepancies,
                    use_fallback=use_fallback,
                    compiler_label=info.escape_id,
                    critical_margin_type=CriticalMarginType.CANDIDATE_TO_MENTIONS,
                    strong_candidates=self.seed_strong_candidates,
                    very_strong_candidates=self.seed_very_strong_candidates,
                    maximum_possible_tallies=maximum_possible_tallies,
                    lowest_strong_candidate=int(lowest_strong_candidate),
                    lowest_strong_tally=lowest_strong_tally,
                )
            self.add_compiler(compiler, info)

    def _print_seeded_graph_flag(self, audit_graph: Any) -> None:
        def names(candidates: frozenset[int]) -> list[str]:
            return [audit_graph.candidate_names[candidate] for candidate in sorted(candidates)]

        print("Detected seeded audit graph.")
        print(f"  very strong: {names(self.seed_very_strong_candidates)}")
        print(f"  strong: {names(self.seed_strong_candidates)}")
        print(f"  weak: {names(self.seed_weak_candidates)}")

    def _find_strong_only_seed_vertex(self, audit_graph: Any) -> Any:
        matches = [
            vertex
            for layer in audit_graph.layers
            for vertex in layer
            if (
                hasattr(vertex.key, "hopefuls")
                and frozenset(vertex.key.hopefuls) == self.seed_strong_candidates
            )
        ]
        if not matches:
            raise ValueError(
                "Seeded audit graph does not contain a vertex whose hopefuls are "
                "exactly the seed strong candidates."
            )
        return min(matches, key=lambda vertex: (vertex.ref.layer, vertex.ref.local_id))

    def _validate_no_election_edges_from_strong_vertex(
        self,
        audit_graph: Any,
        strong_vertex: Any,
    ) -> None:
        election_edges = [
            edge
            for edge in audit_graph.outgoing_edges(strong_vertex.ref)
            if (
                EdgeAction(edge.action).is_election
                and not self._is_forced_fill_edge(strong_vertex, edge)
            )
        ]
        if election_edges:
            label = audit_graph.vertex_label(strong_vertex.ref)
            raise ValueError(
                "Strong-only seed vertex has election edges, so strong candidates "
                f"are not safely below quota. vertex={label}, edges={election_edges}"
            )

    def _is_forced_fill_edge(self, vertex: Any, edge: Any) -> bool:
        return (
            EdgeAction(edge.action) == EdgeAction.FORCE_ELECT
            and hasattr(vertex.key, "hopefuls")
            and len(vertex.key.hopefuls) + int(vertex.degree) == self.m
        )

    def _initialize_seeded_mentions_data(self, audit_graph: Any) -> None:
        wt_vec = self._seeded_prebatch_wt_vec_from_graph(audit_graph)
        masked = self.seed_very_strong_candidates
        self.seed_prebatch_strong_tallies = fpv_tallies_from_matrix(
            audit_graph.ballot_matrix,
            wt_vec,
            int(audit_graph.n_candidates),
            masked_candidates=masked,
        )

        graph_mentions = getattr(audit_graph, "seed_maximum_possible_tallies", None)
        if graph_mentions is None:
            self.seed_maximum_possible_tallies = maximum_possible_tallies_from_matrix(
                audit_graph.ballot_matrix,
                wt_vec,
                int(audit_graph.n_candidates),
                self.seed_strong_candidates,
                masked_candidates=masked,
            )
        else:
            self.seed_maximum_possible_tallies = np.asarray(graph_mentions, dtype=np.float64)

    def _seeded_prebatch_wt_vec_from_graph(self, audit_graph: Any) -> NDArray[np.float64]:
        """
        Return the wt_vec after any very-strong/pre-seed elections.

        For seeded WIGM graphs, seeded_build stores this post-election,
        pre-batch-elimination vector in root_wt_vec. This is intentionally
        different from _wt_vec_from_graph, which prefers _unseeded_root_wt_vec
        for sampling from the original ballot population.
        """
        if not hasattr(audit_graph, "root_wt_vec"):
            raise ValueError(
                "Seeded audit graph must expose root_wt_vec for seeded assertions."
            )
        return np.asarray(audit_graph.root_wt_vec, dtype=np.float64)

    def _make_seed_mentions_compiler_info(
        self,
        audit_graph: Any,
        vertex: Any,
        candidate: int,
    ) -> EscapeCompilerInfo:
        base_label = audit_graph.vertex_label(vertex.ref)
        candidate_name = audit_graph.candidate_names[candidate]
        return EscapeCompilerInfo(
            escape_id=f"{base_label}-M{candidate}",
            base_label=base_label,
            base_layer=int(vertex.ref.layer),
            base_local_id=int(vertex.ref.local_id),
            action=EdgeAction.ELIMINATE,
            candidate=int(candidate),
            candidate_name=candidate_name,
        )

    def deduplicate_compilers(self) -> int:
        seen: dict[tuple[Any, ...], int] = {}
        keep_indices: list[int] = []
        removed = 0

        for idx, compiler in enumerate(self.compilers):
            signature = self._compiler_signature(compiler)
            if signature in seen:
                removed += 1
                continue
            seen[signature] = idx
            keep_indices.append(idx)

        if removed == 0:
            return 0

        self.compilers = [self.compilers[idx] for idx in keep_indices]
        self.compiler_info = [self.compiler_info[idx] for idx in keep_indices]
        self.compiler_index_by_escape_id = {
            info.escape_id.upper(): idx
            for idx, info in enumerate(self.compiler_info)
        }
        print(f"Removed {removed} duplicate compilers.")
        return removed

    def _compiler_signature(
        self,
        compiler: CobraNoiseFilterBase | CobraCompiler,
    ) -> tuple[Any, ...]:
        if isinstance(compiler, CobraNoiseFilterCompiler):
            return (
                "noise-filter",
                compiler.base_vertex.ref,
                compiler.c,
                compiler.l,
                round(float(compiler.MoI), 12),
                round(float(compiler.critical_margin), 12),
            )
        if isinstance(compiler, CobraQuotaNoiseFilterCompiler):
            return (
                "quota-noise-filter",
                compiler.base_vertex.ref,
                compiler.candidate,
                compiler.margin_type,
                round(float(compiler.MoI), 12),
                round(float(compiler.critical_margin), 12),
            )
        if isinstance(compiler, CobraMentionsNoiseFilterCompiler):
            return (
                "mentions-noise-filter",
                compiler.base_vertex.ref,
                compiler.weak_candidate,
                compiler.strong_candidate,
                round(float(compiler.MoI), 12),
                round(float(compiler.critical_margin), 12),
            )
        if not isinstance(compiler, CobraCompiler):
            raise TypeError(f"Unsupported compiler type: {type(compiler).__name__}.")
        return (
            compiler.base_vertex_label,
            compiler.critical_margin_type,
            compiler.canonical_victor,
            compiler.canonical_loser,
            compiler.canonical_winner,
            compiler.canonical_non_winner,
            round(float(compiler.critical_margin), 12),
        )

    def _make_compiler_info(
        self,
        audit_graph: Any,
        vertex: Any,
        candidate: int,
        action: EdgeAction,
    ) -> EscapeCompilerInfo:
        base_label = audit_graph.vertex_label(vertex.ref)
        action_code = "E" if action == EdgeAction.ELIMINATE else "W"
        candidate_name = audit_graph.candidate_names[candidate]
        return EscapeCompilerInfo(
            escape_id=f"{base_label}-{action_code}{candidate}",
            base_label=base_label,
            base_layer=int(vertex.ref.layer),
            base_local_id=int(vertex.ref.local_id),
            action=action,
            candidate=int(candidate),
            candidate_name=candidate_name,
        )

    def run(
        self,
        ballot_matrix: NDArray[np.integer] | None = None,
        num_steps: int | None = None,
        reset_compilers: bool = True,
    ) -> bool:
        if reset_compilers:
            self.reset()

        max_i = self.sample_size
        if num_steps is not None:
            max_i = min(max_i, self.i + int(num_steps))

        while self.i < max_i and not self.is_done:
            row_c, row_b = self._sample_pair(self.i, ballot_matrix)
            self.update_pool(row_c, row_b)
            self.i += 1

            if (
                self.print_diagnostics_every > 0
                and self.i > 0
                and self.i % self.print_diagnostics_every == 0
            ):
                self.print_diagnostics()

        finished = self.is_done
        if finished or self.i >= self.sample_size:
            self.print_summary(success=finished)
        return finished

    def _sample_pair(
        self,
        i: int,
        ballot_matrix: NDArray[np.integer] | None,
    ) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
        if self.BAL is not None and self.CVR is not None:
            return self.CVR[i], self.BAL[i]

        if self.sampler is None:
            raise ValueError("Driver has no sampler and no BAL/CVR matrices.")
        if ballot_matrix is None:
            raise ValueError("ballot_matrix is required when using implicit sampling.")
        return self.sampler.sample(i, ballot_matrix)

    def reset(self) -> None:
        for compiler in self.compilers:
            compiler.reset()
        self.pool = list(range(len(self.compilers)))
        self.certified = set()
        self.certification_order = []
        self.i = 0

    def lookup_compiler(self, escape_id: str) -> CobraNoiseFilterBase | CobraCompiler:
        normalized = escape_id.strip().upper()
        try:
            return self.compilers[self.compiler_index_by_escape_id[normalized]]
        except KeyError:
            known = ", ".join(info.escape_id for info in self.compiler_info[:10])
            if len(self.compiler_info) > 10:
                known += ", ..."
            raise KeyError(
                f"Unknown compiler escape_id {escape_id!r}. Known IDs include: {known}"
            ) from None

    @property
    def is_done(self) -> bool:
        if self.keep_certified_compilers:
            return len(self.certified) == len(self.compilers)
        return len(self.pool) == 0

    def update_pool(
        self,
        row_c: NDArray[np.integer],
        row_b: NDArray[np.integer],
    ) -> None:
        active = list(self.pool)
        still_active: list[int] = []

        for idx in active:
            compiler = self.compilers[idx]
            compiler.update(row_c, row_b)
            if compiler.M >= self.threshold:
                if idx not in self.certified:
                    self.certified.add(idx)
                    self.certification_order.append(idx)
                if self.keep_certified_compilers:
                    still_active.append(idx)
            else:
                still_active.append(idx)

        self.pool = still_active

    def print_diagnostics(self, limit: int = 5) -> None:
        remaining = self._remaining_indices()
        print(
            f"Audit diagnostic after {self.i} samples: "
            f"{len(remaining)} compilers remain below threshold "
            f"({len(self.certified)}/{len(self.compilers)} certified)."
        )
        for idx in self._lowest_capital_indices(remaining, limit):
            self._print_compiler_brief(idx)

    def print_summary(self, success: bool) -> None:
        if success:
            print(
                "Audit certified all compilers. "
                f"Final sample size: {self.i}."
            )
            recent = self.certification_order[-3:]
            if recent:
                print("Last compilers to pass threshold:")
                for idx in recent:
                    self._print_compiler_brief(idx)
            return

        print(
            "Audit did not certify all compilers before sample exhaustion. "
            f"Samples used: {self.i}/{self.sample_size}."
        )
        remaining = self._remaining_indices()
        for idx in self._lowest_capital_indices(remaining, 3):
            self._print_compiler_brief(idx)
            compiler = self.compilers[idx]
            if isinstance(compiler, CobraCompiler):
                compiler.print_info()
            else:
                print(compiler.get_info())

    def _remaining_indices(self) -> list[int]:
        if self.keep_certified_compilers:
            return [
                idx
                for idx in range(len(self.compilers))
                if idx not in self.certified
            ]
        return list(self.pool)

    def _lowest_capital_indices(
        self,
        indices: list[int],
        limit: int,
    ) -> list[int]:
        return sorted(indices, key=lambda idx: self.compilers[idx].M)[:limit]

    def _print_compiler_brief(self, idx: int) -> None:
        compiler = self.compilers[idx]
        info = self.compiler_info[idx]
        if not isinstance(compiler, CobraCompiler):
            print(
                f"  {info.escape_id}: M={compiler.M:.6g}, "
                f"escape={info.action.name} {info.candidate}:{info.candidate_name}, "
                f"margin={compiler.critical_margin}, kind={compiler.compiler_kind}"
            )
            return
        print(
            f"  {info.escape_id}: M={compiler.M:.6g}, "
            f"escape={info.action.name} {info.candidate}:{info.candidate_name}, "
            f"margin={compiler.critical_margin_type.value} "
            f"{self._critical_margin_label(compiler)}"
        )

    def _critical_margin_label(self, compiler: CobraCompiler) -> str:
        if compiler.canonical_victor is not None and compiler.canonical_loser is not None:
            return (
                f"{compiler._candidate_label(compiler.canonical_victor)} > "
                f"{compiler._candidate_label(compiler.canonical_loser)}"
            )
        if compiler.canonical_winner is not None:
            return f"{compiler._candidate_label(compiler.canonical_winner)} >= quota"
        if compiler.canonical_non_winner is not None:
            return (
                f"{compiler._candidate_label(compiler.canonical_non_winner)} "
                "< quota"
            )
        return "(unknown)"
