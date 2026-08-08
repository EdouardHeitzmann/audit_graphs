from collections import defaultdict
from itertools import combinations
from math import comb
from typing import Iterable, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray

from .datatypes import (
    EdgeAction,
    EdgeRef,
    EdgeStatus,
    ElectionStatus,
    VertexRef,
)
from .graph_wigm import WIGMGraphConstructor

try:
    from ..election_graphs.utils import (
        maximum_possible_tallies_from_matrix,
        search_strong_weak_candidates,
        weak_candidates_from_strong,
    )
except ImportError:
    from election_graphs.utils import (
        maximum_possible_tallies_from_matrix,
        search_strong_weak_candidates,
        weak_candidates_from_strong,
    )


class SeededWIGMGraphConstructor(WIGMGraphConstructor):
    """
    WIGM constructor with batch elimination via seeded builds.

    ``seeded_build`` abridges the early layers of a plausible graph: very
    strong candidates are pre-seated, a designated weak set is eliminated en
    masse, and construction restarts from every viable seed vertex containing
    the strong candidates plus any subset of the remaining candidates.
    """

    def _initialize_graph_specific_storage(self) -> None:
        super()._initialize_graph_specific_storage()
        self._seed_connector_ref: VertexRef | None = None
        self.used_seeded_build = False
        self.seed_very_strong_candidates = frozenset()
        self.seed_strong_candidates = frozenset()
        self.seed_weak_candidates = frozenset()
        self.seed_maximum_possible_tallies: NDArray[np.float64] | None = None

    def seeded_build(
        self,
        *,
        weak_candidates: Iterable[int] | None = None,
        strong_candidates: Iterable[int] | None = None,
        very_strong_candidates: Iterable[int] | None = None,
        already_elected: Sequence[Sequence[int | str]] | None = None,
        transfer_values: Sequence[Sequence[float]] | None = None,
        simultaneous: bool = True,
        diagnostics: bool = True,
        diagnostic_interval: int = 1000,
    ):
        """
        Build forward from all viable seeds matching candidate filters.

        A seed excludes every weak and already-elected candidate from hopefuls,
        includes every strong candidate in hopefuls, and chooses any subset of
        the remaining candidates as hopeful. Seeds with too few hopeful
        candidates to fill the remaining seats are skipped. When
        ``very_strong_candidates`` is omitted, candidates above ``quota + MoI``
        are detected and pre-seated iteratively, with each surplus transfer
        applied before detecting the next candidate. Pass an empty iterable to
        disable automatic very-strong detection. Seeded construction uses
        simultaneous election semantics by default; pass ``simultaneous=False``
        to request sequential semantics explicitly.
        """
        if diagnostic_interval <= 0:
            raise ValueError("diagnostic_interval must be positive.")
        self.simultaneous = bool(simultaneous)

        weak = (
            None
            if weak_candidates is None
            else frozenset(int(candidate) for candidate in weak_candidates)
        )
        strong = (
            None
            if strong_candidates is None
            else frozenset(int(candidate) for candidate in strong_candidates)
        )
        auto_detect_very_strong = (
            very_strong_candidates is None
            and already_elected is None
            and transfer_values is None
        )
        requested_very_strong = (
            None
            if auto_detect_very_strong
            else frozenset(
                int(candidate)
                for candidate in (
                    ()
                    if very_strong_candidates is None
                    else very_strong_candidates
                )
            )
        )

        if requested_very_strong and already_elected is not None:
            raise ValueError(
                "very_strong_candidates cannot currently be combined with "
                "already_elected; use one pre-seating mechanism at a time."
            )
        if requested_very_strong and transfer_values is not None:
            raise ValueError(
                "transfer_values cannot currently be combined with "
                "very_strong_candidates."
            )

        elected_groups = self._normalize_already_elected(already_elected)
        already_elected_set = frozenset(
            candidate
            for group in elected_groups
            for candidate in group
        ) | (
            frozenset()
            if requested_very_strong is None
            else requested_very_strong
        )
        self._validate_seed_candidate_sets(
            frozenset() if weak is None else weak,
            frozenset() if strong is None else strong,
            already_elected_set,
        )
        self._reset_seeded_graph_storage()
        if diagnostics:
            self._print_seeded_build_diagnostics(
                "initialized",
                detail=(
                    f"memory_lite={self.memory_lite}, "
                    f"candidates={self.n_candidates}, ballots={len(self.root_wt_vec)}"
                ),
            )

        preseeded_very_strong = frozenset()
        if auto_detect_very_strong or requested_very_strong:
            seeded_wt_vec, seated_at, elected_groups = (
                self._initialize_very_strong_seed(requested_very_strong)
            )
            preseeded_very_strong = frozenset(
                candidate
                for group in elected_groups
                for candidate in group
            )
            already_elected_set = frozenset(
                candidate
                for group in elected_groups
                for candidate in group
            )
        else:
            seeded_wt_vec, seated_at = self._initialize_already_elected_seed(
                elected_groups=elected_groups,
                transfer_values=transfer_values,
            )
        self.root_wt_vec = seeded_wt_vec
        initial_degree = len(already_elected_set)
        remaining_seats = self.m - initial_degree
        if diagnostics:
            self._print_seeded_build_diagnostics(
                "pre-seating complete",
                detail=(
                    f"already_elected={len(already_elected_set)}, "
                    f"remaining_seats={remaining_seats}"
                ),
            )

        if strong is None:
            strong, weak, maximum_possible_tallies = search_strong_weak_candidates(
                self.ballot_matrix,
                seeded_wt_vec,
                self.n_candidates,
                remaining_seats=remaining_seats,
                MoI=float(self.MoI),
                quota=float(self.quota),
                masked_candidates=already_elected_set,
            )
        elif weak is None:
            weak, maximum_possible_tallies = weak_candidates_from_strong(
                self.ballot_matrix,
                seeded_wt_vec,
                self.n_candidates,
                strong,
                MoI=float(self.MoI),
                quota=float(self.quota),
                verify_strong=True,
                masked_candidates=already_elected_set,
            )
        else:
            maximum_possible_tallies = maximum_possible_tallies_from_matrix(
                self.ballot_matrix,
                seeded_wt_vec,
                self.n_candidates,
                strong,
                masked_candidates=already_elected_set,
            )

        self._validate_seed_candidate_sets(weak, strong, already_elected_set)
        self.used_seeded_build = True
        self.seed_very_strong_candidates = preseeded_very_strong
        self.seed_strong_candidates = strong
        self.seed_weak_candidates = weak
        self.seed_maximum_possible_tallies = maximum_possible_tallies
        if diagnostics:
            self._print_seeded_build_diagnostics(
                "candidate partition complete",
                detail=f"strong={len(strong)}, weak={len(weak)}",
            )

        if remaining_seats == 0:
            flexible = ()
        else:
            flexible = tuple(
                candidate
                for candidate in range(self.n_candidates)
                if (
                    candidate not in weak
                    and candidate not in strong
                    and candidate not in already_elected_set
                )
            )

        minimum_flexible = max(remaining_seats - len(strong), 0)
        expected_seed_count = sum(
            comb(len(flexible), subset_size)
            for subset_size in range(minimum_flexible, len(flexible) + 1)
        )
        seed_report_interval = max(
            diagnostic_interval,
            max(expected_seed_count // 20, 1),
        )
        if diagnostics:
            self._print_seeded_build_diagnostics(
                "allocating seeds",
                detail=(
                    f"flexible={len(flexible)}, "
                    f"expected_viable_seeds={expected_seed_count}"
                ),
            )

        seed_refs: list[VertexRef] = []
        for subset_size in range(len(flexible) + 1):
            for subset in combinations(flexible, subset_size):
                hopefuls = strong | frozenset(subset)
                if len(hopefuls) < remaining_seats:
                    continue

                key = self._make_state_key(
                    seated_at=seated_at,
                    hopefuls=frozenset(hopefuls),
                )
                layer = self.n_candidates - len(hopefuls)
                ref, is_new = self._get_or_create_vertex(
                    layer=layer,
                    key=key,
                    degree=initial_degree,
                )

                if not is_new:
                    continue

                vertex = self.vertex(ref)
                vertex.path_multiplicity = 1
                if not self.memory_lite:
                    self.runtime_cache[ref] = self._materialize_cache_from_state(ref)
                self._enqueue(ref)
                seed_refs.append(ref)
                if (
                    diagnostics
                    and len(seed_refs) % seed_report_interval == 0
                ):
                    self._print_seeded_build_diagnostics(
                        "allocating seeds",
                        detail=(
                            f"created={len(seed_refs)}/{expected_seed_count}"
                        ),
                    )

        if not seed_refs:
            raise ValueError("seeded_build did not produce any viable seed vertices.")

        self._seed_refs = seed_refs
        if diagnostics:
            self._print_seeded_build_diagnostics(
                "seed allocation complete",
                detail=f"created={len(seed_refs)}",
            )
        root_key = self._make_root_key()
        root_local = self.layer_index[0].get(root_key) if self.layers else None
        self.root_ref = None if root_local is None else VertexRef(0, root_local)

        expanded_count = 0
        while self.stack:
            ref = self.stack.pop()
            v = self.vertex(ref)

            if v.status != ElectionStatus.UNEXPANDED:
                continue

            self._expand_vertex(ref)
            expanded_count += 1
            if diagnostics and expanded_count % diagnostic_interval == 0:
                self._print_seeded_build_diagnostics(
                    "expanding graph",
                    detail=f"expanded={expanded_count}",
                )

        if diagnostics:
            self._print_seeded_build_diagnostics(
                "construction complete",
                detail=f"expanded={expanded_count}",
            )

        self.coherence_checked = False
        self.tightest_margins_assigned = False
        self.terminal_vertices_by_winner_set = {}

        return self

    def _print_seeded_build_diagnostics(
        self,
        phase: str,
        *,
        detail: str | None = None,
    ) -> None:
        memory = self._seeded_build_array_memory()
        memory_text = ", ".join(
            f"{name}={size / (1024 ** 2):.1f} MiB"
            for name, size in memory.items()
        )
        rss = self._current_rss_mib()
        rss_text = "unknown" if rss is None else f"{rss:.1f} MiB"
        vertex_count = sum(len(layer) for layer in self.layers)
        edge_count = sum(len(layer) for layer in self.edge_layers)

        suffix = "" if detail is None else f"; {detail}"
        print(
            f"[seeded_build:{phase}] vertices={vertex_count}, edges={edge_count}, "
            f"stack={len(self.stack)}, runtime_caches={len(self.runtime_cache)}, "
            f"rss={rss_text}; {memory_text}{suffix}"
        )
        if (
            phase == "seed allocation complete"
            and not self.memory_lite
            and self.runtime_cache
        ):
            print(
                "[seeded_build:memory-note] Each seed currently retains a "
                "ballot-sized runtime cache. Use memory_lite=True to reconstruct "
                "these caches on demand instead."
            )

    def _seeded_build_array_memory(self) -> dict[str, int]:
        def unique_nbytes(arrays) -> int:
            seen: set[int] = set()
            total = 0
            for array in arrays:
                if not isinstance(array, np.ndarray):
                    continue
                identity = id(array)
                if identity in seen:
                    continue
                seen.add(identity)
                total += int(array.nbytes)
            return total

        profile_arrays = unique_nbytes(
            (self.ballot_matrix, self.root_wt_vec, self._unseeded_root_wt_vec)
        )
        stencil_arrays = unique_nbytes(self.candidate_stencils)
        cache_arrays = unique_nbytes(
            array
            for cache in self.runtime_cache.values()
            for array in (
                cache.bool_ballot_matrix,
                cache.pos_vec,
                cache.fpv_vec,
            )
        )
        edge_arrays = unique_nbytes(
            edge.wt_vec
            for layer in self.edge_layers
            for edge in layer
        )
        tally_arrays = unique_nbytes(
            vertex.tallies
            for layer in self.layers
            for vertex in layer
        )
        return {
            "profile": profile_arrays,
            "stencils": stencil_arrays,
            "caches": cache_arrays,
            "edge_weights": edge_arrays,
            "tallies": tally_arrays,
        }

    def _current_rss_mib(self) -> float | None:
        try:
            with open("/proc/self/status", encoding="ascii") as status_file:
                for line in status_file:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except OSError:
            return None
        return None

    def _initialize_very_strong_seed(
        self,
        very_strong: frozenset[int] | None,
    ) -> tuple[
        NDArray[np.float64],
        tuple[EdgeRef | None, ...],
        tuple[tuple[int, ...], ...],
    ]:
        wt_vec = self._unseeded_root_wt_vec.copy()
        bool_ballot_matrix = np.ones_like(self.ballot_matrix, dtype=np.bool_)
        pos_vec = np.zeros(self.ballot_matrix.shape[0], dtype=np.int8)
        fpv_vec = self.ballot_matrix[np.arange(self.ballot_matrix.shape[0]), pos_vec]
        seated_at: list[EdgeRef | None] = [None] * self.m
        seat_idx = 0
        anchor_ref: VertexRef | None = None
        auto_detect = very_strong is None
        remaining = (
            set(range(self.n_candidates))
            if auto_detect
            else set(very_strong)
        )
        elected_groups: list[tuple[int, ...]] = []

        while remaining and seat_idx < self.m:
            tallies = self._compute_tallies_from_fpv(fpv_vec, wt_vec)
            forced = [
                candidate
                for candidate in sorted(remaining)
                if tallies[candidate] > self.quota + self.MoI
            ]
            if not forced:
                break

            if anchor_ref is None:
                anchor_ref = self._make_already_elected_root_anchor()
            anchor = self.vertex(anchor_ref)
            anchor.tallies = tallies
            anchor.next_margin = self._next_margin_for_vertex(anchor, wt_vec)

            remaining_seats = self.m - seat_idx
            if self.simultaneous:
                ranked_forced = sorted(
                    forced,
                    key=lambda candidate: (-tallies[candidate], candidate),
                )
                if len(ranked_forced) > remaining_seats:
                    warnings.warn(
                        "More forced very-strong candidates than remaining seats; "
                        "pre-seating the highest current tallies only.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                group = tuple(sorted(ranked_forced[:remaining_seats]))
            else:
                highest = max(forced, key=lambda candidate: tallies[candidate])
                near_highest = [
                    candidate
                    for candidate in forced
                    if (
                        candidate != highest
                        and tallies[highest] - tallies[candidate] <= self.MoI
                    )
                ]
                if near_highest:
                    warnings.warn(
                        "Multiple very-strong candidates are forced and within "
                        "MoI of the maximum tally; pre-seating only the highest "
                        f"current tally candidate {highest}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                group = (int(highest),)

            group_transfer_values = self._compute_simultaneous_transfer_values(
                group=group,
                fpv_vec=fpv_vec,
                wt_vec=wt_vec,
            )
            for candidate, transfer_value in zip(group, group_transfer_values):
                wt_vec[fpv_vec == candidate] *= transfer_value

            for candidate in group:
                bool_ballot_matrix &= self.candidate_stencils[candidate]
                remaining.remove(candidate)

            child_ref, edge_refs = self._make_already_elected_group_vertex(
                parent_ref=anchor_ref,
                seated_at=seated_at,
                group=group,
                transfer_values=group_transfer_values,
                wt_vec=wt_vec,
                fpv_vec=fpv_vec,
                degree=seat_idx + len(group),
            )

            for edge_ref in edge_refs:
                seated_at[seat_idx] = edge_ref
                seat_idx += 1

            elected_groups.append(group)
            anchor_ref = child_ref
            pos_vec = bool_ballot_matrix.argmax(axis=1).astype(np.int8)
            fpv_vec = self.ballot_matrix[
                np.arange(self.ballot_matrix.shape[0]),
                pos_vec,
            ]
            child = self.vertex(anchor_ref)
            child.tallies = self._compute_tallies_from_fpv(fpv_vec, wt_vec)
            child.next_margin = self._next_margin_for_vertex(child, wt_vec)

        if remaining and not auto_detect:
            warnings.warn(
                "Some very_strong_candidates were not pre-seated because their "
                f"current tallies did not force election: {sorted(remaining)}.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._seed_connector_ref = anchor_ref
        return wt_vec, tuple(seated_at), tuple(elected_groups)

    def _normalize_seed_elected_groups(
        self,
        *,
        very_strong_candidates: frozenset[int],
        already_elected: Sequence[Sequence[int | str]] | None,
        transfer_values: Sequence[Sequence[float]] | None,
    ) -> tuple[tuple[int, ...], ...]:
        user_groups = self._normalize_already_elected(already_elected)
        if not very_strong_candidates:
            return user_groups

        if transfer_values is not None:
            raise ValueError(
                "transfer_values cannot currently be combined with "
                "very_strong_candidates; pass the corresponding candidates via "
                "already_elected instead."
            )

        if self.simultaneous:
            very_strong_groups = (tuple(sorted(very_strong_candidates)),)
        else:
            very_strong_groups = tuple(
                (candidate,)
                for candidate in sorted(very_strong_candidates)
            )

        overlap = very_strong_candidates & frozenset(
            candidate
            for group in user_groups
            for candidate in group
        )
        if overlap:
            raise ValueError(
                "very_strong_candidates cannot overlap already_elected: "
                f"{sorted(overlap)}"
            )

        return very_strong_groups + user_groups

    def _validate_seed_candidate_sets(
        self,
        weak: frozenset[int],
        strong: frozenset[int],
        already_elected: frozenset[int] = frozenset(),
    ) -> None:
        invalid = sorted(
            candidate
            for candidate in weak | strong | already_elected
            if candidate < 0 or candidate >= self.n_candidates
        )
        if invalid:
            raise ValueError(f"seeded_build has invalid candidate indices: {invalid}")

        overlap = weak & strong
        if overlap:
            raise ValueError(
                "seeded_build candidates cannot be both weak and strong: "
                f"{sorted(overlap)}"
            )

        elected_overlap = already_elected & (weak | strong)
        if elected_overlap:
            raise ValueError(
                "seeded_build already_elected candidates cannot also be weak "
                f"or strong: {sorted(elected_overlap)}"
            )

        if len(already_elected) > self.m:
            raise ValueError(
                "seeded_build already_elected contains more candidates than "
                f"available seats: {len(already_elected)} > {self.m}"
            )

    def _normalize_already_elected(
        self,
        already_elected: Sequence[Sequence[int | str]] | None,
    ) -> tuple[tuple[int, ...], ...]:
        if already_elected is None:
            return ()

        candidate_to_index = {
            name: idx
            for idx, name in enumerate(self.candidate_names)
        }
        seen: set[int] = set()
        groups: list[tuple[int, ...]] = []

        for group in already_elected:
            normalized_group = []
            for candidate in group:
                if isinstance(candidate, str):
                    try:
                        candidate_idx = candidate_to_index[candidate]
                    except KeyError as exc:
                        raise ValueError(
                            f"Unknown already_elected candidate name: {candidate}"
                        ) from exc
                else:
                    candidate_idx = int(candidate)

                if candidate_idx in seen:
                    raise ValueError(
                        "already_elected candidates cannot be repeated: "
                        f"{candidate_idx}"
                    )

                seen.add(candidate_idx)
                normalized_group.append(candidate_idx)

            if not normalized_group:
                raise ValueError("already_elected cannot contain an empty group.")

            groups.append(tuple(normalized_group))

        return tuple(groups)

    def _initialize_already_elected_seed(
        self,
        *,
        elected_groups: tuple[tuple[int, ...], ...],
        transfer_values: Sequence[Sequence[float]] | None,
    ) -> tuple[NDArray[np.float64], tuple[EdgeRef | None, ...]]:
        if (
            transfer_values is not None
            and len(transfer_values) != len(elected_groups)
        ):
            raise ValueError(
                "transfer_values must have the same number of groups as "
                "already_elected."
            )

        wt_vec = self._unseeded_root_wt_vec.copy()
        bool_ballot_matrix = np.ones_like(self.ballot_matrix, dtype=np.bool_)
        pos_vec = np.zeros(self.ballot_matrix.shape[0], dtype=np.int8)
        fpv_vec = self.ballot_matrix[np.arange(self.ballot_matrix.shape[0]), pos_vec]
        seated_at: list[EdgeRef | None] = [None] * self.m
        seat_idx = 0
        anchor_ref: VertexRef | None = None

        for group_idx, group in enumerate(elected_groups):
            current_tallies = self._compute_tallies_from_fpv(fpv_vec, wt_vec)
            if transfer_values is None:
                group_transfer_values = self._compute_simultaneous_transfer_values(
                    group=group,
                    fpv_vec=fpv_vec,
                    wt_vec=wt_vec,
                )
            else:
                raw_group_transfer_values = transfer_values[group_idx]
                if len(raw_group_transfer_values) != len(group):
                    raise ValueError(
                        "Each transfer_values group must match the corresponding "
                        "already_elected group length."
                    )
                group_transfer_values = tuple(
                    float(transfer_value)
                    for transfer_value in raw_group_transfer_values
                )

            for candidate, transfer_value in zip(group, group_transfer_values):
                wt_vec[fpv_vec == candidate] *= transfer_value

            for candidate in group:
                bool_ballot_matrix &= self.candidate_stencils[candidate]

            if anchor_ref is None:
                anchor_ref = self._make_already_elected_root_anchor()
            anchor = self.vertex(anchor_ref)
            anchor.tallies = current_tallies
            anchor.next_margin = self._next_margin_for_vertex(anchor, wt_vec)

            child_ref, edge_refs = self._make_already_elected_group_vertex(
                parent_ref=anchor_ref,
                seated_at=seated_at,
                group=group,
                transfer_values=group_transfer_values,
                wt_vec=wt_vec,
                fpv_vec=fpv_vec,
                degree=seat_idx + len(group),
            )

            for edge_ref in edge_refs:
                seated_at[seat_idx] = edge_ref
                seat_idx += 1

            anchor_ref = child_ref

            pos_vec = bool_ballot_matrix.argmax(axis=1).astype(np.int8)
            fpv_vec = self.ballot_matrix[
                np.arange(self.ballot_matrix.shape[0]),
                pos_vec,
            ]
            child = self.vertex(anchor_ref)
            child.tallies = self._compute_tallies_from_fpv(fpv_vec, wt_vec)
            child.next_margin = self._next_margin_for_vertex(child, wt_vec)

        self._seed_connector_ref = anchor_ref

        return wt_vec, tuple(seated_at)

    def _make_already_elected_root_anchor(self) -> VertexRef:
        anchor_ref, _ = self._get_or_create_vertex(
            layer=0,
            key=self._make_root_key(),
            degree=0,
        )
        anchor = self.vertex(anchor_ref)
        anchor.status = ElectionStatus.EXPANDED
        anchor.path_multiplicity = 0
        return anchor_ref

    def _make_already_elected_group_vertex(
        self,
        *,
        parent_ref: VertexRef,
        seated_at: list[EdgeRef | None],
        group: tuple[int, ...],
        transfer_values: tuple[float, ...],
        wt_vec: NDArray[np.float64],
        fpv_vec: NDArray[np.integer] | None = None,
        degree: int,
    ) -> tuple[VertexRef, list[EdgeRef]]:
        parent = self.vertex(parent_ref)
        next_seated_at = list(seated_at)
        first_edge_ref = self._next_edge_ref(parent_ref.layer)
        edge_refs = [
            EdgeRef(parent_ref.layer, first_edge_ref.local_id + offset)
            for offset in range(len(group))
        ]

        for offset, edge_ref in enumerate(edge_refs):
            next_seated_at[parent.degree + offset] = edge_ref

        child_ref = self._create_vertex_no_dedupe(
            layer=parent_ref.layer + len(group),
            key=self._make_state_key(
                seated_at=tuple(next_seated_at),
                hopefuls=parent.key.hopefuls - set(group),
                parent_key=parent.key,
            ),
            degree=degree,
        )
        child = self.vertex(child_ref)
        child.status = ElectionStatus.EXPANDED
        child.path_multiplicity = 0

        action = (
            EdgeAction.FORCE_ELECT
            if len(group) == 1
            else EdgeAction.SIMULTANEOUS_ELECT
        )

        for candidate, transfer_value, edge_ref in zip(
            group,
            transfer_values,
            edge_refs,
        ):
            edge_ref, _ = self._add_edge(
                src=parent_ref,
                dst=child_ref,
                action=action,
                candidate=candidate,
                status=EdgeStatus.CANONICAL,
                transfer_value=transfer_value,
                wt_vec=wt_vec.copy(),
                fpv_vec=(
                    None
                    if fpv_vec is None or not self.store_fpv_vec
                    else fpv_vec.copy()
                ),
                margin=0.0,
                forced_ref=edge_ref,
                skip_dedupe=True,
            )

        child.path_multiplicity = 0
        return child_ref, edge_refs

    def _compute_simultaneous_transfer_values(
        self,
        *,
        group: tuple[int, ...],
        fpv_vec: NDArray[np.integer],
        wt_vec: NDArray[np.float64],
    ) -> tuple[float, ...]:
        transfer_values = []

        for candidate in group:
            tally = float(wt_vec[fpv_vec == candidate].sum())
            if tally == 0:
                raise ValueError(
                    "Cannot compute transfer value for already_elected candidate "
                    f"{candidate} with zero first-preference tally."
                )

            transfer_values.append(float((tally - self.quota) / tally))

        return tuple(transfer_values)

    def _reset_seeded_graph_storage(self) -> None:
        self.layers = []
        self.edge_layers = []
        self.layer_index = []
        self.edge_by_ref = {}
        self.edge_lookup = {}
        self._outgoing_edge_index = defaultdict(list)
        self._incoming_edge_index = defaultdict(list)
        self.stack.clear()
        self.enqueued = set()
        self._deferred_enqueue_refs = None
        self.runtime_cache = {}
        self.primary_parent_edge = {}
        self.pending_primary_children.clear()
        self.root_ref = None
        self.terminal_vertices_by_winner_set = {}
        self.coherence_checked = False
        self.tightest_margins_assigned = False
        self._terminal_winner_set_tripwire = None
        self._pending_incoherent_leaf_error = None
        self._seed_refs = []
        self._seed_connector_ref = None
        self._initialize_graph_specific_storage()
