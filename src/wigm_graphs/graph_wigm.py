from itertools import combinations

from .datatypes import (
    EdgeAction,
    EdgeStatus,
    ElectionStatus,
    ElectionState,
    ElectionEdge,
    StateKey,
    EdgeRef,
    VertexRef,)
from .datatypes import WIGMRuntimeCache as RuntimeCache
try:
    from ..election_graphs import AbstractGraphConstructor, ChildProposal
except ImportError:
    from election_graphs import AbstractGraphConstructor, ChildProposal

import numpy as np
from numpy.typing import NDArray
from typing import Iterable
from collections import defaultdict


class WIGMGraphConstructor(AbstractGraphConstructor):
    def __init__(
        self,
        profile,
        m: int,
        MoI: float,
        *,
        memory_lite: bool = False,
        trip_when_incoherent: bool = False,
        simultaneous: bool = False,
        store_fpv_vec: bool = True,
    ) -> None:
        self.simultaneous = bool(simultaneous)
        self.store_fpv_vec = bool(store_fpv_vec)
        super().__init__(
            profile,
            m,
            MoI,
            memory_lite=memory_lite,
            trip_when_incoherent=trip_when_incoherent,
        )

    # -----------------------------------------------------------------
    # Initial setup
    # -----------------------------------------------------------------

    def _initialize_profile_data(self, profile) -> None:
        if self._has_numpy_profile_arrays(profile):
            self.ballot_matrix, self.root_wt_vec = (
                self._make_padded_numpy_profile_arrays(profile)
            )
        else:
            self.ballot_matrix, self.root_wt_vec = self._make_ballot_matrix(profile)
        self.candidate_stencils = self._make_candidate_stencils(self.ballot_matrix)
        self._unseeded_root_wt_vec = self.root_wt_vec.copy()

        # Droop quota.
        self.quota = np.floor(self.root_wt_vec.sum() / (self.m + 1)) + 1

    def _initialize_graph_specific_storage(self) -> None:
        # Index for post-build natural-edge completion.
        self.same_seated_index: list[
            dict[tuple[EdgeRef | None, ...], set[VertexRef]]
        ] = []

    def _on_new_layer(self, layer: int) -> None:
        self.same_seated_index.append(defaultdict(set))

    def _record_vertex_index(self, layer: int, key: StateKey, ref: VertexRef) -> None:
        self.same_seated_index[layer][self._same_seated_index_key(key)].add(ref)

    def _same_seated_index_key(self, key: StateKey):
        return key.seated_at

    def _make_state_key(
        self,
        *,
        seated_at: tuple[EdgeRef | None, ...],
        hopefuls: frozenset[int],
        parent_key: StateKey | None = None,
    ) -> StateKey:
        return StateKey(seated_at=seated_at, hopefuls=hopefuls)

    def _remap_key_refs(
        self,
        key: StateKey,
        edge_ref_map: dict[EdgeRef, EdgeRef],
    ) -> StateKey:
        seated_at = tuple(
            None if edge_ref is None else edge_ref_map[edge_ref]
            for edge_ref in key.seated_at
        )
        return self._make_state_key(
            seated_at=seated_at,
            hopefuls=key.hopefuls,
            parent_key=key,
        )

    def _state_key_signature(
        self,
        key: StateKey,
    ) -> tuple[tuple[tuple[int, object, int, int] | None, ...], frozenset[int]]:
        seated_edges = tuple(
            None if edge_ref is None else self._seating_edge_signature(edge_ref)
            for edge_ref in key.seated_at
        )
        return seated_edges, key.hopefuls

    def _seating_edge_signature(
        self,
        edge_ref: EdgeRef,
    ) -> tuple[int, object, int, int]:
        edge = self.edge(edge_ref)
        parent = self.vertex(edge.src)
        return (
            edge.src.layer,
            self._state_key_signature(parent.key),
            int(edge.action),
            int(edge.candidate),
        )

    def _make_ballot_matrix(
        self,
        pf,
    ) -> tuple[NDArray[np.integer], NDArray[np.float64]]:
        df = pf.df.copy()

        candidate_to_index = {
            frozenset([name]): i
            for i, name in enumerate(self.candidate_names)
        }
        candidate_to_index[frozenset(["~"])] = int(-127)

        ranking_columns = [c for c in df.columns if c.startswith("Ranking")]
        num_rows = len(df)
        num_cols = len(ranking_columns)

        if num_cols > len(pf.candidates):
            ranking_columns = ranking_columns[: len(pf.candidates)]
            num_cols = len(ranking_columns)

        cells = df[ranking_columns].to_numpy()

        def map_cell(cell):
            try:
                return candidate_to_index[cell]
            except KeyError:
                raise TypeError(f"Found invalid entry: {cell}")

        mapped = np.frompyfunc(map_cell, 1, 1)(cells).astype(np.int8)

        # Padding gives every row an eventual exhausted / sentinel value.
        ballot_matrix: NDArray = np.full(
            (num_rows, num_cols + 1),
            -127,
            dtype=np.int8,
        )
        ballot_matrix[:, :num_cols] = mapped

        wt_vec: NDArray = df["Weight"].astype(np.float64).to_numpy()

        return ballot_matrix, wt_vec

    def _make_candidate_stencils(
        self,
        ballot_matrix: NDArray[np.integer],
    ) -> list[NDArray[np.bool_]]:
        stencil_list = []

        for idx in range(self.n_candidates):
            stencil = ballot_matrix == idx
            stencil_list.append(~stencil)

        return stencil_list

    def _make_root_key(self) -> StateKey:
        return self._make_state_key(
            seated_at=(None,) * self.m,
            hopefuls=frozenset(range(self.n_candidates)),
        )

    def _make_root_cache(self) -> RuntimeCache:
        bool_ballot_matrix = np.ones_like(self.ballot_matrix, dtype=np.bool_)
        pos_vec = np.zeros(bool_ballot_matrix.shape[0], dtype=np.int8)
        fpv_vec = self.ballot_matrix[np.arange(bool_ballot_matrix.shape[0]), pos_vec]

        return RuntimeCache(
            bool_ballot_matrix=bool_ballot_matrix,
            pos_vec=pos_vec,
            fpv_vec=fpv_vec,
        )

    def _derive_child_cache(
        self,
        parent_cache: RuntimeCache,
        incoming_edge: ElectionEdge,
    ) -> RuntimeCache:
        bool_ballot_matrix = parent_cache.bool_ballot_matrix.copy()
        if incoming_edge.action == EdgeAction.SIMULTANEOUS_ELECT:
            removed_candidates = self._simultaneous_group_for_edge(incoming_edge)
        else:
            removed_candidates = (incoming_edge.candidate,)

        for candidate in removed_candidates:
            bool_ballot_matrix &= self.candidate_stencils[candidate]

        needs_update = np.isin(parent_cache.fpv_vec, removed_candidates)

        pos_vec = parent_cache.pos_vec.copy()
        pos_vec[needs_update] = bool_ballot_matrix[needs_update].argmax(axis=1)

        fpv_vec = self.ballot_matrix[np.arange(bool_ballot_matrix.shape[0]), pos_vec]

        return RuntimeCache(
            bool_ballot_matrix=bool_ballot_matrix,
            pos_vec=pos_vec,
            fpv_vec=fpv_vec,
        )

    def _simultaneous_group_for_edge(
        self,
        incoming_edge: ElectionEdge,
    ) -> tuple[int, ...]:
        if incoming_edge.action != EdgeAction.SIMULTANEOUS_ELECT:
            return (incoming_edge.candidate,)

        group = [
            edge.candidate
            for edge in self.edge_layers[incoming_edge.ref.layer]
            if (
                edge.src == incoming_edge.src
                and edge.dst == incoming_edge.dst
                and edge.action == EdgeAction.SIMULTANEOUS_ELECT
            )
        ]
        if not group:
            raise ValueError(
                f"Could not recover simultaneous election group for {incoming_edge.ref}."
            )

        return tuple(group)

    def _materialize_cache_from_state(self, ref: VertexRef) -> RuntimeCache:
        v = self.vertex(ref)
        bool_ballot_matrix = np.ones_like(self.ballot_matrix, dtype=np.bool_)

        unavailable = set(range(self.n_candidates)) - set(v.key.hopefuls)
        for candidate in unavailable:
            bool_ballot_matrix &= self.candidate_stencils[candidate]

        pos_vec = bool_ballot_matrix.argmax(axis=1).astype(np.int8)
        fpv_vec = self.ballot_matrix[np.arange(bool_ballot_matrix.shape[0]), pos_vec]

        return RuntimeCache(
            bool_ballot_matrix=bool_ballot_matrix,
            pos_vec=pos_vec,
            fpv_vec=fpv_vec,
        )

    # -----------------------------------------------------------------
    # Weight-vector / tally logic
    # -----------------------------------------------------------------

    def _wt_vec_for_vertex(
        self,
        v: ElectionState,
        incoming_edge: ElectionEdge | None,
    ) -> NDArray[np.float64]:
        if incoming_edge is not None and incoming_edge.action.is_election:
            if incoming_edge.wt_vec is None:
                raise ValueError("Election edge is missing wt_vec.")
            return incoming_edge.wt_vec

        latest = self._latest_seating_edge(v.key)

        if latest is None:
            return self.root_wt_vec

        e = self.edge(latest)
        if e.wt_vec is None:
            raise ValueError(f"Seating edge {latest} is missing wt_vec.")

        return e.wt_vec

    def _expansion_context(
        self,
        ref: VertexRef,
        v: ElectionState,
        cache: RuntimeCache,
        incoming_edge: ElectionEdge | None,
    ) -> NDArray[np.float64]:
        return self._wt_vec_for_vertex(v, incoming_edge)

    def _latest_seating_edge(self, key: StateKey) -> EdgeRef | None:
        seating_edges = [e for e in key.seated_at if e is not None]
        if not seating_edges:
            return None

        return max(seating_edges, key=lambda e: (e.layer, e.local_id))

    def _compute_tallies(
        self,
        v: ElectionState,
        cache: RuntimeCache,
        wt_vec: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if cache.fpv_vec is None:
            raise ValueError("cache.fpv_vec has not been computed.")

        not_exhausted_mask = cache.fpv_vec >= 0

        return np.bincount(
            cache.fpv_vec[not_exhausted_mask],
            weights=wt_vec[not_exhausted_mask],
            minlength=self.n_candidates,
        ).astype(np.float64)

    def _compute_tallies_from_fpv(
        self,
        fpv_vec: NDArray[np.integer],
        wt_vec: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        not_exhausted_mask = fpv_vec >= 0
        return np.bincount(
            fpv_vec[not_exhausted_mask],
            weights=wt_vec[not_exhausted_mask],
            minlength=self.n_candidates,
        ).astype(np.float64)

    def _edge_fpv_vec_from_cache(
        self,
        cache: RuntimeCache,
    ) -> NDArray[np.integer] | None:
        if not self.store_fpv_vec:
            return None
        if cache.fpv_vec is None:
            return None
        return cache.fpv_vec.copy()

    def _next_margin_for_vertex(
        self,
        v: ElectionState,
        wt_vec: NDArray[np.float64],
    ) -> float | None:
        if v.tallies is None or v.degree == self.m:
            return None

        margins: list[float] = []

        if self.simultaneous:
            hopefuls = np.array(sorted(v.key.hopefuls), dtype=int)
            if len(hopefuls) == 0:
                return None
            forced = bool(np.any(v.tallies[hopefuls] > self.quota + self.MoI))
            if not forced and len(hopefuls) + v.degree == self.m:
                return None

            # New seating groups appear when a hopeful enters the plausible
            # winner window [quota - MoI, quota + MoI] from either side, or
            # (under seat scarcity) when a pairwise head-to-head gap between
            # hopefuls stops exceeding MoI.
            quota_gaps = np.abs(v.tallies[hopefuls] - self.quota)
            margins.extend(float(gap) for gap in quota_gaps if gap > self.MoI)
            hopeful_tallies = v.tallies[hopefuls]
            pairwise_gaps = hopeful_tallies[:, None] - hopeful_tallies[None, :]
            margins.extend(
                float(gap)
                for gap in np.unique(pairwise_gaps[pairwise_gaps > self.MoI])
            )

            lowest_tally = np.min(v.tallies[hopefuls])
            elimination_margin_floor = float(
                max(np.max(v.tallies) - self.quota, 0.0)
            )
            elimination_margins = np.maximum(
                v.tallies[hopefuls] - lowest_tally,
                elimination_margin_floor,
            )
            margins.extend(float(m) for m in elimination_margins if m > self.MoI)
        elif np.any(v.tallies > self.quota + self.MoI):
            highest_tally = np.max(v.tallies)
            margins.extend(
                float(highest_tally - tally)
                for tally in v.tallies
                if highest_tally - tally > self.MoI
            )
        elif len(v.key.hopefuls) + v.degree == self.m:
            return None
        else:
            highest_tally = np.max(v.tallies)
            election_margins = np.maximum(
                np.maximum(highest_tally, self.quota) - v.tallies,
                0.0,
            )
            margins.extend(float(m) for m in election_margins if m > self.MoI)

            hopefuls = np.array(list(v.key.hopefuls), dtype=int)
            if len(hopefuls) > 0:
                lowest_tally = np.min(v.tallies[hopefuls])
                elimination_margin_floor = float(max(highest_tally - self.quota, 0.0))
                elimination_margins = np.maximum(
                    v.tallies[hopefuls] - lowest_tally,
                    elimination_margin_floor,
                )
                margins.extend(float(m) for m in elimination_margins if m > self.MoI)

        if not margins:
            return None
        return min(margins)

    # -----------------------------------------------------------------
    # Child proposal logic
    # -----------------------------------------------------------------

    def _updated_wt_vec_for_election_group(
        self,
        group: tuple[int, ...],
        cache: RuntimeCache,
        wt_vec: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], tuple[float, ...]]:
        updated_wt_vec = wt_vec.copy()
        transfer_values: list[float] = []

        for candidate in group:
            tally = float(wt_vec[cache.fpv_vec == candidate].sum())
            if tally == 0:
                raise ValueError(
                    f"Cannot compute transfer value for candidate {candidate} "
                    "with zero first-preference tally."
                )
            transfer_value = float((tally - self.quota) / tally)
            transfer_values.append(transfer_value)
            updated_wt_vec[cache.fpv_vec == candidate] *= transfer_value

        return updated_wt_vec, tuple(transfer_values)

    def _election_groups_within_moi(
        self,
        v: ElectionState,
    ) -> Iterable[tuple[int, ...]]:
        """Enumerate the simultaneous seating groups DW(v) + W'.

        The definite winner set DW(v) holds the hopefuls more than MoI above
        quota; the plausible winner set W(v) holds those within MoI of quota.
        One seating edge is proposed for every nonempty DW(v) union W' with
        W' a subset of W(v), provided the group fits in the remaining seats.

        Under seat scarcity -- more plausible winners than remaining seats --
        the plausible edges are computed top-down: a maximal group (seating
        every remaining seat) is excluded when some window candidate left
        out of it beats one of its members by more than MoI, and every
        group DW union S with S a subset of a surviving maximal group is
        then plausible as well.
        """
        if v.tallies is None:
            raise ValueError("Cannot choose election groups before tallies exist.")

        remaining_seats = self.m - v.degree
        if remaining_seats <= 0:
            return ()

        hopefuls = np.array(sorted(v.key.hopefuls), dtype=int)
        if len(hopefuls) == 0:
            return ()

        definite = tuple(
            int(candidate)
            for candidate in hopefuls
            if v.tallies[candidate] > self.quota + self.MoI
        )
        if len(definite) > remaining_seats:
            return ()

        plausible = tuple(
            int(candidate)
            for candidate in hopefuls
            if self.quota - self.MoI <= v.tallies[candidate] <= self.quota + self.MoI
        )

        max_extra = min(len(plausible), remaining_seats - len(definite))

        if len(definite) + len(plausible) <= remaining_seats:
            groups = []
            for extra_size in range(0, max_extra + 1):
                for extra in combinations(plausible, extra_size):
                    group = tuple(sorted(definite + extra))
                    if group:
                        groups.append(group)
            return tuple(groups)

        groups_set: set[tuple[int, ...]] = set()
        for maximal_extra in combinations(plausible, max_extra):
            excluded_tallies = [
                float(v.tallies[candidate])
                for candidate in plausible
                if candidate not in maximal_extra
            ]
            member_min = min(
                float(v.tallies[candidate])
                for candidate in definite + maximal_extra
            )
            if excluded_tallies and max(excluded_tallies) - member_min > self.MoI:
                continue
            for subset_size in range(0, max_extra + 1):
                for subset in combinations(maximal_extra, subset_size):
                    group = tuple(sorted(definite + subset))
                    if group:
                        groups_set.add(group)

        return tuple(sorted(groups_set))

    def _simultaneous_seating_margin(
        self,
        v: ElectionState,
        group: tuple[int, ...],
    ) -> float:
        """Plausibility threshold of a simultaneous seating edge.

        The edge is natural once every member reaches quota and every
        non-member hopeful sits at or below it, so the base threshold is the
        largest member shortfall below quota or non-member excess above it.
        Under seat scarcity at that threshold, the group's best maximal
        completion (adding the strongest excluded candidates until every
        remaining seat is filled) must also survive head-to-head, so the
        threshold grows to the gap by which the strongest candidate still
        left out beats the weakest seated member.
        """
        quota = float(self.quota)
        members = frozenset(group)
        excluded_tallies = sorted(
            (
                float(v.tallies[candidate])
                for candidate in v.key.hopefuls
                if candidate not in members
            ),
            reverse=True,
        )
        shortfall = max(
            (quota - float(v.tallies[candidate]) for candidate in group),
            default=0.0,
        )
        excess = max(
            (tally - quota for tally in excluded_tallies),
            default=0.0,
        )
        base = max(shortfall, excess, 0.0)

        window_count = sum(
            1
            for hopeful in v.key.hopefuls
            if v.tallies[hopeful] >= quota - base
        )
        remaining_seats = self.m - v.degree
        if window_count > remaining_seats and excluded_tallies:
            completion = excluded_tallies[: remaining_seats - len(group)]
            left_out = excluded_tallies[remaining_seats - len(group) :]
            if left_out:
                member_min = min(
                    float(v.tallies[candidate]) for candidate in group
                )
                if completion:
                    member_min = min(member_min, min(completion))
                head_to_head = max(left_out) - member_min
                return max(base, head_to_head, 0.0)

        return base

    def _propose_children(
        self,
        v: ElectionState,
        cache: RuntimeCache,
        wt_vec: NDArray[np.float64],
    ) -> Iterable[ChildProposal]:
        if v.tallies is None:
            raise ValueError("Cannot propose children before tallies are computed.")

        hopefuls = np.array(sorted(v.key.hopefuls), dtype=int)
        if len(hopefuls) == 0:
            return

        # Forced election: someone is safely above quota + MoI.
        if np.any(v.tallies[hopefuls] > self.quota + self.MoI):
            edge_fpv_vec = self._edge_fpv_vec_from_cache(cache)
            if self.simultaneous:
                for group in self._election_groups_within_moi(v):
                    updated_wt_vec, transfer_values = (
                        self._updated_wt_vec_for_election_group(group, cache, wt_vec)
                    )
                    action = (
                        EdgeAction.FORCE_ELECT
                        if len(group) == 1
                        else EdgeAction.SIMULTANEOUS_ELECT
                    )
                    yield ChildProposal(
                        action=action,
                        candidate=int(group[0]),
                        transfer_value=transfer_values[0],
                        transfer_values=transfer_values,
                        wt_vec=updated_wt_vec,
                        fpv_vec=edge_fpv_vec,
                        margin=self._simultaneous_seating_margin(v, group),
                        candidates=group,
                    )
                return

            highest_tally = np.max(v.tallies[hopefuls])
            winner_idx_within_moi = hopefuls[
                np.where(v.tallies[hopefuls] >= highest_tally - self.MoI)[0]
            ]

            for candidate in winner_idx_within_moi:
                group = (int(candidate),)
                updated_wt_vec, transfer_values = (
                    self._updated_wt_vec_for_election_group(group, cache, wt_vec)
                )
                yield ChildProposal(
                    action=EdgeAction.FORCE_ELECT,
                    candidate=int(group[0]),
                    transfer_value=transfer_values[0],
                    transfer_values=transfer_values,
                    wt_vec=updated_wt_vec,
                    fpv_vec=edge_fpv_vec,
                    margin=float(highest_tally - v.tallies[candidate]),
                    candidates=group,
                )

        # All remaining hopefuls must be seated.
        elif len(v.key.hopefuls) + v.degree == self.m:
            edge_fpv_vec = self._edge_fpv_vec_from_cache(cache)
            for candidate in hopefuls:
                updated_wt_vec = wt_vec.copy()

                yield ChildProposal(
                    action=EdgeAction.FORCE_ELECT,
                    candidate=int(candidate),
                    transfer_value=0.0,
                    wt_vec=updated_wt_vec,
                    fpv_vec=edge_fpv_vec,
                    margin=0.0,
                )

        else:
            # Optional election edges for candidates within MoI of highest tally.
            highest_tally = np.max(v.tallies)
            elimination_margin_floor = float(max(highest_tally - self.quota, 0.0))
            if highest_tally >= self.quota - self.MoI:
                edge_fpv_vec = self._edge_fpv_vec_from_cache(cache)
                if self.simultaneous:
                    group_iter = self._election_groups_within_moi(v)
                else:
                    winner_idx_within_moi = hopefuls[
                        np.where(
                            v.tallies[hopefuls]
                            >= max(highest_tally, self.quota) - self.MoI
                        )[0]
                    ]
                    group_iter = tuple((int(candidate),) for candidate in winner_idx_within_moi)

                for group in group_iter:
                    updated_wt_vec, transfer_values = (
                        self._updated_wt_vec_for_election_group(group, cache, wt_vec)
                    )
                    if self.simultaneous:
                        group_margin = self._simultaneous_seating_margin(v, group)
                    else:
                        group_margin = max(
                            max(highest_tally, self.quota) - min(v.tallies[list(group)]),
                            0.0,
                        )
                    action = (
                        EdgeAction.ELECT
                        if len(group) == 1
                        else EdgeAction.SIMULTANEOUS_ELECT
                    )

                    yield ChildProposal(
                        action=action,
                        candidate=int(group[0]),
                        transfer_value=transfer_values[0],
                        transfer_values=transfer_values,
                        wt_vec=updated_wt_vec,
                        fpv_vec=edge_fpv_vec,
                        margin=float(group_margin),
                        candidates=group,
                    )

            # Optional elimination edges for candidates within MoI of lowest tally.
            if len(hopefuls) > 0:
                lowest_tally = np.min(v.tallies[hopefuls])
                loser_idx_within_moi = hopefuls[
                    np.where(v.tallies[hopefuls] <= lowest_tally + self.MoI)[0]
                ]

                for candidate in loser_idx_within_moi:
                    margin = max(
                        v.tallies[candidate] - lowest_tally,
                        elimination_margin_floor,
                    )

                    yield ChildProposal(
                        action=EdgeAction.ELIMINATE,
                        candidate=int(candidate),
                        margin=float(margin),
                    )

    # -----------------------------------------------------------------
    # Child insertion
    # -----------------------------------------------------------------

    def _add_election_child(
        self,
        parent_ref: VertexRef,
        proposal: ChildProposal,
    ) -> VertexRef:
        if proposal.candidates is not None and len(proposal.candidates) > 1:
            return self._add_simultaneous_election_child(parent_ref, proposal)

        parent = self.vertex(parent_ref)
        edge_layer = parent_ref.layer

        # WIGM StateKey stores this incoming EdgeRef, so reserve it first.
        edge_ref = self._next_edge_ref(edge_layer)

        child_key, child_degree = self._make_child_key_and_degree(
            parent=parent,
            proposal=proposal,
            incoming_edge_ref=edge_ref,
        )

        # Election children are guaranteed new for WIGM because the child key
        # includes the reserved incoming seating edge.
        child_ref = self._create_vertex_no_dedupe(
            layer=parent_ref.layer + 1,
            key=child_key,
            degree=child_degree,
        )

        created_edge_ref, _ = self._add_edge(
            src=parent_ref,
            dst=child_ref,
            action=proposal.action,
            candidate=proposal.candidate,
            status=proposal.status,
            transfer_value=proposal.transfer_value,
            wt_vec=proposal.wt_vec,
            fpv_vec=proposal.fpv_vec,
            margin=proposal.margin,
            forced_ref=edge_ref,
            skip_dedupe=True,
        )

        assert created_edge_ref == edge_ref

        self.primary_parent_edge[child_ref] = edge_ref

        if self.vertex(child_ref).status != ElectionStatus.TERMINAL:
            self.pending_primary_children[parent_ref] += 1
            self._enqueue(child_ref)
        else:
            self._check_incoherent_leaf_tripwire(child_ref)

        return child_ref

    def _add_simultaneous_election_child(
        self,
        parent_ref: VertexRef,
        proposal: ChildProposal,
    ) -> VertexRef:
        if proposal.wt_vec is None:
            raise ValueError("Simultaneous election proposal is missing wt_vec.")
        if proposal.candidates is None or len(proposal.candidates) <= 1:
            raise ValueError(
                "Simultaneous election proposal must include multiple candidates."
            )

        parent = self.vertex(parent_ref)
        group = tuple(int(candidate) for candidate in proposal.candidates)
        edge_layer = parent_ref.layer

        first_edge_ref = self._next_edge_ref(edge_layer)
        edge_refs = [
            EdgeRef(edge_layer, first_edge_ref.local_id + offset)
            for offset in range(len(group))
        ]

        new_seated_at = list(parent.key.seated_at)
        for offset, edge_ref in enumerate(edge_refs):
            new_seated_at[parent.degree + offset] = edge_ref

        child_key = self._make_state_key(
            seated_at=tuple(new_seated_at),
            hopefuls=parent.key.hopefuls - set(group),
            parent_key=parent.key,
        )
        child_degree = parent.degree + len(group)
        child_ref = self._create_vertex_no_dedupe(
            layer=parent_ref.layer + len(group),
            key=child_key,
            degree=child_degree,
        )

        transfer_values = self._transfer_values_for_simultaneous_proposal(
            group=group,
            proposal=proposal,
        )

        for candidate, edge_ref, transfer_value in zip(
            group,
            edge_refs,
            transfer_values,
        ):
            created_edge_ref, _ = self._add_edge(
                src=parent_ref,
                dst=child_ref,
                action=EdgeAction.SIMULTANEOUS_ELECT,
                candidate=candidate,
                status=proposal.status,
                transfer_value=transfer_value,
                wt_vec=proposal.wt_vec,
                fpv_vec=proposal.fpv_vec,
                margin=proposal.margin,
                forced_ref=edge_ref,
                skip_dedupe=True,
            )
            assert created_edge_ref == edge_ref

        child = self.vertex(child_ref)
        child.path_multiplicity = parent.path_multiplicity
        self.primary_parent_edge[child_ref] = edge_refs[0]

        if child.status != ElectionStatus.TERMINAL:
            self.pending_primary_children[parent_ref] += 1
            self._enqueue(child_ref)
        else:
            self._check_incoherent_leaf_tripwire(child_ref)

        return child_ref

    def _transfer_values_for_simultaneous_proposal(
        self,
        *,
        group: tuple[int, ...],
        proposal: ChildProposal,
    ) -> tuple[float, ...]:
        if len(group) == 0:
            return ()

        if proposal.transfer_values is None:
            raise ValueError("Simultaneous election proposal is missing transfer values.")

        transfer_values = proposal.transfer_values
        if len(transfer_values) != len(group):
            raise ValueError("Simultaneous transfer value count does not match group.")

        return tuple(float(value) for value in transfer_values)

    def _make_child_key_and_degree(
        self,
        parent: ElectionState,
        proposal: ChildProposal,
        incoming_edge_ref: EdgeRef | None,
    ) -> tuple[StateKey, int]:
        if proposal.action == EdgeAction.ELIMINATE:
            return (
                self._make_state_key(
                    seated_at=parent.key.seated_at,
                    hopefuls=parent.key.hopefuls - {proposal.candidate},
                    parent_key=parent.key,
                ),
                parent.degree,
            )

        if proposal.action.is_election:
            new_seated_at = list(parent.key.seated_at)
            new_seated_at[parent.degree] = incoming_edge_ref

            return (
                self._make_state_key(
                    seated_at=tuple(new_seated_at),
                    hopefuls=parent.key.hopefuls - {proposal.candidate},
                    parent_key=parent.key,
                ),
                parent.degree + 1,
            )

        raise ValueError(f"Unknown proposal action: {proposal.action}")

    # -----------------------------------------------------------------
    # Post-build natural edges and tightest margins
    # -----------------------------------------------------------------

    def add_natural_edges(self) -> int:
        """
        Add missing natural elimination edges between adjacent layers.

        These are parent -> child edges where seated_at agrees and the child
        has exactly one fewer hopeful candidate.

        Since these edges were missed by construction, assign them margin MoI.
        Returns the number of new edges added.
        """
        n_added = 0

        for layer_idx, layer in enumerate(self.layers[:-1]):
            next_layer_idx = layer_idx + 1

            for parent in layer:
                candidates = self.same_seated_index[next_layer_idx].get(
                    self._same_seated_index_key(parent.key),
                    set(),
                )

                for child_ref in candidates:
                    child = self.vertex(child_ref)

                    diff = parent.key.hopefuls ^ child.key.hopefuls
                    if len(diff) != 1:
                        continue

                    if not child.key.hopefuls < parent.key.hopefuls:
                        continue

                    eliminated = int(next(iter(diff)))

                    _, edge_is_new = self._add_edge(
                        src=parent.ref,
                        dst=child_ref,
                        action=EdgeAction.ELIMINATE,
                        candidate=eliminated,
                        status=EdgeStatus.DEFAULT,
                        margin=float(self.MoI),
                    )

                    if edge_is_new:
                        n_added += 1
                        if (
                            parent.status == ElectionStatus.TERMINAL
                            and parent.degree < self.m
                        ):
                            parent.status = ElectionStatus.EXPANDED

        return n_added

    # -----------------------------------------------------------------
    # Post-Construction analysis and utilities
    # -----------------------------------------------------------------

    def _winner_set_for_vertex(self, v: ElectionState) -> frozenset[int]:
        winners = set()

        for edge_ref in v.key.seated_at:
            if edge_ref is None:
                continue

            edge = self.edge(edge_ref)
            winners.add(edge.candidate)

        return frozenset(winners)

    def _insecure_winners_for_vertex(self, v: ElectionState) -> list[int]:
        """Winners of v whose tally at their seating vertex was below quota."""
        insecure = []
        for edge_ref in v.key.seated_at:
            if edge_ref is None:
                continue
            edge = self.edge(edge_ref)
            seating_vertex = self.vertex(edge.src)
            if seating_vertex.tallies is None:
                raise ValueError(
                    "Cannot judge winner security: seating vertex "
                    f"{edge.src} has no computed tallies."
                )
            if float(seating_vertex.tallies[edge.candidate]) < float(self.quota):
                insecure.append(int(edge.candidate))
        return insecure

    def security_check(self) -> bool:
        """
        Check that no non-leaf vertex carries two insecure winners.

        A winner is insecure at v when her tally at her seating vertex
        lambda_v(w) was below quota; a non-leaf vertex with two insecure
        winners is very insecure. Returns True when the graph contains no
        very insecure vertex.

        Also stores:
            self.very_insecure_vertices
        """
        very_insecure: list[VertexRef] = []
        for layer in self.layers:
            for v in layer:
                if v.status == ElectionStatus.TERMINAL:
                    continue
                if len(self._insecure_winners_for_vertex(v)) >= 2:
                    very_insecure.append(v.ref)

        self.very_insecure_vertices = tuple(very_insecure)
        self.security_checked = True

        if not very_insecure:
            print("Security check passed: no very insecure vertices.")
            return True

        print("Security check failed.")
        print(f"  very insecure vertices: {len(very_insecure)}")
        for ref in very_insecure[:10]:
            v = self.vertex(ref)
            names = [
                self.candidate_names[w]
                for w in self._insecure_winners_for_vertex(v)
            ]
            print(f"    {self.vertex_label(ref)}: insecure winners {names}")
        if len(very_insecure) > 10:
            print(f"    ... and {len(very_insecure) - 10} more")
        return False

    def _transfer_value_for_seating_edge(self, edge_ref: EdgeRef) -> float | None:
        """
        Return the transfer value associated with a seating edge.

        Prefer the stored edge.transfer_value. If missing, recompute as
        (tally - quota) / tally using the parent tallies.
        """
        edge = self.edge(edge_ref)

        if edge.transfer_value is not None:
            return edge.transfer_value

        parent = self.vertex(edge.src)

        if parent.tallies is None:
            return None

        tally = parent.tallies[edge.candidate]

        if tally == 0:
            return None

        return float((tally - self.quota) / tally)


    def _elected_summary_for_vertex(self, v: ElectionState) -> list[tuple[int, float | None]]:
        """
        Return [(candidate_index, transfer_value), ...] for candidates already seated
        at this vertex.
        """
        elected = []

        for edge_ref in v.key.seated_at:
            if edge_ref is None:
                continue

            edge = self.edge(edge_ref)
            tv = self._transfer_value_for_seating_edge(edge_ref)
            elected.append((edge.candidate, tv))

        return sorted(elected, key=lambda x: x[0])
