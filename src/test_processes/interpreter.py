from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from ..election_graphs.datatypes import (
        SENTINEL,
        EdgeRef,
        ElectionEdge,
        ElectionState,
        VertexRef,
    )
except ImportError:
    from election_graphs.datatypes import (
        SENTINEL,
        EdgeRef,
        ElectionEdge,
        ElectionState,
        VertexRef,
    )


COORDINATE_COLUMNS = {"c": 0, "l": 1, "o": 2}
ThetaKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class VertexCoordinate:
    """One row/column coordinate in the vertex-local t-system."""

    winner_prefix: int
    column: int


def _is_theta_key(value: object) -> bool:
    if not isinstance(value, tuple) or len(value) != 2:
        return False
    return all(isinstance(item, (int, np.integer)) for item in value)


class VertexInterpreter:
    """
    Project profile rows and sampled ballot/CVR rows into a vertex-local
    ``(2**degree, 3)`` t-coordinate system.

    Rows are indexed by the binary winner-prefix mask induced by the vertex's
    seated ancestor edges, in the order those edges appear in
    ``vertex.key.seated_at``. Columns are ``c``, ``l``, and ``o``.
    """

    def __init__(
        self,
        graph: Any,
        vertex: ElectionState | VertexRef | str,
    ) -> None:
        self.graph = graph
        self.vertex = self._resolve_vertex(vertex)
        self.seating_edge_refs = tuple(
            edge_ref for edge_ref in self.vertex.key.seated_at if edge_ref is not None
        )
        if len(self.seating_edge_refs) != self.vertex.degree:
            raise ValueError(
                "Vertex degree does not match the number of seated ancestor edges."
            )
        self.seating_edges = tuple(
            graph.edge(edge_ref) for edge_ref in self.seating_edge_refs
        )
        self.winners = tuple(edge.candidate for edge in self.seating_edges)
        self._prefix_cache: NDArray[np.integer] | None = None
        self._current_fpv_cache: NDArray[np.integer] | None = None
        self._profile_candidate_mass_cache: NDArray[np.float64] | None = None
        self._profile_prefix_mass_cache: NDArray[np.float64] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (1 << self.vertex.degree, 3)

    def winner_prefix_indices(self, *, copy: bool = True) -> NDArray[np.integer]:
        """Return the t-row index of every row in the graph's profile matrix."""
        if self._prefix_cache is not None:
            return self._prefix_cache.copy() if copy else self._prefix_cache

        if self.vertex.degree >= np.iinfo(np.int64).bits - 1:
            raise ValueError(
                "Vertex degree is too large to encode winner prefixes in int64."
            )
        n_rows = self.graph.ballot_matrix.shape[0]
        if not self.seating_edges:
            prefix = np.zeros(n_rows, dtype=np.int64)
        else:
            fpv_matrix = np.stack(
                [self._edge_fpv_vec(edge) for edge in self.seating_edges],
                axis=1,
            )
            winners = np.asarray(self.winners, dtype=fpv_matrix.dtype)
            bit_weights = np.left_shift(
                np.int64(1),
                np.arange(len(self.seating_edges), dtype=np.int64),
            )
            prefix = (fpv_matrix == winners).dot(bit_weights)

        max_prefix = (1 << self.vertex.degree) - 1
        prefix_dtype = np.min_scalar_type(max_prefix)
        self._prefix_cache = prefix.astype(prefix_dtype, copy=False)
        return self._prefix_cache.copy() if copy else self._prefix_cache

    def current_fpv_vec(self, *, copy: bool = True) -> NDArray[np.integer]:
        """Return every profile row's current first hopeful candidate at this vertex."""
        if self._current_fpv_cache is not None:
            return self._current_fpv_cache.copy() if copy else self._current_fpv_cache

        cache = self.graph.runtime_cache.get(self.vertex.ref)
        if cache is None:
            cache = self.graph._materialize_cache_from_state(self.vertex.ref)
        if cache.fpv_vec is None:
            raise ValueError(f"Vertex {self.vertex.ref} has no fpv_vec.")

        self._current_fpv_cache = cache.fpv_vec.copy()
        return self._current_fpv_cache.copy() if copy else self._current_fpv_cache

    def base_point(
        self,
        c: int | str,
        l: int | str,
        *,
        wt_vec: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        Aggregate the graph profile into local t-coordinates for the given
        candidate-to-candidate margin.

        By default this uses the unreweighted CVR profile weights. The partial
        optimizer's margin function then derives transfer weights internally
        from the prefix-row totals. Pass ``wt_vec=...`` only for diagnostics.
        """
        c_idx = self.candidate_index(c)
        l_idx = self.candidate_index(l)
        if c_idx == l_idx:
            raise ValueError("c and l must be distinct candidates.")

        if wt_vec is not None:
            weights = np.asarray(wt_vec, dtype=float)
            fpv = self.current_fpv_vec(copy=False)
            prefixes = self.winner_prefix_indices(copy=False)
            columns = np.full(len(fpv), COORDINATE_COLUMNS["o"], dtype=np.int8)
            columns[fpv == c_idx] = COORDINATE_COLUMNS["c"]
            columns[fpv == l_idx] = COORDINATE_COLUMNS["l"]
            point = np.zeros(self.shape, dtype=np.float64)
            np.add.at(point, (prefixes, columns), weights)
            return point

        candidate_mass, prefix_mass = self.profile_mass_by_prefix_candidate(
            copy=False
        )
        point = np.zeros(self.shape, dtype=np.float64)
        point[:, COORDINATE_COLUMNS["c"]] = candidate_mass[:, c_idx]
        point[:, COORDINATE_COLUMNS["l"]] = candidate_mass[:, l_idx]
        point[:, COORDINATE_COLUMNS["o"]] = (
            prefix_mass - candidate_mass[:, c_idx] - candidate_mass[:, l_idx]
        )
        return point

    def candidate_base_point(
        self,
        candidate: int | str,
        candidate_column: int,
    ) -> NDArray[np.float64]:
        """Build a one-candidate local base point from cached profile masses."""
        candidate_idx = self.candidate_index(candidate)
        column = int(candidate_column)
        if column not in {COORDINATE_COLUMNS["c"], COORDINATE_COLUMNS["l"]}:
            raise ValueError("candidate_column must be the c or l column.")
        candidate_mass, prefix_mass = self.profile_mass_by_prefix_candidate(
            copy=False
        )
        point = np.zeros(self.shape, dtype=np.float64)
        point[:, column] = candidate_mass[:, candidate_idx]
        point[:, COORDINATE_COLUMNS["o"]] = (
            prefix_mass - candidate_mass[:, candidate_idx]
        )
        return point

    def profile_mass_by_prefix_candidate(
        self,
        *,
        copy: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Aggregate profile weight once by winner prefix and current FPV."""
        if (
            self._profile_candidate_mass_cache is None
            or self._profile_prefix_mass_cache is None
        ):
            weights = self.profile_wt_vec()
            fpv = self.current_fpv_vec(copy=False)
            prefixes = self.winner_prefix_indices(copy=False)
            prefix_mass = np.bincount(
                prefixes,
                weights=weights,
                minlength=self.shape[0],
            ).astype(np.float64, copy=False)
            candidate_mass = np.zeros(
                (self.shape[0], int(self.graph.n_candidates)),
                dtype=np.float64,
            )
            valid = (fpv >= 0) & (fpv < int(self.graph.n_candidates))
            np.add.at(
                candidate_mass,
                (prefixes[valid], fpv[valid]),
                weights[valid],
            )
            self._profile_candidate_mass_cache = candidate_mass
            self._profile_prefix_mass_cache = prefix_mass

        if copy:
            return (
                self._profile_candidate_mass_cache.copy(),
                self._profile_prefix_mass_cache.copy(),
            )
        return (
            self._profile_candidate_mass_cache,
            self._profile_prefix_mass_cache,
        )

    def profile_coordinates(
        self,
        c: int | str,
        l: int | str,
    ) -> tuple[VertexCoordinate, ...]:
        """Return the local coordinate of each row of the graph profile."""
        c_idx = self.candidate_index(c)
        l_idx = self.candidate_index(l)
        prefixes = self.winner_prefix_indices(copy=False)
        fpv = self.current_fpv_vec(copy=False)

        columns = np.full(len(fpv), COORDINATE_COLUMNS["o"], dtype=np.int8)
        columns[fpv == c_idx] = COORDINATE_COLUMNS["c"]
        columns[fpv == l_idx] = COORDINATE_COLUMNS["l"]
        return tuple(
            VertexCoordinate(int(prefix), int(column))
            for prefix, column in zip(prefixes, columns)
        )

    def coordinate(
        self,
        row: NDArray[np.integer] | list[int],
        c: int | str,
        l: int | str,
    ) -> VertexCoordinate:
        """Project one ballot row to a local t-coordinate."""
        c_idx = self.candidate_index(c)
        l_idx = self.candidate_index(l)
        ballot_row = np.asarray(row, dtype=int)

        prefix = 0
        for bit, edge in enumerate(self.seating_edges):
            source = self.graph.vertex(edge.src)
            fpv = self._row_fpv(ballot_row, source.key.hopefuls)
            if fpv == edge.candidate:
                prefix |= 1 << bit

        current_fpv = self._row_fpv(ballot_row, self.vertex.key.hopefuls)
        if current_fpv == c_idx:
            column = COORDINATE_COLUMNS["c"]
        elif current_fpv == l_idx:
            column = COORDINATE_COLUMNS["l"]
        else:
            column = COORDINATE_COLUMNS["o"]
        return VertexCoordinate(winner_prefix=prefix, column=column)

    def theta(
        self,
        ballot_row: NDArray[np.integer] | list[int],
        cvr_row: NDArray[np.integer] | list[int],
        c: int | str,
        l: int | str,
    ) -> NDArray[np.float64]:
        """Return ``t(ballot_row) - t(cvr_row)`` in local coordinates."""
        direction = np.zeros(self.shape, dtype=np.float64)
        ballot_coord = self.coordinate(ballot_row, c, l)
        cvr_coord = self.coordinate(cvr_row, c, l)
        direction[ballot_coord.winner_prefix, ballot_coord.column] += 1.0
        direction[cvr_coord.winner_prefix, cvr_coord.column] -= 1.0
        return direction

    def theta_key(
        self,
        ballot_row: NDArray[np.integer] | list[int],
        cvr_row: NDArray[np.integer] | list[int],
        c: int | str,
        l: int | str,
    ) -> ThetaKey:
        """
        Return the canonical compact theta representation.

        The tuple is ``(plus_flat_index, minus_flat_index)``, where the plus
        entry is the sampled ballot coordinate and the minus entry is the CVR
        coordinate under row-major flattening of the ``(2**degree, 3)`` array.
        """
        ballot_coord = self.coordinate(ballot_row, c, l)
        cvr_coord = self.coordinate(cvr_row, c, l)
        return self.theta_key_from_coordinates(ballot_coord, cvr_coord)

    def theta_key_from_coordinates(
        self,
        ballot_coord: VertexCoordinate,
        cvr_coord: VertexCoordinate,
    ) -> ThetaKey:
        """Return the compact theta key for two local coordinates."""
        return (
            self.coordinate_flat_index(ballot_coord),
            self.coordinate_flat_index(cvr_coord),
        )

    def theta_from_key(self, key: ThetaKey) -> NDArray[np.float64]:
        """Expand ``(plus_flat_index, minus_flat_index)`` into a dense theta."""
        plus_index, minus_index = self._normalize_theta_key(key)
        direction = np.zeros(self.shape, dtype=np.float64)
        direction.flat[plus_index] += 1.0
        direction.flat[minus_index] -= 1.0
        return direction

    def profile_wt_vec(self) -> NDArray[np.float64]:
        """
        Return original profile row weights for t-coordinate basepoints.

        Seeded WIGM graphs may mutate ``root_wt_vec`` to a post-preseating
        vector. The transfer recursion represented by a t-basepoint must start
        from the untransferred profile weights, because winner-prefix bits
        encode the earlier transfers explicitly.
        """
        if hasattr(self.graph, "_unseeded_root_wt_vec"):
            return np.asarray(self.graph._unseeded_root_wt_vec, dtype=np.float64)
        return np.asarray(self.graph.root_wt_vec, dtype=np.float64)

    def simultaneous_dead_rows(self) -> tuple[int, ...]:
        """
        Return t-row indices that are structurally zero due to simultaneous winners.

        If two seated winner positions were filled by the same simultaneous
        election edge group, no ballot can have transferred through both before
        the current vertex.
        """
        dead = set()
        for i, left in enumerate(self.seating_edges):
            for j in range(i + 1, len(self.seating_edges)):
                right = self.seating_edges[j]
                if (
                    left.src == right.src
                    and left.dst == right.dst
                    and left.action == right.action
                    and left.action.name == "SIMULTANEOUS_ELECT"
                ):
                    required = (1 << i) | (1 << j)
                    for row in range(self.shape[0]):
                        if row & required == required:
                            dead.add(row)
        return tuple(sorted(dead))

    def coordinate_flat_index(self, coordinate: VertexCoordinate) -> int:
        """Return the row-major flattened index of one local coordinate."""
        if coordinate.winner_prefix < 0 or coordinate.winner_prefix >= self.shape[0]:
            raise ValueError(
                f"winner_prefix must lie in [0, {self.shape[0]}): "
                f"{coordinate.winner_prefix}"
            )
        if coordinate.column < 0 or coordinate.column >= self.shape[1]:
            raise ValueError(
                f"column must lie in [0, {self.shape[1]}): {coordinate.column}"
            )
        return int(coordinate.winner_prefix * self.shape[1] + coordinate.column)

    def candidate_index(self, candidate: int | str) -> int:
        if isinstance(candidate, str):
            try:
                return int(self.graph.candidate_names.index(candidate))
            except ValueError as exc:
                raise ValueError(f"Unknown candidate name: {candidate!r}") from exc

        idx = int(candidate)
        if idx < 0 or idx >= self.graph.n_candidates:
            raise ValueError(f"Candidate index out of range: {idx}")
        return idx

    def _resolve_vertex(self, vertex: ElectionState | VertexRef | str) -> ElectionState:
        if isinstance(vertex, str):
            return self.graph.vertex(self.graph.parse_vertex_label(vertex))
        if isinstance(vertex, VertexRef):
            return self.graph.vertex(vertex)
        return vertex

    def _edge_fpv_vec(self, edge: ElectionEdge) -> NDArray[np.integer]:
        if edge.fpv_vec is not None:
            return edge.fpv_vec

        cache = self.graph.runtime_cache.get(edge.src)
        if cache is None:
            cache = self.graph._materialize_cache_from_state(edge.src)
        if cache.fpv_vec is None:
            raise ValueError(f"Source vertex {edge.src} for edge {edge.ref} has no fpv_vec.")
        return cache.fpv_vec

    def _normalize_theta_key(self, key: ThetaKey) -> ThetaKey:
        if not _is_theta_key(key):
            raise ValueError("theta keys must be a tuple of two integer flat indices.")
        plus_index, minus_index = (int(key[0]), int(key[1]))
        max_index = self.shape[0] * self.shape[1]
        if not (0 <= plus_index < max_index and 0 <= minus_index < max_index):
            raise ValueError(
                f"theta key indices must lie in [0, {max_index}); got {key}."
            )
        return plus_index, minus_index

    @staticmethod
    def _row_fpv(row: NDArray[np.integer], hopefuls: frozenset[int]) -> int:
        for candidate in row:
            idx = int(candidate)
            if idx >= 0 and idx in hopefuls:
                return idx
        return SENTINEL


__all__ = [
    "COORDINATE_COLUMNS",
    "ThetaKey",
    "VertexCoordinate",
    "VertexInterpreter",
]
