from __future__ import annotations

import math
from typing import Any, Type

import numpy as np

try:
    from ..election_graphs import AbstractGraphConstructor, IncoherentLeafError
    from ..meek_graphs import MeekGraphConstructor
except ImportError:
    from election_graphs import AbstractGraphConstructor, IncoherentLeafError
    from meek_graphs import MeekGraphConstructor


def exponential_search(
    profile: Any,
    *,
    m: int,
    starting_lam: float = 10.0,
    memory_lite: bool = True,
    constructor_cls: Type[AbstractGraphConstructor] = MeekGraphConstructor,
    **constructor_kwargs: Any,
) -> AbstractGraphConstructor:
    """
    Naive exponential LAM search.

    Rebuilds the graph from scratch at each doubled LAM until coherence fails.
    Then restricts the first incoherent graph to the smallest tightest_margin
    among terminal vertices whose winner set differs from the inferred recorded
    winner set, checks coherence again, and returns that restricted constructor.
    """
    if starting_lam < 0:
        raise ValueError("starting_lam must be non-negative.")

    lam = float(starting_lam)

    while True:
        constructor = constructor_cls(
            profile,
            m=m,
            LAM=lam,
            memory_lite=memory_lite,
            **constructor_kwargs,
        )
        constructor.build()
        constructor.add_natural_edges()
        constructor.assign_tightest_margins()

        if not constructor.coherence_check():
            break

        if lam == 0:
            lam = 1.0
        else:
            lam *= 2.0

    recorded_winner_set = _infer_recorded_winner_set(constructor)
    restriction_lam = _smallest_incoherent_terminal_margin(
        constructor,
        recorded_winner_set,
    )

    constructor.restrict_margin(restriction_lam)
    constructor.coherence_check()

    return constructor


def heap_based_search(
    profile: Any,
    *,
    m: int,
    memory_lite: bool = True,
    constructor_cls: Type[AbstractGraphConstructor] = MeekGraphConstructor,
    verify_output: bool = False,
    allow_lam_at_or_above_half_quota: bool = False,
    **constructor_kwargs: Any,
) -> AbstractGraphConstructor:
    """
    Naive next-margin expansion search.

    Starts at LAM 1, repeatedly expands to the smallest stored next_margin in
    the current graph, and stops when coherence fails. It then rebuilds the
    returned graph from scratch at floor(smallest incoherent terminal margin),
    avoiding stale edge semantics from earlier LAM values. By default, the
    search stops at the largest integer LAM strictly below half the election
    quota; set ``allow_lam_at_or_above_half_quota`` to retain the unrestricted
    search behavior. When ``verify_output`` is true, a second fresh graph is
    built and compared after normalizing allocation-order-dependent references.
    """
    constructor = constructor_cls(
        profile,
        m=m,
        LAM=1.0,
        memory_lite=memory_lite,
        **constructor_kwargs,
    )
    constructor.build()
    constructor.add_natural_edges()
    constructor.assign_tightest_margins()

    half_quota = None
    maximum_lam = None
    if not allow_lam_at_or_above_half_quota:
        quota = _search_quota(constructor)
        half_quota = quota / 2.0
        maximum_lam = float(math.ceil(half_quota) - 1)
        if maximum_lam < 0:
            raise ValueError(
                "heap_based_search requires a non-negative integer LAM "
                f"strictly below q/2, but q={quota}."
            )

        # The historical initial LAM is 1. Small synthetic elections can have
        # q/2 <= 1, in which case immediately rebuild at the only admissible
        # integer margin rather than searching from an invalid starting point.
        if float(constructor.LAM) >= half_quota:
            constructor = _fresh_graph_at_lam(
                profile=profile,
                m=m,
                lam=maximum_lam,
                memory_lite=memory_lite,
                constructor_cls=constructor_cls,
                constructor_kwargs=constructor_kwargs,
            )
            if verify_output:
                _verify_heap_output_against_fresh_build(
                    constructor,
                    profile=profile,
                    m=m,
                    memory_lite=memory_lite,
                    constructor_cls=constructor_cls,
                    constructor_kwargs=constructor_kwargs,
                )
            return constructor

    stopped_at_half_quota = False
    while constructor.coherence_check():
        next_lam = _smallest_next_margin(constructor)
        if half_quota is not None and next_lam >= half_quota:
            stopped_at_half_quota = True
            break
        constructor.expand_margin(next_lam)
        constructor.add_natural_edges()
        constructor.assign_tightest_margins()

    if stopped_at_half_quota:
        assert maximum_lam is not None
        restriction_lam = maximum_lam
    else:
        recorded_winner_set = _infer_recorded_winner_set(constructor)
        incoherent_lam = _smallest_incoherent_terminal_margin(
            constructor,
            recorded_winner_set,
        )
        restriction_lam = math.floor(incoherent_lam)

    constructor = _fresh_graph_at_lam(
        profile=profile,
        m=m,
        lam=restriction_lam,
        memory_lite=memory_lite,
        constructor_cls=constructor_cls,
        constructor_kwargs=constructor_kwargs,
    )

    if verify_output:
        _verify_heap_output_against_fresh_build(
            constructor,
            profile=profile,
            m=m,
            memory_lite=memory_lite,
            constructor_cls=constructor_cls,
            constructor_kwargs=constructor_kwargs,
        )

    return constructor


def _search_quota(constructor: AbstractGraphConstructor) -> float:
    """Return the fixed/root quota used to bound heap-search LAM values."""
    quota = getattr(constructor, "quota", None)
    if quota is None:
        root_ref = getattr(constructor, "root_ref", None)
        if root_ref is not None:
            quota = getattr(constructor.vertex(root_ref), "quota", None)

    if quota is None:
        raise ValueError(
            "heap_based_search cannot enforce M < q/2 because the graph "
            "constructor exposes neither a quota nor a root-vertex quota. "
            "Set allow_lam_at_or_above_half_quota=True to override this check."
        )

    quota = float(quota)
    if not math.isfinite(quota) or quota <= 0:
        raise ValueError(
            "heap_based_search requires a finite positive quota to enforce "
            f"M < q/2; received q={quota}."
        )
    return quota


def _verify_heap_output_against_fresh_build(
    heap_graph: AbstractGraphConstructor,
    *,
    profile: Any,
    m: int,
    memory_lite: bool,
    constructor_cls: Type[AbstractGraphConstructor],
    constructor_kwargs: dict[str, Any],
) -> None:
    """Rebuild at the final LAM and compare normalized graph contents."""
    fresh_graph = _fresh_graph_at_lam(
        profile=profile,
        m=m,
        lam=float(heap_graph.LAM),
        memory_lite=memory_lite,
        constructor_cls=constructor_cls,
        constructor_kwargs=constructor_kwargs,
    )
    _assert_graphs_identical(heap_graph, fresh_graph)


def _fresh_graph_at_lam(
    *,
    profile: Any,
    m: int,
    lam: float,
    memory_lite: bool,
    constructor_cls: Type[AbstractGraphConstructor],
    constructor_kwargs: dict[str, Any],
) -> AbstractGraphConstructor:
    graph = constructor_cls(
        profile,
        m=m,
        LAM=float(lam),
        memory_lite=memory_lite,
        **constructor_kwargs,
    )
    graph.build()
    graph.add_natural_edges()
    graph.assign_tightest_margins()
    graph.coherence_check()
    return graph


def _assert_graphs_identical(
    heap_graph: AbstractGraphConstructor,
    fresh_graph: AbstractGraphConstructor,
) -> None:
    """Compare graph semantics independently of allocation-order references."""
    metadata = ("m", "n_candidates", "candidate_names", "simultaneous")
    for attribute in metadata:
        if getattr(heap_graph, attribute, None) != getattr(
            fresh_graph, attribute, None
        ):
            raise AssertionError(
                "Heap verification metadata mismatch for "
                f"{attribute}: {getattr(heap_graph, attribute, None)!r} != "
                f"{getattr(fresh_graph, attribute, None)!r}."
            )
    _assert_close("LAM", heap_graph.LAM, fresh_graph.LAM)
    _assert_close(
        "quota",
        getattr(heap_graph, "quota", None),
        getattr(fresh_graph, "quota", None),
    )

    heap_vertices = _vertices_by_semantic_signature(heap_graph)
    fresh_vertices = _vertices_by_semantic_signature(fresh_graph)
    _assert_same_keys("vertices", heap_vertices, fresh_vertices)
    for signature in heap_vertices:
        _assert_vertices_equal(
            signature,
            heap_vertices[signature],
            fresh_vertices[signature],
        )

    heap_edges = _edges_by_semantic_signature(heap_graph, heap_vertices)
    fresh_edges = _edges_by_semantic_signature(fresh_graph, fresh_vertices)
    _assert_same_keys("edges", heap_edges, fresh_edges)
    for signature in heap_edges:
        _assert_edges_equal(
            signature,
            heap_edges[signature],
            fresh_edges[signature],
        )


def _vertices_by_semantic_signature(
    graph: AbstractGraphConstructor,
) -> dict[tuple[Any, ...], Any]:
    memo: dict[Any, tuple[Any, ...]] = {}

    def vertex_signature(ref: Any) -> tuple[Any, ...]:
        existing = memo.get(ref)
        if existing is not None:
            return existing
        vertex = graph.vertex(ref)
        key = vertex.key
        hopefuls = tuple(sorted(int(candidate) for candidate in key.hopefuls))
        if hasattr(key, "elected_candidates"):
            history: tuple[Any, ...] = (
                "elected",
                tuple(int(candidate) for candidate in key.elected_candidates),
            )
        elif hasattr(key, "seated_at"):
            seated = []
            for edge_ref in key.seated_at:
                if edge_ref is None:
                    seated.append(None)
                    continue
                edge = graph.edge(edge_ref)
                seated.append(
                    (
                        int(edge.action),
                        int(edge.candidate),
                        vertex_signature(edge.src),
                    )
                )
            history = ("seated", tuple(seated))
        else:
            raise AssertionError(
                f"Cannot normalize graph state key {type(key).__name__}."
            )
        signature = (
            type(key).__name__,
            hopefuls,
            history,
            int(getattr(key, "seed_id", -1)),
        )
        memo[ref] = signature
        return signature

    by_signature = {}
    for layer in graph.layers:
        for vertex in layer:
            signature = vertex_signature(vertex.ref)
            if signature in by_signature:
                raise AssertionError(
                    "Graph contains duplicate semantic vertex signature: "
                    f"{signature!r}."
                )
            by_signature[signature] = vertex
    return by_signature


def _edges_by_semantic_signature(
    graph: AbstractGraphConstructor,
    vertices: dict[tuple[Any, ...], Any],
) -> dict[tuple[Any, ...], Any]:
    signature_by_ref = {
        vertex.ref: signature for signature, vertex in vertices.items()
    }
    by_signature = {}
    for layer in graph.edge_layers:
        for edge in layer:
            signature = (
                signature_by_ref[edge.src],
                signature_by_ref[edge.dst],
                int(edge.action),
                int(edge.candidate),
                int(edge.status),
            )
            if signature in by_signature:
                raise AssertionError(
                    "Graph contains duplicate semantic edge signature: "
                    f"{signature!r}."
                )
            by_signature[signature] = edge
    return by_signature


def _assert_same_keys(
    kind: str,
    left: dict[tuple[Any, ...], Any],
    right: dict[tuple[Any, ...], Any],
) -> None:
    left_keys = set(left)
    right_keys = set(right)
    if left_keys == right_keys:
        return
    missing = next(iter(right_keys - left_keys), None)
    extra = next(iter(left_keys - right_keys), None)
    raise AssertionError(
        f"Heap verification {kind} mismatch: "
        f"heap={len(left_keys)}, fresh={len(right_keys)}, "
        f"missing_from_heap={missing!r}, extra_in_heap={extra!r}."
    )


def _assert_vertices_equal(signature: Any, left: Any, right: Any) -> None:
    exact_attributes = ("degree", "status", "color", "path_multiplicity")
    for attribute in exact_attributes:
        if getattr(left, attribute) != getattr(right, attribute):
            raise AssertionError(
                f"Heap verification vertex {signature!r} differs in {attribute}."
            )
    for attribute in ("tightest_margin", "next_margin", "quota"):
        _assert_close(
            f"vertex {signature!r} {attribute}",
            getattr(left, attribute, None),
            getattr(right, attribute, None),
        )
    for attribute in ("tallies", "keep_factors"):
        _assert_array_equal(
            f"vertex {signature!r} {attribute}",
            getattr(left, attribute, None),
            getattr(right, attribute, None),
        )


def _assert_edges_equal(signature: Any, left: Any, right: Any) -> None:
    for attribute in ("margin", "transfer_value"):
        _assert_close(
            f"edge {signature!r} {attribute}",
            getattr(left, attribute, None),
            getattr(right, attribute, None),
        )
    for attribute in ("wt_vec", "fpv_vec"):
        _assert_array_equal(
            f"edge {signature!r} {attribute}",
            getattr(left, attribute, None),
            getattr(right, attribute, None),
        )


def _assert_close(label: str, left: Any, right: Any) -> None:
    if left is None or right is None:
        if left is right:
            return
        raise AssertionError(f"Heap verification {label} mismatch: {left} != {right}.")
    if not math.isclose(
        float(left), float(right), rel_tol=1e-10, abs_tol=1e-8
    ):
        raise AssertionError(f"Heap verification {label} mismatch: {left} != {right}.")


def _assert_array_equal(label: str, left: Any, right: Any) -> None:
    if left is None or right is None:
        if left is right:
            return
        raise AssertionError(
            f"Heap verification {label} mismatch: one array is missing."
        )
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or not np.allclose(
        left_array,
        right_array,
        rtol=1e-10,
        atol=1e-8,
        equal_nan=True,
    ):
        raise AssertionError(f"Heap verification {label} array mismatch.")


def hybrid_search(
    profile: Any,
    *,
    m: int,
    starting_lam: float = 1.0,
    memory_lite: bool = True,
    constructor_cls: Type[AbstractGraphConstructor] = MeekGraphConstructor,
    **constructor_kwargs: Any,
) -> AbstractGraphConstructor:
    """
    Expansion-based hybrid LAM search.

    Starts with trip_when_incoherent enabled, doubles LAM in place with
    expand_margin until construction discovers the first incoherent terminal
    leaf, then restricts back to the last coherent LAM and finishes with the
    naive next-margin heap-style refinement.
    """
    if starting_lam < 0:
        raise ValueError("starting_lam must be non-negative.")

    constructor_kwargs.pop("trip_when_incoherent", None)
    last_coherent_lam: float | None = None

    constructor = constructor_cls(
        profile,
        m=m,
        LAM=float(starting_lam),
        memory_lite=memory_lite,
        trip_when_incoherent=True,
        **constructor_kwargs,
    )

    try:
        constructor.build()

        while True:
            next_lam = 1.0 if constructor.LAM == 0 else 2.0 * constructor.LAM
            last_coherent_lam = float(constructor.LAM)
            constructor.expand_margin(next_lam)
    except IncoherentLeafError:
        constructor.trip_when_incoherent = False
        constructor._pending_incoherent_leaf_error = None
        constructor._terminal_winner_set_tripwire = None

        if last_coherent_lam is not None:
            constructor.restrict_margin(last_coherent_lam)
        else:
            constructor = constructor_cls(
                profile,
                m=m,
                LAM=0.0,
                memory_lite=memory_lite,
                trip_when_incoherent=False,
                **constructor_kwargs,
            )
            constructor.build()

    return _heap_refinement_from_constructor(constructor)


def _heap_refinement_from_constructor(
    constructor: AbstractGraphConstructor,
) -> AbstractGraphConstructor:
    constructor.trip_when_incoherent = False
    constructor.add_natural_edges()
    constructor.assign_tightest_margins()

    while constructor.coherence_check():
        next_lam = _smallest_next_margin(constructor)
        constructor.expand_margin(next_lam)
        constructor.add_natural_edges()
        constructor.assign_tightest_margins()

    recorded_winner_set = _infer_recorded_winner_set(constructor)
    incoherent_lam = _smallest_incoherent_terminal_margin(
        constructor,
        recorded_winner_set,
    )
    restriction_lam = math.floor(incoherent_lam)

    constructor.restrict_margin(restriction_lam)
    constructor.coherence_check()

    return constructor


def _smallest_next_margin(
    constructor: AbstractGraphConstructor,
) -> float:
    margins = [
        vertex.next_margin
        for layer in constructor.layers
        for vertex in layer
        if vertex.next_margin is not None
        and vertex.next_margin > constructor.LAM
    ]

    if not margins:
        raise ValueError("No next_margin available before coherence failed.")

    return float(min(margins))


def _infer_recorded_winner_set(
    constructor: AbstractGraphConstructor,
) -> frozenset[int]:
    best_winner_set = None
    best_margin = float("inf")

    for winner_set, refs in constructor.terminal_vertices_by_winner_set.items():
        margins = [
            constructor.vertex(ref).tightest_margin
            for ref in refs
            if constructor.vertex(ref).tightest_margin is not None
        ]

        if not margins:
            continue

        winner_set_margin = min(margins)
        if winner_set_margin < best_margin:
            best_margin = winner_set_margin
            best_winner_set = winner_set

    if best_winner_set is None:
        raise ValueError("Could not infer recorded winner set.")

    return best_winner_set


def _smallest_incoherent_terminal_margin(
    constructor: AbstractGraphConstructor,
    recorded_winner_set: frozenset[int],
) -> float:
    best_margin = float("inf")

    for winner_set, refs in constructor.terminal_vertices_by_winner_set.items():
        if winner_set == recorded_winner_set:
            continue

        for ref in refs:
            margin = constructor.vertex(ref).tightest_margin
            if margin is not None and margin < best_margin:
                best_margin = margin

    if best_margin == float("inf"):
        raise ValueError("No incoherent terminal vertex found.")

    return float(best_margin)
