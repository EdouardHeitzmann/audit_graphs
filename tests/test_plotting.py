from types import SimpleNamespace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.election_graphs.datatypes import EdgeRef, VertexRef
from src.plotting import (
    _compute_layer_positions,
    plot_vertex_densities,
    plot_wigm_graph,
)


def _vertex(layer: int, local_id: int, margin: float = 0.0):
    return SimpleNamespace(
        ref=VertexRef(layer, local_id),
        tightest_margin=margin,
    )


def test_layer_positions_only_include_visible_vertices():
    left = _vertex(0, 0)
    hidden = _vertex(1, 0)
    right = _vertex(1, 1)
    graph = SimpleNamespace(layers=[[left], [hidden, right]])

    positions = _compute_layer_positions(
        graph,
        visible_refs={left.ref, right.ref},
    )

    assert set(positions) == {left.ref, right.ref}
    assert positions[left.ref][0] == 0.0
    assert positions[right.ref][0] == 0.0


def test_plot_can_highlight_only_root_natural_path(monkeypatch):
    refs = [
        VertexRef(0, 0),
        VertexRef(1, 0),
        VertexRef(1, 1),
        VertexRef(2, 0),
    ]
    vertices = [SimpleNamespace(ref=ref, color=None, tightest_margin=0.0) for ref in refs]
    edges = [
        SimpleNamespace(ref=EdgeRef(0, 0), src=refs[0], dst=refs[1], margin=0.0),
        SimpleNamespace(ref=EdgeRef(0, 1), src=refs[0], dst=refs[2], margin=0.0),
        SimpleNamespace(ref=EdgeRef(1, 0), src=refs[1], dst=refs[3], margin=0.0),
        SimpleNamespace(ref=EdgeRef(1, 1), src=refs[2], dst=refs[3], margin=5.0),
    ]

    class Graph:
        layers = [[vertices[0]], [vertices[1], vertices[2]], [vertices[3]]]
        edge_layers = [[edges[0], edges[1]], [edges[2], edges[3]]]
        root_ref = refs[0]
        m = 1
        n_candidates = 2

        def outgoing_edges(self, ref):
            return [edge for layer in self.edge_layers for edge in layer if edge.src == ref]

    monkeypatch.setattr(plt, "show", lambda: None)

    plot_wigm_graph(
        Graph(),
        label_edges=None,
        highlight_natural_path=True,
    )

    axis = plt.gcf().axes[0]
    red_arrows = [
        text
        for text in axis.texts
        if getattr(text, "arrow_patch", None) is not None
        and text.arrow_patch.get_edgecolor()[:3] == (1.0, 0.0, 0.0)
    ]
    assert len(red_arrows) == 2
    plt.close("all")


def test_plot_vertex_densities_counts_eliminated_vertices(monkeypatch):
    refs = [
        VertexRef(0, 0),
        VertexRef(1, 0),
        VertexRef(1, 1),
        VertexRef(1, 2),
        VertexRef(2, 0),
        VertexRef(2, 1),
    ]
    seat_a_ref = EdgeRef(0, 0)
    vertices = [
        SimpleNamespace(
            ref=refs[0],
            key=SimpleNamespace(seated_at=(None,), hopefuls=frozenset({0, 1, 2})),
        ),
        SimpleNamespace(
            ref=refs[1],
            key=SimpleNamespace(seated_at=(None,), hopefuls=frozenset({0, 1})),
        ),
        SimpleNamespace(
            ref=refs[2],
            key=SimpleNamespace(seated_at=(seat_a_ref,), hopefuls=frozenset({1, 2})),
        ),
        SimpleNamespace(
            ref=refs[3],
            key=SimpleNamespace(seated_at=(None,), hopefuls=frozenset({1, 2})),
        ),
        SimpleNamespace(
            ref=refs[4],
            key=SimpleNamespace(seated_at=(None,), hopefuls=frozenset({1})),
        ),
        SimpleNamespace(
            ref=refs[5],
            key=SimpleNamespace(seated_at=(seat_a_ref,), hopefuls=frozenset({1})),
        ),
    ]

    class Graph:
        candidate_names = ("A", "B", "C")
        n_candidates = 3
        layers = [[vertices[0]], vertices[1:4], vertices[4:6]]

        def edge(self, ref):
            assert ref == seat_a_ref
            return SimpleNamespace(ref=seat_a_ref, candidate=0)

    monkeypatch.setattr(plt, "show", lambda: None)

    ax = plot_vertex_densities(Graph(), ["A"])

    np.testing.assert_allclose(
        [patch.get_height() for patch in ax.patches],
        [0.0, 1.0 / 3.0, 1.0 / 2.0],
    )
    assert [text.get_text() for text in ax.texts] == ["0/1", "1/3", "1/2"]
    plt.close("all")

    ax = plot_vertex_densities(
        Graph(),
        ["A"],
        show_complement_of_eliminated=True,
    )

    np.testing.assert_allclose(
        [patch.get_height() for patch in ax.patches],
        [1.0, 2.0 / 3.0, 1.0 / 2.0],
    )
    assert [text.get_text() for text in ax.texts] == ["1/1", "2/3", "1/2"]
    plt.close("all")
