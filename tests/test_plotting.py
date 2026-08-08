from types import SimpleNamespace

from src.election_graphs.datatypes import VertexRef
from src.plotting import _compute_layer_positions


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
