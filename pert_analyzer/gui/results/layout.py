"""
Deterministic AON layout layer for the network canvas.

Computes presentation-only coordinates from the backend graph. Never
modifies GraphModel data; the layout is a view-level concern.

Uses a topological (Kahn) ordering, groups activities by their longest
logical level, and arranges rows deterministically by activity id.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple


NODE_WIDTH = 132
NODE_HEIGHT = 68
H_SPACING = 72
V_SPACING = 28
MARGIN = 40


def build_layout(
    activity_ids: List[str],
    dependencies: List[Any],
    node_width: int = NODE_WIDTH,
    node_height: int = NODE_HEIGHT,
    h_spacing: int = H_SPACING,
    v_spacing: int = V_SPACING,
    margin: int = MARGIN,
) -> Dict[str, Tuple[float, float, float, float]]:
    """Return activity_id -> (x, y, width, height) presentation rects.

    ``dependencies`` accepts objects with ``source``/``target`` attributes
    or (source, target) tuples.
    """
    id_set = set(activity_ids)
    successors: Dict[str, List[str]] = defaultdict(list)
    predecessors: Dict[str, List[str]] = defaultdict(list)
    edge_pairs: List[Tuple[str, str]] = []
    for dep in dependencies:
        if isinstance(dep, tuple):
            src, tgt = dep
        else:
            src = getattr(dep, "source", None)
            tgt = getattr(dep, "target", None)
        if src is None or tgt is None:
            continue
        if src in id_set and tgt in id_set:
            successors[src].append(tgt)
            predecessors[tgt].append(src)
            edge_pairs.append((src, tgt))

    indegree = {aid: len(predecessors.get(aid, [])) for aid in activity_ids}
    ready = deque(sorted(aid for aid in activity_ids if indegree[aid] == 0))
    topo_order: List[str] = []
    while ready:
        aid = ready.popleft()
        topo_order.append(aid)
        for nxt in sorted(successors.get(aid, [])):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    # Any unvisited (disconnected) nodes keep a stable appendix.
    for aid in sorted(activity_ids):
        if aid not in set(topo_order):
            topo_order.append(aid)

    level: Dict[str, int] = {aid: 0 for aid in activity_ids}
    for aid in topo_order:
        for nxt in successors.get(aid, []):
            level[nxt] = max(level[nxt], level[aid] + 1)

    by_level: Dict[int, List[str]] = defaultdict(list)
    for aid in topo_order:
        by_level[level[aid]].append(aid)

    col_step = node_width + h_spacing
    row_step = node_height + v_spacing
    positions: Dict[str, Tuple[float, float, float, float]] = {}
    for lvl in sorted(by_level):
        ids_in_level = sorted(by_level[lvl])
        for row, aid in enumerate(ids_in_level):
            x = margin + lvl * col_step
            y = margin + row * row_step
            positions[aid] = (float(x), float(y), float(node_width), float(node_height))

    return positions


def edge_points(
    positions: Dict[str, Tuple[float, float, float, float]],
    source: str,
    target: str,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return (start, end) points for a source->target dependency arrow."""
    sx, sy, sw, sh = positions[source]
    tx, ty, tw, th = positions[target]
    start = (sx + sw, sy + sh / 2)
    end = (tx, ty + th / 2)
    return start, end