"""
Accuracy measurement layer for the ground-truth benchmark.

Pure *measurement*: this module never calls the CV pipeline and never
modifies production behavior.  It consumes already-produced pipeline
results (reconstruction activities/events, arrow geometry, graph edges)
and compares them against v1.0 ground-truth annotations.

Matching strategy (documented)
-------------------------------
- AON nodes / AOA events are matched geometrically, never by array
  position and never by OCR label alone:
    1. A Greedy best-first assignment over all (gt, detected) pairs.
    2. Score: bounding-box IoU when both bboxes exist, else centre
       distance.  A pair is admissible when IoU >= ``iou_threshold``
       OR (no bbox on either side AND centre distance <= tolerance).
       Without any geometry on either side the node is treated as
       count-only (``N/A`` for precision/recall).
    3. Each node is assigned at most once (one-to-one).
- Dependencies are compared as normalized directed pairs ``(source, target)``.
  Detected edges are re-mapped onto ground-truth node ids via the node
  match map, so duplicates/visual fragments collapse into one logical edge.
- Direction is tracked separately from the undirected pair:
  correct pair + reversed direction counts as ``pair correct, direction wrong``.
- Durations are never silently rounded: exact match uses full float
  equality; absolute and relative error are reported unchanged.
- Any metric whose annotation (or detection geometry) cannot support it
  is reported as ``N/A`` (``None``), never fabricated.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

METRIC_NA = None

# Default geometric tolerances (pixels in original image space).
DEFAULT_IOU_THRESHOLD = 0.20
DEFAULT_CENTER_TOLERANCE = 40.0
ARROW_EVENT_MARGIN = 15.0  # extra px beyond an event circle radius


def _center(item_geometry: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if not item_geometry:
        return None
    pos = item_geometry.get("position")
    if pos is not None and len(pos) >= 2:
        return (float(pos[0]), float(pos[1]))
    bbox = item_geometry.get("bbox")
    if bbox is not None and len(bbox) >= 2:
        cx = float(bbox[0]) + float(bbox[2]) / 2.0
        cy = float(bbox[1]) + float(bbox[3]) / 2.0
        return (cx, cy)
    return None


def _diagonal(item_geometry: Optional[Dict[str, Any]]) -> Optional[float]:
    if not item_geometry:
        return None
    bbox = item_geometry.get("bbox")
    if bbox is not None and len(bbox) >= 4:
        w = float(bbox[2])
        h = float(bbox[3])
        if w > 0 and h > 0:
            return math.hypot(w, h)
    pos = item_geometry.get("position")
    bbox2 = item_geometry.get("bbox")
    if pos is None and bbox2 is not None and len(bbox2) >= 4:
        return math.hypot(float(bbox2[2]), float(bbox2[3]))
    return None


def _iou(bbox_a: List[float], bbox_b: List[float]) -> float:
    ax, ay, aw, ah = (float(v) for v in bbox_a[:4])
    bx, by, bw, bh = (float(v) for v in bbox_b[:4])
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_nodes(
    gt_geometries: Dict[str, Dict[str, Any]],
    det_items: Sequence[Any],
    det_id_fn,
    det_geometry_fn,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    center_tolerance: float = DEFAULT_CENTER_TOLERANCE,
) -> Dict[str, Any]:
    """Greedy one-to-one geometric node matching.

    ``det_items`` are the detected nodes; ``det_id_fn`` and
    ``det_geometry_fn`` extract id (str) and a geometry dict
    (``{"position": [x, y], "bbox": [x, y, w, h]}``) from each.

    Returns::

        {
          "matched": {gt_id: det_id},
          "gt_unmatched": [gt_id, ...],
          "det_unmatched": [det_id, ...],
          "countable": True/False   # False when geometry is absent everywhere
        }
    """
    gt_ids = list(gt_geometries.keys())
    det_ids = [det_id_fn(item) for item in det_items]
    gt_centers = {gid: _center(gt_geometries.get(gid)) for gid in gt_ids}
    det_geoms = {did: det_geometry_fn(item) for did, item in zip(det_ids, det_items)}

    candidates: List[Tuple[float, float, str, str]] = []
    geometry_present = False
    for gid in gt_ids:
        if gt_centers[gid] is None and not (gt_geometries.get(gid) or {}).get("bbox"):
            continue
        for did in det_ids:
            dgeom = det_geoms.get(did) or {}
            dcenter = _center(dgeom)
            if dcenter is None and not dgeom.get("bbox"):
                continue
            geometry_present = True

            gbbox = gt_geometries[gid].get("bbox")
            dbbox = dgeom.get("bbox") or dgeom.get("bounding_box")
            score: Optional[float] = None
            admissible = False
            dist = math.inf
            if gbbox and dbbox and len(gbbox) >= 4 and len(dbbox) >= 4:
                score = _iou(gbbox, dbbox)
                admissible = score >= iou_threshold
            if not admissible:
                gc = gt_centers[gid] or _center(gt_geometries[gid])
                dc = dcenter
                if gc is None or dc is None:
                    continue
                dist = math.hypot(gc[0] - dc[0], gc[1] - dc[1])
                diag = _diagonal(gt_geometries.get(gid))
                ddiag = _diagonal(dgeom)
                reference = max([v for v in (diag, ddiag) if v is not None] or [1.0])
                admissible = dist <= max(center_tolerance, 0.5 * reference)
                if admissible:
                    score = 1.0 - (dist / max(reference, 1e-6))
            if admissible:
                candidates.append((-1.0 if score is None else -score, dist, gid, did))

    if not geometry_present:
        return {
            "matched": {},
            "gt_unmatched": gt_ids,
            "det_unmatched": det_ids,
            "countable": False,
        }

    candidates.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
    assigned_gt: set = set()
    assigned_det: set = set()
    matched: Dict[str, str] = {}
    for _, _, gid, did in candidates:
        if gid in assigned_gt or did in assigned_det:
            continue
        assigned_gt.add(gid)
        assigned_det.add(did)
        matched[gid] = did
    return {
        "matched": matched,
        "gt_unmatched": [g for g in gt_ids if g not in matched],
        "det_unmatched": [d for d in det_ids if d not in matched.values()],
        "countable": True,
    }


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and prec + rec > 0) else None
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


def compare_durations(
    matched_map: Dict[str, str],
    gt_durations: Dict[str, float],
    det_durations: Dict[str, float],
) -> Dict[str, Any]:
    """Compare durations over matched nodes (exact/absolute/relative)."""
    exact_total = 0
    abs_errors: List[float] = []
    rel_errors: List[float] = []
    compared = 0
    for gid, did in matched_map.items():
        if gid not in gt_durations:
            continue
        det = det_durations.get(did)
        if det is None:
            continue
        g = float(gt_durations[gid])
        d = float(det)
        compared += 1
        if abs(g - d) < 1e-9:
            exact_total += 1
        abs_errors.append(abs(g - d))
        if g != 0.0:
            rel_errors.append(abs(g - d) / abs(g))
    out: Dict[str, Any] = {
        "compared": compared,
        "expected_with_duration": len(gt_durations),
    }
    if compared == 0:
        out.update({"exact_match_accuracy": None, "mean_abs_error": None,
                    "mean_relative_error": None, "exact_matches": 0})
        return out
    out["exact_matches"] = exact_total
    out["exact_match_accuracy"] = round(exact_total / compared, 4)
    out["mean_abs_error"] = round(sum(abs_errors) / len(abs_errors), 4)
    out["mean_relative_error"] = (
        round(sum(rel_errors) / len(rel_errors), 4) if rel_errors else None
    )
    return out


def compare_dependencies(
    det_edges: List[Tuple[str, str]],
    gt_edges: List[Tuple[str, str]],
    node_map: Dict[str, str],
) -> Dict[str, Any]:
    """Compare detected directed edges against ground-truth edges.

    ``node_map`` maps detected node ids -> ground-truth node ids.  Edges
    whose endpoints both map are "comparable" and feed precision/recall;
    edges with an unmapped endpoint are counted as extraneous
    (false positives), and logged separately.
    """
    gt_set = {gt_edges} if isinstance(gt_edges, tuple) else set(gt_edges)

    comparable: List[Tuple[str, str]] = []
    incomparable = 0
    for s, t in det_edges:
        gs = node_map.get(s, s)
        gt_ = node_map.get(t, t)
        if gs is None or gt_ is None:
            incomparable += 1
            continue
        if gs not in {a for e in gt_set for a in e} and gt_ not in {a for e in gt_set for a in e}:
            # Edge endpoints still reference real gt nodes even if not matched?
            # node_map always maps through; treat as comparable only if both
            # sides landed in the gt node universe.
            pass
        comparable.append((gs, gt_))

    # Deterministic de-dup of detected edges (visual fragments already
    # collapsed upstream; keep set semantics here as a safety net).
    comparable = list(dict.fromkeys(comparable))
    det_set = set(comparable)

    directed_tp = len(gt_set & det_set)
    fp = len(det_set - gt_set)
    fn = len(gt_set - det_set)

    gt_undirected = {frozenset(e) for e in gt_set}
    det_undirected = {frozenset(e) for e in det_set}
    pair_tp = len(gt_undirected & det_undirected)
    pair_fp = len(det_undirected - gt_undirected)
    pair_fn = len(gt_undirected - det_undirected)

    direction_wrong = 0
    direction_ok = 0
    for pair in gt_undirected & det_undirected:
        if len(pair) != 2:
            # Degenerate self-loop edge (x -> x): a frozenset collapses to a
            # single element, so there is no src/tgt direction to compare.
            element = next(iter(pair))
            if (element, element) in gt_set and (element, element) in det_set:
                direction_ok += 1
            else:
                direction_wrong += 1
            continue
        src, tgt = tuple(pair)
        if (src, tgt) in gt_set:
            expected = (src, tgt)
        else:
            expected = (tgt, src)
        found = [e for e in det_set if frozenset(e) == pair]
        ok = any(e == expected for e in found)
        if ok:
            direction_ok += 1
        else:
            direction_wrong += 1

    out: Dict[str, Any] = {
        "expected_edges": len(gt_set),
        "detected_edges": len(det_set),
        "incomparable_detected_edges": incomparable,
        "directed": precision_recall_f1(directed_tp, fp, fn),
        "undirected_pair": precision_recall_f1(pair_tp, pair_fp, pair_fn),
    }
    matched_pairs = direction_ok + direction_wrong
    out["direction"] = {
        "correct_pairs": direction_ok,
        "reversed_pairs": direction_wrong,
        "direction_accuracy": (
            round(direction_ok / matched_pairs, 4) if matched_pairs else None
        ),
    }
    return out


def _id_accuracy(matched_map: Dict[str, str], gt_ids: List[str],
                 det_ids: Dict[str, str]) -> Dict[str, Any]:
    """OCR/semantic id accuracy over matched nodes."""
    correct = 0
    total = 0
    for gid, did in matched_map.items():
        g = (gid or "").strip()
        d = (det_ids.get(did) or "").strip()
        if not g:
            continue
        total += 1
        if g.lower() == d.lower():
            correct += 1
    out: Dict[str, Any] = {"compared": total}
    out["id_accuracy"] = round(correct / total, 4) if total else None
    return out


# =============================================================================
# Per-diagram-type comparisons
# =============================================================================


def compare_aon(
    annotation, reconstruction, candidate_edges: List[List[str]],
) -> Dict[str, Any]:
    """AON accuracy: activities, ids, durations, dependencies, graph."""
    out: Dict[str, Any] = {"diagram_type": "AON"}

    gt_geoms = annotation.activity_geometries()
    gt_activities = annotation.comparable_activities()

    det_activities = list(getattr(reconstruction, "activities", None) or [])
    det_durations = {
        act.activity_id or act.geometric_node_id or idx: float(act.duration)
        for idx, act in enumerate(det_activities)
        if act.duration is not None
    }
    det_id_labels = {}
    for act in det_activities:
        label = act.semantic_activity_id or act.activity_id
        # Key by the same id space match_nodes used (activity_id), falling
        # back to the geometric id when no semantic id is present.
        det_id_labels[act.activity_id or act.geometric_node_id] = str(label or "")

    def _geom_fn(act):
        pos = act.position
        bbox = act.bounding_box
        geom: Dict[str, Any] = {}
        if pos is not None:
            geom["position"] = [float(pos[0]), float(pos[1])]
        if bbox is not None and hasattr(bbox, "x"):
            geom["bbox"] = [float(bbox.x), float(bbox.y),
                            float(bbox.width), float(bbox.height)]
        elif bbox is not None and isinstance(bbox, (list, tuple)):
            geom["bbox"] = [float(v) for v in bbox[:4]]
        return geom

    match = match_nodes(gt_geoms, det_activities, lambda a: a.activity_id, _geom_fn)

    gt_total = len(gt_activities)
    det_total = len(det_activities)
    tp = len(match["matched"])
    fp = len(match["det_unmatched"])
    fn = len(match["gt_unmatched"])

    if match["countable"]:
        out["activities"] = precision_recall_f1(tp, fp, fn)
    else:
        out["activities"] = {"countable": False,
                             "expected_activities": gt_total,
                             "detected_activities": det_total}
    out["activity_counts"] = {
        "expected": gt_total,
        "detected": det_total,
        "ratio": round(det_total / gt_total, 4) if gt_total else None,
    }
    out["node_matching"] = {
        "matched": len(match["matched"]),
        "gt_unmatched": match["gt_unmatched"],
        "det_unmatched": match["det_unmatched"],
    }

    # ID accuracy over matched nodes with a real id.
    id_acc = _id_accuracy(match["matched"], [a.id for a in gt_activities],
                          det_id_labels)
    out["ids"] = id_acc

    # Durations.
    gt_durations = annotation.duration_map()
    out["durations"] = compare_durations(match["matched"], gt_durations, det_durations)

    # Dependencies.
    gt_edges = annotation.comparable_dependencies()
    det_edges = [
        (str(edge[0].strip()), str(edge[1].strip()))
        for edge in candidate_edges
        if len(edge) >= 2
    ]
    node_map = {v: k for k, v in match["matched"].items()}
    out["dependencies"] = compare_dependencies(det_edges, gt_edges, node_map)

    out["expected_dependencies"] = len(gt_edges)
    out["detected_dependencies"] = len(det_edges)
    return out


def _associate_arrow_events(
    arrows, events, event_positions: Dict[str, Tuple[float, float]],
    event_radii: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Associate every detected arrow endpoint with the nearest event.

    Returns one entry per arrow: ``{"pair": (src, tgt) | None,
    "unmatched_end": 0|1|2}``.  The head is chosen via the arrow
    direction vector (start->end when direction is start_to_end).
    """
    out: List[Dict[str, Any]] = []
    for arrow in arrows:
        start = getattr(arrow, "start", None)
        end = getattr(arrow, "end", None)
        if start is None or end is None:
            out.append({"pair": None, "unmatched_end": 2})
            continue
        tail = (float(start.x), float(start.y))
        head = (float(end.x), float(end.y))
        dv = getattr(arrow, "direction_vector", None)
        if dv is not None and len(dv) >= 2 and float(dv[0]) < 0:
            tail, head = head, tail
        tail_ev = _nearest_event(tail, event_positions, event_radii)
        head_ev = _nearest_event(head, event_positions, event_radii)
        unmatched = (1 if tail_ev is None else 0) + (1 if head_ev is None else 0)
        pair = (tail_ev, head_ev) if (tail_ev is not None and head_ev is not None) else None
        out.append({"pair": pair, "unmatched_end": unmatched})
    return out


def _nearest_event(
    point: Tuple[float, float],
    event_positions: Dict[str, Tuple[float, float]],
    event_radii: Dict[str, float],
) -> Optional[str]:
    best: Optional[str] = None
    best_dist = math.inf
    for eid, center in event_positions.items():
        radius = event_radii.get(eid, 0.0)
        dist = math.hypot(point[0] - center[0], point[1] - center[1])
        limit = radius * 1.4 + ARROW_EVENT_MARGIN
        if dist <= limit and dist < best_dist:
            best_dist = dist
            best = eid
    return best


def compare_aoa(annotation, reconstruction, arrow_result) -> Dict[str, Any]:
    """AOA accuracy: events, arrows (events-pair), direction, ids."""
    out: Dict[str, Any] = {"diagram_type": "AOA"}

    gt_geoms = annotation.event_geometries()
    gt_events = annotation.comparable_events()
    det_events = list(getattr(reconstruction, "events", None) or [])

    def _geom_fn(evt):
        geom: Dict[str, Any] = {}
        if evt.position is not None:
            geom["position"] = [float(evt.position[0]), float(evt.position[1])]
        bb = getattr(evt, "bounding_box", None)
        if bb is not None and hasattr(bb, "x"):
            geom["bbox"] = [float(bb.x), float(bb.y), float(bb.width), float(bb.height)]
        return geom

    match = match_nodes(gt_geoms, det_events, lambda e: e.event_id, _geom_fn)
    tp = len(match["matched"])
    fp = len(match["det_unmatched"])
    fn = len(match["gt_unmatched"])
    if match["countable"]:
        out["events"] = precision_recall_f1(tp, fp, fn)
    else:
        out["events"] = {"countable": False,
                         "expected_events": len(gt_events),
                         "detected_events": len(det_events)}
    out["event_counts"] = {
        "expected": len(gt_events),
        "detected": len(det_events),
        "ratio": round(len(det_events) / len(gt_events), 4) if gt_events else None,
    }
    out["node_matching"] = {
        "matched": len(match["matched"]),
        "gt_unmatched": match["gt_unmatched"],
        "det_unmatched": match["det_unmatched"],
    }

    # AOA arrows (activities) via event pairs.
    gt_edges = annotation.comparable_dependencies()

    event_positions = {}
    event_radii = {}
    for evt in det_events:
        if evt.position is None:
            continue
        event_positions[evt.event_id] = (
            float(evt.position[0]), float(evt.position[1]))
        bb = getattr(evt, "bounding_box", None)
        radius = 0.0
        if bb is not None:
            radius = (float(bb.width) + float(bb.height)) / 4.0
        event_radii[evt.event_id] = radius

    arrows = list(getattr(arrow_result, "arrows", None) or [])
    associations = _associate_arrow_events(arrows, det_events, event_positions, event_radii)

    evt_map = {v: k for k, v in match["matched"].items()}  # det event id -> gt event id
    arrow_pairs: List[Tuple[str, str]] = []
    unresolved_ends = 0
    for assoc in associations:
        pair = assoc["pair"]
        if not pair:
            unresolved_ends += 1
            continue
        g_src = evt_map.get(pair[0])
        g_tgt = evt_map.get(pair[1])
        if g_src is None or g_tgt is None:
            unresolved_ends += 1
            continue
        arrow_pairs.append((g_src, g_tgt))

    out["arrows"] = compare_dependencies(arrow_pairs, gt_edges, {})
    out["arrow_association"] = {
        "detected_arrows": len(arrows),
        "arrows_mapped_to_event_pairs": len(arrow_pairs),
        "arrows_with_unresolved_event_end": unresolved_ends,
        "expected_arrows": len(gt_edges),
    }
    out["activity_counts"] = {
        "expected": len(annotation.comparable_activities()),
        "detected": len(arrow_pairs),
        "detected_including_unresolved": len(arrows),
    }

    # Durations: N/A unless ground truth carries per-arrow durations and
    # we can deterministically pair arrows (event pair) with an activity.
    gt_durations = annotation.duration_map()
    out["durations"] = {
        "compared": 0,
        "expected_with_duration": len(gt_durations),
        "exact_match_accuracy": None,
        "mean_abs_error": None,
        "mean_relative_error": None,
        "note": "AOA per-arrow durations only measured when ground truth provides them",
    }
    return out


def compute_accuracy(
    annotation: Any, result: Any, candidate_edges: List[List[str]]
) -> Dict[str, Any]:
    """Compute full accuracy metrics for one image.

    ``result`` is the pipeline ``PipelineResult`` exposing
    ``_reconstruction`` and ``_arrow_result``; ``candidate_edges`` are the
    reviewed graph dependency edges (source -> target lists).
    """
    reconstruction = getattr(result, "_reconstruction", None)
    arrow_result = getattr(result, "_arrow_result", None)

    out: Dict[str, Any] = {
        "annotation_status": annotation.annotation_status,
        "annotation_uncertain": annotation.uncertain,
        "notes": annotation.notes,
        "expected_activities": annotation.expected_activities,
        "expected_events": annotation.expected_events,
        "expected_dependencies": annotation.expected_dependencies,
        "excluded": {
            "count": (
                len(annotation.uncertain_activities)
                + len(annotation.uncertain_events)
                + len(annotation.uncertain_dependencies)
            ),
            "activities": len(annotation.uncertain_activities),
            "events": len(annotation.uncertain_events),
            "dependencies": len(annotation.uncertain_dependencies),
            "note": "Uncertain/excluded reference items are never part of metric math.",
        },
    }
    if annotation.annotation_status == "UNCERTAIN":
        # A whole-file UNCERTAIN annotation means no reference item is trusted
        # (the draft may even be seeded from detector output).  Never feed it
        # into accuracy math — that would be circular.
        out["available"] = False
        out["reason"] = (
            "annotation_status=UNCERTAIN (not human-verified); excluded from "
            "accuracy math"
        )
        return out
    if annotation.diagram_type == "AON":
        if reconstruction is None:
            out["available"] = False
            return out
        out["available"] = True
        out.update(compare_aon(annotation, reconstruction, candidate_edges))
    elif annotation.diagram_type == "AOA":
        if reconstruction is None or arrow_result is None:
            out["available"] = False
            return out
        out["available"] = True
        out.update(compare_aoa(annotation, reconstruction, arrow_result))
    else:
        out["available"] = False
    return out