"""M3 layout analyzer — deterministic geometry facts from the DOM snapshot.

The grid is built on **content atoms** (text, images, icons, buttons), not on
structural wrappers, so symmetry and alignment are measured against what the user
actually sees. Atoms are grouped into grid cells by their smallest card/box, and
each cell is measured from its largest card edge, which keeps a card's label+value
pair from being mistaken for two page columns. No saliency is required: these are
geometry facts.

Scope:

* ``document_union`` — every distinct element across the scroll slices, deduped by
  DOM identity, used for columns / symmetry / spacing / alignment. A component
  captured in overlapping slices is counted once, never once per slice.
* ``first_view`` — the slice at ``scroll_y == 0``. Fold clipping and the first-view
  ink balance are measured here only.

A symmetric page is not automatically a good page, and an asymmetric page is not
automatically bad: sidebars, off-centre heros and deliberately ragged text are all
legitimate. What this module comments on is **organization**, and only when the
geometry shows it:

* a text block that ends with one dangling word on its own line (:issue:`ORPHAN_LINE`);
* a repeated row that fills far less of its grid than its sibling rows
  (:issue:`ROW_UNDERFILL`);
* a row that leaves a middle column empty while its siblings fill it (:issue:`GRID_GAP`);
* a row that piles its content on one side while its siblings are spread evenly
  (:issue:`BLOCK_IMBALANCE`);
* a column (or cell) whose edge or width drifts inside a repeated grid
  (:issue:`GRID_MISALIGNMENT`).

Everything is also reported as a metric, and ``layout_notes`` carries a short
factual comment for every measured property, so the grid never "goes quiet" on a
page that happens to be well organized. Every finding ships with ``preserve``;
none of them recommends deleting or hiding content.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, fields
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from crawler import BoundingBox, ElementMetadata, ScreenCapture
from complexity_scorer import FlaggedProblemElement, MEDIA_TAGS


# Landmarks that are naturally taller than the viewport; being cut by the fold is
# not "noise", it is the page continuing.
_FOLD_EXCLUDED_TAGS = {"html", "body", "main", "nav", "header", "footer"}


# --------------------------------------------------------------------------- #
# Content atoms
# --------------------------------------------------------------------------- #
@dataclass
class ContentAtom:
    selector: str
    type: str  # "text" | "image" | "icon" | "button"
    bbox: BoundingBox  # document space (union geometry)
    view_bbox: BoundingBox  # source-slice space (first-view occupancy)
    visible_fraction: float
    tag: str
    text: str
    font_px: float
    dom_id: int
    heading_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selector": self.selector,
            "type": self.type,
            "bbox": self.bbox.to_dict(),
            "visible_fraction": round(self.visible_fraction, 3),
            "tag": self.tag,
            "font_px": self.font_px,
            "heading_level": self.heading_level,
        }


def _font_px(el: ElementMetadata) -> float:
    raw = str((el.computed_style or {}).get("fontSize", "")).replace("px", "").strip()
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _doc_bbox(el: ElementMetadata) -> BoundingBox:
    return el.doc_bbox if el.doc_bbox is not None else el.bbox


def _is_icon(el: ElementMetadata) -> bool:
    tag = el.tag.lower()
    if tag == "svg":
        return True
    marker = f"{el.classes} {el.id} {el.role}".lower()
    if any(k in marker for k in ("icon", "glyph", "fa-")):
        return True
    if el.role == "img":
        return True
    if tag == "i" and el.bbox.width <= 48 and el.bbox.height <= 48:
        return True
    return False


def atom_type(el: ElementMetadata) -> Optional[str]:
    """Classify an element as a content atom, or ``None`` if it is a wrapper."""
    if el.is_interactive:
        return "button"
    tag = el.tag.lower()
    # Icons before media: an inline <svg> is an icon, not a photo-like image.
    if _is_icon(el):
        return "icon"
    if tag in MEDIA_TAGS:
        return "image"
    if (el.has_direct_text and (el.text or "").strip()) or el.heading_level > 0:
        return "text"
    # A leaf with a background image and no children reads as an image tile.
    if el.has_background_image and el.child_count == 0:
        return "image"
    return None


def _atom(el: ElementMetadata) -> Optional[ContentAtom]:
    kind = atom_type(el)
    if kind is None:
        return None
    return ContentAtom(
        selector=el.selector,
        type=kind,
        bbox=_doc_bbox(el),
        view_bbox=el.bbox,
        visible_fraction=float(el.visible_fraction),
        tag=el.tag,
        text=(el.text or el.accessible_name or "")[:80],
        font_px=_font_px(el),
        dom_id=el.dom_id,
        heading_level=el.heading_level,
    )


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class LayoutConfig:
    tolerance_px: float = 2.0
    row_band_tolerance_px: float = 10.0
    min_container_px: float = 48.0
    spacing_tolerance_px: float = 2.0
    grid_width_variance: float = 0.15
    fold_clip_min_visible: float = 0.02
    fold_clip_max_visible: float = 0.98
    fold_clip_min_px: float = 8.0
    fold_clip_max_findings: int = 8
    max_atoms: int = 1200
    # --- organization / "design expert" knobs ---
    orphan_max_words: int = 2
    orphan_max_width_ratio: float = 0.30
    orphan_min_prev_chars: int = 12
    orphan_min_words: int = 4
    underfill_ratio: float = 0.62
    imbalance_skew: float = 0.55
    imbalance_delta: float = 0.25
    min_det_cluster_rows: int = 3
    max_grid_findings: int = 12
    orphan_max_findings: int = 6


def _config(thresholds: Optional[Dict[str, Any]]) -> LayoutConfig:
    block = (thresholds or {}).get("layout", {}) if thresholds else {}
    if not isinstance(block, dict):
        block = {}
    cfg = LayoutConfig()
    for f in fields(LayoutConfig):
        if f.name in block:
            try:
                setattr(cfg, f.name, type(f.default)(block[f.name]))
            except (TypeError, ValueError):
                pass
    return cfg


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _area(b: BoundingBox) -> float:
    return max(0.0, float(b.width)) * max(0.0, float(b.height))


def _contains(outer: BoundingBox, inner: BoundingBox, tol: float = 1.0) -> bool:
    return (
        outer.x - tol <= inner.x
        and outer.y - tol <= inner.y
        and outer.x + outer.width + tol >= inner.x + inner.width
        and outer.y + outer.height + tol >= inner.y + inner.height
    )


def _dedupe(screens: List[ScreenCapture]) -> List[ElementMetadata]:
    """One element per DOM identity across overlapping scroll slices."""
    best: Dict[Any, ElementMetadata] = {}
    for scr in screens:
        for el in scr.elements:
            key: Any = el.dom_id if el.dom_id >= 0 else (
                el.selector,
                int(_doc_bbox(el).x),
                int(_doc_bbox(el).y),
                int(_doc_bbox(el).width),
                int(_doc_bbox(el).height),
            )
            prev = best.get(key)
            if prev is None or el.visible_fraction > prev.visible_fraction:
                best[key] = el
    return list(best.values())


def _structural_boxes(elements: List[ElementMetadata]) -> List[Tuple[int, BoundingBox]]:
    out: List[Tuple[int, BoundingBox]] = []
    for el in elements:
        if not el.is_structural_box:
            continue
        b = _doc_bbox(el)
        if b.width <= 0 or b.height <= 0:
            continue
        out.append((el.dom_id, b))
    return out


def _maximal_inner(
    bbox: BoundingBox, all_boxes: List[Tuple[int, BoundingBox]], self_id: int
) -> List[Tuple[int, BoundingBox]]:
    inner = [(i, b) for i, b in all_boxes if i != self_id and _contains(bbox, b)]
    maximal: List[Tuple[int, BoundingBox]] = []
    for i, b in inner:
        if any(
            j != i and _contains(b2, b) and _area(b2) < _area(b)
            for j, b2 in inner
        ):
            continue
        maximal.append((i, b))
    return maximal


def _is_wrapper(
    dom_id: int,
    bbox: BoundingBox,
    all_boxes: List[Tuple[int, BoundingBox]],
    candidate_ids: Set[int],
    tol: float = 0.15,
) -> bool:
    """A grid/list/section wrapper, not a card.

    A wrapper either holds >=3 similarly-sized containers (the repeated cells) or
    holds exactly one card-like container (a region around a grid).
    """
    maximal = _maximal_inner(bbox, all_boxes, dom_id)
    if not maximal:
        return False
    for _, b in maximal:
        similar = sum(
            1
            for j, b2 in maximal
            if j in candidate_ids
            and abs(b2.width - b.width) <= tol * b.width
            and abs(b2.height - b.height) <= tol * b.height
        )
        if similar >= 3:
            return True
    if len(maximal) == 1 and maximal[0][0] in candidate_ids:
        return True
    return False


def _containers(
    elements: List[ElementMetadata], viewport_w: int, cfg: LayoutConfig
) -> Tuple[Dict[int, BoundingBox], List[Tuple[int, BoundingBox]]]:
    """Card-like boxes that can own several atoms in one row."""
    all_boxes = _structural_boxes(elements)
    candidate_ids = {
        el.dom_id
        for el in elements
        if el.is_structural_box and el.child_count >= 1
    }
    boxes: List[Tuple[int, BoundingBox]] = []
    for el in elements:
        if el.dom_id not in candidate_ids:
            continue
        b = _doc_bbox(el)
        if b.width < cfg.min_container_px or b.height < 24:
            continue
        if viewport_w and b.width > 0.95 * viewport_w:
            continue
        if _is_wrapper(el.dom_id, b, all_boxes, candidate_ids):
            continue
        boxes.append((el.dom_id, b))
    return {dom_id: b for dom_id, b in boxes}, boxes


def _smallest_container(b: BoundingBox, boxes: List[Tuple[int, BoundingBox]]) -> Optional[int]:
    best: Optional[int] = None
    best_area = float("inf")
    for dom_id, box in boxes:
        if not _contains(box, b):
            continue
        area = _area(box)
        if area < best_area:
            best, best_area = dom_id, area
    return best


def _largest_container(b: BoundingBox, boxes: List[Tuple[int, BoundingBox]]) -> Optional[int]:
    best: Optional[int] = None
    best_area = -1.0
    for dom_id, box in boxes:
        if not _contains(box, b):
            continue
        area = _area(box)
        if area > best_area:
            best, best_area = dom_id, area
    return best


def _row_bands(atoms: List[ContentAtom], tol: float) -> List[Dict[str, Any]]:
    ordered = sorted(atoms, key=lambda a: (a.bbox.y + a.bbox.height / 2.0, a.bbox.x))
    bands: List[Dict[str, Any]] = []
    for a in ordered:
        cy = a.bbox.y + a.bbox.height / 2.0
        if bands and abs(cy - bands[-1]["cy"]) <= tol:
            band = bands[-1]
            band["atoms"].append(a)
            band["cy"] = sum(x.bbox.y + x.bbox.height / 2.0 for x in band["atoms"]) / len(band["atoms"])
            band["top"] = min(band["top"], a.bbox.y)
            band["bottom"] = max(band["bottom"], a.bbox.y + a.bbox.height)
        else:
            bands.append({
                "cy": cy,
                "atoms": [a],
                "top": a.bbox.y,
                "bottom": a.bbox.y + a.bbox.height,
            })
    return bands


def _cells(
    atoms: List[ContentAtom],
    group_of: Dict[int, int],
    card_of: Dict[int, int],
    container_bbox: Dict[int, BoundingBox],
) -> List[Dict[str, Any]]:
    """Group atoms in one row into grid cells.

    Grouping uses the *smallest* card (so a card's label+value stay one cell);
    geometry uses the *largest* card (so every row of that card reports the same
    edges). Atoms with no card stay separate cells measured by their own edges.
    """
    buckets: Dict[Any, List[ContentAtom]] = {}
    order: List[Any] = []
    for a in sorted(atoms, key=lambda a: a.bbox.x):
        if a.bbox.width < 4 or a.bbox.height < 4:
            continue
        key: Any = group_of.get(a.dom_id)
        if key is None:
            key = ("atom", a.dom_id)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(a)

    cells: List[Dict[str, Any]] = []
    for key in order:
        group = buckets[key]
        card_id = card_of.get(group[0].dom_id)
        if card_id is not None and card_id in container_bbox:
            b = container_bbox[card_id]
            x0, x1 = b.x, b.x + b.width
        else:
            x0 = min(a.bbox.x for a in group)
            x1 = max(a.bbox.x + a.bbox.width for a in group)
        if x1 - x0 < 4:
            continue
        cells.append({
            "x0": x0,
            "x1": x1,
            "atoms": group,
            "container": card_id,
        })
    cells.sort(key=lambda c: c["x0"])
    return cells


def _cluster_rows(
    rows: List[Dict[str, Any]], tol: float
) -> List[List[Dict[str, Any]]]:
    """Group rows that belong to the same grid region.

    Rows are grouped by their shared left edge, and split into separate groups
    when the vertical gap between consecutive rows is a section break rather
    than the grid's own rhythm. The dashboard header (narrow text rows) and the
    metric cards below it can share a left edge, yet they are different grids;
    mixing them would make the header read as an "unfilled" grid row.
    """
    clusters: List[List[Dict[str, Any]]] = []
    for entry in rows:
        x0 = entry["cells"][0]["x0"]
        placed = False
        for cluster in clusters:
            cx0 = median([e["cells"][0]["x0"] for e in cluster])
            if abs(x0 - cx0) > tol:
                continue
            last = cluster[-1]
            gap = entry["top"] - last["bottom"]
            gaps = [
                cluster[i + 1]["top"] - cluster[i]["bottom"]
                for i in range(len(cluster) - 1)
            ]
            typical = float(median([g for g in gaps if g >= 0])) if gaps else 0.0
            threshold = max(320.0, 3.5 * typical)
            if 0 <= gap <= threshold:
                cluster.append(entry)
                placed = True
                break
        if not placed:
            clusters.append([entry])
    return clusters


def _occupancy_boxes(boxes: List[BoundingBox], width: int, height: int, cell: int = 16) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    gw = max(1, int(math.ceil(width / cell)))
    gh = max(1, int(math.ceil(height / cell)))
    grid = np.zeros((gh, gw), dtype=bool)
    for b in boxes:
        x0 = max(0, int(b.x))
        y0 = max(0, int(b.y))
        x1 = min(width, int(b.x + b.width))
        y1 = min(height, int(b.y + b.height))
        if x1 <= x0 or y1 <= y0:
            continue
        grid[int(y0 // cell):int(math.ceil(y1 / cell)), int(x0 // cell):int(math.ceil(x1 / cell))] = True
    return float(grid.mean())


def _overlap_width(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #
def _flag(
    selector: str,
    *,
    issue_type: str,
    bbox: BoundingBox,
    tag: str,
    text: str,
    description: str,
    observed: Dict[str, Any],
    expected: Dict[str, Any],
    severity: str,
    impact: str = "medium",
    confidence: float = 0.7,
    requires_human: bool = True,
) -> FlaggedProblemElement:
    return FlaggedProblemElement(
        selector=selector,
        tag=tag,
        text=text,
        bbox=bbox,
        issue_type=issue_type,
        severity=severity,
        rule_id=issue_type,
        impact=impact,
        confidence=confidence,
        evidence_class="deterministic_geometry",
        description=description,
        observed=observed,
        expected=expected,
        requires_human=requires_human,
        attention_stats={},
    )


# --------------------------------------------------------------------------- #
# Detector: dangling one-word lines (typographic orphans)
# --------------------------------------------------------------------------- #
def _orphan_detection(
    union_elements: List[ElementMetadata], cfg: LayoutConfig
) -> Tuple[List[FlaggedProblemElement], List[str]]:
    """A text block whose *last rendered line* holds a dangling short fragment.

    This needs the crawler's per-line ``text_lines``; an element box alone cannot
    show which words landed on which line. Headings are skipped: titles are
    normally allowed to wrap.
    """
    flags: List[FlaggedProblemElement] = []
    selectors: List[str] = []
    for el in union_elements:
        lines = el.text_lines or []
        if len(lines) < 2 or el.heading_level:
            continue
        if sum(int(l.get("words", 0)) for l in lines) < cfg.orphan_min_words:
            continue
        last, prev = lines[-1], lines[-2]
        block_width = max(float(l.get("width", 0.0)) for l in lines)
        last_width = float(last.get("width", 0.0))
        prev_width = float(prev.get("width", 0.0))
        last_words = int(last.get("words", 0))
        if block_width <= 0 or last_width <= 0:
            continue
        if len((prev.get("text") or "").strip()) < cfg.orphan_min_prev_chars:
            continue
        if last_width > cfg.orphan_max_width_ratio * block_width:
            continue
        if last_words > cfg.orphan_max_words:
            continue
        if last_words > 1 and prev_width < 2.5 * last_width:
            continue
        share = last_width / block_width * 100.0
        fragment = (last.get("text") or "").strip()
        selectors.append(el.selector)
        if len(flags) >= cfg.orphan_max_findings:
            continue
        flags.append(_flag(
            el.selector,
            issue_type="ORPHAN_LINE",
            bbox=el.bbox,
            tag=el.tag,
            text=(el.text or el.accessible_name or "")[:80],
            description=(
                f"This text block wraps onto {len(lines)} lines and its last line holds only "
                f"{fragment!r} ({last_words} word(s), {share:.0f}% of the block width), so one "
                f"word dangles on its own line. Tighten the copy or the container width so the "
                f"lines fill more evenly — do not remove the text."
            ),
            observed={
                "line_count": len(lines),
                "last_line_text": fragment,
                "last_line_words": last_words,
                "last_line_width_px": round(last_width, 1),
                "block_width_px": round(block_width, 1),
                "last_line_share_pct": round(share, 1),
            },
            expected={
                "last_line_share_pct": ">= 30% of the block width, or no dangling line",
            },
            severity="needs_investigation",
            impact="low",
            confidence=0.75,
        ))
    return flags, selectors


# --------------------------------------------------------------------------- #
# Detector: grid organization (holes, underfilled rows, lopsided rows, drift)
# --------------------------------------------------------------------------- #
def _grid_detection(
    clusters: List[List[Dict[str, Any]]],
    cfg: LayoutConfig,
    first_screen_id: str,
) -> Tuple[Dict[str, Any], List[Tuple[str, FlaggedProblemElement]]]:
    """Comment on how the repeated rows organize their content.

    Every detector anchors on the region's *repeated multi-cell shape*: full
    rows (same cell count) define the column references, and only cells that
    actually match those references are judged. Section headers, lead text and
    deliberately wide one-off rows are excluded rather than "fixed".
    """
    metrics: Dict[str, Any] = {
        "grid_holes": 0,
        "underfilled_rows": 0,
        "min_row_fill_ratio": None,
        "skewed_rows": 0,
    }
    flags: List[Tuple[str, FlaggedProblemElement]] = []
    seen_selectors: Set[str] = set()
    hole_seen: Dict[Tuple[int, int], float] = {}  # (cluster_idx, slot) -> last top y

    for cluster_idx, cluster in enumerate(clusters):
        if len(cluster) < cfg.min_det_cluster_rows:
            continue
        cluster_cells = [e["cells"] for e in cluster]
        tops = [float(e["top"]) for e in cluster]
        ncols = max(len(c) for c in cluster_cells)
        full_rows = [c for c in cluster_cells if len(c) == ncols]
        has_grid = ncols >= 2 and len(full_rows) >= 2
        spans = [float(c[-1]["x1"] - c[0]["x0"]) for c in cluster_cells]
        median_span = float(median(spans))
        internal_gaps: List[float] = []
        for cells in cluster_cells:
            for i in range(len(cells) - 1):
                g = cells[i + 1]["x0"] - cells[i]["x1"]
                if g >= 0:
                    internal_gaps.append(float(g))
        gutter = float(median(internal_gaps)) if internal_gaps else 0.0
        extent = (
            min(c[0]["x0"] for c in cluster_cells),
            max(c[-1]["x1"] for c in cluster_cells),
        )

        # Column refs describe the repeated multi-cell shape of this region.
        refs_x0: Dict[int, float] = {}
        refs_w: Dict[int, float] = {}
        pitch = 0.0
        snap = 0.0
        if has_grid:
            for j in range(ncols):
                xs0 = [c[j]["x0"] for c in full_rows]
                ws = [c[j]["x1"] - c[j]["x0"] for c in full_rows]
                refs_x0[j] = float(median(xs0))
                refs_w[j] = float(median(ws))
            pitches = [refs_x0[j + 1] - refs_x0[j] for j in range(ncols - 1)]
            pitch = float(median(pitches)) if pitches else 0.0
            snap = max(6.0, 0.25 * max(1.0, pitch))

        # Row position facts.
        first_grid_idx = next(
            (i for i, c in enumerate(cluster_cells) if len(c) == ncols and ncols >= 2),
            len(cluster_cells),
        )

        # ---- pass 1: per-row facts ----
        rows_info: List[Dict[str, Any]] = []
        for row_idx, cells in enumerate(cluster_cells):
            span = float(cells[-1]["x1"] - cells[0]["x0"])
            fill = span / median_span if median_span > 0 else 1.0

            # A hole is a *skipped middle column* in an otherwise real grid row:
            # every cell maps to a column ref (position and width). The page
            # header (logo left, button right) and deliberate narrow rails do
            # not follow the grid's cell geometry, so they are never "holes"
            # and never "unfilled rows" — they are simply a different layout.
            holes: List[Tuple[int, float]] = []
            grid_like = False
            if has_grid:
                mapped: List[int] = []
                ok = True
                for c in cells:
                    best = min(
                        (k for k in refs_x0),
                        key=lambda k: abs(refs_x0[k] - c["x0"]),
                        default=None,
                    )
                    if best is None or abs(refs_x0[best] - c["x0"]) > snap:
                        ok = False
                        break
                    if abs((c["x1"] - c["x0"]) - refs_w[best]) > max(
                        cfg.tolerance_px, cfg.grid_width_variance * refs_w[best]
                    ):
                        ok = False
                        break
                    mapped.append(best)
                grid_like = (
                    ok
                    and mapped == sorted(mapped)
                    and len(set(mapped)) == len(mapped)
                )
                if grid_like and len(cells) >= 2:
                    covered = set(mapped)
                    lo, hi = min(mapped), max(mapped)
                    for slot in range(lo + 1, hi):
                        if slot not in covered:
                            prev_cell = max(c["x1"] for c in cells if c["x0"] < refs_x0[slot])
                            next_cell = min(c["x0"] for c in cells if c["x1"] > refs_x0[slot])
                            holes.append((slot, float(next_cell - prev_cell)))

            thirds: Optional[Dict[str, float]] = None
            skew: Optional[float] = None
            if len(cells) >= 2 and extent[1] > extent[0]:
                third_w = (extent[1] - extent[0]) / 3.0
                third_bands = [
                    (extent[0], extent[0] + third_w),
                    (extent[0] + third_w, extent[0] + 2 * third_w),
                    (extent[0] + 2 * third_w, extent[1]),
                ]
                shares = [
                    sum(_overlap_width((c["x0"], c["x1"]), band) for c in cells)
                    for band in third_bands
                ]
                total_lr = shares[0] + shares[2]
                skew = abs(shares[0] - shares[2]) / total_lr if total_lr > 0 else 0.0
                if fill < 0.5:
                    skew = None
                else:
                    thirds = {
                        "left": shares[0],
                        "center": shares[1],
                        "right": shares[2],
                    }
            rows_info.append({
                "cells": cells,
                "top": tops[row_idx],
                "row_idx": row_idx,
                "lead": has_grid and row_idx < first_grid_idx,
                "grid_like": grid_like,
                "fill": fill,
                "span": span,
                "holes": holes,
                "thirds": thirds,
                "skew": skew,
            })

        median_skew = float(median([r["skew"] for r in rows_info if r["skew"] is not None])) \
            if any(r["skew"] is not None for r in rows_info) else 0.0
        if has_grid and median_span > 0:
            fills = [r["fill"] for r in rows_info if not r["lead"] and r["grid_like"]]
            if fills:
                ratio = round(min(fills), 2)
                prev = metrics["min_row_fill_ratio"]
                metrics["min_row_fill_ratio"] = ratio if prev is None else min(prev, ratio)

        # ---- pass 2: flags, one per row, first matching detector wins ----
        for info in rows_info:
            if len(flags) >= cfg.max_grid_findings:
                break
            cells = info["cells"]
            widest = max((a for c in cells for a in c["atoms"]), key=lambda a: _area(a.bbox))
            if widest.selector in seen_selectors:
                continue

            if not info["lead"] and has_grid and info["holes"]:
                slot, gap = info["holes"][0]
                key = (cluster_idx, slot)
                last_top = hole_seen.get(key)
                # Sub-bands of the same visual row repeat the same hole; only
                # the first band in each card row reports it.
                if last_top is None or info["top"] - last_top > 100.0:
                    hole_seen[key] = info["top"]
                    seen_selectors.add(widest.selector)
                    metrics["grid_holes"] += 1
                    flags.append((first_screen_id, _flag(
                        widest.selector,
                        issue_type="GRID_GAP",
                        bbox=widest.bbox,
                        tag=widest.tag,
                        text=widest.text,
                        description=(
                            f"In this repeated row the gap between two cells is "
                            f"{gap:.0f}px while the row's normal gutter is "
                            f"{gutter:.0f}px — a middle column is left empty while the sibling "
                            f"rows fill it. Either give the slot content or let the neighbours "
                            f"absorb it so the row reads as one grid."
                        ),
                        observed={
                            "gap_px": round(gap, 1),
                            "normal_gutter_px": round(gutter, 1),
                            "missing_column": slot + 1,
                            "cells_in_row": len(cells),
                            "grid_columns_estimated": ncols,
                        },
                        expected={"gap_px": round(gutter, 1)},
                        severity="potential_improvement",
                        impact="medium",
                        confidence=0.7,
                    )))
                    continue

            if (
                not info["lead"]
                and info["skew"] is not None
                and info["thirds"] is not None
                and info["skew"] >= cfg.imbalance_skew
                and info["skew"] - median_skew >= cfg.imbalance_delta
            ):
                seen_selectors.add(widest.selector)
                metrics["skewed_rows"] += 1
                thirds = info["thirds"]
                total = max(1.0, sum(thirds.values()))
                heavy = "left" if thirds["left"] >= thirds["right"] else "right"
                flags.append((first_screen_id, _flag(
                    widest.selector,
                    issue_type="BLOCK_IMBALANCE",
                    bbox=widest.bbox,
                    tag=widest.tag,
                    text=widest.text,
                    description=(
                        f"This row puts {max(thirds['left'], thirds['right']) / total * 100:.0f}% "
                        f"of its width in the {heavy} slots while its sibling rows spread their "
                        f"content evenly (median skew {median_skew:.0%}). The row reads off "
                        f"balance. Rebalance the cell widths or pair the small cell with a "
                        f"similar one — keep the content."
                    ),
                    observed={
                        "left_third_pct": round(thirds["left"] / total * 100.0, 1),
                        "center_third_pct": round(thirds["center"] / total * 100.0, 1),
                        "right_third_pct": round(thirds["right"] / total * 100.0, 1),
                        "skew_pct": round(info["skew"] * 100.0, 1),
                        "cluster_median_skew_pct": round(median_skew * 100.0, 1),
                    },
                    expected={"skew_pct": f"<= {cfg.imbalance_skew * 100:.0f}%"},
                    severity="needs_investigation",
                    impact="low",
                    confidence=0.6,
                )))
                continue

            if not info["lead"] and has_grid and info["grid_like"] and info["fill"] < cfg.underfill_ratio:
                slot_key = (cluster_idx, "under", 0)
                last_top = hole_seen.get(slot_key)
                # Sub-bands of one visual row repeat the same judgment; only
                # the first band reports it.
                if last_top is None or info["top"] - last_top > 100.0:
                    hole_seen[slot_key] = info["top"]
                    seen_selectors.add(widest.selector)
                    metrics["underfilled_rows"] += 1
                    flags.append((first_screen_id, _flag(
                    widest.selector,
                    issue_type="ROW_UNDERFILL",
                    bbox=widest.bbox,
                    tag=widest.tag,
                    text=widest.text,
                    description=(
                        f"This grid row uses only {info['fill'] * 100:.0f}% of the width its "
                        f"sibling rows use ({info['span']:.0f}px of {median_span:.0f}px), so the "
                        f"line reads as unfilled. Give it a full-width role, pull it into the "
                        f"neighbouring row, or size its cells to match the rest of the grid."
                    ),
                    observed={
                        "row_span_px": round(info["span"], 1),
                        "cluster_span_px": round(median_span, 1),
                        "fill_ratio_pct": round(info["fill"] * 100.0, 1),
                        "cells_in_row": len(cells),
                    },
                    expected={"fill_ratio_pct": f">= {cfg.underfill_ratio * 100:.0f}%"},
                    severity="needs_investigation",
                    impact="low",
                    confidence=0.65,
                )))
                continue

        # ---- column drift inside the region ----
        if not has_grid:
            continue
        for cells in cluster_cells:
            partial_row = len(cells) != ncols
            for j, c in enumerate(cells):
                if j not in refs_x0:
                    break
                # A differently-organized partial row (centred closing row,
                # spread header) is not a drift; only cells that sit near a
                # column line are judged.
                dev_x = abs(c["x0"] - refs_x0[j])
                if partial_row and dev_x > snap:
                    break
                ref_w = refs_w[j]
                cell_w = c["x1"] - c["x0"]
                dev_w = abs(cell_w - ref_w)
                if (
                    dev_x > cfg.tolerance_px
                    and len(flags) < cfg.max_grid_findings
                ):
                    widest = max(c["atoms"], key=lambda a: _area(a.bbox))
                    if widest.selector not in seen_selectors:
                        seen_selectors.add(widest.selector)
                        metrics["off_grid_count"] = int(metrics.get("off_grid_count", 0)) + 1
                        flags.append((first_screen_id, _flag(
                            widest.selector,
                            issue_type="GRID_MISALIGNMENT",
                            bbox=widest.bbox,
                            tag=widest.tag,
                            text=widest.text,
                            description=(
                                f"This cell starts at x={c['x0']:.0f}px while the same column in "
                                f"this grid starts near x={refs_x0[j]:.0f}px (off by "
                                f"{dev_x:.0f}px). Repeated rows do not share a column line, so "
                                f"the grid reads as misaligned."
                            ),
                            observed={
                                "mode": "x0",
                                "column_index": j + 1,
                                "x0_px": round(c["x0"], 1),
                                "column_x0_px": round(refs_x0[j], 1),
                                "deviation_px": round(dev_x, 1),
                            },
                            expected={"deviation_px": 0.0},
                            severity="needs_investigation",
                            impact="medium",
                            confidence=0.65,
                        )))
                # Width drift only makes sense for full rows: a partial row's
                # cells are legitimately not cell-track sized.
                if (
                    not partial_row
                    and ref_w > 0
                    and dev_w > max(cfg.tolerance_px, cfg.grid_width_variance * ref_w)
                    and len(flags) < cfg.max_grid_findings
                ):
                    widest = max(c["atoms"], key=lambda a: _area(a.bbox))
                    if widest.selector not in seen_selectors:
                        seen_selectors.add(widest.selector)
                        metrics["off_grid_count"] = int(metrics.get("off_grid_count", 0)) + 1
                        flags.append((first_screen_id, _flag(
                            widest.selector,
                            issue_type="GRID_MISALIGNMENT",
                            bbox=widest.bbox,
                            tag=widest.tag,
                            text=widest.text,
                            description=(
                                f"This cell is {cell_w:.0f}px wide while its column "
                                f"siblings are {ref_w:.0f}px wide. The repeated row does not "
                                f"keep even cell widths, so the row reads off balance."
                            ),
                            observed={
                                "mode": "width",
                                "column_index": j + 1,
                                "width_px": round(cell_w, 1),
                                "column_width_px": round(ref_w, 1),
                                "width_deviation_px": round(dev_w, 1),
                            },
                            expected={"width_deviation_px": 0.0},
                            severity="needs_investigation",
                            impact="medium",
                            confidence=0.65,
                        )))

    if metrics["min_row_fill_ratio"] is None:
        metrics["min_row_fill_ratio"] = 1.0
    return metrics, flags


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
@dataclass
class LayoutResult:
    metrics: Dict[str, Any]
    flags: List[Tuple[str, FlaggedProblemElement]]  # (screen_id, flag)


def analyze_layout(
    screens: List[ScreenCapture],
    thresholds: Optional[Dict[str, Any]] = None,
) -> LayoutResult:
    """Analyze one (page, view) group. Returns metrics + screen-scoped flags."""
    if not screens:
        return LayoutResult(metrics={}, flags=[])

    cfg = _config(thresholds)
    first = min(screens, key=lambda s: (s.scroll_y, s.scroll_index))
    viewport_w = max(1, int(first.viewport_width))
    notes: List[str] = []

    # ---- document-union atoms (deduped, scroll-noise free) ----
    union_elements = _dedupe(screens)
    atoms: List[ContentAtom] = []
    for el in union_elements:
        a = _atom(el)
        if a is not None and _area(a.bbox) > 0:
            atoms.append(a)
    if len(atoms) > cfg.max_atoms:
        atoms.sort(key=lambda a: _area(a.bbox), reverse=True)
        atoms = atoms[: cfg.max_atoms]

    counts = {"text": 0, "image": 0, "icon": 0, "button": 0}
    for a in atoms:
        counts[a.type] = counts.get(a.type, 0) + 1

    # ---- group atoms into cards, then into grid cells ----
    container_bbox, container_list = _containers(union_elements, viewport_w, cfg)
    group_of: Dict[int, int] = {}
    card_of: Dict[int, int] = {}
    for a in atoms:
        g = _smallest_container(a.bbox, container_list)
        if g is not None:
            group_of[a.dom_id] = g
        c = _largest_container(a.bbox, container_list)
        if c is not None:
            card_of[a.dom_id] = c

    bands = _row_bands(atoms, cfg.row_band_tolerance_px)
    band_entries: List[Dict[str, Any]] = []
    band_rows: List[List[Dict[str, Any]]] = []  # cells only, for shape histograms
    for b in bands:
        cells = _cells(b["atoms"], group_of, card_of, container_bbox)
        if cells:
            band_entries.append({"cells": cells, "top": b["top"], "bottom": b["bottom"]})
            band_rows.append(cells)

    # ---- dominant row shape (page level) ----
    columns = 1
    grid_confidence = 1.0 if band_rows else 0.0
    if band_rows:
        # The dominant row shape is the most common one. A page with a few
        # 3-across button rows but a single-column card layout stays 1-column.
        freq = Counter(len(c) for c in band_rows)
        columns = max(freq, key=lambda k: (freq[k], k))
        grid_confidence = freq[columns] / max(1, len(band_rows))

    # ---- grid regions: rows sharing a left edge, split at section breaks ----
    clusters = _cluster_rows(band_entries, max(8.0, cfg.tolerance_px * 4)) if band_entries else []
    main_cluster = (
        max(clusters, key=lambda cl: sum(len(e["cells"]) for e in cl)) if clusters else []
    )
    # Column metrics describe the repeated shape of the main region: rows whose
    # cell count matches the region's own mode (a stray wider/narrower row is
    # organization data, not a column line).
    main_modal_count = (
        Counter(len(e["cells"]) for e in main_cluster).most_common(1)[0][0]
        if main_cluster else 1
    )
    main_rows = [e["cells"] for e in main_cluster if len(e["cells"]) == main_modal_count]

    # ---- column widths / pitch / gutter/edges (main grid region) ----
    column_widths: List[float] = []
    column_edges: List[List[float]] = []
    column_pitch = 0.0
    gutter_px = 0.0
    if main_rows:
        ncols = main_modal_count
        for j in range(ncols):
            xs0 = [c[j]["x0"] for c in main_rows]
            xs1 = [c[j]["x1"] for c in main_rows]
            if xs0 and xs1:
                column_widths.append(round(float(median([x1 - x0 for x0, x1 in zip(xs0, xs1)])), 1))
                column_edges.append([round(float(median(xs0)), 1), round(float(median(xs1)), 1)])
        if ncols >= 2:
            pitches: List[float] = []
            gaps: List[float] = []
            for cells in main_rows:
                for i in range(len(cells) - 1):
                    pitches.append(cells[i + 1]["x0"] - cells[i]["x0"])
                    g = cells[i + 1]["x0"] - cells[i]["x1"]
                    if g >= 0:
                        gaps.append(float(g))
            if pitches:
                column_pitch = round(float(median(pitches)), 1)
            if gaps:
                gutter_px = round(float(median(gaps)), 1)

    # ---- symmetry: outer margins of each row in the main grid region ----
    sym_errors: List[float] = []
    for cells in main_rows:
        left = min(c["x0"] for c in cells)
        right = viewport_w - max(c["x1"] for c in cells)
        sym_errors.append(abs(left - right))
    symmetry_error_px = round(float(median(sym_errors)), 1) if sym_errors else 0.0
    symmetry_error_ratio = round(symmetry_error_px / viewport_w, 4) if viewport_w else 0.0

    # ---- spacing: horizontal gutters when multi-column, else vertical rhythm ----
    vertical_gaps: List[float] = []
    for i in range(len(bands) - 1):
        g = bands[i + 1]["top"] - bands[i]["bottom"]
        if g >= 0:
            vertical_gaps.append(round(g, 1))

    spacing_axis = "horizontal" if columns >= 2 and main_rows else "vertical"
    spacing_values: List[float] = []
    if spacing_axis == "horizontal":
        for cells in main_rows:
            for i in range(len(cells) - 1):
                g = cells[i + 1]["x0"] - cells[i]["x1"]
                if g >= 0:
                    spacing_values.append(float(g))
    else:
        spacing_values = vertical_gaps
    dominant_spacing: Optional[float] = None
    spacing_outliers = 0
    if spacing_values:
        dominant_spacing = round(float(median(spacing_values)), 1)
        # Only horizontal grid gutters are expected to repeat. Vertical rhythm is
        # intentionally varied by headings/sections, so do not call it outliers.
        if spacing_axis == "horizontal":
            spacing_outliers = sum(
                1 for g in spacing_values if abs(g - dominant_spacing) > cfg.spacing_tolerance_px
            )

    # ---- organization detectors (the "design expert" comments) ----
    grid_metrics, grid_flags = _grid_detection(clusters, cfg, first.screen_id)
    off_grid_count = int(grid_metrics.pop("off_grid_count", 0))
    orphan_flags, orphan_selectors = _orphan_detection(union_elements, cfg)
    orphan_flag_tuples = [(first.screen_id, f) for f in orphan_flags]

    # ---- first view: coverage, fold clipping, ink balance ----
    first_atoms: List[ContentAtom] = []
    for el in first.elements:
        a = _atom(el)
        if a is not None and _area(a.view_bbox) > 0:
            first_atoms.append(a)

    fold_candidates: List[ElementMetadata] = []
    for el in first.elements:
        vf = float(el.visible_fraction)
        if not (cfg.fold_clip_min_visible <= vf < cfg.fold_clip_max_visible):
            continue
        if not (el.is_clipped_bottom or el.is_clipped_top):
            continue
        if el.tag.lower() in _FOLD_EXCLUDED_TAGS:
            continue
        if el.bbox.height < cfg.fold_clip_min_px or el.bbox.width < 24:
            continue
        if not (el.is_structural_box or atom_type(el) is not None):
            continue
        fold_candidates.append(el)

    # Keep the most specific clipped box: a card, not the whole section that holds it.
    fold_components: List[ElementMetadata] = []
    for el in fold_candidates:
        ob = _doc_bbox(el)
        if any(
            other is not el and _contains(ob, _doc_bbox(other))
            for other in fold_candidates
        ):
            continue
        fold_components.append(el)
    fold_components.sort(key=lambda e: _area(_doc_bbox(e)), reverse=True)

    clipped_boxes = [_doc_bbox(e) for e in fold_components]
    clipped_atoms = [a for a in first_atoms if any(_contains(box, a.bbox) for box in clipped_boxes)]
    clipped_ids = {a.dom_id for a in clipped_atoms}
    first_visible_atoms = [a for a in first_atoms if a.dom_id not in clipped_ids]

    visual_boxes: List[BoundingBox] = [a.view_bbox for a in first_visible_atoms]
    for el in first.elements:
        if not el.is_structural_box or el.child_count < 1:
            continue
        if el.tag.lower() in {"html", "body", "main", "section", "article", "nav", "header", "footer"}:
            continue
        if _area(el.bbox) <= 0:
            continue
        if el.bbox.width > 0.95 * first.viewport_width and el.bbox.height > 0.5 * first.viewport_height:
            continue
        visual_boxes.append(el.bbox)
    occupancy = _occupancy_boxes(visual_boxes, first.viewport_width, first.viewport_height)
    whitespace_pct = round(max(0.0, 1.0 - occupancy) * 100.0, 1)

    # first-view ink balance across the viewport thirds (fact, not verdict)
    content_skew_pct = 0.0
    if first_visible_atoms and first.viewport_width > 0:
        edges = (
            0.0,
            first.viewport_width / 3.0,
            2.0 * first.viewport_width / 3.0,
            float(first.viewport_width),
        )
        bands_lr = [(edges[0], edges[1]), (edges[2], edges[3])]
        shares = [
            sum(
                _overlap_width((a.view_bbox.x, a.view_bbox.x + a.view_bbox.width), band)
                for a in first_visible_atoms
            )
            for band in bands_lr
        ]
        total = shares[0] + shares[1]
        content_skew_pct = round(abs(shares[0] - shares[1]) / total * 100.0, 1) if total > 0 else 0.0

    fold_flags: List[Tuple[str, FlaggedProblemElement]] = []
    clipped_selectors: List[str] = []
    for el in fold_components[: cfg.fold_clip_max_findings]:
        vf = float(el.visible_fraction)
        edge = "bottom" if el.is_clipped_bottom else "top"
        doc = _doc_bbox(el)
        hidden_px = max(0, int(round(doc.height * (1.0 - vf))))
        kind = atom_type(el) or "box"
        clipped_selectors.append(el.selector)
        fold_flags.append((
            first.screen_id,
            _flag(
                el.selector,
                issue_type="FOLD_CLIPPED_CONTENT",
                bbox=el.bbox,
                tag=el.tag,
                text=(el.text or el.accessible_name or "")[:80],
                description=(
                    f"Only {vf * 100:.0f}% of this {kind} fits inside the first viewport; "
                    f"the remaining ~{hidden_px}px is cut by the {edge} fold. In the first view "
                    f"it is rendered as just noise — a partial strip, not a complete object — "
                    f"so the viewer cannot read it as a real element. Shift it down (or resize "
                    f"it) so it is either fully inside the first view or fully below the fold."
                ),
                observed={
                    "visible_pct": round(vf * 100.0, 1),
                    "clipped_edge": edge,
                    "clipped_px": hidden_px,
                    "element_type": kind,
                    "covered_atoms": sum(1 for a in first_atoms if _contains(doc, a.bbox)),
                },
                expected={
                    "visible_pct": 100.0,
                    "note": "fully inside the first viewport, or fully below the fold",
                },
                severity="potential_improvement",
                impact="medium",
                confidence=0.8,
            ),
        ))

    # ---- intra-row atom gaps (e.g. 6px / 3px button groups) ----
    intra_gaps: List[int] = []
    for b in bands:
        ordered = sorted(b["atoms"], key=lambda a: a.bbox.x)
        for i in range(len(ordered) - 1):
            gap = ordered[i + 1].bbox.x - (ordered[i].bbox.x + ordered[i].bbox.width)
            if 0 <= gap <= 64:
                intra_gaps.append(int(round(gap)))
    intra_histogram = {str(k): v for k, v in sorted(Counter(intra_gaps).items())}

    # ---- notes: what the grid measured, and what could be organized better ----
    if main_rows and column_edges:
        shape = f"{len(column_edges)}-column"
        notes.append(
            f"Grid: {shape} layout over {len(main_rows)} repeated rows; "
            f"gutter {gutter_px:.0f}px, pitch {column_pitch:.0f}px, widest offset "
            f"{symmetry_error_px:.0f}px from the viewport edge."
        )
    elif main_rows:
        notes.append(
            f"Layout: {len(main_rows)} single-column rows, median width "
            f"{column_widths[0]:.0f}px, margins {symmetry_error_px:.0f}px."
        )
    if orphan_selectors:
        notes.append(
            f"Organization: {len(orphan_selectors)} text block(s) end with a dangling word on "
            f"its own line ({', '.join(orphan_selectors[:3])}) — tighten the copy or the block width."
        )
    if grid_metrics.get("grid_holes"):
        notes.append(
            f"Organization: {grid_metrics['grid_holes']} row(s) leave a middle column empty while "
            f"their siblings fill it — fill the slot or merge the neighbours."
        )
    if grid_metrics.get("underfilled_rows"):
        notes.append(
            f"Organization: {grid_metrics['underfilled_rows']} row(s) use under "
            f"{cfg.underfill_ratio * 100:.0f}% of their grid width — give them a full-width role "
            f"or rebalance the cells."
        )
    if grid_metrics.get("skewed_rows"):
        notes.append(
            f"Balance: {grid_metrics['skewed_rows']} row(s) pile most of their width on one side "
            f"while sibling rows spread evenly — rebalance the cell widths."
        )
    if off_grid_count:
        notes.append(
            f"Alignment: {off_grid_count} cell(s) drift off their column line by more than "
            f"{cfg.tolerance_px:.0f}px toward the edge of their grid."
        )
    if fold_components:
        notes.append(
            f"First view: {len(fold_components)} element(s) straddle the fold at "
            f"{first.viewport_height}px and render as partial strips — shift them down or up."
        )
    if content_skew_pct >= 25.0:
        side = "left" if _first_view_side(first_visible_atoms, first.viewport_width) == "left" else "right"
        notes.append(
            f"Balance (fact, not a verdict): the first view's text and controls sit "
            f"{content_skew_pct:.0f}% more on the {side} side. Deliberate sidebars and "
            f"asymmetric layouts are fine — judge whether the visual mass is intended."
        )
    else:
        notes.append(
            f"Measured: every repeated row stays on its column line within "
            f"{cfg.tolerance_px:.0f}px, rows fill at least "
            f"{round(float(grid_metrics.get('min_row_fill_ratio') or 1.0) * 100)}% of their grid "
            f"width, and no row leaves a centre column empty."
        )
    notes = notes[:8]

    metrics: Dict[str, Any] = {
        "scope": "document_union",
        "first_view_screen_id": first.screen_id,
        "content_atoms": {
            "text": counts.get("text", 0),
            "image": counts.get("image", 0),
            "icon": counts.get("icon", 0),
            "button": counts.get("button", 0),
            "total": len(atoms),
        },
        "first_view_atoms": len(first_visible_atoms),
        "row_bands": len(bands),
        "columns": columns,
        "column_widths_px": column_widths,
        "column_edges_px": column_edges,
        "column_pitch_px": column_pitch,
        "gutter_px": gutter_px,
        "column_histogram": {str(k): v for k, v in sorted(Counter(len(c) for c in band_rows).items())},
        "grid_confidence": round(float(grid_confidence), 2),
        "grid_regions": len(clusters),
        "symmetry_error_px": symmetry_error_px,
        "symmetry_error_ratio": symmetry_error_ratio,
        "spacing_axis": spacing_axis,
        "dominant_spacing_px": dominant_spacing,
        "spacing_outliers": spacing_outliers,
        "atom_gap_histogram": intra_histogram,
        "off_grid_count": off_grid_count,
        "whitespace_pct": whitespace_pct,
        # organization / balance facts
        "orphan_lines": len(orphan_selectors),
        "orphan_selectors": orphan_selectors,
        "grid_holes": grid_metrics.get("grid_holes", 0),
        "underfilled_rows": grid_metrics.get("underfilled_rows", 0),
        "min_row_fill_ratio": grid_metrics.get("min_row_fill_ratio"),
        "skewed_rows": grid_metrics.get("skewed_rows", 0),
        "content_skew_pct": content_skew_pct,
        "layout_notes": notes,
        "fold_clipped_count": len(fold_components),
        "fold_clipped_atom_count": len(clipped_atoms),
        "fold_clipped_pct": round(len(clipped_atoms) / max(1, len(first_atoms)) * 100.0, 1),
        "fold_clipped_selectors": clipped_selectors,
    }

    return LayoutResult(metrics=metrics, flags=grid_flags + orphan_flag_tuples + fold_flags)


def _first_view_side(atoms: Sequence[ContentAtom], viewport_w: int) -> str:
    """Which side holds more first-view ink — a fact for the notes, not a verdict."""
    if not atoms or viewport_w <= 0:
        return "left"
    left_band = (0.0, viewport_w / 3.0)
    right_band = (2.0 * viewport_w / 3.0, float(viewport_w))
    left = sum(
        _overlap_width((a.view_bbox.x, a.view_bbox.x + a.view_bbox.width), left_band)
        for a in atoms
    )
    right = sum(
        _overlap_width((a.view_bbox.x, a.view_bbox.x + a.view_bbox.width), right_band)
        for a in atoms
    )
    return "left" if left >= right else "right"
