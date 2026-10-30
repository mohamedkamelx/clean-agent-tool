import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np
from scipy import ndimage
from scipy.ndimage import maximum_filter

from crawler import ScreenCapture, ElementMetadata, BoundingBox
from config import load_thresholds

try:  # optional, used for fast mode
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


CONTAINER_TAGS = {"main", "section", "article", "header", "footer", "nav", "aside", "body", "html"}
TEXT_TAGS = {
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "span", "a", "button", "label",
    "li", "td", "th", "div", "strong", "em", "small", "dd", "dt", "figcaption",
    "blockquote", "code", "pre", "summary", "legend",
}
MEDIA_TAGS = {"img", "svg", "video", "canvas", "picture", "iframe", "object"}


@dataclass
class ElementScore:
    element: ElementMetadata
    attention_inclusive: float
    attention_exclusive: float
    mean_saliency: float
    peak_saliency: float
    attention_share: float  # exclusive share of total saliency, [0, 1]
    inclusive_share: float
    is_container: bool = False
    is_textual: bool = False
    is_media: bool = False
    is_decorative: bool = False
    is_control: bool = False
    is_heading: bool = False
    is_landmark: bool = False
    is_full_page: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selector": self.element.selector,
            "tag": self.element.tag,
            "id": self.element.id,
            "classes": self.element.classes,
            "text": self.element.text,
            "accessible_name": self.element.accessible_name,
            "is_interactive": self.element.is_interactive,
            "is_structural_box": self.element.is_structural_box,
            "is_container": self.is_container,
            "is_textual": self.is_textual,
            "is_media": self.is_media,
            "is_decorative": self.is_decorative,
            "is_heading": self.is_heading,
            "bbox": self.element.bbox.to_dict(),
            "attention_mass": round(self.attention_exclusive, 6),
            "attention_inclusive": round(self.attention_inclusive, 6),
            "attention_share_pct": round(self.attention_share * 100.0, 2),
            "inclusive_share_pct": round(self.inclusive_share * 100.0, 2),
            "mean_saliency": round(self.mean_saliency, 6),
            "peak_saliency": round(self.peak_saliency, 6),
        }


@dataclass
class CrowdingGroup:
    group_id: str
    elements: List[ElementMetadata]
    bbox: BoundingBox
    min_gap_px: float
    threshold_px: float
    min_target_px: float
    severity_score: float
    is_clear_problem: bool
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "selectors": [e.selector for e in self.elements],
            "texts": [e.text[:60] for e in self.elements],
            "bbox": self.bbox.to_dict(),
            "gap_px": round(self.min_gap_px, 2),
            "threshold_px": round(self.threshold_px, 1),
            "target_min_px": round(self.min_target_px, 1),
            "severity_score": round(self.severity_score, 3),
            "is_clear_problem": self.is_clear_problem,
            "description": self.description,
            "element_count": len(self.elements),
        }


@dataclass
class CrowdingConflict:
    """Backward-compatible pair representation (kept for callers that need it)."""
    element_a: ElementMetadata
    element_b: ElementMetadata
    distance_px: float
    critical_threshold_px: float
    severity_score: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_a": {"selector": self.element_a.selector, "text": self.element_a.text, "bbox": self.element_a.bbox.to_dict()},
            "element_b": {"selector": self.element_b.selector, "text": self.element_b.text, "bbox": self.element_b.bbox.to_dict()},
            "distance_px": round(self.distance_px, 2),
            "critical_threshold_px": round(self.critical_threshold_px, 1),
            "severity_score": round(self.severity_score, 3),
            "description": self.description,
        }


@dataclass
class ComplexStructureCluster:
    cluster_id: str
    container_selector: str
    bbox: BoundingBox
    box_count: int
    total_elements: int
    element_tags: List[str]
    density_score: float  # elements per 10k sq px
    attention_share_pct: float
    severity: str
    description: str
    selectors: List[str] = field(default_factory=list)
    columns: int = 1          # max boxes side-by-side in one row band
    control_count: int = 0    # interactive elements inside the cluster
    controls_per_box: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "container_selector": self.container_selector,
            "bbox": self.bbox.to_dict(),
            "box_count": self.box_count,
            "total_elements": self.total_elements,
            "element_tags": self.element_tags,
            "density_score": round(self.density_score, 2),
            "attention_share_pct": round(self.attention_share_pct, 1),
            "severity": self.severity,
            "description": self.description,
            "selectors": self.selectors,
            "columns": self.columns,
            "control_count": self.control_count,
            "controls_per_box": round(self.controls_per_box, 2),
        }


@dataclass
class FlaggedProblemElement:
    selector: str
    tag: str
    text: str
    bbox: BoundingBox
    issue_type: str
    severity: str
    description: str
    attention_stats: Dict[str, Any]
    rule_id: str = ""
    impact: str = "medium"
    confidence: float = 0.5
    evidence_class: str = "statistical_heuristic"
    observed: Dict[str, Any] = field(default_factory=dict)
    expected: Dict[str, Any] = field(default_factory=dict)
    related_selectors: List[str] = field(default_factory=list)
    requires_human: bool = False
    is_clear_problem: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selector": self.selector,
            "tag": self.tag,
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "issue_type": self.issue_type,
            "severity": self.severity,
            "rule_id": self.rule_id,
            "impact": self.impact,
            "confidence": round(self.confidence, 2),
            "evidence_class": self.evidence_class,
            "observed": self.observed,
            "expected": self.expected,
            "related_selectors": self.related_selectors,
            "requires_human": self.requires_human,
            "is_clear_problem": self.is_clear_problem,
            "description": self.description,
            "attention_stats": self.attention_stats,
        }


@dataclass
class ScreenComplexityReport:
    screen_id: str
    url: str
    page_name: str
    view_mode: str
    scroll_index: int
    image_path: str

    noise_score: float
    focus_competition_score: float
    crowding_score: float
    distraction_score: float
    overall_complexity_score: float
    complexity_level: str

    top_focus_elements: List[ElementScore] = field(default_factory=list)
    crowding_groups: List[CrowdingGroup] = field(default_factory=list)
    crowding_conflicts: List[CrowdingConflict] = field(default_factory=list)
    complex_structure_clusters: List[ComplexStructureCluster] = field(default_factory=list)
    flagged_problem_elements: List[FlaggedProblemElement] = field(default_factory=list)
    metric_details: Dict[str, Any] = field(default_factory=dict)

    scroll_y: int = 0
    viewport_width: int = 0
    viewport_height: int = 0
    document_height: int = 0
    saliency_mode: str = "enabled"
    calibration_status: str = "provisional"
    thresholds_version: str = ""
    scores_version: str = "2.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "screen_id": self.screen_id,
            "url": self.url,
            "page_name": self.page_name,
            "view_mode": self.view_mode,
            "scroll_index": self.scroll_index,
            "scroll_y": self.scroll_y,
            "viewport_width": self.viewport_width,
            "viewport_height": self.viewport_height,
            "document_height": self.document_height,
            "image_path": self.image_path,
            "saliency_mode": self.saliency_mode,
            "calibration_status": self.calibration_status,
            "thresholds_version": self.thresholds_version,
            "scores_version": self.scores_version,
            "noise_score": round(self.noise_score, 1),
            "focus_competition_score": round(self.focus_competition_score, 1),
            "crowding_score": round(self.crowding_score, 1),
            "distraction_score": round(self.distraction_score, 1),
            "overall_complexity_score": round(self.overall_complexity_score, 1),
            "complexity_level": self.complexity_level,
            "top_focus_elements": [e.to_dict() for e in self.top_focus_elements[:5]],
            "crowding_groups": [g.to_dict() for g in self.crowding_groups],
            "crowding_conflicts": [c.to_dict() for c in self.crowding_conflicts],
            "complex_structure_clusters": [c.to_dict() for c in self.complex_structure_clusters],
            "flagged_problem_elements": [f.to_dict() for f in self.flagged_problem_elements],
            "metric_details": self.metric_details,
        }


class VisualComplexityScorer:
    """
    Computes visual cognitive complexity by combining deterministic DOM/layout
    facts (geometry, target size, spacing) with DeepGaze saliency.

    Saliency is attributed exclusively: each pixel belongs to exactly one
    element (the deepest/smallest eligible element). Inclusive per-element
    mass is retained only as a diagnostic.
    """

    def __init__(
        self,
        mobile_crowding_threshold: Optional[float] = None,
        desktop_crowding_threshold: Optional[float] = None,
        thresholds: Optional[Dict[str, Any]] = None,
        thresholds_path: Optional[str] = None,
    ):
        self.config = thresholds or load_thresholds(thresholds_path)
        crowding = self.config.get("crowding", {})
        self.mobile_threshold = (
            mobile_crowding_threshold
            if mobile_crowding_threshold is not None
            else float(crowding.get("mobile", {}).get("gap_px", 16.0))
        )
        self.desktop_threshold = (
            desktop_crowding_threshold
            if desktop_crowding_threshold is not None
            else float(crowding.get("desktop", {}).get("gap_px", 8.0))
        )
        self.thresholds_version = str(self.config.get("version", "unknown"))
        self.provisional = bool(self.config.get("provisional", True))

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_bounding_box_distance(b1: BoundingBox, b2: BoundingBox) -> float:
        dx = max(0, max(b1.x - (b2.x + b2.width), b2.x - (b1.x + b1.width)))
        dy = max(0, max(b1.y - (b2.y + b2.height), b2.y - (b1.y + b1.height)))
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _bbox_contains(outer: BoundingBox, inner: BoundingBox, tolerance: float = 2.0) -> bool:
        return (
            outer.x - tolerance <= inner.x
            and outer.y - tolerance <= inner.y
            and outer.x + outer.width + tolerance >= inner.x + inner.width
            and outer.y + outer.height + tolerance >= inner.y + inner.height
        )

    @staticmethod
    def _is_ancestor(ancestor_idx: int, child_idx: int, elements: List[ElementMetadata]) -> bool:
        if ancestor_idx < 0 or child_idx < 0 or ancestor_idx >= len(elements) or child_idx >= len(elements):
            return False
        return ancestor_idx in elements[child_idx].ancestor_indices

    def _is_nested(self, a: ElementMetadata, b: ElementMetadata) -> bool:
        """DOM ancestry first, then geometric containment as a fallback."""
        if a.element_index >= 0 and b.element_index >= 0:
            if a.element_index in b.ancestor_indices or b.element_index in a.ancestor_indices:
                return True
        if self._bbox_contains(a.bbox, b.bbox) or self._bbox_contains(b.bbox, a.bbox):
            return True
        return False

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def _classify(self, el: ElementMetadata, width: int, height: int) -> Dict[str, bool]:
        tag = el.tag.lower()
        is_control = bool(el.is_interactive)
        is_heading = el.heading_level > 0
        is_landmark = tag in CONTAINER_TAGS and not is_control
        has_text = bool(el.has_direct_text and (el.text or "").strip())
        is_textual = has_text or (is_heading and bool((el.accessible_name or el.text or "").strip()))
        is_media = tag in MEDIA_TAGS
        is_container = (not is_control) and (not is_textual) and (not is_media) and el.child_count > 0
        is_decorative = (
            (is_media and not (el.alt or "").strip() and not (el.accessible_name or "").strip())
            or (not is_control and not is_textual and not is_media and el.has_background_image)
        )
        is_full_page = el.bbox.width >= width * 0.95 and el.bbox.height >= height * 0.90
        return {
            "is_control": is_control,
            "is_heading": is_heading,
            "is_landmark": is_landmark,
            "is_textual": is_textual,
            "is_media": is_media,
            "is_container": is_container,
            "is_decorative": is_decorative,
            "is_full_page": is_full_page,
        }

    @staticmethod
    def _priority(flags: Dict[str, bool]) -> int:
        if flags["is_control"]:
            return 0
        if flags["is_media"]:
            return 1
        if flags["is_textual"]:
            return 2
        if flags["is_container"]:
            return 3
        return 4

    # ------------------------------------------------------------------ #
    # Saliency attribution
    # ------------------------------------------------------------------ #
    def _attribute_saliency(
        self,
        heatmap: np.ndarray,
        screen: ScreenCapture,
        width: int,
        height: int,
    ) -> Tuple[List[ElementScore], np.ndarray, float]:
        total = float(np.sum(heatmap))
        if total <= 0:
            total = 1.0

        scores: List[ElementScore] = []
        owner = np.full((height, width), -1, dtype=np.int32)

        # Claim pixels. Controls and media claim before text; text before containers.
        order = []
        for idx, el in enumerate(screen.elements):
            flags = self._classify(el, width, height)
            order.append((self._priority(flags), -int(el.dom_depth), el.bbox.area, idx, flags))
        order.sort(key=lambda t: (t[0], t[1], t[2]))

        for _, _, _, idx, _ in order:
            el = screen.elements[idx]
            bx = max(0, min(width - 1, el.bbox.x))
            by = max(0, min(height - 1, el.bbox.y))
            bw = max(1, min(width - bx, el.bbox.width))
            bh = max(1, min(height - by, el.bbox.height))
            if bw <= 0 or bh <= 0:
                continue
            sub = owner[by:by + bh, bx:bx + bw]
            sub[sub == -1] = idx

        owned_pixels = owner >= 0
        exclusive_mass = np.bincount(
            owner[owned_pixels],
            weights=heatmap[owned_pixels],
            minlength=len(screen.elements),
        ).astype(np.float64)
        unassigned = float(np.sum(heatmap[owner == -1]))
        element_mask = owner >= 0

        for idx, el in enumerate(screen.elements):
            flags = self._classify(el, width, height)
            bx = max(0, min(width - 1, el.bbox.x))
            by = max(0, min(height - 1, el.bbox.y))
            bw = max(1, min(width - bx, el.bbox.width))
            bh = max(1, min(height - by, el.bbox.height))
            patch = heatmap[by:by + bh, bx:bx + bw] if bw > 0 and bh > 0 else np.array([0.0])
            inclusive = float(np.sum(patch))
            exc = float(exclusive_mass[idx]) if idx < len(exclusive_mass) else 0.0
            scores.append(ElementScore(
                element=el,
                attention_inclusive=inclusive,
                attention_exclusive=exc,
                mean_saliency=float(np.mean(patch)) if patch.size else 0.0,
                peak_saliency=float(np.max(patch)) if patch.size else 0.0,
                attention_share=exc / total,
                inclusive_share=inclusive / total,
                is_container=flags["is_container"],
                is_textual=flags["is_textual"],
                is_media=flags["is_media"],
                is_decorative=flags["is_decorative"],
                is_control=flags["is_control"],
                is_heading=flags["is_heading"],
                is_landmark=flags["is_landmark"],
                is_full_page=flags["is_full_page"],
            ))

        return scores, owner, unassigned

    # ------------------------------------------------------------------ #
    # Content selection
    # ------------------------------------------------------------------ #
    def _select_distinct_content(
        self,
        scores: List[ElementScore],
        width: int,
        height: int,
        limit: Optional[int] = None,
    ) -> List[ElementScore]:
        candidates = []
        for s in scores:
            if s.is_container or s.is_landmark or s.is_full_page:
                continue
            if s.element.bbox.width >= width * 0.90 and s.element.bbox.height >= height * 0.70:
                continue
            if s.attention_inclusive <= 0:
                continue
            if not (s.is_textual or s.is_control or s.is_media):
                continue
            candidates.append(s)

        candidates.sort(key=lambda s: s.attention_inclusive, reverse=True)
        selected: List[ElementScore] = []
        for cand in candidates:
            if limit is not None and len(selected) >= limit:
                break
            nested = False
            for sel in selected:
                if self._is_nested(cand.element, sel.element):
                    nested = True
                    break
            if nested:
                continue
            selected.append(cand)
        return selected

    # ------------------------------------------------------------------ #
    # Noise / focus / distraction features
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bounded_score(features: Dict[str, float], section: Dict[str, Any]) -> float:
        """Score from absolute, published thresholds — not corpus statistics.

        Each feature contributes `weight * clamp((value - zero)/(sat - zero), 0, 1)`,
        i.e. a bounded 0..1 ramp anchored on a fixed UX reference point (`zero`) and
        a saturation point (`sat`). The final score is a weighted mean scaled to 0..100.

        Why this shape:
        - `zero`/`sat` are absolute (e.g. "density_outlier > 40% of the viewport is
          saturated"), so the score means the same thing on any design. It is NOT
          centered on whatever pages happened to be in the calibration set.
        - Every feature's contribution is capped by its weight, so no single feature
          can swing the score across the whole 0..100 range (the failure mode of the
          previous fitted-logistic model, where tiny_text alone moved noise by 60 pts).
        - Missing/unknown features contribute 0 rather than a guessed default.
        """
        spec = section.get("features", {}) or {}
        weighted_sum = 0.0
        weight_total = 0.0
        for name, rule in spec.items():
            if name not in features:
                continue
            try:
                value = float(features[name])
                zero = float(rule.get("zero", 0.0))
                sat = float(rule.get("sat", 1.0))
                weight = float(rule.get("weight", 1.0))
            except (TypeError, ValueError):
                continue
            span = sat - zero
            frac = 0.0 if span == 0 else (value - zero) / span
            frac = float(np.clip(frac, 0.0, 1.0))
            weighted_sum += weight * frac
            weight_total += abs(weight)
        if weight_total <= 0:
            return 0.0
        return float(np.clip(100.0 * weighted_sum / weight_total, 0.0, 100.0))

    # Backward-compatible alias (older configs / callers used the fitted-logistic name).
    _calibrated_score = staticmethod(_bounded_score.__func__)

    @staticmethod
    def _font_px(score) -> float:
        raw = str(score.element.computed_style.get("fontSize", "")).replace("px", "")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _peak_metrics(heatmap: np.ndarray) -> Tuple[int, float, float]:
        height, width = heatmap.shape[:2]
        size = max(9, int(min(height, width) * 0.02) * 2 + 1)
        maxf = maximum_filter(heatmap, size=size)
        threshold = float(np.percentile(heatmap, 99.0))
        peaks = (heatmap >= maxf) & (heatmap > threshold)
        labels, n = ndimage.label(peaks)
        if n <= 0:
            return 0, 0.0, 0.0
        masses = []
        radius = max(6, size // 2)
        for label_id in range(1, n + 1):
            ys, xs = np.where(labels == label_id)
            if ys.size == 0:
                continue
            cy, cx = int(ys.mean()), int(xs.mean())
            y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
            x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
            masses.append(float(np.sum(heatmap[y0:y1, x0:x1])))
        total = float(np.sum(masses))
        if total <= 0:
            return n, 0.0, 0.0
        p = np.array(masses, dtype=np.float64) / total
        effective = float(1.0 / np.sum(p * p)) if np.sum(p * p) > 0 else 0.0
        top1 = float(np.max(p))
        return n, effective, top1

    @staticmethod
    def _density_metrics(owner: np.ndarray, scores: List[ElementScore], width: int, height: int) -> Tuple[float, float]:
        content_mask = np.zeros((height, width), dtype=np.uint8)
        for s in scores:
            if not (s.is_textual or s.is_control or s.is_media):
                continue
            b = s.element.bbox
            x0, y0 = max(0, b.x), max(0, b.y)
            x1, y1 = min(width, b.x + b.width), min(height, b.y + b.height)
            if x1 > x0 and y1 > y0:
                content_mask[y0:y1, x0:x1] = 1
        content_coverage = float(content_mask.mean() * 100.0)
        tile = 64
        means = []
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                block = content_mask[y:y + tile, x:x + tile]
                if block.size:
                    means.append(float(block.mean()))
        density_outlier = float(np.mean(np.array(means) > 0.60) * 100.0) if means else 0.0
        return content_coverage, density_outlier

    # ------------------------------------------------------------------ #
    # Complex structures
    # ------------------------------------------------------------------ #
    def _looks_like_box(self, el: ElementMetadata, screen_w: int, screen_h: int) -> bool:
        # Size is judged in DOCUMENT space when available: a card clipped by a
        # scroll-slice boundary has a truncated viewport bbox (e.g. 17px of a 162px
        # card) but is still a real card. Using the visible height here would drop
        # every card that straddles a slice edge — a major recall gap on long pages.
        doc = el.doc_bbox
        box_w = doc.width if doc is not None else el.bbox.width
        box_h = doc.height if doc is not None else el.bbox.height
        if box_w < 60 or box_h < 35:
            return False
        if box_w >= screen_w * 0.98 and box_h >= screen_h * 0.92:
            return False
        # Reject only true slivers (a card barely peeking in at the very edge).
        if el.visible_fraction < 0.15:
            return False
        style = el.computed_style or {}
        radius = str(style.get("borderRadius", "0px"))
        has_radius = radius not in ("0px", "0", "")
        class_id = f"{el.classes} {el.id}".lower()
        explicit = any(k in class_id for k in ("card", "panel", "tile", "widget"))
        return bool(
            explicit
            or (el.has_background_color and (el.has_border or el.has_shadow or has_radius))
            or el.has_shadow
        )

    def _detect_complex_structures(
        self,
        scores: List[ElementScore],
        screen: ScreenCapture,
        width: int,
        height: int,
        total_saliency: float,
    ) -> List[ComplexStructureCluster]:
        clusters: List[ComplexStructureCluster] = []
        view_mode = screen.view_mode
        elements = screen.elements

        box_scores: List[ElementScore] = []
        for s in scores:
            el = s.element
            if s.is_control or s.is_textual or s.is_landmark or s.is_full_page:
                continue
            if not el.is_structural_box:
                continue
            if not self._looks_like_box(el, width, height):
                continue
            box_scores.append(s)

        n = len(box_scores)
        if n < 2:
            return clusters

        # Adjacency graph (spatial proximity, not DOM nesting).
        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
        proximity = 32.0 if view_mode == "mobile" else 20.0
        for i in range(n):
            for j in range(i + 1, n):
                if self._is_nested(box_scores[i].element, box_scores[j].element):
                    continue
                dist = self._compute_bounding_box_distance(box_scores[i].element.bbox, box_scores[j].element.bbox)
                if dist <= proximity:
                    adj[i].add(j)
                    adj[j].add(i)

        # Connected components: process EVERY component (the original loop bug).
        visited: Set[int] = set()
        components: List[List[int]] = []
        for start in range(n):
            if start in visited:
                continue
            component: List[int] = []
            queue = [start]
            visited.add(start)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)

        for component in components:
            if len(component) < 2:
                continue

            comp_boxes = [box_scores[i] for i in component]
            leaf_boxes = [
                b for b in comp_boxes
                if not any(other is not b and self._bbox_contains(other.element.bbox, b.element.bbox) for other in comp_boxes)
            ]
            if len(leaf_boxes) < 2:
                leaf_boxes = comp_boxes
            box_count = len(leaf_boxes)

            min_x = min(int(b.element.bbox.x) for b in leaf_boxes)
            min_y = min(int(b.element.bbox.y) for b in leaf_boxes)
            max_x = max(int(b.element.bbox.x + b.element.bbox.width) for b in leaf_boxes)
            max_y = max(int(b.element.bbox.y + b.element.bbox.height) for b in leaf_boxes)
            cluster_w = max(1, max_x - min_x)
            cluster_h = max(1, max_y - min_y)
            cluster_area = float(cluster_w * cluster_h)

            # ---- Grid vs list: how many boxes sit side-by-side in one row band? ----
            # A multi-column grid (>=2 boxes sharing a row) is genuinely complex.
            # A single-column list (one box per row, stacked) is a scannable pattern
            # and should NOT be flagged just because it has repeated rows.
            row_tol = max(24.0, 0.5 * min(b.element.bbox.height for b in leaf_boxes))
            bands: List[Dict[str, Any]] = []
            for b in sorted(leaf_boxes, key=lambda x: x.element.bbox.center_y):
                cy = b.element.bbox.center_y
                if bands and abs(cy - bands[-1]["cy"]) <= row_tol:
                    bands[-1]["boxes"].append(b)
                    k = len(bands[-1]["boxes"])
                    bands[-1]["cy"] = (bands[-1]["cy"] * (k - 1) + cy) / k
                else:
                    bands.append({"cy": cy, "boxes": [b]})
            columns = max((len(band["boxes"]) for band in bands), default=1)

            box_selectors = {b.element.selector for b in leaf_boxes}
            contained: List[ElementScore] = []
            control_count = 0
            for s in scores:
                if s.element.selector in box_selectors:
                    if s.is_control:
                        control_count += 1
                    continue
                if s.is_container or s.is_landmark:
                    continue
                if not (s.is_textual or s.is_control or s.is_media):
                    continue
                cx = s.element.bbox.center_x
                cy = s.element.bbox.center_y
                if min_x - 4 <= cx <= max_x + 4 and min_y - 4 <= cy <= max_y + 4:
                    contained.append(s)
                    if s.is_control:
                        control_count += 1

            total_elements = len(contained)
            density = (total_elements * 10000.0) / max(1.0, cluster_area)
            controls_per_box = control_count / float(box_count)

            # ---- Attention: exclusive masses are disjoint, so this never exceeds 100%. ----
            attention_mass = 0.0
            for s in scores:
                cx = s.element.bbox.center_x
                cy = s.element.bbox.center_y
                if min_x - 4 <= cx <= max_x + 4 and min_y - 4 <= cy <= max_y + 4:
                    attention_mass += s.attention_exclusive
            attention_pct = min(100.0, (attention_mass / max(total_saliency, 1e-12)) * 100.0)

            # ---- Decision: complexity = (grid OR dense-interactive) AND draws attention. ----
            # DeepGaze gates this: a region the model says nobody looks at is not a
            # visual-complexity problem, however many boxes it technically contains.
            is_grid = columns >= 2 and box_count >= 3 and total_elements >= 6
            is_dense_interactive = (columns == 1 and box_count >= 4 and controls_per_box >= 2.0)
            draws_attention = attention_pct >= 6.0
            if not ((is_grid or is_dense_interactive) and draws_attention):
                continue

            # Lowest common ancestor of the leaf boxes.
            common: Optional[Set[int]] = None
            for b in leaf_boxes:
                ancestors = set(b.element.ancestor_indices)
                common = ancestors if common is None else (common & ancestors)
            container_selector = "box-cluster"
            if common:
                candidate_indices = [i for i in common if 0 <= i < len(elements)]
                if candidate_indices:
                    lca_idx = max(candidate_indices, key=lambda i: elements[i].dom_depth)
                    container_selector = elements[lca_idx].selector

            layout = f"{columns}-column grid" if columns >= 2 else "single-column list"
            tags = sorted({s.element.tag for s in contained})
            desc = (
                f"Repeated {layout} under '{container_selector}': {box_count} boxes "
                f"({control_count} controls, {controls_per_box:.1f}/box) and {total_elements} content "
                f"elements in a {cluster_w}x{cluster_h}px area (density {density:.1f}/10k px2) on "
                f"{view_mode}. DeepGaze predicts {attention_pct:.1f}% of attention lands here."
            )

            clusters.append(ComplexStructureCluster(
                cluster_id=f"cluster_{len(clusters) + 1}_{view_mode}",
                container_selector=container_selector,
                bbox=BoundingBox(x=min_x, y=min_y, width=cluster_w, height=cluster_h),
                box_count=box_count,
                total_elements=total_elements,
                element_tags=tags,
                density_score=density,
                attention_share_pct=attention_pct,
                severity="HIGH" if attention_pct >= 15.0 else "MEDIUM",
                description=desc,
                selectors=[b.element.selector for b in leaf_boxes],
                columns=columns,
                control_count=control_count,
                controls_per_box=controls_per_box,
            ))

        return clusters

    # ------------------------------------------------------------------ #
    # Crowding
    # ------------------------------------------------------------------ #
    def _detect_crowding_groups(
        self,
        scores: List[ElementScore],
        view_mode: str,
        total_saliency: float,
    ) -> Tuple[List[CrowdingGroup], List[CrowdingConflict], float]:
        threshold = self.mobile_threshold if view_mode == "mobile" else self.desktop_threshold
        crowding_cfg = self.config.get("crowding", {}).get(view_mode, {})
        clear_gap = float(crowding_cfg.get("clear_gap_px", 4.0 if view_mode == "desktop" else 8.0))
        comfortable_target = float(crowding_cfg.get("comfortable_target_px", 44.0))
        clear_target = float(crowding_cfg.get("clear_target_px", 24.0))

        controls = [s for s in scores if s.is_control and s.element.bbox.area > 0]
        n = len(controls)
        pairs: List[Tuple[int, int, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = controls[i], controls[j]
                if self._is_nested(a.element, b.element):
                    continue
                dist = self._compute_bounding_box_distance(a.element.bbox, b.element.bbox)
                if dist < threshold:
                    pairs.append((i, j, dist))

        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
        for i, j, _ in pairs:
            adj[i].add(j)
            adj[j].add(i)

        visited: Set[int] = set()
        groups: List[CrowdingGroup] = []
        conflicts: List[CrowdingConflict] = []
        total_penalty = 0.0

        for start in range(n):
            if start in visited or not adj[start]:
                continue
            component: List[int] = []
            queue = [start]
            visited.add(start)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(component) < 2:
                continue

            comp_controls = [controls[i] for i in component]
            comp_pairs = [(i, j, d) for i, j, d in pairs if i in component and j in component]
            min_gap = min(d for _, _, d in comp_pairs)
            min_target = min(min(s.element.bbox.width, s.element.bbox.height) for s in comp_controls)

            min_x = min(s.element.bbox.x for s in comp_controls)
            min_y = min(s.element.bbox.y for s in comp_controls)
            max_x = max(s.element.bbox.x + s.element.bbox.width for s in comp_controls)
            max_y = max(s.element.bbox.y + s.element.bbox.height for s in comp_controls)

            is_clear = (min_target < clear_target) or (min_gap < clear_gap and min_target < comfortable_target)
            closeness = max(0.0, 1.0 - (min_gap / max(threshold, 1e-9)))
            severity = closeness * (1.0 if is_clear else 0.45)
            total_penalty += severity

            label = "touch" if view_mode == "mobile" else "pointer"
            desc = (
                f"{len(comp_controls)} adjacent controls are separated by {min_gap:.1f}px "
                f"(threshold {threshold:.0f}px); smallest target is "
                f"{min(comp_controls, key=lambda s: min(s.element.bbox.width, s.element.bbox.height)).element.bbox.width}x"
                f"{min(comp_controls, key=lambda s: min(s.element.bbox.width, s.element.bbox.height)).element.bbox.height}px. "
                f"Check {label} spacing and target size."
            )
            groups.append(CrowdingGroup(
                group_id=f"crowd_{len(groups) + 1}_{view_mode}",
                elements=[s.element for s in comp_controls],
                bbox=BoundingBox(x=int(min_x), y=int(min_y), width=int(max_x - min_x), height=int(max_y - min_y)),
                min_gap_px=min_gap,
                threshold_px=threshold,
                min_target_px=min_target,
                severity_score=severity,
                is_clear_problem=is_clear,
                description=desc,
            ))

            for i, j, dist in comp_pairs:
                a, b = controls[i], controls[j]
                conflicts.append(CrowdingConflict(
                    element_a=a.element,
                    element_b=b.element,
                    distance_px=dist,
                    critical_threshold_px=threshold,
                    severity_score=max(0.0, 1.0 - dist / max(threshold, 1e-9)),
                    description=f"'{a.element.selector}' and '{b.element.selector}' are {dist:.1f}px apart.",
                ))

        weight = float(self.config.get("crowding", {}).get("severity_weight", 25.0))
        crowding_score = float(np.clip(total_penalty * weight, 0.0, 100.0))
        return groups, conflicts, crowding_score

    # ------------------------------------------------------------------ #
    # Main analysis
    # ------------------------------------------------------------------ #
    def analyze_screen(
        self,
        heatmap: Optional[np.ndarray],
        screen_capture: ScreenCapture,
        saliency_mode: str = "enabled",
    ) -> ScreenComplexityReport:
        if heatmap is None:
            height = max(1, int(screen_capture.viewport_height))
            width = max(1, int(screen_capture.viewport_width))
            heatmap = np.full((height, width), 1.0 / (height * width), dtype=np.float32)
            saliency_mode = "disabled_uniform"
        heatmap = np.asarray(heatmap, dtype=np.float64)
        if heatmap.ndim != 2:
            raise ValueError(f"heatmap must be 2D, got shape {heatmap.shape}")
        height, width = heatmap.shape[:2]
        total_saliency = float(np.sum(heatmap))
        if total_saliency <= 0:
            heatmap = np.full((height, width), 1.0 / (height * width), dtype=np.float64)
            total_saliency = 1.0

        view_mode = screen_capture.view_mode
        scores, owner, unassigned = self._attribute_saliency(heatmap, screen_capture, width, height)

        # ---------- noise features ----------
        container_mass = sum(s.attention_exclusive for s in scores if s.is_container)
        decorative_mass = sum(s.attention_exclusive for s in scores if s.is_decorative)
        control_mass = sum(s.attention_exclusive for s in scores if s.is_control)
        textual_mass = sum(s.attention_exclusive for s in scores if s.is_textual)
        media_mass = sum(s.attention_exclusive for s in scores if s.is_media)
        n_peaks, effective_peaks, peak_top1 = self._peak_metrics(heatmap)
        content_coverage, density_outlier = self._density_metrics(owner, scores, width, height)
        unexplained_pct = (unassigned / total_saliency) * 100.0
        container_pct = (container_mass / total_saliency) * 100.0
        decorative_pct = (decorative_mass / total_saliency) * 100.0
        control_pct = (control_mass / total_saliency) * 100.0
        textual_pct = (textual_mass / total_saliency) * 100.0
        media_pct = (media_mass / total_saliency) * 100.0
        effective_peaks_norm = float(np.clip(math.log2(max(1.0, effective_peaks)) / math.log2(8.0), 0.0, 1.5))

        # Deterministic DOM-density features. These are independent of saliency and
        # therefore remain meaningful when saliency is politely uncertain.
        visible_count = len(scores)
        content_scores = [s for s in scores if s.is_textual or s.is_control or s.is_media]
        text_scores = [s for s in scores if s.is_textual]
        control_scores = [s for s in scores if s.is_control]
        area_factor = max(1.0, (width * height) / 100000.0)
        element_density = visible_count / area_factor
        content_density = len(content_scores) / area_factor
        control_density = len(control_scores) / area_factor
        text_density = len(text_scores) / area_factor

        font_sizes = []
        for s in text_scores:
            raw_size = str(s.element.computed_style.get("fontSize", "")).replace("px", "")
            try:
                font_sizes.append(float(raw_size))
            except Exception:
                pass
        tiny_text_pct = (
            (sum(1 for size in font_sizes if size < 12.0) / len(font_sizes)) * 100.0
            if font_sizes else 0.0
        )
        font_size_variety = len(set(round(size, 1) for size in font_sizes))
        small_controls = [
            s for s in control_scores
            if min(s.element.bbox.width, s.element.bbox.height) < 24
        ]
        small_control_pct = (len(small_controls) / len(control_scores) * 100.0) if control_scores else 0.0

        # ---------- content selection / focus features ----------
        selected = self._select_distinct_content(scores, width, height)
        masses = np.array([s.attention_inclusive for s in selected], dtype=np.float64)
        if masses.size and masses.sum() > 0:
            q = masses / masses.sum()
            effective_targets = float(1.0 / np.sum(q * q))
            top1_share = float(q[0])
            balance = float(1.0 - top1_share)
        else:
            effective_targets = 1.0
            top1_share = 0.0
            balance = 0.0

        control_selected = [s for s in selected if s.is_control]
        if len(control_selected) >= 1:
            cm = np.array([s.attention_inclusive for s in control_selected], dtype=np.float64)
            cq = cm / cm.sum() if cm.sum() > 0 else np.array([1.0])
            control_effective_targets = float(1.0 / np.sum(cq * cq))
        else:
            control_effective_targets = 0.0

        heading_selected = [s for s in selected if s.is_heading]
        primary_heading = max(heading_selected, key=lambda s: s.attention_inclusive) if heading_selected else None
        secondary_noninteractive = [
            s for s in selected
            if (not s.is_control) and s is not primary_heading
        ]
        secondary_mass = sum(s.attention_inclusive for s in secondary_noninteractive)
        secondary_content_share_pct = (secondary_mass / total_saliency) * 100.0
        primary_heading_share_pct = (primary_heading.attention_inclusive / total_saliency * 100.0) if primary_heading else 0.0

        # ---------- calibrated scores ----------
        noise_features = {
            "unexplained_pct": unexplained_pct,
            "container_pct": container_pct,
            "decorative_pct": decorative_pct,
            "effective_peaks_norm": effective_peaks_norm,
            "density_outlier_pct": density_outlier,
            "content_coverage_pct": content_coverage,
            "element_density": element_density,
            "content_density": content_density,
            "control_density": control_density,
            "text_density": text_density,
            "tiny_text_pct": tiny_text_pct,
            "font_size_variety": float(font_size_variety),
            "small_control_pct": small_control_pct,
            "secondary_content_share_pct": secondary_content_share_pct,
        }
        focus_features = {
            "effective_targets": effective_targets,
            "balance": balance,
            "control_effective_targets": control_effective_targets,
            "secondary_content_share_pct": secondary_content_share_pct,
            "top1_share": top1_share,
        }
        distraction_features = {
            "secondary_content_share_pct": secondary_content_share_pct,
            "decorative_pct": decorative_pct,
            "unexplained_pct": unexplained_pct,
            "effective_peaks_norm": effective_peaks_norm,
        }
        noise_score = self._bounded_score(noise_features, self.config.get("noise", {}))
        focus_competition_score = self._bounded_score(focus_features, self.config.get("focus", {}))
        distraction_score = self._bounded_score(distraction_features, self.config.get("distraction", {}))

        # ---------- structures & crowding ----------
        complex_clusters = self._detect_complex_structures(scores, screen_capture, width, height, total_saliency)
        crowding_groups, crowding_conflicts, crowding_score = self._detect_crowding_groups(
            scores, view_mode, total_saliency
        )

        # Structural clusters contribute to crowding but are also reported separately.
        if complex_clusters:
            cluster_boost = sum(min(25.0, max(0.0, (c.total_elements - 4) * 2.0)) for c in complex_clusters)
            crowding_score = float(np.clip(crowding_score + cluster_boost, 0.0, 100.0))

        # ---------- overall ----------
        weights = self.config.get("weights", {})
        overall = (
            float(weights.get("noise", 0.25)) * noise_score
            + float(weights.get("focus", 0.35)) * focus_competition_score
            + float(weights.get("crowding", 0.30)) * crowding_score
            + float(weights.get("distraction", 0.10)) * distraction_score
        )
        overall = float(np.clip(overall, 0.0, 100.0))
        levels = self.config.get("levels", {"low": 40.0, "high": 65.0})
        if overall < float(levels.get("low", 40.0)):
            complexity_level = "LOW"
        elif overall < float(levels.get("high", 65.0)):
            complexity_level = "MODERATE"
        else:
            complexity_level = "OVER_COMPLEX"

        # ---------- flags ----------
        flagged: List[FlaggedProblemElement] = []
        flagged_selectors: Set[str] = set()

        for cluster in complex_clusters:
            flagged.append(FlaggedProblemElement(
                selector=cluster.container_selector,
                tag="cluster",
                text=f"{cluster.box_count} boxes / {cluster.total_elements} elements",
                bbox=cluster.bbox,
                issue_type="COMPLEX_STRUCTURE",
                severity="potential_improvement",
                rule_id="CLUSTER_REPEATED_BOXES",
                impact="medium",
                confidence=0.7,
                evidence_class="statistical_heuristic",
                observed={
                    "box_count": cluster.box_count,
                    "total_elements": cluster.total_elements,
                    "density": round(cluster.density_score, 2),
                    "columns": cluster.columns,
                    "control_count": cluster.control_count,
                    "controls_per_box": round(cluster.controls_per_box, 2),
                    "layout": "grid" if cluster.columns >= 2 else "dense_list",
                },
                expected={"box_count": max(1, cluster.box_count - 2)},
                related_selectors=cluster.selectors,
                description=cluster.description,
                attention_stats={"attention_share_pct": round(cluster.attention_share_pct, 2)},
            ))

        crowd_cfg = self.config.get("crowding", {}).get(view_mode, {})
        for group in crowding_groups:
            severity = "clear_problem" if group.is_clear_problem else "potential_improvement"
            first = group.elements[0]
            flagged.append(FlaggedProblemElement(
                selector=first.selector,
                tag=first.tag,
                text=f"{len(group.elements)} controls, gap {group.min_gap_px:.1f}px",
                bbox=group.bbox,
                issue_type="CROWDING_VIOLATION",
                severity=severity,
                rule_id="CROWDING_GROUP",
                impact="high" if group.is_clear_problem else "medium",
                confidence=0.9 if group.is_clear_problem else 0.6,
                evidence_class="deterministic_rule" if group.is_clear_problem else "statistical_heuristic",
                observed={"gap_px": round(group.min_gap_px, 2), "target_min_px": round(group.min_target_px, 1), "group_size": len(group.elements)},
                expected={"gap_px": group.threshold_px, "target_px": float(crowd_cfg.get("comfortable_target_px", 44.0))},
                related_selectors=[e.selector for e in group.elements],
                requires_human=False,
                is_clear_problem=group.is_clear_problem,
                description=group.description,
                attention_stats={"gap_px": round(group.min_gap_px, 2), "severity": round(group.severity_score, 3)},
            ))

        # Focus competition: only when a second target is genuinely prominent, not
        # merely present. A heading and its subtitle are not competitors just because
        # both are visible.
        runner_up_share = selected[1].inclusive_share if len(selected) >= 2 else 0.0
        top_share = selected[0].inclusive_share if selected else 0.0
        comparable = (
            len(selected) >= 2
            and runner_up_share >= 0.08
            and (top_share <= 0.0 or (runner_up_share / top_share) >= 0.60)
            and effective_targets >= 2.4
            and focus_competition_score >= 60.0
        )
        if comparable:
            runner_up = selected[1]
            if runner_up.element.selector not in flagged_selectors:
                flagged_selectors.add(runner_up.element.selector)
                flagged.append(FlaggedProblemElement(
                    selector=runner_up.element.selector,
                    tag=runner_up.element.tag,
                    text=runner_up.element.text,
                    bbox=runner_up.element.bbox,
                    issue_type="FOCUS_COMPETITION",
                    severity="potential_improvement",
                    rule_id="FOCUS_MULTIPLE_TARGETS",
                    impact="medium",
                    confidence=0.55,
                    evidence_class="statistical_heuristic",
                    observed={"effective_targets": round(effective_targets, 2), "runner_up_share_pct": round(runner_up.inclusive_share * 100.0, 2)},
                    expected={"effective_targets": 1.5},
                    related_selectors=[selected[0].element.selector],
                    description=(
                        f"'{runner_up.element.selector}' competes with '{selected[0].element.selector}' "
                        f"for visual prominence ({runner_up.attention_inclusive:.3f} vs {selected[0].attention_inclusive:.3f} inclusive mass)."
                    ),
                    attention_stats={"effective_targets": round(effective_targets, 2), "top1_share": round(top1_share, 3)},
                ))

        # Distraction: attention on secondary non-interactive content.
        if distraction_score >= 55.0 and secondary_noninteractive:
            top_secondary = max(secondary_noninteractive, key=lambda s: s.attention_inclusive)
            if top_secondary.element.selector not in flagged_selectors:
                flagged_selectors.add(top_secondary.element.selector)
                flagged.append(FlaggedProblemElement(
                    selector=top_secondary.element.selector,
                    tag=top_secondary.element.tag,
                    text=top_secondary.element.text,
                    bbox=top_secondary.element.bbox,
                    issue_type="VISUAL_DISTRACTION",
                    severity="needs_investigation",
                    rule_id="SECONDARY_CONTENT_ATTENTION",
                    impact="medium",
                    confidence=0.5,
                    evidence_class="statistical_heuristic",
                    observed={"secondary_content_share_pct": round(secondary_content_share_pct, 2), "element_share_pct": round(top_secondary.inclusive_share * 100.0, 2)},
                    expected={"secondary_content_share_pct": 10.0},
                    related_selectors=[primary_heading.element.selector] if primary_heading else [],
                    requires_human=True,
                    description=(
                        f"Secondary non-interactive content absorbs {secondary_content_share_pct:.1f}% of predicted attention "
                        f"(element: '{top_secondary.element.selector}', {top_secondary.inclusive_share * 100.0:.1f}%). "
                        f"Check whether it is necessary for the task before reducing or moving it."
                    ),
                    attention_stats={"secondary_content_share_pct": round(secondary_content_share_pct, 2)},
                ))

        # Legibility: reported as its OWN finding, scored against an absolute
        # font-size threshold. It is deliberately NOT part of noise/clutter.
        legibility_cfg = self.config.get("legibility", {}) or {}
        tiny_px = float(legibility_cfg.get("tiny_text_threshold_px", 12.0))
        if font_sizes:
            tiny_count = sum(1 for size in font_sizes if size < tiny_px)
            smallest = min(font_sizes)
            if tiny_text_pct >= float(legibility_cfg.get("tiny_text_pct", {}).get("zero", 5.0)):
                # Anchor the finding on the smallest offending text element.
                tiny_el = next(
                    (s for s in text_scores
                     if self._font_px(s) < tiny_px),
                    text_scores[0] if text_scores else None,
                )
                if tiny_el is not None and tiny_el.element.selector not in flagged_selectors:
                    flagged_selectors.add(tiny_el.element.selector)
                    flagged.append(FlaggedProblemElement(
                        selector=tiny_el.element.selector,
                        tag=tiny_el.element.tag,
                        text=tiny_el.element.text,
                        bbox=tiny_el.element.bbox,
                        issue_type="TEXT_TOO_SMALL",
                        severity="potential_improvement" if tiny_text_pct < 25.0 else "clear_problem",
                        rule_id="MIN_FONT_SIZE",
                        impact="medium" if tiny_text_pct < 25.0 else "high",
                        confidence=0.85,
                        evidence_class="deterministic_rule",
                        observed={"tiny_text_pct": round(tiny_text_pct, 2), "smallest_px": round(smallest, 1), "text_elements": len(font_sizes)},
                        expected={"threshold_px": tiny_px},
                        related_selectors=[],
                        description=(
                            f"{tiny_count} of {len(font_sizes)} text elements are under {tiny_px:.0f}px "
                            f"(smallest {smallest:.1f}px; {tiny_text_pct:.0f}% below threshold). "
                            f"Small text is a legibility issue, not clutter — keep the content, raise the size."
                        ),
                        attention_stats={"tiny_text_pct": round(tiny_text_pct, 2)},
                    ))

        metric_details = {
            # noise
            "entropy_bits": None,
            "norm_entropy": None,
            "unexplained_saliency_pct": round(unexplained_pct, 2),
            "container_saliency_pct": round(container_pct, 2),
            "decorative_saliency_pct": round(decorative_pct, 2),
            "interactive_saliency_pct": round(control_pct, 2),
            "textual_saliency_pct": round(textual_pct, 2),
            "media_saliency_pct": round(media_pct, 2),
            "detected_peaks_count": n_peaks,
            "effective_peaks": round(effective_peaks, 2),
            "peak_top1_share": round(peak_top1, 3),
            "content_coverage_pct": round(content_coverage, 2),
            "density_outlier_pct": round(density_outlier, 2),
            "visible_element_count": visible_count,
            "content_element_count": len(content_scores),
            "control_count": len(control_scores),
            "element_density": round(element_density, 3),
            "content_density": round(content_density, 3),
            "control_density": round(control_density, 3),
            "text_density": round(text_density, 3),
            "tiny_text_pct": round(tiny_text_pct, 2),
            "font_size_variety": font_size_variety,
            "small_control_pct": round(small_control_pct, 2),
            # focus
            "effective_targets": round(effective_targets, 2),
            "top1_share": round(top1_share, 3),
            "balance": round(balance, 3),
            "control_effective_targets": round(control_effective_targets, 2),
            "secondary_content_share_pct": round(secondary_content_share_pct, 2),
            "primary_heading_share_pct": round(primary_heading_share_pct, 2),
            # crowding
            "crowding_groups_count": len(crowding_groups),
            "crowding_conflicts_count": len(crowding_conflicts),
            "complex_clusters_count": len(complex_clusters),
            # raw features used by calibration
            "features": {
                **{k: round(v, 4) for k, v in noise_features.items()},
                **{k: round(v, 4) for k, v in focus_features.items()},
                "distraction_secondary_content_share_pct": round(secondary_content_share_pct, 4),
                "distraction_decorative_pct": round(decorative_pct, 4),
                "distraction_unexplained_pct": round(unexplained_pct, 4),
            },
            "labels": {
                "noise_score": round(noise_score, 2),
                "focus_competition_score": round(focus_competition_score, 2),
                "distraction_score": round(distraction_score, 2),
                "crowding_score": round(crowding_score, 2),
            },
            "component_status": {
                "noise": "active" if saliency_mode == "enabled" else "disabled_uniform_heatmap",
                "focus": "active" if saliency_mode == "enabled" else "disabled_uniform_heatmap",
                "distraction": "active" if saliency_mode == "enabled" else "disabled_uniform_heatmap",
            },
        }

        # Entropy on the probability map (kept as a diagnostic, not scored).
        prob = heatmap / max(total_saliency, 1e-12)
        nz = prob > 1e-12
        entropy = float(-np.sum(prob[nz] * np.log2(prob[nz])))
        metric_details["entropy_bits"] = round(entropy, 3)
        metric_details["effective_support_pct"] = round((2.0 ** entropy) / (width * height) * 100.0, 4)

        return ScreenComplexityReport(
            screen_id=screen_capture.screen_id,
            url=screen_capture.url,
            page_name=screen_capture.page_name,
            view_mode=view_mode,
            scroll_index=screen_capture.scroll_index,
            scroll_y=screen_capture.scroll_y,
            viewport_width=screen_capture.viewport_width,
            viewport_height=screen_capture.viewport_height,
            document_height=screen_capture.document_height,
            image_path=screen_capture.image_path,
            saliency_mode=saliency_mode,
            calibration_status="provisional" if self.provisional else "calibrated",
            thresholds_version=self.thresholds_version,
            noise_score=noise_score,
            focus_competition_score=focus_competition_score,
            crowding_score=crowding_score,
            distraction_score=distraction_score,
            overall_complexity_score=overall,
            complexity_level=complexity_level,
            top_focus_elements=selected[:5],
            crowding_groups=crowding_groups,
            crowding_conflicts=crowding_conflicts,
            complex_structure_clusters=complex_clusters,
            flagged_problem_elements=flagged,
            metric_details=metric_details,
        )
