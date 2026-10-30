import argparse
import contextlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from heatmap import DeepGazeHeatmap
from score_modifiers import apply_score_modifiers, view_modifiers_summary
from crawler import WebAppCrawler, ScreenCapture
from complexity_scorer import (
    VisualComplexityScorer,
    ScreenComplexityReport,
    FlaggedProblemElement,
    ComplexStructureCluster,
)
from findings import build_findings, summarize_findings, diff_findings

TOOL_NAME = "interface-audit"
TOOL_VERSION = "0.2.0"
SCHEMA_VERSION = "1.1"

LIMITATIONS = [
    "Saliency predicts free-viewing attention on a static screenshot, not task importance.",
    "Spacing/alignment findings are heuristics unless severity is 'clear_problem'.",
    "Semantic suggestions (removal, dropdown placement, label clarity) require human review.",
    "Dynamic states (hover, focus, open menus, errors, empty/loading) are not captured.",
    "Only the configured desktop (1920x1080) and mobile (390x844) viewports are analyzed.",
]


# --------------------------------------------------------------------------- #
# Visual overlays
# --------------------------------------------------------------------------- #
def save_heatmap_image(
    heatmap: np.ndarray,
    output_path: str,
    upscale_to: Optional[Tuple[int, int]] = None,
) -> None:
    """Persist the raw DeepGaze saliency map as a jet-colored image.

    ``upscale_to`` is an optional (width, height) to resize to the original
    screenshot size so the heatmap lines up 1:1 with the page.
    """
    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze()
    arr = np.clip(arr, 0.0, 1.0)
    peak = float(arr.max()) if arr.size else 0.0
    if peak > 0:
        arr = arr / peak
    h, w = arr.shape[:2]
    jet = cv2.applyColorMap((arr * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if upscale_to is not None:
        tw, th = upscale_to
        if tw > 0 and th > 0 and (w != tw or h != th):
            jet = cv2.resize(jet, (tw, th), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, jet)


def _dashed_rect(
    image: np.ndarray,
    p1: tuple,
    p2: tuple,
    color: tuple,
    thickness: int = 2,
    dash: int = 8,
) -> None:
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    for x in range(x1, x2, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_diagnostic_overlay(
    original_image_path: str,
    heatmap: np.ndarray,
    report: ScreenComplexityReport,
    output_path: str,
    alpha: float = 0.45,
    screen: Optional[ScreenCapture] = None,
    fold_flags: Optional[List[Any]] = None,
    layout_metrics: Optional[Dict[str, Any]] = None,
    extra_flags: Optional[List[Any]] = None,
) -> None:
    image_bgr = cv2.imread(original_image_path)
    if image_bgr is None:
        return

    height, width = image_bgr.shape[:2]
    saliency_uint8 = (np.clip(heatmap, 0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap_jet = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
    if heatmap_jet.shape[:2] != (height, width):
        heatmap_jet = cv2.resize(heatmap_jet, (width, height))

    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_jet, alpha, 0)

    COLOR_CLUSTER = (200, 30, 180)
    COLOR_CROWDING = (0, 0, 240)
    COLOR_COMPETITION = (0, 140, 255)
    COLOR_DISTRACTION = (0, 220, 220)
    COLOR_BOX = (255, 200, 0)     # BGR: cyan-ish blue for structural boxes
    COLOR_TEXT = (0, 220, 0)      # green for text
    COLOR_CONTROL = (255, 120, 0) # blue for interactive controls
    COLOR_FOLD = (0, 165, 255)    # amber for fold-clipped slivers
    COLOR_GRID = (230, 0, 230)    # magenta for grid organization flags
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ---- layout grid guides (vertical column edges, first view only) ----
    # These are the measured card/atom edges, so alignment can be checked by eye
    # against the real content rather than a guessed CSS grid.
    if layout_metrics and screen is not None and screen.scroll_y == 0:
        for edge in (layout_metrics.get("column_edges_px") or [])[:10]:
            for ex in edge:
                gx = int(round(float(ex)))
                if not (0 <= gx < width):
                    continue
                for gy in range(62, height - 2, 22):
                    cv2.line(
                        overlay,
                        (gx, gy),
                        (gx, min(gy + 11, height - 1)),
                        (140, 140, 140),
                        1,
                    )

    # ---- Detected-object inventory (drawn first, so problem annotations sit on top) ----
    # Every element the crawler/scorer saw is outlined and tallied, so a clean page
    # that produces no findings still visibly shows that its text/boxes were caught.
    n_text = n_control = n_box = n_clipped = 0
    fold_selectors = {f.selector for f in (fold_flags or [])}
    if screen is not None:
        for el in screen.elements:
            b = el.bbox
            bw, bh = int(b.width), int(b.height)
            if bw < 8 or bh < 8:
                continue
            if bw >= width * 0.92 and bh >= height * 0.85:
                continue  # skip full-page containers/landmarks
            x0 = max(0, min(width - 1, int(b.x)))
            y0 = max(0, min(height - 1, int(b.y)))
            x1 = max(0, min(width - 1, int(b.x + b.width)))
            y1 = max(0, min(height - 1, int(b.y + b.height)))
            if x1 <= x0 or y1 <= y0:
                continue
            if el.selector in fold_selectors:
                # A folded sliver is noise, not a scorable object: mark it, do not
                # count it in the text/control/box inventory.
                _dashed_rect(overlay, (x0, y0), (x1, y1), COLOR_FOLD, 2, 6)
                n_clipped += 1
                continue
            if el.is_structural_box:
                n_box += 1
            if el.is_interactive:
                color, n_control = COLOR_CONTROL, n_control + 1
                thickness = 2
            elif el.has_direct_text or (el.text or "").strip():
                color, n_text = COLOR_TEXT, n_text + 1
                thickness = 1
            elif el.is_structural_box:
                color, thickness = COLOR_BOX, 2
            else:
                continue
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness)

        # Tally banner (below the score banner) so the counts are legible at a glance.
        banner = f"Detected: {n_text} text | {n_control} controls | {n_box} boxes"
        if n_clipped:
            banner += f" | {n_clipped} fold-clipped (noise)"
        (tw, th), _ = cv2.getTextSize(banner, font, 0.6, 2)
        by0 = 40
        cv2.rectangle(overlay, (0, by0), (tw + 14, by0 + th + 14), (0, 0, 0), -1)
        cv2.putText(overlay, banner, (7, by0 + th + 6), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        # Legend swatches
        lx = tw + 26
        for label, col in (("text", COLOR_TEXT), ("ctrl", COLOR_CONTROL), ("box", COLOR_BOX), ("fold", COLOR_FOLD)):
            cv2.rectangle(overlay, (lx, by0 + 5), (lx + 12, by0 + 17), col, -1)
            cv2.putText(overlay, label, (lx + 16, by0 + 17), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            lx += 16 + len(label) * 9 + 12

    # ---- first-view fold line + clipped slivers ----
    if screen is not None and screen.scroll_y == 0:
        cv2.line(overlay, (0, max(0, height - 3)), (width - 1, max(0, height - 3)), COLOR_FOLD, 2)
    for flagged in (fold_flags or []):
        bx = max(0, min(width - 1, int(flagged.bbox.x)))
        by = max(0, min(height - 1, int(flagged.bbox.y)))
        bw = max(2, min(width - bx, int(flagged.bbox.width)))
        bh = max(2, min(height - by, int(flagged.bbox.height)))
        _dashed_rect(overlay, (bx, by), (bx + bw, by + bh), COLOR_FOLD, 2, 8)
        pct = float(flagged.observed.get("visible_pct", 0.0))
        label = f"[FOLD-CLIPPED {pct:.0f}% visible] {flagged.selector}"
        (label_w, label_h), _ = cv2.getTextSize(label, font, 0.42, 1)
        label_y = max(18, min(height - 2, by - 5))
        cv2.rectangle(overlay, (bx, label_y - label_h - 4), (bx + label_w + 6, label_y + 2), COLOR_FOLD, -1)
        cv2.putText(overlay, label, (bx + 3, label_y - 2), font, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

    for cluster in report.complex_structure_clusters:
        bx = max(0, min(width - 1, cluster.bbox.x))
        by = max(0, min(height - 1, cluster.bbox.y))
        bw = max(4, min(width - bx, cluster.bbox.width))
        bh = max(4, min(height - by, cluster.bbox.height))
        sub_img = overlay[by:by + bh, bx:bx + bw]
        tint_rect = np.full(sub_img.shape, (160, 20, 140), dtype=np.uint8)
        cv2.addWeighted(sub_img, 0.85, tint_rect, 0.15, 0, sub_img)
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), COLOR_CLUSTER, 3)
        cv2.rectangle(overlay, (bx + 2, by + 2), (bx + bw - 2, by + bh - 2), (255, 255, 255), 1)
        if cluster.columns >= 2:
            layout = f"{cluster.columns}-col grid"
        else:
            layout = "dense list"
        label = (
            f"BOX CLUSTER: {cluster.container_selector} "
            f"({cluster.box_count} boxes, {layout}, {cluster.attention_share_pct:.0f}% att)"
        )
        (lw, lh), _ = cv2.getTextSize(label, font, 0.45, 1)
        tag_y = max(18, by - 6)
        cv2.rectangle(overlay, (bx, tag_y - lh - 4), (bx + lw + 8, tag_y + 2), COLOR_CLUSTER, -1)
        cv2.putText(overlay, label, (bx + 4, tag_y - 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    for flagged in list(report.flagged_problem_elements) + list(extra_flags or []):
        if flagged.issue_type == "COMPLEX_STRUCTURE":
            continue
        bx = max(0, min(width - 1, flagged.bbox.x))
        by = max(0, min(height - 1, flagged.bbox.y))
        bw = max(2, min(width - bx, flagged.bbox.width))
        bh = max(2, min(height - by, flagged.bbox.height))
        if flagged.issue_type == "CROWDING_VIOLATION":
            color = COLOR_CROWDING
        elif flagged.issue_type == "FOCUS_COMPETITION":
            color = COLOR_COMPETITION
        elif flagged.issue_type == "FOLD_CLIPPED_CONTENT":
            color = COLOR_FOLD
        elif flagged.issue_type in ("GRID_MISALIGNMENT", "GRID_GAP", "ROW_UNDERFILL", "BLOCK_IMBALANCE"):
            color = COLOR_GRID
        elif flagged.issue_type == "ORPHAN_LINE":
            color = COLOR_FOLD
        else:
            color = COLOR_DISTRACTION
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, 3)
        if flagged.issue_type == "FOLD_CLIPPED_CONTENT":
            pct = float(flagged.observed.get("visible_pct", 0.0))
            label = f"[FOLD-CLIPPED {pct:.0f}% visible] {flagged.selector}"
        elif flagged.issue_type == "ORPHAN_LINE":
            frag = str(flagged.observed.get("last_line_text", ""))[:12]
            share = float(flagged.observed.get("last_line_share_pct", 0.0))
            label = f"[ORPHAN '{frag}' {share:.0f}% of width] {flagged.selector}"
        elif flagged.issue_type == "GRID_GAP":
            gap = float(flagged.observed.get("gap_px", 0.0))
            gutter = float(flagged.observed.get("normal_gutter_px", 0.0))
            label = f"[GRID GAP {gap:.0f}px vs gutter {gutter:.0f}px] {flagged.selector}"
        elif flagged.issue_type == "ROW_UNDERFILL":
            fill = float(flagged.observed.get("fill_ratio_pct", 100.0))
            label = f"[ROW UNDERFILL {fill:.0f}% of grid width] {flagged.selector}"
        elif flagged.issue_type == "BLOCK_IMBALANCE":
            skew = float(flagged.observed.get("skew_pct", 0.0))
            side = "left" if float(flagged.observed.get("left_third_pct", 0.0)) >= float(flagged.observed.get("right_third_pct", 0.0)) else "right"
            label = f"[IMBALANCE {skew:.0f}% skew, {side}-heavy] {flagged.selector}"
        elif flagged.issue_type == "GRID_MISALIGNMENT":
            if flagged.observed.get("mode") == "width":
                dev = float(flagged.observed.get("width_deviation_px", 0.0))
                label = f"[CELL WIDTH +-{dev:.0f}px vs column] {flagged.selector}"
            else:
                dev = float(flagged.observed.get("deviation_px", 0.0))
                label = f"[COLUMN OFF {dev:.0f}px] {flagged.selector}"
        else:
            label = f"[{flagged.issue_type}] {flagged.tag}"
            if flagged.text:
                label += f": {flagged.text[:18]}"
        (label_w, label_h), _ = cv2.getTextSize(label, font, 0.42, 1)
        label_y = max(18, by - 5)
        cv2.rectangle(overlay, (bx, label_y - label_h - 4), (bx + label_w + 6, label_y + 2), color, -1)
        cv2.putText(overlay, label, (bx + 3, label_y - 2), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    banner_h = 36
    cv2.rectangle(overlay, (0, 0), (width, banner_h), (30, 30, 30), -1)
    status_color = (80, 220, 80) if report.complexity_level == "LOW" else (
        (0, 180, 255) if report.complexity_level == "MODERATE" else (0, 0, 255)
    )
    banner_text = (
        f"{report.screen_id} | Score: {report.overall_complexity_score:.1f} ({report.complexity_level}) | "
        f"Noise: {report.noise_score:.0f} | Focus: {report.focus_competition_score:.0f} | "
        f"Crowding: {report.crowding_score:.0f} | Distraction: {report.distraction_score:.0f} | "
        f"Clusters: {len(report.complex_structure_clusters)}"
    )
    cv2.putText(overlay, banner_text, (10, 24), font, 0.52, status_color, 2, cv2.LINE_AA)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, overlay)


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _weighted_view_score(reports: List[ScreenComplexityReport]) -> Dict[str, Any]:
    """Weight slices by newly-covered document height (avoids double counting overlap)."""
    if not reports:
        return {"score": None, "valid": False, "screens": 0}
    ordered = sorted(reports, key=lambda r: r.scroll_y)
    weights = []
    previous_bottom = 0
    for r in ordered:
        top = max(0, r.scroll_y)
        bottom = top + max(0, r.viewport_height)
        unique = max(0, bottom - max(top, previous_bottom))
        if unique <= 0:
            unique = max(1, r.viewport_height)
        weights.append(float(unique))
        previous_bottom = max(previous_bottom, bottom)
    total_weight = sum(weights) or 1.0
    scores = {
        "overall": sum(r.overall_complexity_score * w for r, w in zip(ordered, weights)) / total_weight,
        "noise": sum(r.noise_score * w for r, w in zip(ordered, weights)) / total_weight,
        "focus_competition": sum(r.focus_competition_score * w for r, w in zip(ordered, weights)) / total_weight,
        "crowding": sum(r.crowding_score * w for r, w in zip(ordered, weights)) / total_weight,
        "distraction": sum(r.distraction_score * w for r, w in zip(ordered, weights)) / total_weight,
    }
    saliency_valid = all(r.saliency_mode == "enabled" for r in ordered)
    result = {
        "score": round(scores["overall"], 1),
        **{k: round(v, 1) for k, v in scores.items()},
        "level": _level_for(scores["overall"]),
        "valid": saliency_valid,
        "screens": len(ordered),
        "scroll_y": [r.scroll_y for r in ordered],
        "saliency_mode": "enabled" if saliency_valid else "disabled_uniform_heatmap",
    }
    modifiers = view_modifiers_summary(
        [r.metric_details.get("score_modifiers") for r in ordered], weights
    )
    if modifiers:
        result["score_modifiers"] = modifiers
    return result


def _level_for(score: float) -> str:
    if score < 40.0:
        return "LOW"
    if score < 65.0:
        return "MODERATE"
    return "OVER_COMPLEX"


def _compare_views(page: Dict[str, Any]) -> Dict[str, Any]:
    desktop = page["views"].get("desktop")
    mobile = page["views"].get("mobile")
    if not desktop or not mobile:
        return {
            "desktop_score": desktop.get("score") if desktop else None,
            "mobile_score": mobile.get("score") if mobile else None,
            "score_diff": None,
            "coverage": "missing_view",
            "mobile_degradation_detected": None,
            "reason": "Both desktop and mobile views are required for comparison.",
        }
    d_score = desktop.get("score")
    m_score = mobile.get("score")
    if d_score is None or m_score is None:
        return {
            "desktop_score": d_score,
            "mobile_score": m_score,
            "score_diff": None,
            "coverage": "missing_score",
            "mobile_degradation_detected": None,
            "reason": "One or both views have no score (saliency disabled).",
        }
    if not desktop.get("valid", False) or not mobile.get("valid", False):
        return {
            "desktop_score": d_score,
            "mobile_score": m_score,
            "score_diff": None,
            "coverage": "saliency_disabled",
            "mobile_degradation_detected": None,
            "reason": "Diff requires saliency-enabled scores; run without --fast.",
        }
    diff = round(m_score - d_score, 1)
    degradation = (m_score >= 60.0 and diff >= 15.0) or (
        mobile.get("level") == "OVER_COMPLEX" and desktop.get("level") != "OVER_COMPLEX"
    )
    return {
        "desktop_score": d_score,
        "mobile_score": m_score,
        "score_diff": diff,
        "coverage": "complete",
        "mobile_degradation_detected": bool(degradation),
        "reason": None,
    }


def _build_report(
    target: str,
    screens: List[ScreenCapture],
    reports: List[ScreenComplexityReport],
    findings: List[Dict[str, Any]],
    config: Dict[str, Any],
    run_id: str,
    started_at: str,
    duration_s: float,
    output_dir: str,
    artifacts: Dict[str, Any],
    errors: Optional[List[Dict[str, Any]]] = None,
    baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pages: Dict[str, Dict[str, Any]] = {}
    for r in reports:
        page = pages.setdefault(r.page_name, {"page_name": r.page_name, "url": r.url, "views": {}})
        page["views"].setdefault(r.view_mode, {"screens": 0, "reports": []})
        page["views"][r.view_mode]["screens"] += 1
        page["views"][r.view_mode]["reports"].append(r)

    page_list = []
    for page_name in sorted(pages):
        page = pages[page_name]
        view_data = {}
        for view_mode, data in page["views"].items():
            view_data[view_mode] = _weighted_view_score(data["reports"])
            layout_metrics = None
            for rep in data["reports"]:
                lm = rep.metric_details.get("layout")
                if lm:
                    layout_metrics = lm
                    break
            if layout_metrics:
                view_data[view_mode]["layout"] = layout_metrics
        page_out = {
            "page_name": page_name,
            "url": page["url"],
            "status": "ok",
            "views": view_data,
            "findings": [f for f in findings if f["page"] == page_name],
        }
        page_out["comparison"] = _compare_views(page_out)
        page_list.append(page_out)

    screen_rows = []
    for r in reports:
        screen_rows.append({
            "screen_id": r.screen_id,
            "page": r.page_name,
            "view": r.view_mode,
            "scroll_y": r.scroll_y,
            "url": r.url,
            "image_path": r.image_path,
            "saliency_mode": r.saliency_mode,
            "calibration_status": r.calibration_status,
            "thresholds_version": r.thresholds_version,
            "scores": {
                "overall": round(r.overall_complexity_score, 2),
                "noise": round(r.noise_score, 2),
                "focus_competition": round(r.focus_competition_score, 2),
                "crowding": round(r.crowding_score, 2),
                "distraction": round(r.distraction_score, 2),
            },
            "score_modifiers": r.metric_details.get("score_modifiers"),
            "level": r.complexity_level,
            "metrics": r.metric_details,
            "findings_count": len(r.flagged_problem_elements),
        })

    measured = ["layout_geometry", "dom_structure", "spacing", "target_size", "crowding"]
    if any(r.saliency_mode == "enabled" for r in reports):
        measured.extend(["saliency_prominence", "noise", "focus_competition", "distraction"])

    not_measured = [
        "task_importance",
        "safe_removal",
        "dropdown_placement",
        "naming_clarity",
        "technical_dependencies",
        "dynamic_ui_states",
    ]

    diff = None
    if baseline and isinstance(baseline.get("findings"), list):
        diff = diff_findings(baseline["findings"], findings)

    modifiers_applied = bool(
        config.get("scoring_modifiers", {}).get("enabled")
        and any(r.metric_details.get("score_modifiers") for r in reports)
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "tool": {
            "name": TOOL_NAME,
            "version": TOOL_VERSION,
            "thresholds_version": config.get("version", "unknown"),
            "calibration": "provisional" if config.get("provisional", True) else "calibrated",
        },
        "target": target,
        "started_at": started_at,
        "duration_s": round(duration_s, 2),
        "config": {
            "views": sorted({r.view_mode for r in reports}),
            "max_pages": None,
            "device": None,
            "fast": not any(r.saliency_mode == "enabled" for r in reports),
            "thresholds_version": config.get("version", "unknown"),
        },
        "coverage": {
            "pages_found": len(page_list),
            "pages_analyzed": sum(1 for p in page_list if p["status"] == "ok"),
            "screens": len(reports),
            "missing": [],
            "pages_failed": errors or [],
            "measured": measured,
            "not_measured": not_measured,
        },
        "limitations": LIMITATIONS,
        "summary": {
            **summarize_findings(findings),
            "score_modifiers_applied": modifiers_applied,
        },
        "pages": page_list,
        "screens": screen_rows,
        "findings": findings,
        "diff": diff,
        "artifacts": artifacts,
    }


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #
def run_analysis(
    target: str,
    output_dir: str = "output_analysis",
    max_pages: int = 5,
    save_overlays: bool = True,
    save_crops: bool = True,
    device: Optional[str] = None,
    views: Optional[List[str]] = None,
    fast: bool = False,
    use_cache: bool = True,
    allow_remote: bool = False,
    project_root: Optional[str] = None,
    baseline_path: Optional[str] = None,
    page_timeout_ms: int = 20000,
) -> Dict[str, Any]:
    if not str(target).startswith(("http://", "https://", "file://")):
        target = os.path.abspath(str(target))
    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    output_dir = os.path.abspath(output_dir)
    screenshots_dir = os.path.join(output_dir, "screenshots")
    overlays_dir = os.path.join(output_dir, "diagnostic_overlays")
    heatmaps_dir = os.path.join(output_dir, "heatmaps")
    crops_dir = os.path.join(output_dir, "crops")
    for d in [output_dir, screenshots_dir]:
        os.makedirs(d, exist_ok=True)
    if save_overlays:
        os.makedirs(overlays_dir, exist_ok=True)
        os.makedirs(heatmaps_dir, exist_ok=True)
    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    print("==================================================================", file=sys.stderr)
    print("INTERFACE AUDIT: LAYOUT, CROWDING & VISUAL PROMINENCE", file=sys.stderr)
    print("==================================================================", file=sys.stderr)
    print(f"Target:      {target}", file=sys.stderr)
    print(f"Output Dir:  {output_dir}", file=sys.stderr)
    print(f"Views:       {views or ['desktop', 'mobile']}", file=sys.stderr)
    print(f"Fast mode:   {fast}", file=sys.stderr)

    crawler = WebAppCrawler(headless=True)
    try:
        screens = crawler.crawl(
            target=target,
            output_dir=screenshots_dir,
            max_pages=max_pages,
            views=views,
            allow_remote=allow_remote,
            project_root=project_root,
            page_timeout_ms=page_timeout_ms,
        )
    except (FileNotFoundError, PermissionError) as e:
        raise RuntimeError(f"Invalid target: {e}") from e

    if not screens:
        raise RuntimeError("No screens were captured. Check the target path/URL and page load.")

    dom_snapshot_path = os.path.join(output_dir, "dom_snapshot.json")
    with open(dom_snapshot_path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict(include_elements=True) for s in screens], f, indent=2, ensure_ascii=False)

    deepgaze = None
    if not fast:
        try:
            deepgaze = DeepGazeHeatmap(
                device=device,
                cache_dir=os.path.join(output_dir, ".cache", "heatmaps"),
                use_cache=use_cache,
            )
        except Exception as e:
            raise RuntimeError(f"DeepGaze initialization failed: {e}") from e

    scorer = VisualComplexityScorer()
    reports: List[ScreenComplexityReport] = []
    screens_by_id: Dict[str, ScreenCapture] = {s.screen_id: s for s in screens}
    heatmap_paths: Dict[str, str] = {}

    # ---- M3: deterministic layout pass, one per (page, view) ----
    # Runs on the deduped document union + the first view. It is heatmap-free, so it
    # runs before the scoring loop and its fold-clipped boxes can be drawn on the
    # overlays. Findings are attached to the matching screen report after the loop.
    layout_by_screen: Dict[str, Any] = {}
    try:
        from layout_analyzer import analyze_layout
    except Exception as exc:  # pragma: no cover - defensive
        analyze_layout = None  # type: ignore[assignment]
        print(f"  [warn] layout analyzer unavailable: {exc}", file=sys.stderr)

    if analyze_layout is not None:
        layout_groups: Dict[Tuple[str, str], List[ScreenCapture]] = {}
        for scr in screens:
            layout_groups.setdefault((scr.page_name, scr.view_mode), []).append(scr)
        for (page_name, view_mode), group_screens in layout_groups.items():
            try:
                layout_result = analyze_layout(group_screens, thresholds=scorer.config)
            except Exception as exc:
                print(
                    f"  [warn] layout analysis failed for {page_name}/{view_mode}: {exc}",
                    file=sys.stderr,
                )
                continue
            for scr in group_screens:
                layout_by_screen[scr.screen_id] = layout_result

    for idx, screen in enumerate(screens, 1):
        print(f"[{idx}/{len(screens)}] Processing '{screen.screen_id}' ({screen.view_mode})...", file=sys.stderr)
        if deepgaze is not None:
            heatmap = deepgaze.get_heatmap(
                screen.image_path,
                output_format="float32",
                normalize="probability",
                use_cache=use_cache,
            )
            report = scorer.analyze_screen(heatmap=heatmap, screen_capture=screen, saliency_mode="enabled")
        else:
            report = scorer.analyze_screen(heatmap=None, screen_capture=screen, saliency_mode="disabled")
            heatmap = np.full(
                (max(1, screen.viewport_height), max(1, screen.viewport_width)),
                1.0 / max(1, screen.viewport_height * screen.viewport_width),
                dtype=np.float32,
            )
        reports.append(report)

        if save_overlays:
            overlay_path = os.path.join(overlays_dir, f"{screen.screen_id}_diagnostic.jpg")
            layout_result = layout_by_screen.get(screen.screen_id)
            fold_flags = []
            extra_flags = []
            if layout_result is not None:
                for flag_screen_id, flag in layout_result.flags:
                    if flag_screen_id != screen.screen_id:
                        continue
                    if flag.issue_type == "FOLD_CLIPPED_CONTENT":
                        fold_flags.append(flag)
                    else:
                        extra_flags.append(flag)
            draw_diagnostic_overlay(
                original_image_path=screen.image_path,
                heatmap=heatmap,
                report=report,
                output_path=overlay_path,
                screen=screen,
                fold_flags=fold_flags,
                layout_metrics=layout_result.metrics if layout_result is not None else None,
                extra_flags=extra_flags,
            )
            # Raw DeepGaze heatmap, upscaled to screenshot size for 1:1 alignment.
            heatmap_path = os.path.join(heatmaps_dir, f"{screen.screen_id}_heatmap.jpg")
            try:
                save_heatmap_image(
                    heatmap,
                    heatmap_path,
                    upscale_to=(screen.viewport_width, screen.viewport_height),
                )
                heatmap_paths[screen.screen_id] = os.path.abspath(heatmap_path)
            except Exception as e:
                print(f"  [warn] could not save heatmap for {screen.screen_id}: {e}", file=sys.stderr)

    baseline = None
    if baseline_path:
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)

    # ---- attach M3 layout metrics + fold findings to their screen reports ----
    report_by_screen_id = {r.screen_id: r for r in reports}
    for screen_id, layout_result in layout_by_screen.items():
        rep = report_by_screen_id.get(screen_id)
        if rep is None:
            continue
        rep.metric_details["layout"] = layout_result.metrics
        for flag_screen_id, flag in layout_result.flags:
            if flag_screen_id == screen_id:
                rep.flagged_problem_elements.append(flag)

    findings = build_findings(reports, screens_by_id, crops_dir if save_crops else None)
    config = scorer.config
    # COMPLEX_STRUCTURE and the grid-organization findings apply a bounded,
    # config-driven penalty on the reported overall (see score_modifiers.py).
    # This runs after findings exist and before the report/view aggregation,
    # so every score the agent reads already includes it.
    apply_score_modifiers(reports, findings, config)
    duration = time.time() - started

    artifacts = {
        "report_json": None,  # filled by caller
        "dom_snapshot": os.path.abspath(dom_snapshot_path),
        "screenshots_dir": os.path.abspath(screenshots_dir),
        "overlays_dir": os.path.abspath(overlays_dir) if save_overlays else None,
        "heatmaps": heatmap_paths if save_overlays else {},
        "crops_dir": os.path.abspath(crops_dir) if save_crops else None,
    }

    report = _build_report(
        target=target,
        screens=screens,
        reports=reports,
        findings=findings,
        config=config,
        run_id=run_id,
        started_at=started_at,
        duration_s=duration,
        output_dir=output_dir,
        artifacts=artifacts,
        baseline=baseline,
    )
    report["config"]["max_pages"] = max_pages
    if deepgaze is None:
        report["config"]["device"] = None
    else:
        report["config"]["device"] = device or deepgaze.device
    return report


# --------------------------------------------------------------------------- #
# CLI commands
# --------------------------------------------------------------------------- #
def _cmd_analyze(args: argparse.Namespace) -> int:
    output_dir = os.path.abspath(args.output)
    json_out = args.json_out
    report_path = os.path.join(output_dir, "complexity_report.json")
    if json_out and json_out != "-":
        report_path = os.path.abspath(json_out)

    try:
        # Keep stdout machine-readable: all crawl/model/library chatter goes to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            report = run_analysis(
                target=args.target,
                output_dir=output_dir,
                max_pages=args.max_pages,
                save_overlays=not args.no_overlays,
                save_crops=not args.no_crops,
                device=args.device,
                views=[v.strip() for v in args.views.split(",") if v.strip()],
                fast=args.fast,
                use_cache=not args.no_cache,
                allow_remote=args.allow_remote,
                project_root=os.path.abspath(os.getcwd()),
                baseline_path=args.baseline,
                page_timeout_ms=args.page_timeout_ms,
            )
    except RuntimeError as e:
        print(json.dumps({"status": "error", "error": str(e)}), file=sys.stderr)
        return 2

    report["artifacts"]["report_json"] = report_path
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.json or json_out == "-":
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif not args.quiet:
        _print_summary(report, report_path)

    counts = report["summary"]
    order = {"clear_problem": 0, "potential_improvement": 1, "needs_investigation": 2}
    if args.fail_on in order:
        threshold = order[args.fail_on]
        if any(order.get(f["severity"], 9) <= threshold for f in report["findings"]):
            return 4
    if report["coverage"]["screens"] == 0:
        return 2
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    try:
        with open(args.before, "r", encoding="utf-8") as f:
            before = json.load(f)
        with open(args.after, "r", encoding="utf-8") as f:
            after = json.load(f)
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}), file=sys.stderr)
        return 2
    diff = diff_findings(before.get("findings", []), after.get("findings", []))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "before_run_id": before.get("run_id"),
        "after_run_id": after.get("run_id"),
        "summary": diff.pop("counts"),
        **diff,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    checks: Dict[str, Any] = {"status": "ok", "checks": {}}
    ok = True
    with contextlib.redirect_stdout(sys.stderr):
        for module_name in ["numpy", "scipy", "cv2", "torch", "playwright", "yaml"]:
            try:
                module = __import__(module_name)
                checks["checks"][module_name] = {"ok": True, "version": getattr(module, "__version__", "unknown")}
            except Exception as e:
                checks["checks"][module_name] = {"ok": False, "error": str(e)}
                ok = False

        try:
            import deepgaze_pytorch  # noqa: F401
            checks["checks"]["deepgaze_pytorch"] = {"ok": True}
        except Exception as e:
            checks["checks"]["deepgaze_pytorch"] = {"ok": False, "error": str(e)}
            ok = False

        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                browser.close()
            checks["checks"]["playwright_chromium"] = {"ok": True}
        except Exception as e:
            checks["checks"]["playwright_chromium"] = {"ok": False, "error": str(e)}
            ok = False

        checks["checks"]["thresholds"] = {"ok": True, "version": VisualComplexityScorer().thresholds_version}
    checks["status"] = "ok" if ok else "error"
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return 0 if ok else 3


def _print_summary(report: Dict[str, Any], report_path: str) -> None:
    print("", file=sys.stderr)
    print("=" * 100, file=sys.stderr)
    print(
        f"SCHEMA {report['schema_version']} | calibration: {report['tool']['calibration']} | "
        f"thresholds: {report['tool']['thresholds_version']} | {report['duration_s']}s",
        file=sys.stderr,
    )
    print("=" * 100, file=sys.stderr)
    summary = report["summary"]
    print(
        f"Findings: {summary['total']} total | clear: {summary['clear_problem']} | "
        f"potential: {summary['potential_improvement']} | investigate: {summary['needs_investigation']} | "
        f"human review: {summary['requires_human']}",
        file=sys.stderr,
    )
    for page in report["pages"]:
        for view, data in page["views"].items():
            score = data.get("score")
            print(
                f"  {page['page_name']:<18} {view:<8} score={score} level={data.get('level')} "
                f"screens={data.get('screens')} valid={data.get('valid')}",
                file=sys.stderr,
            )
        comp = page.get("comparison", {})
        print(
            f"    comparison: desktop={comp.get('desktop_score')} mobile={comp.get('mobile_score')} "
            f"diff={comp.get('score_diff')} degradation={comp.get('mobile_degradation_detected')} "
            f"coverage={comp.get('coverage')}",
            file=sys.stderr,
        )

    print("", file=sys.stderr)
    for f in report["findings"][:12]:
        print(
            f"  [{f['severity']}] {f['issue']} {f['selector']} ({f['view']}) "
            f"impact={f['impact']} confidence={f['confidence']}",
            file=sys.stderr,
        )
        if f.get("why"):
            print(f"      why: {f['why'][:160]}", file=sys.stderr)
    if len(report["findings"]) > 12:
        print(f"  ... {len(report['findings']) - 12} more findings in the JSON report", file=sys.stderr)
    print(f"\nReport: {report_path}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Backward-compatible entry point
# --------------------------------------------------------------------------- #
def run_pipeline(
    target: str,
    output_dir: str = "output_analysis",
    max_pages: int = 5,
    save_overlays: bool = True,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    report = run_analysis(
        target=target,
        output_dir=output_dir,
        max_pages=max_pages,
        save_overlays=save_overlays,
        device=device,
        project_root=os.getcwd(),
    )
    report_path = os.path.join(os.path.abspath(output_dir), "complexity_report.json")
    report["artifacts"]["report_json"] = report_path
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Interface audit: calibrated layout, crowding and visual-prominence findings.",
    )
    sub = parser.add_subparsers(dest="command")

    p_analyze = sub.add_parser("analyze", help="Crawl a target and produce a findings report")
    p_analyze.add_argument("--target", required=True, help="Local HTML path or localhost URL")
    p_analyze.add_argument("--output", default="output_analysis", help="Artifact directory")
    p_analyze.add_argument("--json-out", default=None, help="Write JSON here (use '-' for stdout)")
    p_analyze.add_argument("--json", action="store_true", help="Print JSON to stdout")
    p_analyze.add_argument("--quiet", action="store_true", help="Suppress progress/summary on stderr")
    p_analyze.add_argument("--max-pages", type=int, default=5)
    p_analyze.add_argument("--views", default="desktop,mobile", help="Comma-separated views")
    p_analyze.add_argument("--no-overlays", action="store_true")
    p_analyze.add_argument("--no-crops", action="store_true")
    p_analyze.add_argument("--no-cache", action="store_true")
    p_analyze.add_argument("--fast", action="store_true", help="Skip DeepGaze; layout/DOM findings only")
    p_analyze.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p_analyze.add_argument("--allow-remote", action="store_true", help="Allow non-localhost targets")
    p_analyze.add_argument("--baseline", default=None, help="Previous report JSON for diffing")
    p_analyze.add_argument("--page-timeout-ms", type=int, default=20000)
    p_analyze.add_argument(
        "--fail-on",
        default=None,
        choices=["clear_problem", "potential_improvement", "needs_investigation"],
        help="Exit 4 when a finding at or above this severity exists",
    )
    p_analyze.set_defaults(func=_cmd_analyze)

    p_diff = sub.add_parser("diff", help="Compare two report JSON files")
    p_diff.add_argument("--before", required=True)
    p_diff.add_argument("--after", required=True)
    p_diff.set_defaults(func=_cmd_diff)

    p_check = sub.add_parser("check", help="Check dependencies, browser and configuration")
    p_check.set_defaults(func=_cmd_check)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Backward compatibility: `main.py --target ...` means `analyze`.
    if argv and argv[0] not in {"analyze", "diff", "check"}:
        argv = ["analyze"] + argv
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
