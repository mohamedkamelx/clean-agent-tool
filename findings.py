"""Convert raw per-screen flags into stable, agent-facing findings.

A finding is a *unit of action*: stable ID, evidence, confidence, impact,
a suggested fix, and a verification step. Findings are aggregated across
scroll slices so the same element is not reported five times.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Iterable, List, Optional

from crawler import BoundingBox, ElementMetadata, ScreenCapture
from complexity_scorer import ScreenComplexityReport
from crops import make_crop


SEVERITY_ORDER = {"clear_problem": 0, "potential_improvement": 1, "needs_investigation": 2}
IMPACT_ORDER = {"high": 0, "medium": 1, "low": 2}


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return "f_" + digest[:10]


def _normalize_severity(raw: str) -> str:
    if raw in SEVERITY_ORDER:
        return raw
    if raw in {"CRITICAL", "HIGH"}:
        return "clear_problem" if raw == "CRITICAL" else "potential_improvement"
    return "potential_improvement"


def _fix_for(issue_type: str, rule_id: str, observed: Dict[str, Any], expected: Dict[str, Any]) -> str:
    if issue_type == "CROWDING_VIOLATION":
        target = observed.get("target_min_px", expected.get("target_px", 24))
        gap = observed.get("gap_px", 0)
        return (
            f"Increase the gap in this control group to at least {expected.get('gap_px', 8)}px "
            f"(current {gap}px), or raise the smallest target to at least 44x44px "
            f"(current {target}px). Add padding/gap in the shared action-row container."
        )
    if issue_type == "COMPLEX_STRUCTURE":
        return (
            "Reduce the number of visible boxes in this region (for mobile, keep one column and "
            "collapse secondary cards into an accordion or 'Show more' section). Preserve the "
            "primary metric and its main action above the fold."
        )
    if issue_type == "FOCUS_COMPETITION":
        return (
            "Strengthen one primary focus: increase the size/weight/contrast of the intended "
            "primary element, and demote the competing element (smaller, muted, or move it below "
            "the primary action). Do not remove required controls."
        )
    if issue_type == "VISUAL_DISTRACTION":
        return (
            "Reduce the visual weight of this secondary content (contrast, size, or decoration), "
            "or move it out of the primary viewing area. Verify it is not required for the task "
            "before changing or hiding it."
        )
    if issue_type == "FOLD_CLIPPED_CONTENT":
        return (
            "Shift this element down (increase the space above it) or resize it so it is either "
            "fully inside the first viewport or fully below the fold. Do not leave it straddling "
            "the fold, where the remainder renders as a clipped strip."
        )
    if issue_type == "GRID_MISALIGNMENT":
        return (
            "Make this cell use the same column track width as its row siblings (equal widths or a "
            "declared grid template), so the repeated row stays aligned at both edges."
        )
    if issue_type == "ORPHAN_LINE":
        return (
            "Tighten the copy or widen/narrow the block so no line dangles: shorten the sentence, "
            "split it at a natural point, or use a balanced wrap (e.g. text-wrap: balance). Keep "
            "the full text."
        )
    if issue_type == "ROW_UNDERFILL":
        return (
            "Organize this row into the grid: give it the same width role as its sibling rows "
            "(full-width, or cells sized to the shared column tracks). Keep the content; do not "
            "delete it to make the line look filled."
        )
    if issue_type == "GRID_GAP":
        return (
            "Fill or close the empty middle slot: either give that column cell content of its own, "
            "or let the neighbouring cells span across it (declared grid template), so the row "
            "reads as one continuous grid."
        )
    if issue_type == "BLOCK_IMBALANCE":
        return (
            "Rebalance this row: equalize the cell widths against the grid's column tracks, or pair "
            "the small cell with a similar one, so the visual mass is distributed like the sibling "
            "rows. Keep all content."
        )
    return "Investigate this element in context before changing it."


def _verify_for(issue_type: str) -> str:
    if issue_type == "CROWDING_VIOLATION":
        return "Re-run the audit; this issue_id should disappear and no new crowding group should appear."
    if issue_type == "COMPLEX_STRUCTURE":
        return "Re-run the audit; box_count in this cluster should fall and the mobile density should decrease."
    if issue_type == "FOCUS_COMPETITION":
        return "Re-run the audit; effective_targets and the runner-up share should decrease without losing primary content."
    if issue_type == "VISUAL_DISTRACTION":
        return "Re-run the audit; secondary_content_share_pct should decrease. Confirm the content is still reachable."
    if issue_type == "FOLD_CLIPPED_CONTENT":
        return (
            "Re-run at the first view (scroll_y = 0); this element's visible_fraction should be "
            ">= 0.98 or exactly 0, and this issue_id should disappear."
        )
    if issue_type == "GRID_MISALIGNMENT":
        return "Re-run the audit; the row's cell widths should match and this issue_id should disappear."
    if issue_type == "ORPHAN_LINE":
        return (
            "Re-run the audit; the block's last line should no longer be a single dangling word "
            "(last_line_share_pct >= 30) and this issue_id should disappear."
        )
    if issue_type == "ROW_UNDERFILL":
        return "Re-run the audit; fill_ratio_pct should reach the threshold and this issue_id should disappear."
    if issue_type == "GRID_GAP":
        return "Re-run the audit; the internal gap should match the normal gutter and this issue_id should disappear."
    if issue_type == "BLOCK_IMBALANCE":
        return "Re-run the audit; skew_pct should fall to the sibling-row level and this issue_id should disappear."
    return "Re-run the audit and compare the finding by issue_id."


def _preserve_for(issue_type: str, protected: bool) -> str:
    if protected:
        return (
            "Preserve this element: keep it visible, reachable, and semantically intact. "
            "Change only spacing, size, grouping, or visual weight as described; do not remove or hide it."
        )
    if issue_type == "COMPLEX_STRUCTURE":
        return (
            "Preserve primary content, metrics, and actions. Collapse or regroup secondary cards "
            "only after confirming which content is task-critical."
        )
    if issue_type == "VISUAL_DISTRACTION":
        return (
            "Preserve the content unless a human confirms it is optional. Reduce its visual weight "
            "or move it; do not delete it based on saliency alone."
        )
    return (
        "Preserve task-critical content and controls. Do not remove or hide elements solely because "
        "they attract attention; change layout or hierarchy first."
    )


def _is_protected(finding: Dict[str, Any], screen: ScreenCapture) -> bool:
    """Conservative guardrail: anything interactive/landmark/heading/form-like."""
    selector = finding.get("selector", "")
    for el in screen.elements:
        if el.selector != selector:
            continue
        tag = el.tag.lower()
        if el.is_interactive or tag in {"a", "button", "input", "select", "textarea", "form", "nav", "header", "footer", "main"}:
            return True
        if el.heading_level > 0 or el.aria_live or el.required or el.href:
            return True
        break
    return False


def _finding_from_flag(
    report: ScreenComplexityReport,
    screen: ScreenCapture,
    flag,
    crops_dir: Optional[str],
) -> Dict[str, Any]:
    severity = _normalize_severity(flag.severity)
    issue_id = _stable_id(
        report.page_name,
        report.view_mode,
        flag.issue_type,
        flag.rule_id or flag.issue_type,
        flag.selector,
    )

    crop_path = None
    if crops_dir:
        crop_path = make_crop(
            screen.image_path,
            flag.bbox,
            os.path.join(crops_dir, f"{issue_id}.png"),
        )

    protected = _is_protected({"selector": flag.selector}, screen)
    requires_human = bool(flag.requires_human) or severity == "needs_investigation"
    if flag.issue_type == "VISUAL_DISTRACTION":
        requires_human = True
    if protected and flag.issue_type in {"FOCUS_COMPETITION", "VISUAL_DISTRACTION", "COMPLEX_STRUCTURE"}:
        requires_human = True

    return {
        "issue_id": issue_id,
        "rule_id": flag.rule_id or flag.issue_type,
        "issue": flag.issue_type,
        "severity": severity,
        "impact": flag.impact,
        "confidence": round(float(flag.confidence), 2),
        "requires_human": requires_human,
        "page": report.page_name,
        "view": report.view_mode,
        "screen_id": report.screen_id,
        "scroll_y": report.scroll_y,
        "selector": flag.selector,
        "tag": flag.tag,
        "text": (flag.text or "")[:120],
        "related_selectors": flag.related_selectors,
        "bbox": flag.bbox.to_dict(),
        "crop_path": os.path.abspath(crop_path) if crop_path else None,
        "observed": flag.observed,
        "expected": flag.expected,
        "evidence": {
            "class": flag.evidence_class,
            "slices_seen": 1,
            "screen_ids": [report.screen_id],
            "thresholds_version": report.thresholds_version,
        },
        "why": flag.description,
        "fix": _fix_for(flag.issue_type, flag.rule_id, flag.observed, flag.expected),
        "verify": _verify_for(flag.issue_type),
        "preserve": _preserve_for(flag.issue_type, protected),
        "attention_stats": flag.attention_stats,
        "protected": protected,
    }


def build_findings(
    reports: List[ScreenComplexityReport],
    screens_by_id: Dict[str, ScreenCapture],
    crops_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build and aggregate findings across all screens."""
    by_id: Dict[str, Dict[str, Any]] = {}

    for report in reports:
        screen = screens_by_id.get(report.screen_id)
        if screen is None:
            continue
        for flag in report.flagged_problem_elements:
            finding = _finding_from_flag(report, screen, flag, crops_dir)
            issue_id = finding["issue_id"]
            if issue_id not in by_id:
                by_id[issue_id] = finding
                continue

            existing = by_id[issue_id]
            existing["confidence"] = round(max(existing["confidence"], finding["confidence"]), 2)
            existing["evidence"]["slices_seen"] += 1
            existing["evidence"]["screen_ids"].append(report.screen_id)
            if _normalize_severity(finding["severity"]) == "clear_problem":
                existing["severity"] = "clear_problem"
            if IMPACT_ORDER.get(finding["impact"], 3) < IMPACT_ORDER.get(existing["impact"], 3):
                existing["impact"] = finding["impact"]
            if not existing.get("crop_path") and finding.get("crop_path"):
                existing["crop_path"] = finding["crop_path"]

    findings = list(by_id.values())
    # Group identical issues that differ only by scroll position are already merged by issue_id.
    findings.sort(key=lambda f: (
        SEVERITY_ORDER.get(f["severity"], 9),
        IMPACT_ORDER.get(f["impact"], 9),
        -float(f["confidence"]),
        f["page"],
        f["view"],
        f["issue_id"],
    ))
    return findings


def summarize_findings(findings: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    findings = list(findings)
    counts = {"clear_problem": 0, "potential_improvement": 0, "needs_investigation": 0}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    return {
        **counts,
        "total": len(findings),
        "requires_human": sum(1 for f in findings if f.get("requires_human")),
        "top_issue_ids": [f["issue_id"] for f in findings[:5]],
    }


def diff_findings(before: List[Dict[str, Any]], after: List[Dict[str, Any]]) -> Dict[str, Any]:
    before_map = {f["issue_id"]: f for f in before}
    after_map = {f["issue_id"]: f for f in after}
    new_ids = sorted(set(after_map) - set(before_map))
    fixed_ids = sorted(set(before_map) - set(after_map))
    persisting_ids = sorted(set(before_map) & set(after_map))
    return {
        "new": [after_map[i] for i in new_ids],
        "fixed": [before_map[i] for i in fixed_ids],
        "persisting": [after_map[i] for i in persisting_ids],
        "counts": {"new": len(new_ids), "fixed": len(fixed_ids), "persisting": len(persisting_ids)},
    }
