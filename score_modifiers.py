"""Bounded, finding-driven score modifiers.

The geometric/saliency score (noise, focus competition, crowding, distraction)
stays pure. On top of it, findings that describe *structure and organization*
apply a small, capped penalty to the reported overall so the number the agent
sees rhymes with what the findings say:

- ``COMPLEX_STRUCTURE`` — a repeated many-box cluster raises the overall.
- ``GRID_GAP`` / ``ROW_UNDERFILL`` / ``BLOCK_IMBALANCE`` / ``ORPHAN_LINE`` /
  ``GRID_MISALIGNMENT`` — grid-organization comments raise the overall by a
  smaller amount (they are pointers, not verdicts; needs_investigation items
  weigh less and every finding is confidence-scaled).

Everything is driven by ``config["scoring_modifiers"]`` and hard-capped per
screen, so the base scale stays dominant and can be restored exactly by
setting ``enabled: false``. Every affected finding carries ``score_effect``
and every screen's metric_details carries ``score_modifiers`` in the JSON.
"""

from typing import Any, Dict, List, Optional

STRUCTURE_ISSUE = "COMPLEX_STRUCTURE"
GRID_ISSUE_TYPES = {
    "GRID_GAP",
    "ROW_UNDERFILL",
    "BLOCK_IMBALANCE",
    "ORPHAN_LINE",
    "GRID_MISALIGNMENT",
}

_DEFAULT_SEVERITY_WEIGHT = {
    "clear_problem": 1.0,
    "potential_improvement": 0.7,
    "needs_investigation": 0.45,
}
_DEFAULT_IMPACT_WEIGHT = {"high": 1.25, "medium": 1.0, "low": 0.7}


def _level_for(score: float, levels: Dict[str, Any]) -> str:
    low = float(levels.get("low", 40.0))
    high = float(levels.get("high", 65.0))
    if score < low:
        return "LOW"
    if score < high:
        return "MODERATE"
    return "OVER_COMPLEX"


def _points_for_finding(finding: Dict[str, Any], mods: Dict[str, Any]) -> float:
    sev_w = {**_DEFAULT_SEVERITY_WEIGHT, **(mods.get("severity_weight") or {})}
    imp_w = {**_DEFAULT_IMPACT_WEIGHT, **(mods.get("impact_weight") or {})}
    conf_w = float(mods.get("confidence_weight", 0.5))
    points_key = (
        "structure_points"
        if finding.get("issue") == STRUCTURE_ISSUE
        else "grid_organization_points"
    )
    base = float(mods.get(points_key, 0.0))
    if base <= 0:
        return 0.0
    severity = str(finding.get("severity", "") or "needs_investigation")
    impact = str(finding.get("impact", "") or "medium").lower()
    confidence = float(finding.get("confidence", 0.5) or 0.0)
    confidence = max(0.0, min(1.0, confidence))
    return base * sev_w.get(severity, 0.45) * imp_w.get(impact, 1.0) * (
        conf_w + (1.0 - conf_w) * confidence
    )


def apply_score_modifiers(
    reports: List[Any],
    findings: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    """Apply structural/grid penalties to each screen's overall (in place).

    Safe to call with modifiers disabled: then nothing is annotated and the
    reports keep their base scores.
    """
    mods = config.get("scoring_modifiers") or {}
    if not mods or not mods.get("enabled"):
        return
    levels = config.get("levels") or {"low": 40.0, "high": 65.0}
    structure_cap = float(mods.get("structure_cap", 7.5))
    grid_cap = float(mods.get("grid_organization_cap", 9.0))

    by_screen = {r.screen_id: r for r in reports}
    for finding in findings:
        issue = finding.get("issue") or finding.get("issue_type")
        if issue == STRUCTURE_ISSUE:
            is_structure = True
        elif issue in GRID_ISSUE_TYPES:
            is_structure = False
        else:
            continue
        points = _points_for_finding(finding, mods)
        if points <= 0:
            continue
        evidence = finding.get("evidence") or {}
        screen_ids = list(evidence.get("screen_ids") or [])
        if not screen_ids and finding.get("screen_id"):
            screen_ids = [finding["screen_id"]]
        if not screen_ids:
            continue
        finding["score_effect"] = {
            "channel": "structure" if is_structure else "grid_organization",
            "points_per_screen": round(points, 2),
            "screens": len(screen_ids),
        }
        for screen_id in screen_ids:
            report = by_screen.get(screen_id)
            if report is None:
                continue
            detail = report.metric_details.setdefault(
                "score_modifiers",
                {
                    "base_overall": round(float(report.overall_complexity_score), 2),
                    "structure_points": 0.0,
                    "grid_organization_points": 0.0,
                },
            )
            cap = structure_cap if is_structure else grid_cap
            key = "structure_points" if is_structure else "grid_organization_points"
            detail[key] = round(min(cap, float(detail[key]) + points), 2)

    # ---- apply the capped totals and recompute level ----
    for report in reports:
        detail = report.metric_details.get("score_modifiers")
        if not detail:
            continue
        structure_pts = float(detail.get("structure_points", 0.0))
        grid_pts = float(detail.get("grid_organization_points", 0.0))
        total = structure_pts + grid_pts
        if total <= 0:
            continue
        base = float(report.overall_complexity_score)
        new_overall = float(min(100.0, base + total))
        report.overall_complexity_score = new_overall
        report.complexity_level = _level_for(new_overall, levels)
        detail.update({
            "total_points": round(total, 2),
            "final_overall": round(new_overall, 2),
            "level": report.complexity_level,
            "crossed_to_over_complex": (
                new_overall >= float(levels.get("high", 65.0))
                and base < float(levels.get("high", 65.0))
            ),
        })


def view_modifiers_summary(
    modifier_details: List[Optional[Dict[str, Any]]], weights: List[float]
) -> Optional[Dict[str, Any]]:
    """Weighted-average the per-screen modifier totals for the view JSON."""
    usable = [
        (m, w) for m, w in zip(modifier_details, weights) if m
    ]
    if not usable:
        return None
    total_w = sum(w for _, w in usable) or 1.0
    structure = sum(float(m.get("structure_points", 0.0)) * w for m, w in usable) / total_w
    grid = sum(float(m.get("grid_organization_points", 0.0)) * w for m, w in usable) / total_w
    summary = {
        "structure_points": round(structure, 2),
        "grid_organization_points": round(grid, 2),
        "total_points": round(structure + grid, 2),
    }
    if any(m.get("crossed_to_over_complex") for m, _ in usable):
        summary["crossed_to_over_complex"] = True
    return summary
