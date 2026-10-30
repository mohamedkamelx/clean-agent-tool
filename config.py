"""Versioned threshold/calibration configuration.

The defaults are deliberately provisional. `calibrate.py` rewrites
`config/thresholds.yaml` from a labeled corpus and sets `provisional: false`
once there are enough labels.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "version": "0.1-provisional",
    "provisional": True,
    "calibration": {
        "n_screens": 0,
        "noise_auc": None,
        "focus_auc": None,
        "labeled_positive": 0,
        "labeled_negative": 0,
    },
    # NOTE: scoring uses ABSOLUTE, published thresholds (`zero`/`sat`/`weight`),
    # not corpus statistics. Each feature contributes a bounded 0..1 ramp; no single
    # feature can swing the score across the whole range. Legibility (tiny text) is
    # deliberately EXCLUDED from `noise` — it is a readability signal, not clutter —
    # and is reported as its own finding.
    "noise": {
        "features": {
            "density_outlier_pct": {"zero": 5.0, "sat": 40.0, "weight": 1.0},
            "control_density": {"zero": 0.8, "sat": 5.0, "weight": 1.0},
            "element_density": {"zero": 2.0, "sat": 14.0, "weight": 0.6},
            "content_density": {"zero": 1.5, "sat": 10.0, "weight": 0.8},
            "effective_peaks_norm": {"zero": 0.25, "sat": 1.0, "weight": 0.9},
            "content_coverage_pct": {"zero": 35.0, "sat": 78.0, "weight": 0.7},
            "unexplained_pct": {"zero": 3.0, "sat": 25.0, "weight": 0.9},
        },
        "level_low": 40.0,
        "level_high": 65.0,
    },
    "focus": {
        "features": {
            "effective_targets": {"zero": 1.5, "sat": 10.0, "weight": 1.0},
            "balance": {"zero": 0.35, "sat": 0.90, "weight": 0.8},
            "control_effective_targets": {"zero": 1.5, "sat": 8.0, "weight": 0.7},
            "secondary_content_share_pct": {"zero": 12.0, "sat": 50.0, "weight": 0.8},
        },
        "level_low": 40.0,
        "level_high": 65.0,
    },
    "distraction": {
        "features": {
            "secondary_content_share_pct": {"zero": 10.0, "sat": 45.0, "weight": 1.0},
            "decorative_pct": {"zero": 2.0, "sat": 20.0, "weight": 0.8},
            "unexplained_pct": {"zero": 3.0, "sat": 25.0, "weight": 0.9},
            "effective_peaks_norm": {"zero": 0.3, "sat": 1.0, "weight": 0.6},
        },
        "level_low": 40.0,
        "level_high": 65.0,
    },
    # Legibility: absolute font-size thresholds, reported as its own finding.
    # These do NOT feed the noise/distraction/overall scores.
    "legibility": {
        "tiny_text_pct": {"zero": 5.0, "sat": 40.0},   # % of text elements < 12px
        "tiny_text_threshold_px": 12.0,
    },
    "crowding": {
        "desktop": {
            "gap_px": 8.0,
            "target_px": 24.0,
            "clear_gap_px": 4.0,
            "clear_target_px": 24.0,
            "comfortable_target_px": 44.0,
        },
        "mobile": {
            "gap_px": 16.0,
            "target_px": 24.0,
            "clear_gap_px": 8.0,
            "clear_target_px": 24.0,
            "comfortable_target_px": 44.0,
        },
        "severity_weight": 25.0,
    },
    "weights": {
        "noise": 0.25,
        "focus": 0.35,
        "crowding": 0.30,
        "distraction": 0.10,
    },
    "levels": {"low": 40.0, "high": 65.0},
    # Bounded, finding-driven penalties applied ON TOP of the geometric score.
    # COMPLEX_STRUCTURE (repeated-box clusters) and the grid-organization
    # findings (GRID_GAP / ROW_UNDERFILL / BLOCK_IMBALANCE / ORPHAN_LINE /
    # GRID_MISALIGNMENT) each add points per screen where they were seen,
    # weighted by severity, impact and confidence, then capped per screen so
    # the base geometric/saliency scale always stays dominant. Set
    # enabled: false to restore the base-score-only behaviour.
    "scoring_modifiers": {
        "enabled": True,
        "structure_points": 2.5,
        "structure_cap": 7.5,
        "grid_organization_points": 1.5,
        "grid_organization_cap": 9.0,
        "severity_weight": {
            "clear_problem": 1.0,
            "potential_improvement": 0.7,
            "needs_investigation": 0.45,
        },
        "impact_weight": {"high": 1.25, "medium": 1.0, "low": 0.7},
        "confidence_weight": 0.5,  # points scale linearly 0.5x..1.0x by confidence
    },
    # M3 layout analyzer — deterministic geometry, not a score. Every value is a
    # tolerance/fact threshold; none of them assume a preferred design style.
    "layout": {
        "tolerance_px": 2.0,
        "row_band_tolerance_px": 10.0,
        "cell_merge_gap_px": 8.0,
        "min_container_px": 48.0,
        "symmetry_warn_ratio": 0.04,
        "spacing_tolerance_px": 2.0,
        "grid_width_variance": 0.15,
        "fold_clip_min_visible": 0.02,
        "fold_clip_max_visible": 0.98,
        "fold_clip_min_px": 8.0,
        "fold_clip_max_findings": 8,
        "max_atoms": 1200,
        # organization detectors ("design expert" comments on how rows are organized)
        "orphan_max_words": 2,
        "orphan_max_width_ratio": 0.30,
        "orphan_min_prev_chars": 12,
        "orphan_min_words": 4,
        "orphan_max_findings": 6,
        "underfill_ratio": 0.62,
        "imbalance_skew": 0.55,
        "imbalance_delta": 0.25,
        "min_det_cluster_rows": 3,
        "max_grid_findings": 12,
    },
}

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "thresholds.yaml")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_thresholds(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        return copy.deepcopy(DEFAULT_THRESHOLDS)
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return _deep_merge(DEFAULT_THRESHOLDS, loaded)
    except Exception:
        return copy.deepcopy(DEFAULT_THRESHOLDS)


def save_thresholds(config: Dict[str, Any], path: Optional[str] = None) -> str:
    path = path or DEFAULT_CONFIG_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import yaml  # type: ignore

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path
