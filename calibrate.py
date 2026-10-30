"""Calibration *diagnostics* for the interface-audit scorer.

Important design note
---------------------
Base scoring is **absolute**, not fitted. Every feature is scored against a fixed UX
reference point (`zero`) and saturation point (`sat`) declared in `config/thresholds.yaml`.
That means the score means the same thing on any design and does NOT depend on which
pages happened to be in a calibration corpus.

This script therefore does NOT rewrite scoring weights. Refitting weights from a tiny
corpus is exactly what produced the earlier bias: with only two sample apps, a feature
that merely distinguishes those two apps (e.g. "uses small text") gets a huge weight and
starts standing in for interface quality. Instead, this harness:

  * measures per-feature AUC on the labeled corpus (diagnostic only),
  * measures whether the absolute scores separate a dense reference from a simple one,
  * records provenance (n, method, warnings) into the `calibration` block of the config,
  * refuses to claim calibration when the corpus is small.

Use it to *audit* the absolute thresholds, not to train a model.

Usage:
    python3 calibrate.py
    python3 calibrate.py --labels calibration/labels.csv --out config/thresholds.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import DEFAULT_CONFIG_PATH, load_thresholds, save_thresholds


SECTIONS = ("noise", "focus", "distraction", "crowding", "overall")


def _load_rows(labels_path: str, issue_type: str) -> List[Tuple[int, Dict[str, Any]]]:
    rows: List[Tuple[int, Dict[str, Any]]] = []
    labels_dir = os.path.dirname(os.path.abspath(labels_path))
    with open(labels_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("issue_type") or "").strip() != issue_type:
                continue
            report_path = (row.get("report_path") or "").strip()
            if not report_path:
                continue
            if not os.path.isabs(report_path):
                report_path = os.path.join(labels_dir, report_path)
            if not os.path.exists(report_path):
                print(f"[calibrate] Warning: missing report {report_path}", file=sys.stderr)
                continue
            label = int(str(row.get("label", "")).strip())
            screen_filter = (row.get("screen_id") or "*").strip() or "*"
            with open(report_path, "r", encoding="utf-8") as rf:
                report = json.load(rf)
            for screen in report.get("screens") or []:
                if screen_filter not in ("*", "") and screen.get("screen_id") != screen_filter:
                    continue
                rows.append((label, screen))
    return rows


def _feature_values(rows: Sequence[Tuple[int, Dict[str, Any]]], name: str) -> Tuple[np.ndarray, np.ndarray]:
    values = []
    labels = []
    for label, screen in rows:
        feats = (screen.get("metrics") or {}).get("features") or {}
        values.append(float(feats.get(name, 0.0) or 0.0))
        labels.append(int(label))
    return np.asarray(values, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _score_values(rows: Sequence[Tuple[int, Dict[str, Any]]], score_key: str) -> Tuple[np.ndarray, np.ndarray]:
    values = []
    labels = []
    for label, screen in rows:
        scores = screen.get("scores") or {}
        if score_key == "crowding":
            val = scores.get("crowding")
        else:
            val = scores.get(score_key)
        values.append(float(val or 0.0))
        labels.append(int(label))
    return np.asarray(values, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Rank-based AUC (Mann-Whitney U), ties count 0.5.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # Average ranks for ties.
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    sum_pos = float(ranks[labels == 1].sum())
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _feature_diagnostics(rows: Sequence[Tuple[int, Dict[str, Any]]], section_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    spec = (section_cfg or {}).get("features", {}) or {}
    out = []
    labels_all = np.array([y for y, _ in rows])
    for name in spec:
        vals, labels = _feature_values(rows, name)
        if vals.size == 0:
            continue
        a = auc(labels, vals)
        pos_mean = float(vals[labels == 1].mean()) if np.any(labels == 1) else 0.0
        neg_mean = float(vals[labels == 0].mean()) if np.any(labels == 0) else 0.0
        out.append({
            "feature": name,
            "auc": round(float(a), 4) if not math.isnan(a) else None,
            "positive_mean": round(pos_mean, 4),
            "negative_mean": round(neg_mean, 4),
            "oriented_correctly": bool(pos_mean >= neg_mean),
        })
    return out


def calibrate(
    labels_path: str,
    out_path: str,
    report_path: Optional[str] = None,
    min_auc: float = 0.80,
    dry_run: bool = False,
) -> Dict[str, Any]:
    config = load_thresholds(out_path)

    # Aggregate diagnostics across all issue sections present in the labels.
    per_section: Dict[str, Any] = {}
    n_pos = n_neg = 0
    for section_name in SECTIONS:
        rows = _load_rows(labels_path, section_name)
        if not rows:
            per_section[section_name] = {"available": False, "reason": "no labels"}
            continue
        labels_all = np.array([y for y, _ in rows])
        n_pos = max(n_pos, int((labels_all == 1).sum()))
        n_neg = max(n_neg, int((labels_all == 0).sum()))
        if section_name in ("noise", "focus", "distraction"):
            vals, labels = _score_values(rows, section_name)
            a = auc(labels, vals) if vals.size else float("nan")
            per_section[section_name] = {
                "available": True,
                "absolute_score_auc": round(float(a), 4) if not math.isnan(a) else None,
                "meets_min_auc": bool(not math.isnan(a) and a >= min_auc),
                "feature_diagnostics": _feature_diagnostics(rows, config.get(section_name, {})),
            }
        else:
            vals, labels = _score_values(rows, section_name)
            a = auc(labels, vals) if vals.size else float("nan")
            per_section[section_name] = {
                "available": True,
                "absolute_score_auc": round(float(a), 4) if not math.isnan(a) else None,
                "meets_min_auc": bool(not math.isnan(a) and a >= min_auc),
                "note": "deterministic/composite score; reported only",
            }

    n_screens = max(n_pos, 0) + max(n_neg, 0)
    small_corpus = (n_pos < 30) or (n_neg < 30)

    warnings: List[str] = []
    if small_corpus:
        warnings.append(
            f"Corpus is small ({n_pos} positive / {n_neg} negative screens). "
            "Base scoring is absolute and does not depend on it; do not treat these "
            "AUCs as generalization estimates."
        )
    for name in ("noise", "distraction", "overall"):
        sec = per_section.get(name) or {}
        if sec.get("available") and sec.get("absolute_score_auc") is not None and not sec.get("meets_min_auc"):
            warnings.append(
                f"Absolute {name} score separates the corpus weakly (AUC "
                f"{sec['absolute_score_auc']} < {min_auc}). Review the zero/sat anchors "
                "in config/thresholds.yaml against real screens."
            )

    config["version"] = "2026-10-29.absolute"
    config["provisional"] = True  # severity bands are provisional until n >= 30/class
    config["calibration"] = {
        "base_scoring": "absolute",
        "method": "absolute-threshold scoring; corpus used only as a sanity check",
        "n_screens": n_screens,
        "labeled_positive": n_pos,
        "labeled_negative": n_neg,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "sections": per_section,
        "warnings": warnings,
        "note": "Weights/anchors are fixed UX thresholds, NOT fitted to this corpus.",
    }

    payload = {
        "status": "ok" if not warnings else "ok_with_warnings",
        "base_scoring": "absolute",
        "provisional": True,
        "n_screens": n_screens,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "sections": per_section,
        "warnings": warnings,
        "config_version": config["version"],
    }

    if not dry_run:
        saved = save_thresholds(config, out_path)
        payload["config_path"] = saved
        if report_path:
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            payload["report_path"] = os.path.abspath(report_path)

        # Feature AUC table for reviewers.
        table_dir = os.path.dirname(os.path.abspath(report_path)) if report_path else os.path.dirname(os.path.abspath(out_path))
        table_path = os.path.join(table_dir, "feature_auc.csv")
        with open(table_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["section", "feature", "auc", "positive_mean", "negative_mean", "oriented_correctly"])
            for section_name, sec in per_section.items():
                for row in sec.get("feature_diagnostics", []) or []:
                    writer.writerow([
                        section_name, row["feature"], row["auc"],
                        row["positive_mean"], row["negative_mean"], row["oriented_correctly"],
                    ])
        payload["feature_auc_csv"] = table_path

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit absolute scoring thresholds against labeled screens")
    parser.add_argument("--labels", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration", "labels.csv"))
    parser.add_argument("--out", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--report", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration", "calibration_report.json"))
    parser.add_argument("--min-auc", type=float, default=0.80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    payload = calibrate(
        labels_path=args.labels,
        out_path=args.out,
        report_path=args.report,
        min_auc=args.min_auc,
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    for w in payload.get("warnings", []):
        print(f"[calibrate] WARNING: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
