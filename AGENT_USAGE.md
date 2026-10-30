# Agent usage guide

This tool audits a rendered interface for **layout, crowding, and visual prominence**.
It produces machine-readable findings, evidence crops, and a versioned report. It is
designed to be called by an agent, not to make product decisions on its own.

## What the tool does and does not decide

The tool measures:

- element geometry, spacing, target sizes, and crowding groups;
- DOM structure, repeated box clusters, and content density;
- saliency attribution from DeepGaze (exclusive attention mass per element);
- noise, focus competition, and distraction **scores** derived from calibrated features.

The tool does **not** decide:

- which content is important to the user or the business;
- that salient or frequent elements should be removed;
- that a dropdown should be moved, a label renamed, or a feature deleted;
- whether a design change is technically feasible in the host codebase.

A finding is evidence plus a suggested next check. It is not an instruction to delete
anything. In particular, saliency must never be used to justify removing or hiding
content. Use it to prioritize inspection and to compare competing focal points.

## Install and check

```bash
python3 -m pip install -r requirements.txt
python3 -m playwright install chromium
python3 main.py check
```

`check` prints JSON and exits `0` when the core dependencies, Chromium, and the
threshold file are usable. It exits `3` when a dependency is missing.

## Commands

### Analyze

```bash
python3 main.py analyze \
  --target ./modern_app/index.html \
  --output ./modern_results \
  --json-out ./modern_results/complexity_report.json \
  --quiet
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--target` | Local HTML file or localhost URL. Remote URLs require `--allow-remote`. |
| `--output` | Directory for screenshots, overlays, crops, DOM snapshot, and the report. |
| `--json-out PATH` | Write the report to an explicit path. `-` prints it to stdout. |
| `--json` | Print the report to stdout. Progress still goes to stderr. |
| `--quiet` | Suppress the human-readable summary on stderr. |
| `--views desktop,mobile` | Views to crawl. Default: both. |
| `--fast` | Skip DeepGaze. Layout/DOM findings only; saliency scores are invalid. |
| `--no-crops` / `--no-overlays` | Skip evidence crops or diagnostic overlays. |
| `--no-cache` | Recompute heatmaps instead of using the output directory cache. |
| `--baseline PATH` | Add an inline `diff` against a previous report. |
| `--fail-on clear_problem` | Exit `4` if a finding at or above that severity exists. |
| `--fail-on potential_improvement` | Exit `4` for potential improvements or worse. |
| `--fail-on needs_investigation` | Exit `4` for any finding (all severities qualify). |

The legacy form `python3 main.py --target ...` is treated as `analyze`.

### Diff two reports

```bash
python3 main.py diff --before old/report.json --after new/report.json
```

The output lists added, resolved, and persisting findings by stable `issue_id`, plus
count changes. Use it to verify that a fix removed the intended finding without
introducing a new one.

### Check environment

```bash
python3 main.py check
```

## Output contract

### Streams

- **stdout**: JSON only, when `--json` or `--json-out -` is used. `check` and `diff`
  always print JSON to stdout.
- **stderr**: crawl progress, model/library chatter, the human summary, and errors.
- Files: the report and all artifacts are written under `--output`. Every path in the
  report is absolute.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Analysis completed and no `--fail-on` threshold was crossed. |
| `2` | Invalid/unreachable target, crawl failure, or no screens captured. Error JSON is on stderr. |
| `3` | `check` found a missing dependency or browser. |
| `4` | `--fail-on` threshold crossed. The report is still written. |

Unhandled internal errors use the process default (`1`); treat stderr as diagnostic and
do not parse stdout in that case.

### Report shape

Top-level keys:

```json
{
  "schema_version": "1.1",
  "run_id": "20261029T152437Z_ab12cd",
  "tool": {
    "name": "interface-audit",
    "version": "...",
    "thresholds_version": "2026-10-29.bootstrap",
    "calibration": "provisional"
  },
  "target": "/absolute/path/to/index.html",
  "started_at": "2026-10-29T15:24:37+00:00",
  "duration_s": 58.2,
  "config": {
    "views": ["desktop", "mobile"],
    "max_pages": 1,
    "device": "cuda",
    "fast": false,
    "thresholds_version": "2026-10-29.bootstrap"
  },
  "coverage": {
    "pages_found": 1,
    "pages_analyzed": 1,
    "screens": 5,
    "missing": [],
    "pages_failed": [],
    "measured": ["layout_geometry", "dom_structure", "spacing", "target_size", "crowding", "saliency_prominence", "noise", "focus_competition", "distraction"],
    "not_measured": ["task_importance", "safe_removal", "dropdown_placement", "naming_clarity", "technical_dependencies", "dynamic_ui_states"]
  },
  "limitations": ["..."],
  "summary": {
    "clear_problem": 6,
    "potential_improvement": 10,
    "needs_investigation": 4,
    "total": 20,
    "requires_human": 4,
    "top_issue_ids": ["f_..."]
  },
  "pages": ["..."],
  "screens": ["..."],
  "findings": ["..."],
  "diff": null,
  "artifacts": {
    "report_json": "/abs/path/complexity_report.json",
    "dom_snapshot": "/abs/path/dom_snapshot.json",
    "screenshots_dir": "/abs/path/screenshots",
    "overlays_dir": "/abs/path/diagnostic_overlays",
    "heatmaps": {
      "desktop_index_s0": "/abs/path/heatmaps/desktop_index_s0_heatmap.jpg",
      "mobile_index_s0": "/abs/path/heatmaps/mobile_index_s0_heatmap.jpg"
    },
    "crops_dir": "/abs/path/crops"
  }
}
```

`pages[].views[]` contains the weighted view score:

```json
{
  "score": 60.0,
  "overall": 60.0,
  "noise": 63.8,
  "focus_competition": 76.3,
  "crowding": 37.3,
  "distraction": 61.5,
  "level": "MODERATE",
  "valid": true,
  "screens": 2,
  "scroll_y": [0, 900],
  "saliency_mode": "enabled",
  "layout": {
    "scope": "document_union",
    "first_view_screen_id": "desktop_index_s0",
    "content_atoms": {"text": 33, "image": 0, "icon": 0, "button": 23, "total": 56},
    "first_view_atoms": 37,
    "columns": 3,
    "column_widths_px": [367.0, 367.0, 367.0],
    "column_edges_px": [[393.0, 760.0], [776.0, 1143.0], [1160.0, 1527.0]],
    "column_pitch_px": 383.5,
    "gutter_px": 16.5,
    "column_histogram": {"1": 7, "2": 1, "3": 9},
    "grid_confidence": 0.53,
    "symmetry_error_px": 0.0,
    "spacing_axis": "horizontal",
    "dominant_spacing_px": 16.5,
    "spacing_outliers": 0,
    "atom_gap_histogram": {"5": 2, "6": 5, "7": 1, "8": 2, "20": 1, "31": 1, "32": 2, "53": 2, "54": 5, "55": 5},
    "off_grid_count": 0,
    "whitespace_pct": 70.9,
    "orphan_lines": 0,
    "orphan_selectors": [],
    "grid_holes": 0,
    "underfilled_rows": 0,
    "min_row_fill_ratio": 1.0,
    "skewed_rows": 0,
    "content_skew_pct": 9.0,
    "layout_notes": [
      "Grid: 3-column layout over 8 repeated rows; gutter 16px, pitch 384px, widest offset 0px from the viewport edge.",
      "Measured: every repeated row stays on its column line within 2px, rows fill at least 100% of their grid width, and no row leaves a centre column empty."
    ],
    "fold_clipped_count": 3,
    "fold_clipped_atom_count": 9,
    "fold_clipped_pct": 19.6,
    "fold_clipped_selectors": ["#card-latency", "#card-tokens", "#card-error"]
  }
}
```

- `layout.scope` is `document_union`: the grid is measured on the deduped
  document (one entry per DOM element across scroll slices), not per slice. A
  component captured in overlapping slices is counted once.
- **Score modifiers** — two finding families add a bounded penalty on top of the
  weighted geometric score, so the number rhymes with what the findings say:
  `COMPLEX_STRUCTURE` (channel `structure`) and the grid-organization findings
  `GRID_GAP` / `ROW_UNDERFILL` / `BLOCK_IMBALANCE` / `ORPHAN_LINE` /
  `GRID_MISALIGNMENT` (channel `grid_organization`). Each finding contributes
  `points × severity_weight × impact_weight × (0.5 + 0.5 × confidence)` **per
  screen where it was seen**, capped per screen (`structure_cap` 7.5,
  `grid_organization_cap` 9.0); the two caps together top out at +16.5.
  Where to read it in the JSON:
  - `summary.score_modifiers_applied` — true whenever the channel is active.
  - `pages[].views[view].score_modifiers` — weighted `{structure_points,
    grid_organization_points, total_points}` for the view.
  - `screens[].score_modifiers` — `{base_overall, structure_points,
    grid_organization_points, total_points, final_overall, level,
    crossed_to_over_complex}`; the `overall` next to it is the final number.
  - `findings[].score_effect` — `{channel, points_per_screen, screens}` on
    exactly the findings that contributed; everything else (crowding, text
    size, saliency) leaves the score untouched.
  Fold clipping is still a metric + finding only, **not** a score input.
  Disable with `scoring_modifiers.enabled: false` in `config/thresholds.yaml`
  to get the pure base score for A/B comparisons.
- `content_atoms` counts text / image / icon / button elements. The grid is built
  from these atoms, grouped by their enclosing card, so a card's label+value pair
  is one column, not two. `columns` is the most common row shape; `column_histogram`
  shows every row shape so a few 3-across button rows are not mistaken for a
  1-column page.
- `gutter_px` is the content gap between adjacent cells (it includes card padding,
  so it can be larger than the CSS gap). `column_pitch_px` is the centre-to-centre
  column distance.
- `symmetry_error_px` is the median difference between the left and right outer
  margins of the main grid rows. It is a **fact**, not a verdict: a deliberately
  left-aligned page will have a large value and is not automatically a problem.
- `fold_clipped_*` is first-view only. See "First view, fold clipping, and layout
  geometry" below.
- `layout_notes` is the grid's own commentary — short factual lines about what was
  measured and what could be organized better (an unilled row, a dangling word,
  a skipped column). They are generated on every run, including well-organized
  pages, so the grid never "goes quiet"; treat them as pointers, and treat the
  corresponding findings as the actionable items.
- `orphan_lines` / `orphan_selectors`: text blocks whose last *rendered line*
  holds a dangling word. Measured from the crawler's per-line `text_lines`
  (word-level client rects), not from the element box.
- `grid_holes` counts rows that skip a middle grid column; `underfilled_rows`
  counts grid rows that fill far less than their siblings;
  `min_row_fill_ratio` is the smallest fill among rows that actually belong to
  the grid.
- `content_skew_pct` (first view) is how much more first-view ink sits on one
  side than the other. It is a **fact, not a verdict**: a deliberate sidebar or
  off-centre layout will show a high value and is not flagged by it.
- `valid` is `false` when saliency was skipped (`--fast`) or could not be computed.
- `score` is the weighted overall score, kept for comparison only. Prefer the
  component scores and findings.
- `level` is `LOW`, `MODERATE`, or `OVER_COMPLEX`. It is a coarse band, not a
  precise measurement. Current thresholds are provisional.

`pages[].comparison` is explicit about coverage:

```json
{
  "desktop_score": 60.0,
  "mobile_score": 83.8,
  "score_diff": 23.8,
  "coverage": "complete",
  "mobile_degradation_detected": true,
  "reason": null
}
```

`coverage` is one of `complete`, `missing_view`, `missing_score`, or
`saliency_disabled`. When it is not `complete`, `score_diff` and
`mobile_degradation_detected` are `null`. Do not infer mobile degradation from a
missing or invalid view.

`screens[]` contains per-screen scores, metrics, and evidence paths:

```json
{
  "screen_id": "mobile_index_s1",
  "page": "index",
  "view": "mobile",
  "scroll_y": 900,
  "image_path": "/abs/path/screenshots/mobile_index_s1.png",
  "saliency_mode": "enabled",
  "calibration_status": "provisional",
  "thresholds_version": "2026-10-29.bootstrap",
  "scores": {
    "overall": 96.09,
    "noise": 98.41,
    "focus_competition": 99.58,
    "crowding": 100.0,
    "distraction": 66.4
  },
  "level": "OVER_COMPLEX",
  "metrics": {"..." : "..."},
  "findings_count": 5
}
```

## Finding shape

Every finding has the same shape:

```json
{
  "issue_id": "f_1b13f1d212",
  "rule_id": "CROWDING_GROUP",
  "issue": "CROWDING_VIOLATION",
  "severity": "clear_problem",
  "impact": "high",
  "confidence": 0.9,
  "requires_human": false,
  "page": "index",
  "view": "mobile",
  "screen_id": "mobile_index_s0",
  "scroll_y": 0,
  "selector": "#btn-quota",
  "tag": "button",
  "text": "Quota",
  "related_selectors": ["#btn-quota", "#btn-cost"],
  "bbox": {"x": 44, "y": 439, "width": 302, "height": 30},
  "crop_path": "/abs/path/crops/f_1b13f1d212.png",
  "observed": {"gap_px": 3.0, "target_min_px": 30, "group_size": 2},
  "expected": {"gap_px": 16.0, "target_px": 44.0},
  "evidence": {
    "class": "deterministic_rule",
    "slices_seen": 1,
    "screen_ids": ["mobile_index_s2"],
    "thresholds_version": "2026-10-29.bootstrap"
  },
  "why": "2 adjacent controls are separated by 3.0px (threshold 16px); ...",
  "fix": "Increase the gap between these controls to at least 16px ...",
  "verify": "Re-run the audit; this issue_id should disappear ...",
  "preserve": "Preserve this element: keep it visible, reachable, and semantically intact. ...",
  "attention_stats": {"gap_px": 3.0, "severity": 0.812},
  "protected": true
}
```

Fields an agent should act on:

- `severity`: what kind of claim this is (see below).
- `impact`: expected user-facing consequence, `high` / `medium` / `low`.
- `confidence`: evidence strength, `0.0`–`1.0`. Deterministic geometry is `0.9`+.
- `why`: the measured fact and the rule that fired.
- `fix`: the smallest recommended change, not a mandatory rewrite.
- `verify`: the check to run after a change.
- `preserve`: what must remain visible, reachable, or semantically intact. When it
  conflicts with a tempting simplification, `preserve` wins.
- `observed` / `expected`: machine-readable comparison.
- `crop_path`: visual evidence. Open it when the claim is visual.
- `protected` and `requires_human`: guardrails, described next.

## Severity semantics

| Severity | Meaning | Agent behavior |
| --- | --- | --- |
| `clear_problem` | A deterministic rule is violated (for example, controls overlap or a touch group is below the spacing/size floor). | Fix the underlying geometry. Do not suppress the finding. |
| `potential_improvement` | A calibrated or heuristic pattern suggests friction, but context may justify it. | Investigate the crop and surrounding content. Recommend, do not assume. |
| `needs_investigation` | A semantic hypothesis (for example, attention may be pulled away from the primary task). | Ask a human to review; do not make a unilateral change. |

`requires_human` is `true` for `needs_investigation`, for distraction findings, and for
removal-like findings (focus, distraction, structure) on a protected target. A
deterministic spacing fix on a protected control can still be automated, but the
element itself must remain. `protected` means the element is interactive, a heading,
a landmark, or otherwise likely to carry task-critical content. **Never remove, hide,
or demote a protected element based on this tool's output.** Change its spacing, size,
grouping, or surrounding hierarchy only when the finding supports that specific change.

## How to use the findings

1. Run `analyze` with saliency enabled unless you only need layout checks.
2. Start with `clear_problem` findings. These are deterministic; fix geometry first.
3. For `potential_improvement`, read `why`, open `crop_path`, and check whether the
   element is protected. If it is protected, treat the fix as a layout change, not a
   removal.
4. For `needs_investigation`, route to a human with the crop and `why` text. Do not
   act on the score alone.
5. Apply the smallest change described by `fix`, then run `verify` (usually a new
   analysis and `diff`).
6. Use `diff` to confirm the issue is resolved and no new issue replaced it.

Recommended agent loop:

```bash
python3 main.py analyze --target ./app/index.html --output ./audit \
  --json-out ./audit/report.json --quiet
python3 - <<'PY'
import json
r = json.load(open("audit/report.json"))
for f in r["findings"]:
    print(f["severity"], f["issue"], f["selector"], f["why"])
PY
# Make one targeted change, then:
python3 main.py analyze --target ./app/index.html --output ./audit2 \
  --json-out ./audit2/report.json --baseline ./audit/report.json --quiet
```

## Noise, focus, distraction, and scores

- **Noise** measures *arrangement and clutter only*: element/control density, how
  much of the viewport is packed with content, attention fragmentation (peaks), and
  attention landing outside content. It deliberately excludes text legibility — a
  clean, well-spaced page that happens to use compact type is not "noisy." It is not
  a removal score.
- **Focus competition** describes whether attention is split between comparable
  content targets. The score is reported even when no finding fires, because a
  moderate score can still be useful context.
- **Distraction** describes attention on secondary, non-interactive content relative
  to the primary target. It is scored per screen.
- **Legibility** (small text) is reported as its own `TEXT_TOO_SMALL` finding, scored
  against an absolute font-size threshold (`< 12px`). It is intentionally **not** part
  of noise, because readability and clutter are different problems. Keep the content
  and raise the size — do not treat small text as a reason to remove anything.
- **Crowding** is deterministic and view-specific (desktop vs mobile thresholds).
- **Overall score** is a weighted summary for triage and comparison. It is **not** a
  percentage of "bad UI" and must not be optimized directly. Prefer reducing specific
  findings and comparing the component scores before/after.

Attention shares are exclusive per screen: each pixel belongs to at most one element,
and the sum of all element shares plus unassigned attention is at most 100%. If a
report ever shows a share above 100%, treat it as a bug and do not use the number.

### What counts as a `COMPLEX_STRUCTURE` cluster

The cluster catcher reports a region as a complexity problem only when it is both
structurally dense **and** draws real DeepGaze attention (`attention_share_pct >= 6`):

- **Grid** — `columns >= 2` boxes side-by-side in one row band, `box_count >= 3`, and
  `total_elements >= 6`. A multi-column card grid is the canonical case.
- **Dense list** — a single-column stack (`columns == 1`) that is interaction-dense:
  `box_count >= 4` and `controls_per_box >= 2.0` (e.g. a mobile card stack with 2–3
  buttons per card).

A scannable **single-column list** (one box per row, one action each) is a good design
and is deliberately **not** flagged just because it repeats. The finding's `observed`
carries `columns`, `control_count`, `controls_per_box`, and `layout` so you can see
which rule fired and why.

### What gets counted

Every element the crawler captures is classified as control / text / media /
container / landmark / decorative and **scored — nothing is silently dropped**. The
counts below are reported per screen (`visible_element_count`, `content_element_count`,
`control_count`, `text_density`, `content_density`, …):

- **Noise** is driven by the full element set, including `element_density`,
  `content_density` (text + controls + media per area), `control_density`,
  `density_outlier_pct`, `content_coverage_pct`, and `effective_peaks_norm`. Object
  count therefore directly affects the score; it is not inferred from a few samples.
- **Focus / distraction** use **all distinct content elements** (nested wrappers
  deduped), not an arbitrary top-N, so a page with 46 competing objects and a page
  with 31 can score differently. The report still displays only the top 5.
- **Box identity uses document-space size.** A card clipped by a scroll-slice
  boundary (e.g. only 17 px of a 162 px card visible at the edge) is still
  recognized as a box via its `doc_bbox`; only true slivers (`visible_fraction <
  0.15`) are skipped. This prevents long pages from losing whole cards between
  slices.

### First view, fold clipping, and layout geometry

The layout pass answers two questions: **what does one shot of the page show?**
and **do the visible objects line up?**

- **First view** is the slice at `scroll_y == 0`. It is the only slice used for
  fold clipping and for the "what one shot shows" object tally. Later slices are
  used only for the deduped document union (grid, symmetry, spacing).
- **Scroll noise is deduped.** The union is keyed by `dom_id`, so an element
  captured in two overlapping slices is counted once. Nothing is inflated by
  scrolling.
- **Fold-clipped content** is an element that is only partly inside the first
  viewport (`0.02 <= visible_fraction < 0.98`) and touches the top or bottom edge.
  It renders as a partial strip, which reads as visual noise rather than as a
  complete object. Such a component **and its descendants** are excluded from the
  first-view object tally (`first_view_atoms`) and reported as
  `FOLD_CLIPPED_CONTENT`. The finding says how much is visible, which edge cuts it,
  and recommends shifting it down (or resizing it) so it is either fully inside the
  first view or fully below the fold. It is **not** a removal instruction.
  Elements that start entirely below the fold (`visible_fraction == 0`) are not
  clipped slivers and are not flagged. `fold_clipped_count` counts clipped
  **components** (cards/boxes); `fold_clipped_atom_count` / `fold_clipped_pct` count
  the text/control atoms those components cover. A card whose visible strip contains
  no atoms can therefore have `fold_clipped_count == 1` with `fold_clipped_pct == 0`.
- **The grid is built from content atoms** — text, images, icons, and buttons — not
  from wrapper `div`s. Atoms in the same row are grouped by their enclosing card, so
  a card's label + value pair is one column rather than two. `columns` is the most
  common row shape; `column_histogram` shows all row shapes, and
  `grid_confidence` is the share of rows that match.
- **Symmetry is a metric, not a verdict.** `symmetry_error_px` reports the median
  difference between the left and right outer margins of the main grid rows. A
  deliberately left-aligned or asymmetric design can have a large value and is not
  automatically wrong. There is no "make it symmetric" finding.
- **`GRID_MISALIGNMENT`** fires only inside a *repeated* grid region (at least three
  rows sharing a left edge) when a column's start drifts by more than
  `layout.tolerance_px` from the same column in the other rows. It is a
  `needs_investigation` finding with `requires_human: true`; it does not imply
  removal. Isolated one-off rows are ignored to avoid false positives.
- **Organization findings** (the "design expert" comments) fire only inside a
  repeated multi-cell grid region, and only for rows that actually follow the
  grid's cell geometry. Deliberate narrow rails, spread page headers (logo left,
  CTA right), centred closing rows, and section lead text are never judged as
  cells, so they are never "broken rows":
  - **`ORPHAN_LINE`** — a text block whose last rendered line holds a dangling
    word ("...one lonely word trailing on its own line"). Needs the crawler's
    per-line `text_lines`; headings are exempt because titles are allowed to wrap.
  - **`GRID_GAP`** — a row that skips a *middle* column: a `potential_improvement`
    with `observed.missing_column`, `gap_px`, and `normal_gutter_px`.
  - **`ROW_UNDERFILL`** — a grid row that keeps the grid's cell shape but spans
    far less than its sibling rows (`observed.fill_ratio_pct`).
  - **`BLOCK_IMBALANCE`** — a row that piles most of its width on one side while
    its sibling rows spread evenly (`observed.skew_pct` vs
    `cluster_median_skew_pct`). A consistent lopsided layout is a metric
    (`content_skew_pct`), never a finding.
  All four are facts + suggested organization, with `preserve`; none recommend
  deleting or hiding content.
- **Fold clipping is currently surfaced as a metric + finding, not wired into the
  numeric noise score.** The score stays stable while the finding gives the agent a
  concrete, verifiable fix.
- **`preserve` is always present.** Every finding tells the agent what to keep. The
  layout pass never recommends deleting or hiding content based on attention.

The diagnostic overlay marks fold-clipped components with **amber dashed boxes**
and a `[FOLD-CLIPPED NN% visible]` label, adds `| N fold-clipped (noise)` to the
inventory tally, and draws an amber line at the first-view bottom edge. Fold-clipped
slivers are not counted as text/control/box objects in that tally. Grid
organization findings draw **magenta boxes**: `[ORPHAN 'word' NN% of width]`,
`[GRID GAP NNNpx vs gutter NNpx]`, `[ROW UNDERFILL NN% of grid width]`,
`[IMBALANCE NN% skew, left-heavy]`, `[COLUMN OFF NNpx]`, and
`[CELL WIDTH +-NNpx vs column]`; `ORPHAN_LINE` reuses the amber family.
`bias_probes/layout_organization.html` is a demo page that exercises every
organization detector (a widow line, a skipped middle column, an underfilled
closing row, and a deliberate sidebar that must stay unflagged).

## Calibration status and `--fast`

### Absolute scoring, not corpus-fitted

Every feature is scored against a **fixed, published threshold** — a `zero` point
(where it starts contributing) and a `sat` point (where it saturates) — declared in
`config/thresholds.yaml`. These anchors come from general UX norms (density,
spacing, attention concentration), **not** from the average of whichever pages were
in a calibration corpus. Each feature contributes a bounded 0..1 ramp, combined as
a weighted mean, so:

- the score means the same thing on any design;
- no single feature can swing the score across the whole 0..100 range;
- the score does not depend on a reference set at all.

This matters because a model *fit* to a small corpus learns to separate the pages it
saw, not interface quality. Earlier, the only two sample apps happened to differ in
font size, so a fitted model gave "uses small text" a huge weight and started
penalizing any clean layout that used compact type. Absolute thresholds remove that
failure mode: a font-size-only change on an identical layout now moves `noise` by
~4 points, and the tiny text is reported as its own `TEXT_TOO_SMALL` finding instead
of masquerading as clutter.

`config/thresholds.yaml` is still marked **provisional** — the *severity bands*
(`level_low` / `level_high`) are provisional until there are enough labels, and the
bootstrap corpus below is only a sanity check that the absolute scores separate a
dense reference from a simple one. On that corpus (10 screens, 5 dense / 5 simple):

| Metric | AUC on bootstrap corpus |
| --- | --- |
| noise | 0.80 |
| distraction | 0.88 |
| crowding | 0.92 (deterministic) |
| overall | 0.80 |
| focus | 0.50 — weak; focus stays a heuristic |

Do not present these as production-generalization estimates; the corpus is 10
screens. To audit the thresholds against new labels:

```bash
python3 calibrate.py \
  --labels calibration/labels.csv \
  --out config/thresholds.yaml \
  --report calibration/calibration_report.json
```

`calibrate.py` does **not** rewrite scoring weights. It measures per-feature and
per-score AUC as diagnostics, records provenance (n, method, warnings) into the
config, and warns loudly when the corpus is too small to support conclusions. It
writes `calibration/calibration_report.json` and `calibration/feature_auc.csv`. The
bootstrap labels are in `calibration/labels.csv`; sample compact reports are in
`calibration/reports/`.

`--fast` skips DeepGaze. Layout and crowding findings remain valid. Noise, focus,
distraction, and overall scores are computed from a uniform heatmap and are marked
`valid: false` / `saliency_mode: "disabled_uniform_heatmap"`. Never use `--fast`
output for prominence, focus, or distraction claims.

## Artifacts

| Path | Contents |
| --- | --- |
| `complexity_report.json` | Full report (default name; may be overridden by `--json-out`). |
| `screenshots/` | Per-scroll screenshots. |
| `diagnostic_overlays/` | Screenshots with the DeepGaze attention overlay, **every detected object outlined** (green = text, blue = control, cyan = structural box) plus a `Detected: N text \| M controls \| K boxes` tally, then problem annotations on top (magenta = cluster, red = crowding, orange = competition, yellow = distraction). **Amber dashed boxes** mark fold-clipped content (`[FOLD-CLIPPED NN% visible]`, counted as `fold-clipped (noise)` instead of as an object) and an amber line marks the first-view bottom edge. **Faint gray dashed vertical lines** mark the measured column edges on the first-view overlay. A page with no findings still shows its full object inventory, so "nothing flagged" is visibly different from "nothing detected". |
| `heatmaps/` | Raw DeepGaze saliency map per screen (`<screen_id>_heatmap.jpg`, jet-colormapped, aligned to screenshot size). Shows exactly where the model predicts attention lands. |
| `crops/` | Evidence crop per finding (`<issue_id>.png`). |
| `dom_snapshot.json` | Crawled DOM metadata used for the audit. |
| `.cache/heatmaps/` | DeepGaze heatmap cache; delete or use `--no-cache` to recompute. |

## Safety and scope

- Remote targets are rejected unless `--allow-remote` is passed. Prefer local files
  or localhost.
- The crawler disables animations and waits for stable layout, but it cannot cover
  every dynamic state. Treat findings on hover, menus, modals, and async content as
  hypotheses.
- The tool does not inspect source code, API behavior, or business logic. A finding
  that is valid visually may still be infeasible or undesirable in the product.
- Never use saliency to justify deleting content. Use it to ask better questions:
  what is competing, what is visually noisy, and what needs human review.

## Tests

```bash
python3 -m pytest -q tests/
```

The suite covers exclusive attribution bounds, cluster processing, focus deduplication,
missing-view comparisons, calibration separation, threshold provenance, and report
schema/crop integrity.
