# Setup & Run — step by step

From zero (after `git clone`) to your first audit report. Every command is
copy-paste ready. The project scans rendered pages for layout, crowding and
visual-prominence problems and writes JSON findings + diagnostic overlays.

---

## 0. What you need

| Requirement | Notes |
|---|---|
| **Python 3.10 or 3.11** | `python3 --version` to check. 3.10 tested. |
| **pip + venv** | Ships with Python. |
| **git** | To clone. |
| **NVIDIA GPU (optional)** | Speeds up DeepGaze saliency a lot. CPU-only works too — the tool falls back to CPU automatically, or force it with `--device cpu`. |
| **Disk** | ~1 GB for PyTorch wheels, ~300 MB for the Chromium browser, ~120 MB for the DeepGaze weights. |
| **Linux** | Tested on Linux. On Windows use WSL2 with these same commands. |

What will be installed, in order:
1. PyTorch + torchvision (CPU or CUDA build)
2. Project requirements — numpy, scipy, opencv-python, playwright, PyYAML, deepgaze-pytorch
3. Chromium browser (for Playwright)
4. DeepGaze IIE pretrained weights (auto-download on first run)

---

## 1. Clone and enter the project

```bash
git clone https://github.com/*YOUR-USER*/*YOUR-REPO*.git
cd *YOUR-REPO*        # the folder that contains main.py
```

Confirm you are in the right place:

```bash
ls
# you should see: main.py  crawler.py  layout_analyzer.py  findings.py
#                 complexity_scorer.py  heatmap.py  config.py  config/  DeepGaze/
```

> If the terminal is showing `python` instead of `python3` anywhere below,
> just use `python` consistently — everything here works either way.

---

## 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS / WSL
# Windows PowerShell: .venv\Scripts\Activate.ps1
```

Your prompt now starts with `(.venv)` — every `pip install` below goes into this
environment and never touches your system Python. Never run `pip install` with
`sudo`.

To leave the environment later: `deactivate`. To re-enter after reopening the
terminal: `source .venv/bin/activate`.

---

## 3. Install PyTorch + torchvision first

### Option A — CPU only (simplest, works everywhere)

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Option B — NVIDIA GPU (CUDA 11.8; faster DeepGaze saliency)

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

(For a CUDA 12.x driver use `cu121` instead of `cu118` in the URL. Check your
driver with `nvidia-smi` — the "CUDA Version" shown there must be >= the wheel
version you pick.)

Installing torch **before** the rest matters: the later `requirements.txt`
install then sees torch is already satisfied and won't rebuild it.

---

## 4. Install the project requirements

```bash
python3 -m pip install -r requirements.txt
```

This installs: `numpy`, `scipy`, `opencv-python`, `playwright`, `PyYAML`,
`deepgaze-pytorch` (plus deps pulled in automatically).

> You may see a one-line warning when importing SciPy about the NumPy version
> range (`A NumPy version >=... <... is required ... detected version ...`).
> It is harmless for this tool — ignore it.

---

## 5. DeepGaze — two ways

`heatmap.py` imports `import deepgaze_pytorch`, so the package must be visible
to your venv.

### Option A (default) — from PyPI  *(already done by step 4)*

Nothing more to do. `requirements.txt` installs `deepgaze-pytorch`.

### Option B — use the copy bundled in this repo (pinned, offline-friendly)

```bash
python3 -m pip uninstall -y deepgaze-pytorch
python3 -m pip install -e ./DeepGaze
```

The `-e` installs the local `DeepGaze/` folder in editable mode. This guarantees
the exact DeepGaze revision this tool was validated with.

---

## 6. Install the Chromium browser (for Playwright)

```bash
python3 -m playwright install chromium
```

On a **fresh server / bare Linux** where a browser fails to launch, also install
the system libraries (asks for sudo):

```bash
python3 -m playwright install-deps chromium
```

Verify the browser installed:

```bash
python3 -m playwright --version          # prints e.g. "Version 1.40.0"
ls ~/.cache/ms-playwright/               # should contain a chromium-* folder
```

---

## 7. DeepGaze model weights (auto-download, one time)

The first non-`--fast` run downloads the pretrained DeepGaze IIE weights
automatically:

- File: `deepgaze2e.pth` (~120 MB)
- URL: `https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/deepgaze2e.pth`
- Saved to: `~/.cache/torch/hub/checkpoints/deepgaze2e.pth`

**Manual download** (offline machine, or blocked network) — run:

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
curl -L -o ~/.cache/torch/hub/checkpoints/deepgaze2e.pth \
  https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/deepgaze2e.pth
```

Subsequent runs load from this cache instantly.

---

## 8. Verify the environment

```bash
python3 main.py check
```

Exit code `0` and every block below set to `"ok": true` means you are ready:

```json
"numpy":   { "ok": true, ... },  "torch":   { "ok": true },
"playwright": { "ok": true },    "deepgaze_pytorch": { "ok": true },
"playwright_chromium": { "ok": true },
"thresholds": { "ok": true, "version": "..." }
```

Exit code non-zero → read the JSON on stdout: the blocks with `"ok": false`
tell you exactly what is missing (usually step 6 or step 4).

---

## 9. Run your first audit

The repo ships two demo pages. Try the "clean" one first:

```bash
python3 main.py analyze \
  --target ./modern_app_clean/index.html \
  --output modern_clean_results \
  --json-out modern_clean_results/report.json
```

What you'll see: progress per screen on stderr, then a summary
(`summary`, scores per view, top issues) on stdout.

**Quiet mode** (JSON only on stdout if `--json` is added; logs stay on stderr):

```bash
python3 main.py analyze \
  --target ./modern_app_clean/index.html \
  --output modern_clean_results \
  --json-out modern_clean_results/report.json \
  --quiet
```

Exit codes: `0` = audit completed (severity counts are in the summary),
`2` = configuration / environment / crawl error, and `4` = findings exist at or
above the severity you passed with `--fail-on` (e.g. `--fail-on clear_problem`
for a CI gate).

### Useful analyze flags

| Flag | Effect |
|---|---|
| `--fast` | Skip DeepGaze saliency (layout/DOM findings only) — much faster, CPU friendly |
| `--device cpu` / `--device cuda` | Force the saliency device |
| `--views desktop,mobile` | Which viewport families to crawl |
| `--no-overlays` / `--no-crops` | Skip diagnostic overlay images / finding crops |
| `--no-cache` | Ignore the heatmap cache and recompute |
| `--max-pages N` | Cap how many pages get crawled for a directory/site target |
| `--allow-remote` | Allow non-localhost URLs (default refuses remote targets) |
| `--baseline old_report.json` | Also diff against a previous run in the same report |

---

## 10. Read the results

Everything lands under the `--output` directory:

```
modern_clean_results/
├── report.json               # machine-readable report (also --json-out copy)
├── complexity_report.json    # same content (stable artifact name)
├── dom_snapshot.json         # extracted DOM geometry, one entry per screen
├── screenshots/              # one PNG per scroll slice per view
├── diagnostic_overlays/      # screenshots + score annotations (share these)
├── heatmaps/                 # DeepGaze saliency maps
└── crops/                    # cropped image per finding (evidence)
```

The JSON the agent/you read:

- `summary` — counts by severity + `score_modifiers_applied`
- `pages[].views[desktop|mobile]` — `overall`, `noise`, `focus_competition`,
  `crowding`, `distraction`, `level` (LOW / MODERATE / OVER_COMPLEX),
  then `layout` (grid facts + `layout_notes`) and `score_modifiers`
- `screens[]` — per scroll slice scores + `metrics.layout`
- `findings[]` — every issue: `what`, `why`, `recommended_fix`,
  `verify`, `preserve`, `severity` (clear_problem / potential_improvement /
  needs_investigation), `requires_human`, `evidence` with selectors,
  plus `score_effect` on findings that moved the score

Pretty-print the JSON report instead of reading it raw:

```bash
python3 main.py analyze ... --quiet --json | python3 -m json.tool | head -50
```

---

## 11. Diff two runs (did the fix work?)

```bash
python3 main.py diff \
  --before path/to/old_report.json \
  --after  path/to/new_report.json
```

Prints `new` / `fixed` / `persisting` issue lists keyed by `issue_id` —
run `analyze` again on the same target after editing the page, then
diff, to verify each finding's `verify` text actually passed.

---

## 12. Full walkthrough on the bundled demo pair

```bash
# busy enterprise console (should point at many issues)
python3 main.py analyze --target ./modern_app/index.html \
  --output modern_results --json-out modern_results/report.json --quiet

# calm editorial page (should stay quiet, LOW)
python3 main.py analyze --target ./modern_app_clean/index.html \
  --output modern_clean_results --json-out modern_clean_results/report.json --quiet

# show what differs between them
python3 main.py diff \
  --before modern_clean_results/report.json \
  --after  modern_results/report.json | python3 -m json.tool | head -60
```

Expected on the demo pair: the console scores OVER_COMPLEX (~70–76 overall)
while the ledger page stays LOW (~22–29). Look at
`modern_results/diagnostic_overlays/*_diagnostic.jpg` — red/amber marked
problem boxes with labels, amber dashed boxes at the fold, magenta boxes for
grid organization comments.

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `check` shows `playwright_chromium: ok: false` | Run `python3 -m playwright install chromium` (step 6), or `--install-deps` variant on bare servers. |
| `ImportError: deepgaze_pytorch` | `pip install -r requirements.txt` again, or `pip install -e ./DeepGaze` (step 5B). |
| Torch says CUDA unavailable but you have a GPU | The torch wheel you installed was the CPU build. Re-run step 3B, then `pip install -r requirements.txt` again. Or just use CPU (`--device cpu` not needed — auto). |
| `CUDA out of memory` | Add `--device cpu`, or `--fast` (skips saliency entirely). |
| Very slow first run | The boot run downloads weights (~120 MB) + model warmup. Later runs reuse `~/.cache/torch/hub/checkpoints/` and `.cache/heatmaps/`. |
| `Target local file does not exist` | `--target` must be an existing HTML file (or a localhost URL). Relative paths are resolved from your current directory. |
| Remote URL refused | The tool only audits `localhost` targets by default; pass `--allow-remote`. |
| Browser "missing libraries" on a server | `python3 -m playwright install-deps chromium` (needs sudo once). |
| NumPy/SciPy version warning at import | Harmless for this tool's usage; ignore. |
| Scores look different between two runs of the same file | Make sure both runs used the same `--fast` setting; saliency mode is recorded per screen (`saliency_mode`). Diff requires both sides saliency-enabled. |

---

## 14. Command cheat sheet

```bash
source .venv/bin/activate

python3 main.py check                                                  # env sanity
python3 main.py analyze --target ./page_or_folder_or_url --output out --json-out out/report.json   # audit
python3 main.py analyze ... --fast                                     # layout-only quick pass
python3 main.py analyze ... --device cpu                               # force CPU saliency
python3 main.py analyze ... --fail-on clear_problem                    # CI gate: exit 4 on clear problems
python3 main.py diff --before old.json --after new.json                # compare two runs
```

Full CLI reference lives inside `AGENT_USAGE.md` at the repo root — the agent
contract (every JSON key, every finding type, exit codes, thresholds, bias
rules) is documented there.
