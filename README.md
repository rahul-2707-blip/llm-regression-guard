# LLM Regression Guard

CI/CD-style regression detection for LLM-powered features. Every prompt or model change is automatically tested against a hand-curated golden dataset; quality regressions are caught with statistical thresholds and reported via HTML diff + Slack alerts before they reach production.

## Why this exists

Most teams ship prompt changes blind — a tweak that fixes one case silently breaks five others, and no one notices until users complain. This project treats prompts like code: versioned, diffed, and CI'd against a golden dataset on every PR.

## Architecture

```
prompts/*.yaml          ← versioned prompt configs (the "code" being tested)
golden_dataset/*.json   ← hand-labeled test cases with edge cases
src/regression_guard/
  ├── classifier.py     ← the LLM feature under test
  ├── judge.py          ← LLM-as-judge for summary scoring
  ├── runner.py         ← parallel test execution
  ├── diff.py           ← run-to-run comparison + statistical thresholds
  ├── storage.py        ← SQLite history of runs
  ├── reporter.py       ← HTML reports with trend charts
  ├── alerts.py         ← Slack webhook alerts
  └── rate_limit.py     ← token-bucket rate limiter
.github/workflows/eval.yml  ← GitHub Action — runs on PRs touching prompts/
```

## What gets measured (multi-dimensional scoring)

For every test case:
1. **Exact category match** (binary) — did the classifier pick the right bucket?
2. **Summary relevance** (1–5, LLM-as-judge) — is the summary good?
3. **Latency per request** (ms)
4. **Token usage** (input + output)

Every dimension is stored per case, per run, in SQLite.

## What gets compared (run-to-run diffing)

Each new run is automatically compared against the previous run:
- Overall accuracy delta (% points)
- Per-category accuracy delta
- Mean summary score delta
- Mean latency delta
- **Specific cases that flipped pass → fail (regressions)**
- **Specific cases that flipped fail → pass (improvements)**

A severity is assigned via configurable thresholds:
- `pass`: no significant change
- `warn`: 3–8% accuracy drop
- `critical`: ≥8% accuracy drop — blocks PR merge

A **slow-drift detector** also flags gradual degradation invisible to per-run checks (7-run trailing average vs. older baseline).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env
# fill GROQ_API_KEY (free tier from console.groq.com)
# fill SLACK_WEBHOOK_URL (optional; falls back to console)

python -m regression_guard evaluate --prompt email_classifier_v1
# → runs the eval, prints summary, generates reports/report_<id>.html
```

To demonstrate regression detection, run the deliberately-regressed v2:

```bash
python -m regression_guard evaluate --prompt email_classifier_v2
# → CRITICAL severity, 7 regressions detected, exit code 1
```

## CI/CD integration

Push to GitHub with `GROQ_API_KEY` and `SLACK_WEBHOOK_URL` as repo secrets. The included GitHub Action runs on every PR that touches `prompts/`, comments on the PR with results, and blocks merge on critical regressions.

## Tech stack

- Python 3.11+
- Pydantic (typed contracts for prompts, outputs, results)
- Groq (Llama 3.3 70B) — free tier; swap to OpenAI/Anthropic by replacing the client
- SQLite (run history)
- Jinja2 (HTML reports)
- Chart.js (trend visualization)
- GitHub Actions (CI)
- Slack incoming webhooks (alerts)

## Design rationale (one I'm proud of)

**Why slow-drift detection is separate from per-run regression detection.**

The obvious approach is a single threshold: "if accuracy drops by N% vs the previous run, alert." That catches sudden breaks — someone ships a bad prompt, every case fails, alarm fires. But it misses the more dangerous failure mode: **gradual rot.**

Imagine accuracy drops 0.5% per week for 12 weeks. No single PR ever triggers a threshold, but the system is now 6% worse than three months ago. The team has no idea, because every individual change "looked fine."

So this project tracks two signals independently:

1. **Per-run delta** — short-horizon. Compares the latest run to the immediately previous run. Catches sudden breaks. Configurable warn at 3%, critical at 8%.
2. **Slow drift** — long-horizon. Compares the 7-run trailing mean to an older baseline (runs 8–14). Catches gradual degradation that per-run checks never see. Fires a "slow drift" warning even when no single run looks bad.

This mirrors how production monitoring works (request-level alerts vs. SLO burn-rate alerts) and is the kind of thinking that distinguishes someone who has actually run production AI from someone who has just integrated an API.

## Interview talking points

1. **Why the golden dataset matters.** Evaluation quality is bounded by data quality. The dataset is hand-labeled (no LLM-generated ground truth), deliberately includes edge cases (multilingual, sarcasm, multi-intent, typos), and tags each case with difficulty so you can track regression patterns by complexity.
2. **Why LLM-as-judge, and its limits.** Category match is binary and trivial; summary quality needs a judge. Using the same model family for judging introduces bias — a production setup would use a stronger/different family. The score is treated as "no signal" (0) on judge failure, not as a 1.
3. **Why thresholds, not raw deltas.** A 2/80 case flip is noise. A statistical-significance threshold (configurable: 3% warn, 8% critical) prevents alert fatigue while catching real regressions.
4. **Why slow-drift detection.** Per-run thresholds miss death-by-a-thousand-cuts degradation. A 7-run trailing-average detector catches gradual prompt rot that no single PR triggered.
