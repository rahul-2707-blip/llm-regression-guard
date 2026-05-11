"""Compare two eval runs to detect regressions and improvements."""
from typing import Literal
from pydantic import BaseModel
from .contracts import EvalRun, CaseResult


Severity = Literal["pass", "warn", "critical"]

# Statistical significance thresholds (configurable in real deployment)
WARN_THRESHOLD_PCT = 3.0
CRITICAL_THRESHOLD_PCT = 8.0


class CaseDiff(BaseModel):
    case_id: str
    old_match: bool
    new_match: bool
    old_summary_score: int
    new_summary_score: int
    old_output: str | None
    new_output: str | None


class RunDiff(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    overall_accuracy_delta_pct: float
    summary_score_delta: float
    latency_delta_ms: float
    regressions: list[CaseDiff]          # pass -> fail
    improvements: list[CaseDiff]         # fail -> pass
    per_category_accuracy_delta: dict[str, float]
    severity: Severity
    headline: str


def _categories_for_run(run: EvalRun, dataset_by_id: dict[str, str]) -> dict[str, list[CaseResult]]:
    """Group case results by their expected category (from the dataset)."""
    grouped: dict[str, list[CaseResult]] = {}
    for r in run.case_results:
        cat = dataset_by_id.get(r.case_id, "unknown")
        grouped.setdefault(cat, []).append(r)
    return grouped


def diff_runs(
    baseline: EvalRun,
    candidate: EvalRun,
    dataset_by_id: dict[str, str],
) -> RunDiff:
    """Compute a structured diff between two runs."""
    base_results = {r.case_id: r for r in baseline.case_results}
    cand_results = {r.case_id: r for r in candidate.case_results}

    regressions: list[CaseDiff] = []
    improvements: list[CaseDiff] = []
    for case_id in sorted(base_results.keys() & cand_results.keys()):
        b = base_results[case_id]
        c = cand_results[case_id]
        if b.category_match and not c.category_match:
            regressions.append(_to_diff(case_id, b, c))
        elif not b.category_match and c.category_match:
            improvements.append(_to_diff(case_id, b, c))

    acc_delta = (candidate.overall_accuracy - baseline.overall_accuracy) * 100
    sum_delta = candidate.mean_summary_score - baseline.mean_summary_score
    lat_delta = candidate.mean_latency_ms - baseline.mean_latency_ms

    base_by_cat = _categories_for_run(baseline, dataset_by_id)
    cand_by_cat = _categories_for_run(candidate, dataset_by_id)
    cat_delta: dict[str, float] = {}
    for cat in set(base_by_cat) | set(cand_by_cat):
        b_acc = _acc(base_by_cat.get(cat, []))
        c_acc = _acc(cand_by_cat.get(cat, []))
        cat_delta[cat] = (c_acc - b_acc) * 100

    severity = _classify(acc_delta)
    headline = _headline(severity, acc_delta, regressions, baseline, candidate)

    return RunDiff(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        overall_accuracy_delta_pct=acc_delta,
        summary_score_delta=sum_delta,
        latency_delta_ms=lat_delta,
        regressions=regressions,
        improvements=improvements,
        per_category_accuracy_delta=cat_delta,
        severity=severity,
        headline=headline,
    )


def _acc(rs: list[CaseResult]) -> float:
    return sum(1 for r in rs if r.category_match) / len(rs) if rs else 0.0


def _to_diff(case_id: str, b: CaseResult, c: CaseResult) -> CaseDiff:
    return CaseDiff(
        case_id=case_id,
        old_match=b.category_match,
        new_match=c.category_match,
        old_summary_score=b.summary_score,
        new_summary_score=c.summary_score,
        old_output=b.parsed.model_dump_json() if b.parsed else b.raw_output,
        new_output=c.parsed.model_dump_json() if c.parsed else c.raw_output,
    )


def _classify(delta_pct: float) -> Severity:
    if delta_pct <= -CRITICAL_THRESHOLD_PCT:
        return "critical"
    if delta_pct <= -WARN_THRESHOLD_PCT:
        return "warn"
    return "pass"


def _headline(
    severity: Severity,
    delta: float,
    regressions: list[CaseDiff],
    baseline: EvalRun,
    candidate: EvalRun,
) -> str:
    emoji = {"pass": "OK", "warn": "WARN", "critical": "CRITICAL"}[severity]
    return (
        f"[{emoji}] {len(regressions)} regression(s) detected — "
        f"accuracy {baseline.overall_accuracy*100:.1f}% → {candidate.overall_accuracy*100:.1f}% "
        f"(delta {delta:+.1f}%)"
    )


def detect_slow_drift(runs: list[EvalRun], window: int = 7, threshold_pct: float = 5.0) -> bool:
    """Return True if the trailing-window mean accuracy has dropped below baseline by threshold.

    Catches gradual degradation that per-run checks miss.
    """
    if len(runs) < window + 1:
        return False
    recent = runs[:window]
    older = runs[window : window * 2] if len(runs) >= window * 2 else runs[window:]
    if not older:
        return False
    recent_avg = sum(r.overall_accuracy for r in recent) / len(recent)
    older_avg = sum(r.overall_accuracy for r in older) / len(older)
    return (older_avg - recent_avg) * 100 >= threshold_pct
