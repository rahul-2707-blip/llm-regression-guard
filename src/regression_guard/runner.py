"""Run the eval pipeline: every test case through the LLM, scored multi-dimensionally."""
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

from .classifier import classify_email
from .contracts import EvalRun, CaseResult, PromptConfig, TestCase
from .judge import judge_summary


def _run_one(case: TestCase, prompt: PromptConfig, use_judge: bool) -> CaseResult:
    call = classify_email(case.input, prompt)

    category_match = False
    summary_score = 0
    if call.parsed:
        category_match = call.parsed.category == case.expected_category
        if use_judge:
            summary_score = judge_summary(
                case.input, case.ideal_summary, call.parsed.summary
            )

    return CaseResult(
        case_id=case.id,
        raw_output=call.raw_output,
        parsed=call.parsed,
        parse_error=call.parse_error,
        category_match=category_match,
        summary_score=summary_score,
        latency_ms=call.latency_ms,
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
    )


def run_eval(
    prompt: PromptConfig,
    cases: list[TestCase],
    use_judge: bool = True,
    concurrency: int = 8,
    on_progress: Optional[callable] = None,
) -> EvalRun:
    """Execute every test case in parallel and return a complete EvalRun."""
    results: list[CaseResult] = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_run_one, c, prompt, use_judge): c for c in cases}
        for i, fut in enumerate(as_completed(futures), start=1):
            results.append(fut.result())
            if on_progress:
                on_progress(i, len(cases))

    results.sort(key=lambda r: r.case_id)

    return EvalRun(
        run_id=str(uuid.uuid4())[:8],
        prompt_fingerprint=prompt.fingerprint,
        prompt_version=prompt.version,
        model=prompt.model,
        timestamp=datetime.now(timezone.utc),
        case_results=results,
    )
