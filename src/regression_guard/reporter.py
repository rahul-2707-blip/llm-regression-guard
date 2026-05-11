"""Generate HTML reports from eval runs and diffs."""
from pathlib import Path
from typing import Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape
from .contracts import EvalRun
from .diff import RunDiff


TEMPLATES_DIR = Path(__file__).parent / "templates"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def render_report(
    run: EvalRun,
    diff: Optional[RunDiff] = None,
    trend_runs: Optional[list[EvalRun]] = None,
    out_path: Optional[Path] = None,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.j2")

    trend_labels: list[str] = []
    trend_data: list[float] = []
    if trend_runs:
        for r in reversed(trend_runs):
            trend_labels.append(r.timestamp.strftime("%m-%d %H:%M"))
            trend_data.append(round(r.overall_accuracy * 100, 1))

    headline = diff.headline if diff else (
        f"Eval run · accuracy {run.overall_accuracy*100:.1f}% · {len(run.case_results)} cases"
    )

    html = template.render(
        run=run,
        diff=diff,
        headline=headline,
        trend_labels=trend_labels,
        trend_data=trend_data,
    )
    out_path = out_path or REPORTS_DIR / f"report_{run.run_id}.html"
    out_path.write_text(html)
    return out_path
