"""CLI: `python -m regression_guard eval --prompt email_classifier_v1`."""
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .alerts import send_slack_alert
from .dataset import load_dataset
from .diff import detect_slow_drift, diff_runs
from .prompts import list_prompts, load_prompt
from .reporter import render_report
from .runner import run_eval
from .storage import init_db, latest_runs, previous_run, save_run


load_dotenv()
console = Console()


@click.group()
def cli():
    """LLM Regression Guard — CI for prompts."""
    init_db()


@cli.command()
@click.option("--prompt", "prompt_name", required=True, help="Prompt YAML name (no extension)")
@click.option("--dataset", "dataset_name", default="cases_v1", help="Golden dataset name")
@click.option("--no-judge", is_flag=True, help="Skip LLM-as-judge summary scoring (faster)")
@click.option("--concurrency", default=4, help="Parallel LLM calls (lower on free tiers)")
@click.option("--fail-on", default="critical", type=click.Choice(["never", "warn", "critical"]),
              help="Exit non-zero when severity meets/exceeds threshold")
def evaluate(prompt_name: str, dataset_name: str, no_judge: bool, concurrency: int, fail_on: str):
    """Run the eval pipeline against a prompt + dataset."""
    prompt = load_prompt(prompt_name)
    cases = load_dataset(dataset_name)
    console.print(f"[bold]Running[/] {prompt.fingerprint} against {len(cases)} cases…")

    def progress(i, total):
        console.print(f"  · {i}/{total} cases complete", end="\r")

    run = run_eval(prompt, cases, use_judge=not no_judge, concurrency=concurrency, on_progress=progress)
    save_run(run)
    console.print()

    # Compare against previous run if one exists
    prev = previous_run(run.run_id)
    diff = None
    if prev:
        dataset_by_id = {c.id: c.expected_category for c in cases}
        diff = diff_runs(prev, run, dataset_by_id)

    # Slow drift check across last ~14 runs
    history = latest_runs(limit=20)
    slow_drift = detect_slow_drift(history)

    # Render HTML
    report_path = render_report(run, diff=diff, trend_runs=history[:10])
    console.print(f"[green]✓[/] Report: file://{report_path.absolute()}")

    # Slack
    send_slack_alert(run, diff, report_url=None, slow_drift=slow_drift)

    # Print summary table
    _print_summary(run, diff)

    # Exit code for CI
    if diff:
        order = {"pass": 0, "warn": 1, "critical": 2}
        threshold = {"never": 99, "warn": 1, "critical": 2}[fail_on]
        if order[diff.severity] >= threshold:
            sys.exit(1)
    sys.exit(0)


def _print_summary(run, diff):
    t = Table(title=f"Run {run.run_id}", show_header=True, header_style="bold")
    t.add_column("Metric"); t.add_column("Value", justify="right")
    t.add_row("Cases", str(len(run.case_results)))
    t.add_row("Accuracy", f"{run.overall_accuracy*100:.1f}%")
    t.add_row("Mean summary score", f"{run.mean_summary_score:.2f}/5")
    t.add_row("Mean latency", f"{run.mean_latency_ms:.0f} ms")
    if diff:
        t.add_row("Severity", diff.severity.upper())
        t.add_row("Regressions", str(len(diff.regressions)))
        t.add_row("Improvements", str(len(diff.improvements)))
    console.print(t)


@cli.command(name="list-prompts")
def list_prompts_cmd():
    """List available prompt configs."""
    for p in list_prompts():
        console.print(f"  · {p}")


@cli.command()
def history():
    """Show recent eval runs."""
    runs = latest_runs(limit=10)
    if not runs:
        console.print("[dim]No runs yet.[/]")
        return
    t = Table(title="Recent runs", show_header=True, header_style="bold")
    t.add_column("Run ID"); t.add_column("Prompt"); t.add_column("Accuracy", justify="right"); t.add_column("Timestamp")
    for r in runs:
        t.add_row(r.run_id, r.prompt_fingerprint, f"{r.overall_accuracy*100:.1f}%", r.timestamp.strftime("%Y-%m-%d %H:%M"))
    console.print(t)


def main():
    cli()


if __name__ == "__main__":
    main()
