"""Slack alerting via incoming webhooks."""
import json
import os
import urllib.request
from typing import Optional
from .contracts import EvalRun
from .diff import RunDiff


SLACK_COLORS = {
    "pass": "#16a34a",
    "warn": "#d97706",
    "critical": "#dc2626",
}


def send_slack_alert(
    run: EvalRun,
    diff: Optional[RunDiff],
    report_url: Optional[str] = None,
    slow_drift: bool = False,
) -> bool:
    """Post a structured message to Slack. Returns True on success.

    If SLACK_WEBHOOK_URL is unset, prints to stdout as a stub so local runs
    still produce visible output.
    """
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    severity = diff.severity if diff else "pass"
    color = SLACK_COLORS[severity]

    fields = [
        {"title": "Run", "value": run.run_id, "short": True},
        {"title": "Prompt", "value": run.prompt_fingerprint, "short": True},
        {"title": "Accuracy", "value": f"{run.overall_accuracy*100:.1f}%", "short": True},
        {"title": "Summary score", "value": f"{run.mean_summary_score:.2f}/5", "short": True},
    ]
    if diff:
        fields.append(
            {"title": "Regressions", "value": str(len(diff.regressions)), "short": True}
        )
        fields.append(
            {"title": "Improvements", "value": str(len(diff.improvements)), "short": True}
        )

    title = diff.headline if diff else f"Eval run {run.run_id}"
    if slow_drift:
        title = f"SLOW DRIFT — {title}"
        color = SLACK_COLORS["warn"] if severity == "pass" else color

    payload = {
        "attachments": [
            {
                "color": color,
                "title": title,
                "title_link": report_url,
                "fields": fields,
                "footer": "regression-guard",
            }
        ]
    }

    if not webhook or "xxx" in webhook:
        print(f"[slack-stub] no webhook configured — would send:\n{json.dumps(payload, indent=2)}")
        return False

    req = urllib.request.Request(
        webhook,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[slack] post failed: {e}")
        return False
