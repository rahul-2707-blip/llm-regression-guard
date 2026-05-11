FROM python:3.11-slim

WORKDIR /app

# Install only what's needed for the eval runner (no dev tooling)
COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
COPY golden_dataset ./golden_dataset

RUN pip install --no-cache-dir -e .

# Persisted volumes (runs DB + HTML reports)
VOLUME ["/app/eval_runs", "/app/reports"]

# Required env vars (set at runtime):
#   GROQ_API_KEY            — required
#   SLACK_WEBHOOK_URL       — optional (falls back to stdout)
# Optional threshold overrides via env (read by future config loader):
#   REGRESSION_WARN_PCT     — default 3.0
#   REGRESSION_CRITICAL_PCT — default 8.0

ENTRYPOINT ["python", "-m", "regression_guard"]
CMD ["evaluate", "--prompt", "email_classifier_v1"]
