"""The LLM-powered email classifier feature under test."""
import json
import os
import time
from typing import Optional
from groq import Groq
from .contracts import PromptConfig, ClassifierOutput
from .rate_limit import with_rate_limit_and_retry


_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        _client = Groq(api_key=api_key, max_retries=6)
    return _client


class ClassificationCall:
    """Holds the result of one classifier invocation including raw output and metrics."""

    def __init__(
        self,
        raw_output: Optional[str],
        parsed: Optional[ClassifierOutput],
        parse_error: Optional[str],
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
    ):
        self.raw_output = raw_output
        self.parsed = parsed
        self.parse_error = parse_error
        self.latency_ms = latency_ms
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def classify_email(email_text: str, prompt: PromptConfig) -> ClassificationCall:
    """Run one email through the classifier.

    Returns the parsed output + raw metrics. Does NOT raise on parse failure —
    that's a signal the eval engine needs to record, not a crash.
    """
    client = _get_client()
    messages = [{"role": "system", "content": prompt.system_prompt}]
    for ex in prompt.few_shot_examples:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": email_text})

    @with_rate_limit_and_retry
    def _call():
        return client.chat.completions.create(
            model=prompt.model,
            temperature=prompt.temperature,
            response_format={"type": "json_object"},
            messages=messages,
        )

    start = time.perf_counter()
    response = _call()
    latency_ms = int((time.perf_counter() - start) * 1000)

    raw = response.choices[0].message.content
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    parsed: Optional[ClassifierOutput] = None
    parse_error: Optional[str] = None
    try:
        data = json.loads(raw or "{}")
        parsed = ClassifierOutput(**data)
    except Exception as e:
        parse_error = str(e)

    return ClassificationCall(
        raw_output=raw,
        parsed=parsed,
        parse_error=parse_error,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
