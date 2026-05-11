"""LLM-as-judge for grading summary relevance on a 1-5 scale.

Uses the same Groq Llama model as the classifier — in a real production setup
you'd use a STRONGER model (different family) as the judge to reduce bias.
For this MVP we accept the same-model trade-off and document it.
"""
import json
import os
from typing import Optional
from groq import Groq
from .rate_limit import with_rate_limit_and_retry


JUDGE_MODEL = "llama-3.3-70b-versatile"
_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=6)
    return _client


JUDGE_SYSTEM = """You grade summaries of customer support emails. Given:
- The original email
- An ideal summary (gold standard)
- A candidate summary produced by a model

Score the candidate on a 1-5 scale:
5 = captures the customer's ask perfectly; equivalent to the ideal
4 = captures the ask but is slightly less precise or misses a minor detail
3 = mostly correct but missing or distorting one important element
2 = partially correct; significant omission or wrong emphasis
1 = wrong, hallucinated, or misses the point entirely

Respond ONLY with JSON: {"score": 1-5, "reason": "one-sentence justification"}"""


def judge_summary(email: str, ideal: str, candidate: str) -> int:
    """Return a 1-5 score. Returns 0 on judge failure (treated as 'no signal')."""
    if not candidate:
        return 0
    client = _get_client()

    @with_rate_limit_and_retry
    def _call():
        return client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Email:\n{email}\n\nIdeal summary:\n{ideal}\n\n"
                        f"Candidate summary:\n{candidate}\n\nGrade now."
                    ),
                },
            ],
        )

    try:
        response = _call()
        data = json.loads(response.choices[0].message.content or "{}")
        score = int(data.get("score", 0))
        return max(1, min(5, score)) if score else 0
    except Exception:
        return 0
