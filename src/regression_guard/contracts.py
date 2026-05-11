"""Typed contracts for prompts, outputs, and eval results."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


Category = Literal["billing", "technical", "account", "general"]
Difficulty = Literal["easy", "medium", "hard"]


class PromptConfig(BaseModel):
    """A versioned prompt configuration loaded from a YAML file."""
    id: str
    version: str
    created_at: str
    model: str
    temperature: float = 0.1
    description: str = ""
    system_prompt: str
    few_shot_examples: list[dict] = Field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        """Stable identifier including version, model, and temp."""
        return f"{self.id}@{self.version}-{self.model}-t{self.temperature}"


class ClassifierOutput(BaseModel):
    """What the LLM is required to return."""
    category: Category
    summary: str


class TestCase(BaseModel):
    """One hand-labeled golden test case."""
    id: str
    input: str
    expected_category: Category
    ideal_summary: str
    difficulty: Difficulty = "medium"
    notes: str = ""


class CaseResult(BaseModel):
    """Result of running one test case through the LLM."""
    case_id: str
    raw_output: Optional[str]
    parsed: Optional[ClassifierOutput]
    parse_error: Optional[str] = None
    category_match: bool = False
    summary_score: int = 0  # 1-5 from LLM-as-judge
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class EvalRun(BaseModel):
    """A complete run of the eval pipeline."""
    run_id: str
    prompt_fingerprint: str
    prompt_version: str
    model: str
    timestamp: datetime
    case_results: list[CaseResult]

    @property
    def overall_accuracy(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(1 for r in self.case_results if r.category_match) / len(self.case_results)

    @property
    def mean_summary_score(self) -> float:
        scored = [r.summary_score for r in self.case_results if r.summary_score > 0]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def mean_latency_ms(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(r.latency_ms for r in self.case_results) / len(self.case_results)
