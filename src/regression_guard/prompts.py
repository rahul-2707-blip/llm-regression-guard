"""Load versioned prompt configs from YAML."""
from pathlib import Path
import yaml
from .contracts import PromptConfig


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str) -> PromptConfig:
    """Load a prompt YAML by filename stem (e.g. 'email_classifier_v1')."""
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return PromptConfig(**data)


def list_prompts() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.yaml"))
