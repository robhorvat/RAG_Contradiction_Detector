from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.llm_clients import LocalRuleBasedJSONClient
from src.agent.schemas import LLMResponse


def main() -> None:
    client = LocalRuleBasedJSONClient()
    user_message = (
        "**Topic of Interest:** Does vitamin D reduce fractures?\n\n"
        "**Retrieved Passages from Paper 1 (ID: 19307517):**\n"
        "Higher received vitamin D dose reduced nonvertebral fractures in older adults.\n\n"
        "**Retrieved Passages from Paper 2 (ID: 8554248):**\n"
        "Results do not show a decrease in fractures after vitamin D supplementation."
    )
    payload, _ = client.generate_json(system_prompt="ignored", user_message=user_message)
    validated = LLMResponse.model_validate(payload)
    print(f"Smoke check verdict: {validated.analysis.verdict}")


if __name__ == "__main__":
    main()
