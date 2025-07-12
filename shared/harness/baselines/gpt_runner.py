from pathlib import Path
from typing import Dict, Any


class GPTRunner:
    """Placeholder runner that will call an external LLM to answer cases.

    Not implemented yet – raises NotImplementedError by default.
    """

    def __init__(self, model: str = "gpt-4o-2024-07-10"):
        self.model = model

    def run_case(self, case_path: Path | str) -> Dict[str, Any]:
        raise NotImplementedError(
            "The GPT baseline is not implemented yet. Provide an OpenAI API key and logic to complete it."
        ) 