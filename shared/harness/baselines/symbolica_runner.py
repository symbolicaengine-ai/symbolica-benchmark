import time
from pathlib import Path
from typing import Dict, Any

import yaml

from symbolica import Engine, facts


class SymbolicaRunner:
    """Run benchmark cases using Symbolica engine.

    The runner loads a global ruleset once and reuses the Engine to evaluate
    each test case.  It also registers a deterministic fake PROMPT() so that
    the seed cases can run without an external LLM service.
    """

    def __init__(self, rules_path: Path | str):
        self.rules_path = Path(rules_path)
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {self.rules_path}")

        # Load engine from YAML rules file
        self.engine = Engine.from_file(self.rules_path)

        # Register a stub PROMPT() implementation so that rules calling PROMPT
        # succeed without network I/O.
        self.engine.register_function("PROMPT", self._fake_prompt, allow_unsafe=True)

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def run_case(self, case_path: Path | str) -> Dict[str, Any]:
        """Execute a single YAML test case and return detailed metrics."""
        case_path = Path(case_path)
        start_time = time.perf_counter()

        with case_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Extract facts and handle any special temporal data encoding.
        case_facts = dict(data.get("facts", {}))

        # Handle synthetic temporal series -> feed into TemporalStore
        spending_series = case_facts.pop("spending_series", None)
        if spending_series is not None:
            for value in spending_series:
                # Key is fixed as 'spend' for now; can be generalised later.
                self.engine.store_datapoint("spend", float(value))

        # Execute rules
        result = self.engine.reason(facts(**case_facts))

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Compare verdict with expectation if present
        expected_verdict = data.get("expected_verdict", {})
        verdict_correct = expected_verdict == result.verdict if expected_verdict else None

        return {
            "id": data.get("id", case_path.stem),
            "suite": case_path.parent.parent.name,  # s1_symbolic, etc.
            "verdict_correct": verdict_correct,
            "trace_exists": bool(result.reasoning),
            "latency_ms": latency_ms,
            "verdict": result.verdict,
            "expected_verdict": expected_verdict,
        }

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _fake_prompt(question: str, return_type: str | None = None, *args, **kwargs):
        """A deterministic PROMPT() substitute used for offline benchmarking.

        Very simple heuristics based on the question string content.
        """
        q_lower = question.lower()
        if "sentiment" in q_lower:
            # Positive sentiment if words like 'love' or 'great' present.
            return "positive" if any(w in q_lower for w in ["love", "great", "excellent", "good"]) else "negative"
        if "employer risk" in q_lower or "assess employer risk" in q_lower:
            return "high_risk" if "crypto" in q_lower else "low_risk"
        # Default fallback returns an empty string of the requested type
        if return_type == "bool":
            return False
        if return_type == "int":
            return 0
        if return_type == "float":
            return 0.0
        return "" 