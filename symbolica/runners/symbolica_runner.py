#!/usr/bin/env python3
"""
Symbolica Rule Engine Runner
===========================

Dedicated runner for evaluating Symbolica's hybrid rule engine performance.
"""

import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import yaml


@dataclass
class SymbolicaResult:
    """Results from a Symbolica evaluation."""
    case_id: str
    suite: str
    
    # Decision results
    decision_correct: bool
    decision_expected: Dict[str, Any]
    decision_actual: Dict[str, Any]
    
    # Performance metrics
    latency_ms: float
    llm_calls: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    
    # Quality metrics
    has_reasoning: bool = False
    reasoning_trace: str = ""
    
    # Error handling
    error: Optional[str] = None


class SymbolicaRunner:
    """Symbolica rule engine runner with comprehensive evaluation."""
    
    def __init__(self, rules_path: Path):
        self.rules_path = rules_path
        self.engine = None
        self.stats = {
            'total_llm_calls': 0,
            'total_tokens': 0,
            'total_cost': 0.0
        }
    
    def setup(self) -> None:
        """Initialize Symbolica engine with rules and LLM client."""
        from symbolica import Engine
        
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        # Setup OpenAI client for PROMPT() functions
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                
                # Load rules from directory
                if self.rules_path.is_dir():
                    self.engine = Engine.from_directory(self.rules_path, llm_client=client)
                else:
                    self.engine = Engine.from_file(self.rules_path, llm_client=client)
                
                print(f"✓ Symbolica engine initialized with OpenAI integration")
                
            except ImportError:
                print("Warning: openai package not installed, PROMPT() functions will fail")
                self.engine = Engine.from_directory(self.rules_path)
        else:
            print("Warning: OPENAI_API_KEY not found, PROMPT() functions will fail")
            self.engine = Engine.from_directory(self.rules_path)
    
    def run_case(self, case_data: Dict[str, Any]) -> SymbolicaResult:
        """Execute a single test case with Symbolica."""
        start_time = time.perf_counter()
        
        try:
            from symbolica import facts
            
            # Extract case information
            case_id = case_data["id"]
            suite = self._extract_suite(case_data)
            expected_decision = case_data["expected_decision"]
            case_facts = case_data["facts"]
            
            # Execute reasoning
            result = self.engine.reason(facts(**case_facts))
            
            # Calculate metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            decision_correct = self._compare_decisions(expected_decision, result.verdict)
            
            # Extract LLM usage stats
            llm_calls, tokens_input, tokens_output, cost_usd = self._extract_llm_stats()
            
            return SymbolicaResult(
                case_id=case_id,
                suite=suite,
                decision_correct=decision_correct,
                decision_expected=expected_decision,
                decision_actual=result.verdict,
                latency_ms=latency_ms,
                llm_calls=llm_calls,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost_usd=cost_usd,
                has_reasoning=bool(result.reasoning),
                reasoning_trace=result.reasoning or ""
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return SymbolicaResult(
                case_id=case_data.get("id", "unknown"),
                suite=self._extract_suite(case_data),
                decision_correct=False,
                decision_expected=case_data.get("expected_decision", {}),
                decision_actual={},
                latency_ms=latency_ms,
                error=str(e)
            )
    
    def _extract_suite(self, case_data: Dict[str, Any]) -> str:
        """Extract suite name from case data."""
        case_id = case_data.get("id", "")
        if "age" in case_id or "credit_score" in case_id:
            return "s1_symbolic"
        elif "sentiment" in case_id:
            return "s2_hybrid"
        elif "spending" in case_id or "fraud" in case_id:
            return "s3_temporal"
        elif "pipeline" in case_id or "workflow" in case_id:
            return "s4_workflow"
        else:
            return "unknown"
    
    def _compare_decisions(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Compare expected vs actual decisions."""
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            if actual[key] != expected_value:
                return False
        return True
    
    def _extract_llm_stats(self) -> tuple:
        """Extract LLM usage statistics from the engine."""
        llm_calls = 0
        tokens_input = 0
        tokens_output = 0
        cost_usd = 0.0
        
        if hasattr(self.engine, '_prompt_evaluator') and self.engine._prompt_evaluator:
            try:
                # Get stats from the LLM adapter
                adapter = self.engine._prompt_evaluator.llm_adapter
                if hasattr(adapter, 'get_stats'):
                    stats = adapter.get_stats()
                    llm_calls = stats.get('total_calls', 0)
                    total_tokens = stats.get('total_tokens', 0)
                    tokens_input = total_tokens // 2  # Rough approximation
                    tokens_output = total_tokens // 2
                    cost_usd = stats.get('total_cost', 0.0)
            except:
                pass  # Fallback to defaults
        
        return llm_calls, tokens_input, tokens_output, cost_usd
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the runner."""
        return {
            'runner_type': 'symbolica',
            'rules_path': str(self.rules_path),
            'llm_integration': hasattr(self.engine, '_prompt_evaluator') and self.engine._prompt_evaluator is not None,
            'total_stats': self.stats
        }


def discover_test_cases(test_cases_dir: Path) -> List[Dict[str, Any]]:
    """Discover and load all test cases from the shared directory."""
    test_cases = []
    
    for yaml_file in test_cases_dir.rglob("*.yaml"):
        try:
            with yaml_file.open("r") as f:
                case_data = yaml.safe_load(f)
                if "id" in case_data and "expected_decision" in case_data:
                    test_cases.append(case_data)
        except Exception as e:
            print(f"Warning: Could not load {yaml_file}: {e}")
    
    return test_cases


def main():
    """Main evaluation function for Symbolica runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Symbolica Rule Engine Evaluation")
    parser.add_argument("--rules", type=Path, default=Path("../rules"),
                       help="Rules directory")
    parser.add_argument("--test-cases", type=Path, default=Path("../../shared/test_cases"),
                       help="Test cases directory")
    parser.add_argument("--output", type=Path, help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = SymbolicaRunner(args.rules)
    runner.setup()
    
    # Load test cases
    test_cases = discover_test_cases(args.test_cases)
    if not test_cases:
        print(f"No test cases found in {args.test_cases}")
        return
    
    print(f"Found {len(test_cases)} test cases")
    
    # Run evaluation
    results = []
    for i, case_data in enumerate(test_cases, 1):
        if args.verbose:
            print(f"Running case {i}/{len(test_cases)}: {case_data['id']}")
        
        result = runner.run_case(case_data)
        results.append(result)
        
        if args.verbose:
            status = "✓" if result.decision_correct else "✗"
            print(f"  {status} {result.case_id} ({result.latency_ms:.1f}ms)")
            if result.error:
                print(f"    Error: {result.error}")
    
    # Generate summary
    total_cases = len(results)
    correct_cases = sum(1 for r in results if r.decision_correct)
    accuracy = correct_cases / total_cases * 100 if total_cases > 0 else 0
    avg_latency = sum(r.latency_ms for r in results) / total_cases if total_cases > 0 else 0
    total_cost = sum(r.cost_usd for r in results)
    
    print(f"\n==== SYMBOLICA EVALUATION RESULTS ====")
    print(f"Total Cases: {total_cases}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Avg Latency: {avg_latency:.1f}ms")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"Errors: {sum(1 for r in results if r.error)}")
    
    # Suite breakdown
    suite_stats = {}
    for result in results:
        suite = result.suite
        if suite not in suite_stats:
            suite_stats[suite] = {"total": 0, "correct": 0}
        suite_stats[suite]["total"] += 1
        if result.decision_correct:
            suite_stats[suite]["correct"] += 1
    
    print("\nSuite Breakdown:")
    for suite, stats in suite_stats.items():
        accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {suite}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
    
    # Save results if requested
    if args.output:
        import json
        results_data = [
            {
                'case_id': r.case_id,
                'suite': r.suite,
                'decision_correct': r.decision_correct,
                'latency_ms': r.latency_ms,
                'llm_calls': r.llm_calls,
                'tokens_input': r.tokens_input,
                'tokens_output': r.tokens_output,
                'cost_usd': r.cost_usd,
                'has_reasoning': r.has_reasoning,
                'error': r.error,
                'expected_decision': r.decision_expected,
                'actual_decision': r.decision_actual
            }
            for r in results
        ]
        
        with args.output.open('w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main() 