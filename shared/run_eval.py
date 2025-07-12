#!/usr/bin/env python3
"""
Symbolica vs LLM Benchmark Harness
==================================

Comprehensive evaluation framework comparing Symbolica hybrid engine
against pure LLM reasoning across multiple dimensions.
"""

import argparse
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import yaml


@dataclass
class BenchmarkResult:
    """Results from a single test case evaluation."""
    case_id: str
    suite: str
    difficulty: str
    category: str
    
    # Decision results
    decision_correct: bool
    decision_expected: Dict[str, Any]
    decision_actual: Dict[str, Any]
    
    # Performance metrics
    latency_ms: float
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    cost_usd: Optional[float] = None
    
    # Quality metrics
    has_reasoning: bool = False
    reasoning_quality: Optional[str] = None
    
    # Error handling
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            'case_id': self.case_id,
            'suite': self.suite,
            'difficulty': self.difficulty,
            'category': self.category,
            'decision_correct': self.decision_correct,
            'latency_ms': self.latency_ms,
            'tokens_input': self.tokens_input or 0,
            'tokens_output': self.tokens_output or 0,
            'cost_usd': self.cost_usd or 0.0,
            'has_reasoning': self.has_reasoning,
            'error': self.error or '',
            'expected_decision': json.dumps(self.decision_expected),
            'actual_decision': json.dumps(self.decision_actual)
        }


class BenchmarkRunner:
    """Base class for benchmark runners."""
    
    def __init__(self, name: str):
        self.name = name
    
    def run_case(self, case_data: Dict[str, Any]) -> BenchmarkResult:
        """Run a single test case. Override in subclasses."""
        raise NotImplementedError
    
    def setup(self) -> None:
        """Setup runner. Override if needed."""
        pass
    
    def teardown(self) -> None:
        """Cleanup runner. Override if needed."""
        pass


class SymbolicaRunner(BenchmarkRunner):
    """Symbolica rule engine runner."""
    
    def __init__(self, rules_path: Path):
        super().__init__("symbolica")
        self.rules_path = rules_path
        self.engine = None
    
    def setup(self) -> None:
        """Initialize Symbolica engine with rules."""
        from symbolica import Engine
        import os
        
        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # dotenv not required
        
        # Get OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                # Load rules from directory or file
                if self.rules_path.is_dir():
                    self.engine = Engine.from_directory(self.rules_path, llm_client=client)
                else:
                    self.engine = Engine.from_file(self.rules_path, llm_client=client)
                print(f"SymbolicaRunner: Using real OpenAI API for PROMPT() functions")
            except ImportError:
                print("Warning: openai package not installed, PROMPT() functions will fail")
                # Load rules from directory or file
                if self.rules_path.is_dir():
                    self.engine = Engine.from_directory(self.rules_path)
                else:
                    self.engine = Engine.from_file(self.rules_path)
        else:
            print("Warning: OPENAI_API_KEY not found, PROMPT() functions will fail")
            # Load rules from directory or file
            if self.rules_path.is_dir():
                self.engine = Engine.from_directory(self.rules_path)
            else:
                self.engine = Engine.from_file(self.rules_path)
    

    
    def run_case(self, case_data: Dict[str, Any]) -> BenchmarkResult:
        """Execute a test case with Symbolica."""
        start_time = time.perf_counter()
        
        try:
            from symbolica import facts
            
            # Extract case info
            case_id = case_data["id"]
            suite = self._extract_suite_from_case(case_data)
            difficulty = case_data.get("difficulty", "unknown")
            category = case_data.get("category", "unknown")
            expected_decision = case_data["expected_decision"]
            case_facts = case_data["facts"]
            
            # Execute rules
            result = self.engine.reason(facts(**case_facts))
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Check correctness
            decision_correct = self._compare_decisions(expected_decision, result.verdict)
            
            # Get LLM usage stats if available
            tokens_input = 0
            tokens_output = 0
            cost_usd = 0.0
            
            # Extract LLM stats from engine if available
            if hasattr(self.engine, '_prompt_evaluator') and self.engine._prompt_evaluator:
                try:
                    llm_stats = self.engine._prompt_evaluator.llm_adapter.get_stats()
                    # Estimate token usage and cost (rough approximation)
                    if llm_stats.get('total_calls', 0) > 0:
                        tokens_input = llm_stats.get('total_tokens', 0) // 2  # Rough split
                        tokens_output = llm_stats.get('total_tokens', 0) // 2
                        cost_usd = llm_stats.get('total_cost', 0.0)
                except:
                    pass  # Fallback to defaults
            
            return BenchmarkResult(
                case_id=case_id,
                suite=suite,
                difficulty=difficulty,
                category=category,
                decision_correct=decision_correct,
                decision_expected=expected_decision,
                decision_actual=result.verdict,
                latency_ms=latency_ms,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost_usd=cost_usd,
                has_reasoning=bool(result.reasoning)
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return BenchmarkResult(
                case_id=case_data.get("id", "unknown"),
                suite=self._extract_suite_from_case(case_data),
                difficulty=case_data.get("difficulty", "unknown"),
                category=case_data.get("category", "unknown"),
                decision_correct=False,
                decision_expected=case_data.get("expected_decision", {}),
                decision_actual={},
                latency_ms=latency_ms,
                error=str(e)
            )
    
    def _extract_suite_from_case(self, case_data: Dict[str, Any]) -> str:
        """Extract suite name from case ID or other indicators."""
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
        # Check if all expected fields are present and match
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            if actual[key] != expected_value:
                return False
        return True


class GPTRunner(BenchmarkRunner):
    """OpenAI GPT runner for comparison."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("gpt")
        self.model = model
        self.client = None
    
    def setup(self) -> None:
        """Initialize OpenAI client."""
        import os
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for GPT runner")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def run_case(self, case_data: Dict[str, Any]) -> BenchmarkResult:
        """Execute a test case with GPT."""
        start_time = time.perf_counter()
        
        try:
            # Construct prompt
            prompt = self._build_prompt(case_data)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,  # Deterministic
                max_tokens=500
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Parse response
            response_text = response.choices[0].message.content
            decision_actual = self._parse_response(response_text)
            
            # Extract metrics
            usage = response.usage
            tokens_input = usage.prompt_tokens
            tokens_output = usage.completion_tokens
            cost_usd = self._calculate_cost(tokens_input, tokens_output)
            
            # Check correctness
            expected_decision = case_data["expected_decision"]
            decision_correct = self._compare_decisions(expected_decision, decision_actual)
            
            return BenchmarkResult(
                case_id=case_data["id"],
                suite=self._extract_suite_from_case(case_data),
                difficulty=case_data.get("difficulty", "unknown"),
                category=case_data.get("category", "unknown"),
                decision_correct=decision_correct,
                decision_expected=expected_decision,
                decision_actual=decision_actual,
                latency_ms=latency_ms,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost_usd=cost_usd,
                has_reasoning=len(response_text) > 50
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return BenchmarkResult(
                case_id=case_data.get("id", "unknown"),
                suite=self._extract_suite_from_case(case_data),
                difficulty=case_data.get("difficulty", "unknown"),
                category=case_data.get("category", "unknown"),
                decision_correct=False,
                decision_expected=case_data.get("expected_decision", {}),
                decision_actual={},
                latency_ms=latency_ms,
                error=str(e)
            )
    
    def _build_prompt(self, case_data: Dict[str, Any]) -> str:
        """Build prompt for GPT evaluation."""
        scenario = case_data.get("scenario", "")
        facts = case_data["facts"]
        
        # Format facts nicely
        facts_str = "\n".join(f"- {k}: {v}" for k, v in facts.items())
        
        prompt = f"""You are a banking decision system. Analyze the following scenario and make a decision.

Scenario: {scenario}

Customer Information:
{facts_str}

Please provide your decision as a JSON object with the appropriate fields. 
For example: {{"approved": true, "reason": "meets_criteria"}}

Your response should be valid JSON only, no additional text."""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse GPT response into decision dictionary."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: try to parse simple yes/no responses
                text_lower = response_text.lower()
                if "approved" in text_lower or "eligible" in text_lower:
                    return {"approved": True}
                else:
                    return {"approved": False}
        except:
            return {"error": "parse_failed"}
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage."""
        # GPT-4o-mini pricing (as of 2024)
        input_cost_per_1k = 0.00015  # $0.15 per 1M tokens
        output_cost_per_1k = 0.0006  # $0.60 per 1M tokens
        
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
    def _extract_suite_from_case(self, case_data: Dict[str, Any]) -> str:
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


def discover_test_cases(benchmark_dir: Path) -> List[Dict[str, Any]]:
    """Discover and load all test cases."""
    test_cases = []
    
    for yaml_file in benchmark_dir.rglob("*.yaml"):
        # Skip rules files
        if "rules" in yaml_file.parts:
            continue
        
        try:
            with yaml_file.open("r") as f:
                case_data = yaml.safe_load(f)
                if "id" in case_data and "expected_decision" in case_data:
                    test_cases.append(case_data)
        except Exception as e:
            print(f"Warning: Could not load {yaml_file}: {e}")
    
    return test_cases


def generate_summary(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Generate benchmark summary statistics."""
    if not results:
        return {}
    
    total_cases = len(results)
    correct_cases = sum(1 for r in results if r.decision_correct)
    accuracy = correct_cases / total_cases * 100
    
    avg_latency = sum(r.latency_ms for r in results) / total_cases
    total_cost = sum(r.cost_usd or 0 for r in results)
    total_tokens = sum((r.tokens_input or 0) + (r.tokens_output or 0) for r in results)
    
    # Suite breakdown
    suite_stats = {}
    for result in results:
        suite = result.suite
        if suite not in suite_stats:
            suite_stats[suite] = {"total": 0, "correct": 0}
        suite_stats[suite]["total"] += 1
        if result.decision_correct:
            suite_stats[suite]["correct"] += 1
    
    # Add accuracy percentages
    for suite, stats in suite_stats.items():
        stats["accuracy"] = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
    
    return {
        "total_cases": total_cases,
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "total_cost_usd": total_cost,
        "total_tokens": total_tokens,
        "suite_breakdown": suite_stats,
        "errors": sum(1 for r in results if r.error)
    }


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Symbolica vs LLM Benchmark")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("../test_cases"),
                       help="Directory containing benchmark test cases")
    parser.add_argument("--rules", type=Path, default=Path("../rules/"),
                       help="Rules directory or file for Symbolica")
    parser.add_argument("--runner", choices=["symbolica", "gpt"], default="symbolica",
                       help="Which runner to use")
    parser.add_argument("--output", type=Path, help="Output CSV file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Discover test cases
    test_cases = discover_test_cases(args.benchmark_dir)
    if not test_cases:
        print(f"No test cases found in {args.benchmark_dir}")
        return
    
    print(f"Found {len(test_cases)} test cases")
    
    # Initialize runner
    if args.runner == "symbolica":
        runner = SymbolicaRunner(args.rules)
    elif args.runner == "gpt":
        runner = GPTRunner()
    else:
        raise ValueError(f"Unknown runner: {args.runner}")
    
    # Setup runner
    try:
        runner.setup()
    except Exception as e:
        print(f"Failed to setup {args.runner} runner: {e}")
        return
    
    # Run benchmark
    results = []
    for i, case_data in enumerate(test_cases, 1):
        if args.verbose:
            print(f"Running case {i}/{len(test_cases)}: {case_data['id']}")
        
        result = runner.run_case(case_data)
        results.append(result)
        
        if args.verbose and result.error:
            print(f"  Error: {result.error}")
    
    # Cleanup
    runner.teardown()
    
    # Generate summary
    summary = generate_summary(results)
    
    print(f"\n==== {runner.name.upper()} Benchmark Results ====")
    print(f"Total Cases: {summary['total_cases']}")
    print(f"Accuracy: {summary['accuracy']:.1f}%")
    print(f"Avg Latency: {summary['avg_latency_ms']:.1f}ms")
    if summary['total_cost_usd'] > 0:
        print(f"Total Cost: ${summary['total_cost_usd']:.4f}")
        print(f"Total Tokens: {summary['total_tokens']}")
    print(f"Errors: {summary['errors']}")
    
    print("\nSuite Breakdown:")
    for suite, stats in summary['suite_breakdown'].items():
        print(f"  {suite}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")
    
    # Save results
    if args.output:
        fieldnames = ['case_id', 'suite', 'difficulty', 'category', 'decision_correct', 
                     'latency_ms', 'tokens_input', 'tokens_output', 'cost_usd', 
                     'has_reasoning', 'error', 'expected_decision', 'actual_decision']
        
        with args.output.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([result.to_dict() for result in results])
        
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main() 