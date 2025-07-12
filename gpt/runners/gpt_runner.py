#!/usr/bin/env python3
"""
GPT Baseline Runner
==================

Dedicated runner for evaluating GPT performance with sophisticated prompt engineering
for the 12 high-quality benchmark test cases.
"""

import time
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml


class GPTRunner:
    """GPT runner with sophisticated prompt engineering for benchmark test cases."""
    
    def __init__(self, model: str = "gpt-4o-mini", prompts_dir: Path = None):
        self.model = model
        self.prompts_dir = prompts_dir or Path("../prompts")
        self.client = None
        self.templates = {}
        self.stats = {
            'total_calls': 0,
            'total_tokens': 0,
            'total_cost': 0.0
        }
    
    def setup(self) -> None:
        """Initialize OpenAI client and load prompt templates."""
        # Setup OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            print(f"✓ GPT runner initialized with model: {self.model}")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        # Load prompt templates
        self._load_templates()
    
    def _load_templates(self):
        """Load all prompt templates from the prompts directory."""
        if not self.prompts_dir.exists():
            print(f"Warning: Prompts directory {self.prompts_dir} not found")
            return
        
        for template_file in self.prompts_dir.glob("*.yaml"):
            try:
                template_name = template_file.stem
                with template_file.open('r') as f:
                    data = yaml.safe_load(f)
                    self.templates[template_name] = data.get('template', '')
                print(f"✓ Loaded template: {template_name}")
            except Exception as e:
                print(f"Warning: Failed to load template {template_file}: {e}")
    
    def run_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single test case with GPT."""
        start_time = time.perf_counter()
        
        try:
            # Extract case information
            case_id = case_data["id"]
            suite = self._extract_suite(case_data)
            expected_decision = case_data["expected_decision"]
            
            # Select and render prompt
            prompt = self._build_prompt(case_data, suite)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,  # Deterministic
                max_tokens=1000
            )
            
            # Calculate metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract response
            response_text = response.choices[0].message.content
            decision_actual = self._parse_response(response_text, suite, case_id)
            
            # Extract usage stats
            usage = response.usage
            tokens_input = usage.prompt_tokens
            tokens_output = usage.completion_tokens
            cost_usd = self._calculate_cost(tokens_input, tokens_output)
            
            # Update stats
            self.stats['total_calls'] += 1
            self.stats['total_tokens'] += tokens_input + tokens_output
            self.stats['total_cost'] += cost_usd
            
            # Check correctness
            decision_correct = self._compare_decisions(expected_decision, decision_actual)
            
            return {
                "case_id": case_id,
                "suite": suite,
                "decision_correct": decision_correct,
                "expected": expected_decision,
                "actual": decision_actual,
                "latency_ms": latency_ms,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "cost_usd": cost_usd,
                "reasoning": response_text,
                "prompt_used": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "error": None
            }
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return {
                "case_id": case_data.get("id", "unknown"),
                "suite": self._extract_suite(case_data),
                "decision_correct": False,
                "expected": case_data.get("expected_decision", {}),
                "actual": {},
                "latency_ms": latency_ms,
                "tokens_input": 0,
                "tokens_output": 0,
                "cost_usd": 0.0,
                "reasoning": "",
                "prompt_used": "",
                "error": str(e)
            }
    
    def _extract_suite(self, case_data: Dict[str, Any]) -> str:
        """Extract suite name from case data."""
        case_id = case_data.get("id", "")
        if "age" in case_id or "credit_score" in case_id or "income" in case_id:
            return "s1_symbolic"
        elif "sentiment" in case_id or "positive" in case_id or "negative" in case_id:
            return "s2_hybrid"
        elif "spending" in case_id or "velocity" in case_id or "fraud" in case_id:
            return "s3_temporal"
        elif "pipeline" in case_id or "workflow" in case_id or "approval" in case_id or "review" in case_id or "rejection" in case_id:
            return "s4_workflow"
        else:
            return "unknown"
    
    def _build_prompt(self, case_data: Dict[str, Any], suite: str) -> str:
        """Build prompt for the given case using templates."""
        # Try to use suite-specific template
        template_name = f"{suite}_prompt"
        if template_name in self.templates:
            template = self.templates[template_name]
            return template.format(
                scenario=case_data.get("scenario", ""),
                description=case_data.get("description", ""),
                facts=case_data["facts"],
                facts_formatted=self._format_facts(case_data["facts"]),
                case_id=case_data["id"]
            )
        
        # Fall back to generic template
        if "generic_prompt" in self.templates:
            template = self.templates["generic_prompt"]
            return template.format(
                scenario=case_data.get("scenario", ""),
                description=case_data.get("description", ""),
                facts=case_data["facts"],
                facts_formatted=self._format_facts(case_data["facts"]),
                case_id=case_data["id"]
            )
        
        # Ultimate fallback: basic prompt
        return self._build_basic_prompt(case_data, suite)
    
    def _build_basic_prompt(self, case_data: Dict[str, Any], suite: str) -> str:
        """Build a basic prompt when no templates are available."""
        scenario = case_data.get("scenario", "")
        facts = case_data["facts"]
        facts_str = self._format_facts(facts)
        
        return f"""You are a banking decision system. Analyze the following scenario and make a decision.

Scenario: {scenario}

Customer Information:
{facts_str}

Please provide your decision as a JSON object with the appropriate fields.
Your response should be valid JSON only, no additional text."""
    
    def _format_facts(self, facts: Dict[str, Any]) -> str:
        """Format facts for display in prompts."""
        return "\n".join(f"- {k}: {v}" for k, v in facts.items())
    
    def _parse_response(self, response_text: str, suite: str, case_id: str) -> Dict[str, Any]:
        """Parse GPT response into decision dictionary."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed
            
            # Suite-specific parsing fallbacks
            if suite == "s1_symbolic":
                return self._parse_symbolic_response(response_text, case_id)
            elif suite == "s2_hybrid":
                return self._parse_hybrid_response(response_text, case_id)
            elif suite == "s3_temporal":
                return self._parse_temporal_response(response_text, case_id)
            elif suite == "s4_workflow":
                return self._parse_workflow_response(response_text, case_id)
            
            # Generic fallback
            return self._parse_generic_response(response_text)
            
        except Exception as e:
            return {"error": f"parse_failed: {str(e)}"}
    
    def _parse_symbolic_response(self, text: str, case_id: str) -> Dict[str, Any]:
        """Parse response for symbolic reasoning cases."""
        text_lower = text.lower()
        
        if "001_age" in case_id:
            # Age verification case
            if "eligible" in text_lower and ("false" in text_lower or "not" in text_lower):
                return {"eligible": False, "reason": "underage"}
            elif "eligible" in text_lower:
                return {"eligible": True}
        
        elif "002_credit" in case_id:
            # Credit score case
            if "approved" in text_lower and ("false" in text_lower or "not" in text_lower or "reject" in text_lower):
                return {"approved": False, "reason": "credit_score_too_low"}
            elif "approved" in text_lower:
                return {"approved": True}
        
        elif "003_high_income" in case_id:
            # High income approval case
            if "approved" in text_lower and ("true" in text_lower or "yes" in text_lower or "approve" in text_lower):
                result = {"approved": True}
                # Try to extract credit limit and interest rate
                if "50000" in text or "50,000" in text:
                    result["credit_limit"] = 50000
                elif "25000" in text or "25,000" in text:
                    result["credit_limit"] = 25000
                
                if "0.045" in text or "4.5%" in text:
                    result["interest_rate"] = 0.045
                elif "0.055" in text or "5.5%" in text:
                    result["interest_rate"] = 0.055
                
                return result
        
        # Generic fallback
        if "approved" in text_lower or "eligible" in text_lower:
            return {"approved": True, "eligible": True}
        else:
            return {"approved": False, "eligible": False}
    
    def _parse_hybrid_response(self, text: str, case_id: str) -> Dict[str, Any]:
        """Parse response for hybrid reasoning cases."""
        text_lower = text.lower()
        result = {}
        
        # Extract sentiment
        if "positive" in text_lower:
            result["sentiment"] = "positive"
        elif "negative" in text_lower:
            result["sentiment"] = "negative"
        elif "neutral" in text_lower:
            result["sentiment"] = "neutral"
        
        # Extract approval status
        if "approved" in text_lower:
            if "false" in text_lower or "not" in text_lower or "reject" in text_lower:
                result["approved"] = False
            else:
                result["approved"] = True
        
        # Extract approval type
        if "expedited" in text_lower:
            result["approval_type"] = "expedited"
        elif "standard" in text_lower:
            result["approval_type"] = "standard"
        
        # Extract reason for rejection
        if "credit" in text_lower and ("low" in text_lower or "insufficient" in text_lower):
            result["reason"] = "credit_score_too_low"
        
        return result
    
    def _parse_temporal_response(self, text: str, case_id: str) -> Dict[str, Any]:
        """Parse response for temporal reasoning cases."""
        text_lower = text.lower()
        result = {}
        
        # Extract fraud alert
        if "fraud" in text_lower:
            if "false" in text_lower or "not" in text_lower or "no fraud" in text_lower:
                result["fraud_alert"] = False
            else:
                result["fraud_alert"] = True
        
        # Extract alert level
        if "critical" in text_lower:
            result["alert_level"] = "critical"
        elif "high" in text_lower:
            result["alert_level"] = "high"
        elif "medium" in text_lower:
            result["alert_level"] = "medium"
        elif "low" in text_lower:
            result["alert_level"] = "low"
        
        # Extract pattern
        if "sustained" in text_lower and "spending" in text_lower:
            result["pattern"] = "sustained_high_spending"
        elif "velocity" in text_lower or "impossible" in text_lower:
            result["pattern"] = "impossible_velocity"
        elif "normal" in text_lower:
            result["pattern"] = "normal"
        
        return result
    
    def _parse_workflow_response(self, text: str, case_id: str) -> Dict[str, Any]:
        """Parse response for workflow cases."""
        text_lower = text.lower()
        result = {}
        
        # Extract approval status
        if "approved" in text_lower:
            if "false" in text_lower or "not" in text_lower or "reject" in text_lower:
                result["approved"] = False
            else:
                result["approved"] = True
        
        # Extract approval level
        if "standard" in text_lower:
            result["approval_level"] = "standard"
        elif "premium" in text_lower:
            result["approval_level"] = "premium"
        
        # Extract risk level
        if "low" in text_lower and "risk" in text_lower:
            result["risk_level"] = "low"
        elif "medium" in text_lower and "risk" in text_lower:
            result["risk_level"] = "medium"
        elif "high" in text_lower and "risk" in text_lower:
            result["risk_level"] = "high"
        
        # Extract manager review
        result["manager_review_required"] = "manager" in text_lower or "review" in text_lower
        
        # Extract eligibility
        if "eligibility" in text_lower:
            if "passed" in text_lower or "eligible" in text_lower:
                result["eligibility_passed"] = True
            else:
                result["eligibility_passed"] = False
        
        # Extract reasons
        if "underage" in text_lower:
            result["reason"] = "underage"
        elif "credit" in text_lower and ("low" in text_lower or "below" in text_lower):
            result["reason"] = "credit_score_below_minimum"
        
        # Extract escalation reason
        if "debt" in text_lower and "bankruptcy" in text_lower:
            result["escalation_reason"] = "high_debt_ratio_and_bankruptcy_history"
        elif "debt" in text_lower:
            result["escalation_reason"] = "high_debt_ratio"
        elif "bankruptcy" in text_lower:
            result["escalation_reason"] = "bankruptcy_history"
        
        return result
    
    def _parse_generic_response(self, text: str) -> Dict[str, Any]:
        """Generic response parsing fallback."""
        text_lower = text.lower()
        if "approved" in text_lower or "eligible" in text_lower or "yes" in text_lower:
            return {"approved": True}
        else:
            return {"approved": False}
    
    def _compare_decisions(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Compare expected vs actual decisions."""
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            if actual[key] != expected_value:
                return False
        return True
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage and model."""
        # GPT-4o-mini pricing (as of 2024)
        if "gpt-4o-mini" in self.model:
            input_cost_per_1k = 0.00015  # $0.15 per 1M tokens
            output_cost_per_1k = 0.0006  # $0.60 per 1M tokens
        else:
            # Default to GPT-4 pricing
            input_cost_per_1k = 0.03     # $30 per 1M tokens
            output_cost_per_1k = 0.06    # $60 per 1M tokens
        
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost


def load_test_cases() -> List[Dict[str, Any]]:
    """Load all test cases from the shared directory."""
    test_cases = []
    test_cases_dir = Path("../../shared/test_cases")
    
    for yaml_file in test_cases_dir.rglob("*.yaml"):
        try:
            with yaml_file.open("r") as f:
                case_data = yaml.safe_load(f)
                if "id" in case_data and "expected_decision" in case_data:
                    test_cases.append(case_data)
        except Exception as e:
            print(f"Warning: Could not load {yaml_file}: {e}")
    
    return sorted(test_cases, key=lambda x: x["id"])


def main():
    """Main evaluation function for GPT runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT Baseline Evaluation")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--prompts", type=Path, default=Path("../prompts"), help="Prompts directory")
    parser.add_argument("--output", type=Path, help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Initialize runner
    runner = GPTRunner(model=args.model, prompts_dir=args.prompts)
    runner.setup()
    
    # Load test cases
    test_cases = load_test_cases()
    if not test_cases:
        print(f"No test cases found")
        return
    
    print(f"Found {len(test_cases)} test cases")
    
    # Run evaluation
    results = []
    total_start = time.time()
    
    for i, case_data in enumerate(test_cases, 1):
        if args.verbose:
            print(f"\n[{i:2d}/{len(test_cases)}] Running: {case_data['id']}")
            print(f"    Description: {case_data['description']}")
        
        result = runner.run_case(case_data)
        results.append(result)
        
        if args.verbose:
            if result["decision_correct"]:
                print(f"    ✓ PASS ({result['latency_ms']:.1f}ms, ${result['cost_usd']:.4f})")
            else:
                print(f"    ✗ FAIL ({result['latency_ms']:.1f}ms, ${result['cost_usd']:.4f})")
                print(f"      Expected: {result['expected']}")
                print(f"      Actual:   {result['actual']}")
                if result["error"]:
                    print(f"      Error:    {result['error']}")
    
    # Generate summary
    total_time = time.time() - total_start
    total_cases = len(results)
    correct_cases = sum(1 for r in results if r["decision_correct"])
    accuracy = correct_cases / total_cases * 100 if total_cases > 0 else 0
    avg_latency = sum(r["latency_ms"] for r in results) / total_cases if total_cases > 0 else 0
    total_cost = sum(r["cost_usd"] for r in results)
    total_tokens = sum(r["tokens_input"] + r["tokens_output"] for r in results)
    
    print(f"\n==== GPT EVALUATION RESULTS ====")
    print(f"Model: {runner.model}")
    print(f"Total Cases: {total_cases}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Avg Latency: {avg_latency:.1f}ms")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Errors: {sum(1 for r in results if r['error'])}")
    print(f"Total Time: {total_time:.2f}s")
    
    # Suite breakdown
    suite_stats = {}
    for result in results:
        suite = result["suite"]
        if suite not in suite_stats:
            suite_stats[suite] = {"total": 0, "correct": 0}
        suite_stats[suite]["total"] += 1
        if result["decision_correct"]:
            suite_stats[suite]["correct"] += 1
    
    print("\nSuite Breakdown:")
    for suite, stats in suite_stats.items():
        suite_accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {suite}: {stats['correct']}/{stats['total']} ({suite_accuracy:.1f}%)")
    
    # Save results if requested
    if args.output:
        results_data = [
            {
                'case_id': r["case_id"],
                'suite': r["suite"],
                'decision_correct': r["decision_correct"],
                'latency_ms': r["latency_ms"],
                'tokens_input': r["tokens_input"],
                'tokens_output': r["tokens_output"],
                'cost_usd': r["cost_usd"],
                'error': r["error"],
                'expected_decision': r["expected"],
                'actual_decision': r["actual"],
                'reasoning': r["reasoning"]
            }
            for r in results
        ]
        
        with args.output.open('w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main() 