#!/usr/bin/env python3
"""
GPT Benchmark Runner
===================

Simple runner to test the GPT baseline with the 12 high-quality benchmark examples.
"""

import os
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any

def load_test_cases() -> List[Dict[str, Any]]:
    """Load all test cases from the shared directory."""
    test_cases = []
    test_cases_dir = Path("shared/test_cases")
    
    for yaml_file in test_cases_dir.rglob("*.yaml"):
        try:
            with yaml_file.open("r") as f:
                case_data = yaml.safe_load(f)
                if "id" in case_data and "expected_decision" in case_data:
                    test_cases.append(case_data)
        except Exception as e:
            print(f"Warning: Could not load {yaml_file}: {e}")
    
    return sorted(test_cases, key=lambda x: x["id"])

def run_gpt_test(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single test case with GPT."""
    try:
        # Import the GPT runner
        import sys
        sys.path.append("gpt/runners")
        from gpt_runner import GPTRunner
        
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        # Create and setup runner
        runner = GPTRunner(model="gpt-4o-mini", prompts_dir=Path("gpt/prompts"))
        runner.setup()
        
        # Run the test case
        result = runner.run_case(case_data)
        
        return {
            "case_id": result["case_id"],
            "decision_correct": result["decision_correct"],
            "expected": result["expected"],
            "actual": result["actual"],
            "latency_ms": result["latency_ms"],
            "cost_usd": result["cost_usd"],
            "tokens_total": result["tokens_input"] + result["tokens_output"],
            "reasoning": result["reasoning"][:200] + "..." if len(result["reasoning"]) > 200 else result["reasoning"],
            "error": result["error"]
        }
        
    except Exception as e:
        return {
            "case_id": case_data.get("id", "unknown"),
            "decision_correct": False,
            "expected": case_data.get("expected_decision", {}),
            "actual": {},
            "latency_ms": 0,
            "cost_usd": 0.0,
            "tokens_total": 0,
            "reasoning": "",
            "error": str(e)
        }

def main():
    """Main benchmark execution."""
    print("🤖 GPT Benchmark - 12 High-Quality Examples")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Load test cases
    test_cases = load_test_cases()
    print(f"Found {len(test_cases)} test cases")
    
    if len(test_cases) == 0:
        print("No test cases found!")
        return
    
    # Run tests
    results = []
    total_start = time.time()
    total_cost = 0.0
    
    for i, case_data in enumerate(test_cases, 1):
        print(f"\n[{i:2d}/{len(test_cases)}] Running: {case_data['id']}")
        print(f"    Description: {case_data['description']}")
        
        result = run_gpt_test(case_data)
        results.append(result)
        total_cost += result["cost_usd"]
        
        # Print result
        if result["decision_correct"]:
            print(f"    ✓ PASS ({result['latency_ms']:.0f}ms, ${result['cost_usd']:.4f}, {result['tokens_total']} tokens)")
        else:
            print(f"    ✗ FAIL ({result['latency_ms']:.0f}ms, ${result['cost_usd']:.4f}, {result['tokens_total']} tokens)")
            print(f"      Expected: {result['expected']}")
            print(f"      Actual:   {result['actual']}")
            if result["error"]:
                print(f"      Error:    {result['error']}")
    
    # Summary
    total_time = time.time() - total_start
    correct_count = sum(1 for r in results if r["decision_correct"])
    accuracy = correct_count / len(results) * 100
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    total_tokens = sum(r["tokens_total"] for r in results)
    
    print(f"\n" + "=" * 50)
    print(f"📊 GPT BENCHMARK RESULTS")
    print(f"=" * 50)
    print(f"Total Cases:     {len(results)}")
    print(f"Correct:         {correct_count}")
    print(f"Accuracy:        {accuracy:.1f}%")
    print(f"Avg Latency:     {avg_latency:.0f}ms")
    print(f"Total Cost:      ${total_cost:.4f}")
    print(f"Total Tokens:    {total_tokens}")
    print(f"Total Time:      {total_time:.2f}s")
    print(f"Errors:          {sum(1 for r in results if r['error'])}")
    
    # Suite breakdown
    suite_stats = {}
    for result in results:
        case_id = result["case_id"]
        if "age" in case_id or "credit_score" in case_id or "income" in case_id:
            suite = "S1_Symbolic"
        elif "sentiment" in case_id or "positive" in case_id or "negative" in case_id:
            suite = "S2_Hybrid"
        elif "spending" in case_id or "velocity" in case_id or "fraud" in case_id:
            suite = "S3_Temporal"
        elif "pipeline" in case_id or "workflow" in case_id or "approval" in case_id or "review" in case_id or "rejection" in case_id:
            suite = "S4_Workflow"
        else:
            suite = "Unknown"
        
        if suite not in suite_stats:
            suite_stats[suite] = {"total": 0, "correct": 0, "cost": 0.0}
        suite_stats[suite]["total"] += 1
        suite_stats[suite]["cost"] += result["cost_usd"]
        if result["decision_correct"]:
            suite_stats[suite]["correct"] += 1
    
    print(f"\nSuite Breakdown:")
    for suite, stats in suite_stats.items():
        suite_accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {suite}: {stats['correct']}/{stats['total']} ({suite_accuracy:.1f}%) - ${stats['cost']:.4f}")
    
    if accuracy == 100:
        print(f"\n🎉 Perfect score! All {len(results)} test cases passed!")
        print(f"💰 Total cost: ${total_cost:.4f}")
    else:
        failed_cases = [r["case_id"] for r in results if not r["decision_correct"]]
        print(f"\n⚠️  Failed cases: {', '.join(failed_cases)}")
        print(f"💰 Total cost: ${total_cost:.4f}")

if __name__ == "__main__":
    main() 