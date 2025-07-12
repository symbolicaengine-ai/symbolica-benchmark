#!/usr/bin/env python3
"""
Symbolica Benchmark Runner
=========================

Simple runner to test the 12 high-quality benchmark examples.
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

def run_symbolica_test(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single test case with Symbolica."""
    try:
        from symbolica import Engine, facts
        
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
            except ImportError:
                print("Warning: openai package not installed")
                client = None
        else:
            print("Warning: OPENAI_API_KEY not found")
            client = None
        
        # Determine which rules to use based on case ID
        case_id = case_data["id"]
        if case_id.startswith("001_") or case_id.startswith("002_") or case_id.startswith("003_"):
            if "age" in case_id or "credit_score" in case_id or "income" in case_id:
                rules_file = "symbolica/rules/s1_symbolic_rules.yaml"
            elif "sentiment" in case_id or "positive" in case_id or "negative" in case_id:
                rules_file = "symbolica/rules/s2_hybrid_rules.yaml"
            elif "spending" in case_id or "velocity" in case_id or "fraud" in case_id:
                rules_file = "symbolica/rules/s3_temporal_rules.yaml"
            elif "pipeline" in case_id or "workflow" in case_id or "approval" in case_id or "review" in case_id or "rejection" in case_id:
                rules_file = "symbolica/rules/s4_workflow_rules.yaml"
            else:
                # Load all rules
                rules_file = "symbolica/rules/"
        else:
            rules_file = "symbolica/rules/"
        
        # Create engine
        if client:
            engine = Engine.from_yaml(rules_file, llm_client=client)
        else:
            engine = Engine.from_yaml(rules_file)
        
        # Execute rules
        start_time = time.perf_counter()
        result = engine.reason(facts(**case_data["facts"]))
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Check correctness
        expected = case_data["expected_decision"]
        actual = result.verdict
        
        decision_correct = True
        for key, expected_value in expected.items():
            if key not in actual:
                decision_correct = False
                break
            if actual[key] != expected_value:
                decision_correct = False
                break
        
        return {
            "case_id": case_data["id"],
            "decision_correct": decision_correct,
            "expected": expected,
            "actual": actual,
            "latency_ms": latency_ms,
            "reasoning": result.reasoning,
            "error": None
        }
        
    except Exception as e:
        return {
            "case_id": case_data.get("id", "unknown"),
            "decision_correct": False,
            "expected": case_data.get("expected_decision", {}),
            "actual": {},
            "latency_ms": 0,
            "reasoning": "",
            "error": str(e)
        }

def main():
    """Main benchmark execution."""
    print("🔧 Symbolica Benchmark - 12 High-Quality Examples")
    print("=" * 50)
    
    # Load test cases
    test_cases = load_test_cases()
    print(f"Found {len(test_cases)} test cases")
    
    if len(test_cases) == 0:
        print("No test cases found!")
        return
    
    # Run tests
    results = []
    total_start = time.time()
    
    for i, case_data in enumerate(test_cases, 1):
        print(f"\n[{i:2d}/{len(test_cases)}] Running: {case_data['id']}")
        print(f"    Description: {case_data['description']}")
        
        result = run_symbolica_test(case_data)
        results.append(result)
        
        # Print result
        if result["decision_correct"]:
            print(f"    ✓ PASS ({result['latency_ms']:.1f}ms)")
        else:
            print(f"    ✗ FAIL ({result['latency_ms']:.1f}ms)")
            print(f"      Expected: {result['expected']}")
            print(f"      Actual:   {result['actual']}")
            if result["error"]:
                print(f"      Error:    {result['error']}")
    
    # Summary
    total_time = time.time() - total_start
    correct_count = sum(1 for r in results if r["decision_correct"])
    accuracy = correct_count / len(results) * 100
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    
    print(f"\n" + "=" * 50)
    print(f"📊 BENCHMARK RESULTS")
    print(f"=" * 50)
    print(f"Total Cases:     {len(results)}")
    print(f"Correct:         {correct_count}")
    print(f"Accuracy:        {accuracy:.1f}%")
    print(f"Avg Latency:     {avg_latency:.1f}ms")
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
            suite_stats[suite] = {"total": 0, "correct": 0}
        suite_stats[suite]["total"] += 1
        if result["decision_correct"]:
            suite_stats[suite]["correct"] += 1
    
    print(f"\nSuite Breakdown:")
    for suite, stats in suite_stats.items():
        suite_accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {suite}: {stats['correct']}/{stats['total']} ({suite_accuracy:.1f}%)")
    
    if accuracy == 100:
        print(f"\n🎉 Perfect score! All {len(results)} test cases passed!")
    else:
        failed_cases = [r["case_id"] for r in results if not r["decision_correct"]]
        print(f"\n⚠️  Failed cases: {', '.join(failed_cases)}")

if __name__ == "__main__":
    main() 