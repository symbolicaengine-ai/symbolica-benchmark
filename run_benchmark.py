#!/usr/bin/env python3
"""
Symbolica Benchmark Runner - Multi-Record Support
================================================

Enhanced runner to test consistency across multiple records per test case,
demonstrating Symbolica's deterministic reliability vs. LLM variance.
"""

import os
import time
import yaml
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

def load_test_cases() -> List[Dict[str, Any]]:
    """Load all test cases from the shared directory."""
    test_cases = []
    test_cases_dir = Path("shared/test_cases")
    
    for txt_file in test_cases_dir.rglob("*.txt"):
        try:
            case_data = parse_txt_test_case(txt_file)
            if case_data and "id" in case_data and "expected_decision" in case_data:
                test_cases.append(case_data)
        except Exception as e:
            print(f"Warning: Could not load {txt_file}: {e}")
    
    return sorted(test_cases, key=lambda x: x["id"])

def parse_txt_test_case(txt_file: Path) -> Dict[str, Any]:
    """Parse a TXT test case file and extract multiple customer records."""
    with txt_file.open("r") as f:
        content = f.read()
    
    # Extract test case ID
    id_match = re.search(r'TEST CASE: (\w+)', content)
    if not id_match:
        return None
    
    case_id = id_match.group(1)
    
    # Extract description
    desc_match = re.search(r'Description: (.+)', content)
    description = desc_match.group(1) if desc_match else ""
    
    # Extract business problem
    business_problem_match = re.search(r'BUSINESS PROBLEM STATEMENT:\s*-{40}\s*(.+?)\s*CUSTOMER APPLICATION DATA', content, re.DOTALL)
    business_problem = business_problem_match.group(1).strip() if business_problem_match else ""
    
    # Extract multiple customer records
    customer_records = parse_multiple_customer_records(content)
    
    # Extract expected decision
    expected_decision = parse_expected_decision_from_txt(content)
    
    return {
        "id": case_id,
        "description": description,
        "business_problem": business_problem,
        "customer_records": customer_records,
        "expected_decision": expected_decision
    }

def parse_multiple_customer_records(content: str) -> List[Dict[str, Any]]:
    """Parse multiple customer records from the test case content."""
    records = []
    
    # Find the customer data section
    customer_data_match = re.search(r'CUSTOMER APPLICATION DATA \(MULTIPLE RECORDS\):\s*-{40}\s*(.+?)\s*EXPECTED SYSTEM DECISION', content, re.DOTALL)
    if not customer_data_match:
        return records
    
    customer_data_text = customer_data_match.group(1).strip()
    
    # Split by record boundaries
    record_sections = re.split(r'\n\s*Record \d+:', customer_data_text)
    
    for i, section in enumerate(record_sections):
        if i == 0:  # Skip the first empty section
            continue
            
        record_data = {}
        lines = section.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line and not line.startswith('Application'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Convert to appropriate types
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.lower() == 'none':
                    value = None
                elif value.replace('.', '').replace('-', '').isdigit():
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]  # Remove quotes
                
                record_data[key] = value
        
        if record_data:
            records.append(record_data)
    
    return records

def parse_expected_decision_from_txt(content: str) -> Dict[str, Any]:
    """Parse expected decision from the test case content."""
    expected_match = re.search(r'EXPECTED SYSTEM DECISION \(ALL RECORDS\):\s*-{40}\s*(.+?)\s*GROUND TRUTH', content, re.DOTALL)
    if not expected_match:
        return {}
    
    expected_text = expected_match.group(1).strip()
    expected_decision = {}
    
    for line in expected_text.split('\n'):
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Convert to appropriate types
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'none':
                value = None
            elif value.replace('.', '').replace('-', '').isdigit():
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]  # Remove quotes
            
            expected_decision[key] = value
    
    return expected_decision

def run_symbolica_test(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single test case with multiple records through Symbolica."""
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
        rules_file = get_rules_file(case_data["id"])
        
        # Create engine
        if client:
            engine = Engine.from_yaml(rules_file, llm_client=client)
        else:
            engine = Engine.from_yaml(rules_file)
        
        # Execute rules for each record
        record_results = []
        total_latency = 0
        
        for i, record in enumerate(case_data["customer_records"]):
            start_time = time.perf_counter()
            result = engine.reason(facts(**record))
            latency_ms = (time.perf_counter() - start_time) * 1000
            total_latency += latency_ms
            
            # Check correctness
            expected = case_data["expected_decision"]
            actual = result.verdict
            
            decision_correct = compare_decisions(expected, actual)
            
            record_results.append({
                "record_index": i + 1,
                "decision_correct": decision_correct,
                "expected": expected,
                "actual": actual,
                "latency_ms": latency_ms,
                "reasoning": result.reasoning
            })
        
        # Calculate consistency metrics
        consistency_metrics = calculate_consistency_metrics(record_results)
        
        return {
            "case_id": case_data["id"],
            "total_records": len(case_data["customer_records"]),
            "record_results": record_results,
            "consistency_metrics": consistency_metrics,
            "total_latency_ms": total_latency,
            "avg_latency_ms": total_latency / len(case_data["customer_records"]),
            "error": None
        }
        
    except Exception as e:
        return {
            "case_id": case_data.get("id", "unknown"),
            "total_records": len(case_data.get("customer_records", [])),
            "record_results": [],
            "consistency_metrics": {},
            "total_latency_ms": 0,
            "avg_latency_ms": 0,
            "error": str(e)
        }

def get_rules_file(case_id: str) -> str:
    """Determine which rules file to use based on case ID."""
    if "age" in case_id or "credit_score" in case_id or "income" in case_id:
        return "symbolica/rules/s1_symbolic_rules.yaml"
    elif "sentiment" in case_id or "positive" in case_id or "negative" in case_id:
        return "symbolica/rules/s2_hybrid_rules.yaml"
    elif "spending" in case_id or "velocity" in case_id or "fraud" in case_id:
        return "symbolica/rules/s3_temporal_rules.yaml"
    elif "pipeline" in case_id or "workflow" in case_id or "approval" in case_id or "review" in case_id or "rejection" in case_id:
        return "symbolica/rules/s4_workflow_rules.yaml"
    elif "ambiguous" in case_id or "sentiment_analysis" in case_id or "risk_assessment" in case_id:
        return "symbolica/rules/gpt_favorable_rules.yaml"
    else:
        return "symbolica/rules/"

def compare_decisions(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    """Compare expected vs actual decisions."""
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        if actual[key] != expected_value:
            return False
    return True

def calculate_consistency_metrics(record_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate consistency metrics across multiple records."""
    if not record_results:
        return {}
    
    # Check if all decisions are correct
    all_correct = all(r["decision_correct"] for r in record_results)
    
    # Check if all actual decisions are identical
    first_actual = record_results[0]["actual"]
    all_identical = all(r["actual"] == first_actual for r in record_results)
    
    # Calculate correctness percentage
    correct_count = sum(1 for r in record_results if r["decision_correct"])
    correctness_percentage = (correct_count / len(record_results)) * 100
    
    # Calculate latency statistics
    latencies = [r["latency_ms"] for r in record_results]
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    latency_variance = sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)
    
    return {
        "all_correct": all_correct,
        "all_identical": all_identical,
        "correct_count": correct_count,
        "correctness_percentage": correctness_percentage,
        "avg_latency_ms": avg_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "latency_variance": latency_variance,
        "deterministic_performance": all_correct and all_identical
    }

def main():
    """Main benchmark execution."""
    print("🔧 Symbolica Benchmark - Multi-Record Consistency Testing")
    print("=" * 60)
    
    # Load test cases
    test_cases = load_test_cases()
    print(f"Found {len(test_cases)} test cases")
    
    if len(test_cases) == 0:
        print("No test cases found!")
        return
    
    # Run tests
    results = []
    total_start = time.time()
    total_records = 0
    
    for i, case_data in enumerate(test_cases, 1):
        print(f"\n[{i:2d}/{len(test_cases)}] Running: {case_data['id']}")
        print(f"    Description: {case_data['description']}")
        print(f"    Records: {len(case_data['customer_records'])}")
        
        result = run_symbolica_test(case_data)
        results.append(result)
        total_records += result["total_records"]
        
        # Print result
        metrics = result["consistency_metrics"]
        if metrics.get("deterministic_performance", False):
            print(f"    ✓ DETERMINISTIC ({metrics['correct_count']}/{result['total_records']} correct, {metrics['avg_latency_ms']:.1f}ms avg)")
        elif metrics.get("all_correct", False):
            print(f"    ✓ ALL CORRECT ({metrics['correct_count']}/{result['total_records']} correct, {metrics['avg_latency_ms']:.1f}ms avg)")
        else:
            print(f"    ✗ INCONSISTENT ({metrics['correct_count']}/{result['total_records']} correct, {metrics['avg_latency_ms']:.1f}ms avg)")
            
        if result["error"]:
            print(f"      Error: {result['error']}")
    
    # Summary
    total_time = time.time() - total_start
    
    # Calculate overall metrics
    total_case_correct = sum(1 for r in results if r["consistency_metrics"].get("all_correct", False))
    total_deterministic = sum(1 for r in results if r["consistency_metrics"].get("deterministic_performance", False))
    
    total_record_correct = sum(r["consistency_metrics"].get("correct_count", 0) for r in results)
    case_accuracy = total_case_correct / len(results) * 100
    record_accuracy = total_record_correct / total_records * 100
    deterministic_rate = total_deterministic / len(results) * 100
    
    avg_latency = sum(r["consistency_metrics"].get("avg_latency_ms", 0) for r in results) / len(results)
    
    print(f"\n" + "=" * 60)
    print(f"📊 MULTI-RECORD BENCHMARK RESULTS")
    print(f"=" * 60)
    print(f"Total Cases:          {len(results)}")
    print(f"Total Records:        {total_records}")
    print(f"Cases All Correct:    {total_case_correct}/{len(results)} ({case_accuracy:.1f}%)")
    print(f"Records Correct:      {total_record_correct}/{total_records} ({record_accuracy:.1f}%)")
    print(f"Deterministic Cases:  {total_deterministic}/{len(results)} ({deterministic_rate:.1f}%)")
    print(f"Avg Latency:          {avg_latency:.1f}ms")
    print(f"Total Time:           {total_time:.2f}s")
    print(f"Errors:               {sum(1 for r in results if r['error'])}")
    
    # Suite breakdown
    suite_stats = {}
    for result in results:
        suite = get_suite_name(result["case_id"])
        
        if suite not in suite_stats:
            suite_stats[suite] = {
                "total_cases": 0, 
                "correct_cases": 0, 
                "deterministic_cases": 0,
                "total_records": 0,
                "correct_records": 0
            }
        
        suite_stats[suite]["total_cases"] += 1
        suite_stats[suite]["total_records"] += result["total_records"]
        suite_stats[suite]["correct_records"] += result["consistency_metrics"].get("correct_count", 0)
        
        if result["consistency_metrics"].get("all_correct", False):
            suite_stats[suite]["correct_cases"] += 1
        
        if result["consistency_metrics"].get("deterministic_performance", False):
            suite_stats[suite]["deterministic_cases"] += 1
    
    print(f"\nSuite Breakdown:")
    for suite, stats in suite_stats.items():
        case_acc = (stats["correct_cases"] / stats["total_cases"] * 100) if stats["total_cases"] > 0 else 0
        record_acc = (stats["correct_records"] / stats["total_records"] * 100) if stats["total_records"] > 0 else 0
        deterministic = (stats["deterministic_cases"] / stats["total_cases"] * 100) if stats["total_cases"] > 0 else 0
        
        print(f"  {suite}:")
        print(f"    Cases: {stats['correct_cases']}/{stats['total_cases']} ({case_acc:.1f}%)")
        print(f"    Records: {stats['correct_records']}/{stats['total_records']} ({record_acc:.1f}%)")
        print(f"    Deterministic: {stats['deterministic_cases']}/{stats['total_cases']} ({deterministic:.1f}%)")
    
    if deterministic_rate == 100:
        print(f"\n🎉 Perfect deterministic performance! All {len(results)} test cases were completely consistent!")
    else:
        inconsistent_cases = [r["case_id"] for r in results if not r["consistency_metrics"].get("deterministic_performance", False)]
        print(f"\n⚠️  Inconsistent cases: {', '.join(inconsistent_cases)}")

def get_suite_name(case_id: str) -> str:
    """Get suite name from case ID."""
    if "age" in case_id or "credit_score" in case_id or "income" in case_id:
        return "S1_Symbolic"
    elif "sentiment" in case_id or "positive" in case_id or "negative" in case_id:
        return "S2_Hybrid"
    elif "spending" in case_id or "velocity" in case_id or "fraud" in case_id:
        return "S3_Temporal"
    elif "pipeline" in case_id or "workflow" in case_id or "approval" in case_id or "review" in case_id or "rejection" in case_id:
        return "S4_Workflow"
    elif "ambiguous" in case_id or "sentiment_analysis" in case_id or "risk_assessment" in case_id:
        return "GPT_Favorable"
    else:
        return "Unknown"

if __name__ == "__main__":
    main() 