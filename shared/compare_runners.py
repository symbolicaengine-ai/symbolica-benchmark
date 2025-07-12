#!/usr/bin/env python3
"""
Symbolica vs GPT Benchmark Comparison
=====================================

Comprehensive comparison script that runs both Symbolica and GPT evaluations
and provides detailed analysis of their performance differences.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys

# Add runners to path
sys.path.append(str(Path(__file__).parent.parent / "symbolica" / "runners"))
sys.path.append(str(Path(__file__).parent.parent / "gpt" / "runners"))

from symbolica_runner import SymbolicaRunner, discover_test_cases as discover_symbolica_cases
from gpt_runner import GPTRunner, discover_test_cases as discover_gpt_cases


def run_symbolica_evaluation(rules_path: Path, test_cases_path: Path, verbose: bool = False) -> Tuple[List[Any], Dict[str, Any]]:
    """Run Symbolica evaluation and return results."""
    print("🔧 Running Symbolica Evaluation...")
    
    # Initialize runner
    runner = SymbolicaRunner(rules_path)
    runner.setup()
    
    # Load test cases
    test_cases = discover_symbolica_cases(test_cases_path)
    if not test_cases:
        raise ValueError(f"No test cases found in {test_cases_path}")
    
    # Run evaluation
    results = []
    start_time = time.time()
    
    for i, case_data in enumerate(test_cases, 1):
        if verbose:
            print(f"  Running case {i}/{len(test_cases)}: {case_data['id']}")
        
        result = runner.run_case(case_data)
        results.append(result)
        
        if verbose:
            status = "✓" if result.decision_correct else "✗"
            print(f"    {status} {result.case_id} ({result.latency_ms:.1f}ms)")
    
    total_time = time.time() - start_time
    
    # Generate summary
    summary = {
        'runner_type': 'symbolica',
        'total_cases': len(results),
        'correct_cases': sum(1 for r in results if r.decision_correct),
        'accuracy': sum(1 for r in results if r.decision_correct) / len(results) * 100,
        'avg_latency_ms': sum(r.latency_ms for r in results) / len(results),
        'total_cost_usd': sum(r.cost_usd for r in results),
        'total_tokens': sum(r.tokens_input + r.tokens_output for r in results),
        'errors': sum(1 for r in results if r.error),
        'total_time_seconds': total_time
    }
    
    return results, summary


def run_gpt_evaluation(model: str, prompts_path: Path, test_cases_path: Path, verbose: bool = False) -> Tuple[List[Any], Dict[str, Any]]:
    """Run GPT evaluation and return results."""
    print(f"🤖 Running GPT Evaluation (model: {model})...")
    
    # Initialize runner
    runner = GPTRunner(model=model, prompts_dir=prompts_path)
    runner.setup()
    
    # Load test cases
    test_cases = discover_gpt_cases(test_cases_path)
    if not test_cases:
        raise ValueError(f"No test cases found in {test_cases_path}")
    
    # Run evaluation
    results = []
    start_time = time.time()
    
    for i, case_data in enumerate(test_cases, 1):
        if verbose:
            print(f"  Running case {i}/{len(test_cases)}: {case_data['id']}")
        
        result = runner.run_case(case_data)
        results.append(result)
        
        if verbose:
            status = "✓" if result.decision_correct else "✗"
            print(f"    {status} {result.case_id} ({result.latency_ms:.1f}ms, ${result.cost_usd:.4f})")
    
    total_time = time.time() - start_time
    
    # Generate summary
    summary = {
        'runner_type': 'gpt',
        'model': model,
        'total_cases': len(results),
        'correct_cases': sum(1 for r in results if r.decision_correct),
        'accuracy': sum(1 for r in results if r.decision_correct) / len(results) * 100,
        'avg_latency_ms': sum(r.latency_ms for r in results) / len(results),
        'total_cost_usd': sum(r.cost_usd for r in results),
        'total_tokens': sum(r.tokens_input + r.tokens_output for r in results),
        'errors': sum(1 for r in results if r.error),
        'total_time_seconds': total_time
    }
    
    return results, summary


def analyze_suite_performance(symbolica_results: List[Any], gpt_results: List[Any]) -> Dict[str, Dict[str, Any]]:
    """Analyze performance by test suite."""
    suite_analysis = {}
    
    # Group results by suite
    symbolica_by_suite = {}
    gpt_by_suite = {}
    
    for result in symbolica_results:
        suite = result.suite
        if suite not in symbolica_by_suite:
            symbolica_by_suite[suite] = []
        symbolica_by_suite[suite].append(result)
    
    for result in gpt_results:
        suite = result.suite
        if suite not in gpt_by_suite:
            gpt_by_suite[suite] = []
        gpt_by_suite[suite].append(result)
    
    # Analyze each suite
    all_suites = set(symbolica_by_suite.keys()) | set(gpt_by_suite.keys())
    
    for suite in all_suites:
        sym_results = symbolica_by_suite.get(suite, [])
        gpt_results_suite = gpt_by_suite.get(suite, [])
        
        sym_accuracy = sum(1 for r in sym_results if r.decision_correct) / len(sym_results) * 100 if sym_results else 0
        gpt_accuracy = sum(1 for r in gpt_results_suite if r.decision_correct) / len(gpt_results_suite) * 100 if gpt_results_suite else 0
        
        sym_latency = sum(r.latency_ms for r in sym_results) / len(sym_results) if sym_results else 0
        gpt_latency = sum(r.latency_ms for r in gpt_results_suite) / len(gpt_results_suite) if gpt_results_suite else 0
        
        sym_cost = sum(r.cost_usd for r in sym_results) if sym_results else 0
        gpt_cost = sum(r.cost_usd for r in gpt_results_suite) if gpt_results_suite else 0
        
        suite_analysis[suite] = {
            'symbolica': {
                'cases': len(sym_results),
                'accuracy': sym_accuracy,
                'avg_latency_ms': sym_latency,
                'total_cost_usd': sym_cost
            },
            'gpt': {
                'cases': len(gpt_results_suite),
                'accuracy': gpt_accuracy,
                'avg_latency_ms': gpt_latency,
                'total_cost_usd': gpt_cost
            },
            'comparison': {
                'accuracy_advantage_symbolica': sym_accuracy - gpt_accuracy,
                'latency_advantage_symbolica': gpt_latency - sym_latency,  # Positive means Symbolica is faster
                'cost_advantage_symbolica': gpt_cost - sym_cost  # Positive means Symbolica is cheaper
            }
        }
    
    return suite_analysis


def print_comparison_report(symbolica_summary: Dict[str, Any], gpt_summary: Dict[str, Any], suite_analysis: Dict[str, Dict[str, Any]]):
    """Print comprehensive comparison report."""
    print("\n" + "="*80)
    print("SYMBOLICA vs GPT BENCHMARK COMPARISON")
    print("="*80)
    
    # Overall comparison
    print("\n📊 OVERALL PERFORMANCE")
    print("-" * 40)
    print(f"{'Metric':<20} {'Symbolica':<15} {'GPT':<15} {'Advantage':<15}")
    print("-" * 40)
    
    accuracy_diff = symbolica_summary['accuracy'] - gpt_summary['accuracy']
    latency_diff = gpt_summary['avg_latency_ms'] - symbolica_summary['avg_latency_ms']
    cost_diff = gpt_summary['total_cost_usd'] - symbolica_summary['total_cost_usd']
    
    print(f"{'Accuracy':<20} {symbolica_summary['accuracy']:<14.1f}% {gpt_summary['accuracy']:<14.1f}% {accuracy_diff:+.1f}%")
    print(f"{'Avg Latency':<20} {symbolica_summary['avg_latency_ms']:<14.1f}ms {gpt_summary['avg_latency_ms']:<14.1f}ms {latency_diff:+.1f}ms")
    print(f"{'Total Cost':<20} ${symbolica_summary['total_cost_usd']:<13.4f} ${gpt_summary['total_cost_usd']:<13.4f} ${cost_diff:+.4f}")
    print(f"{'Total Tokens':<20} {symbolica_summary['total_tokens']:<14} {gpt_summary['total_tokens']:<14} {gpt_summary['total_tokens'] - symbolica_summary['total_tokens']:+}")
    print(f"{'Errors':<20} {symbolica_summary['errors']:<14} {gpt_summary['errors']:<14} {gpt_summary['errors'] - symbolica_summary['errors']:+}")
    
    # Suite-by-suite analysis
    print("\n🔍 SUITE-BY-SUITE ANALYSIS")
    print("-" * 80)
    
    for suite, analysis in suite_analysis.items():
        print(f"\n{suite.upper().replace('_', ' ')}")
        print("-" * 30)
        
        sym = analysis['symbolica']
        gpt = analysis['gpt']
        comp = analysis['comparison']
        
        print(f"  Accuracy:     Symbolica {sym['accuracy']:.1f}% vs GPT {gpt['accuracy']:.1f}% ({comp['accuracy_advantage_symbolica']:+.1f}%)")
        print(f"  Latency:      Symbolica {sym['avg_latency_ms']:.1f}ms vs GPT {gpt['avg_latency_ms']:.1f}ms ({comp['latency_advantage_symbolica']:+.1f}ms)")
        print(f"  Cost:         Symbolica ${sym['total_cost_usd']:.4f} vs GPT ${gpt['total_cost_usd']:.4f} (${comp['cost_advantage_symbolica']:+.4f})")
    
    # Key insights
    print("\n💡 KEY INSIGHTS")
    print("-" * 40)
    
    if accuracy_diff > 0:
        print(f"✓ Symbolica achieves {accuracy_diff:.1f}% higher accuracy")
    else:
        print(f"⚠ GPT achieves {-accuracy_diff:.1f}% higher accuracy")
    
    if latency_diff > 0:
        print(f"✓ Symbolica is {latency_diff:.1f}ms faster on average")
    else:
        print(f"⚠ GPT is {-latency_diff:.1f}ms faster on average")
    
    if cost_diff > 0:
        print(f"✓ Symbolica saves ${cost_diff:.4f} in total costs")
    else:
        print(f"⚠ GPT saves ${-cost_diff:.4f} in total costs")
    
    # Recommendations
    print("\n🎯 RECOMMENDATIONS")
    print("-" * 40)
    
    best_suites = [suite for suite, analysis in suite_analysis.items() 
                   if analysis['comparison']['accuracy_advantage_symbolica'] > 0]
    
    if best_suites:
        print(f"✓ Use Symbolica for: {', '.join(best_suites)}")
    
    worst_suites = [suite for suite, analysis in suite_analysis.items() 
                    if analysis['comparison']['accuracy_advantage_symbolica'] < 0]
    
    if worst_suites:
        print(f"⚠ Consider GPT for: {', '.join(worst_suites)}")
    
    print("\n" + "="*80)


def save_results(symbolica_results: List[Any], gpt_results: List[Any], 
                symbolica_summary: Dict[str, Any], gpt_summary: Dict[str, Any],
                suite_analysis: Dict[str, Dict[str, Any]], output_path: Path):
    """Save comprehensive results to JSON file."""
    results_data = {
        'timestamp': time.time(),
        'symbolica': {
            'summary': symbolica_summary,
            'results': [
                {
                    'case_id': r.case_id,
                    'suite': r.suite,
                    'decision_correct': r.decision_correct,
                    'latency_ms': r.latency_ms,
                    'cost_usd': r.cost_usd,
                    'has_reasoning': r.has_reasoning,
                    'error': r.error,
                    'expected_decision': r.decision_expected,
                    'actual_decision': r.decision_actual
                }
                for r in symbolica_results
            ]
        },
        'gpt': {
            'summary': gpt_summary,
            'results': [
                {
                    'case_id': r.case_id,
                    'suite': r.suite,
                    'decision_correct': r.decision_correct,
                    'latency_ms': r.latency_ms,
                    'tokens_input': r.tokens_input,
                    'tokens_output': r.tokens_output,
                    'cost_usd': r.cost_usd,
                    'has_reasoning': r.has_reasoning,
                    'error': r.error,
                    'expected_decision': r.decision_expected,
                    'actual_decision': r.decision_actual
                }
                for r in gpt_results
            ]
        },
        'suite_analysis': suite_analysis
    }
    
    with output_path.open('w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to {output_path}")


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Symbolica vs GPT Benchmark Comparison")
    parser.add_argument("--symbolica-rules", type=Path, default=Path("../symbolica/rules"),
                       help="Symbolica rules directory")
    parser.add_argument("--gpt-prompts", type=Path, default=Path("../gpt/prompts"),
                       help="GPT prompts directory")
    parser.add_argument("--test-cases", type=Path, default=Path("test_cases"),
                       help="Test cases directory")
    parser.add_argument("--gpt-model", default="gpt-4o-mini",
                       help="GPT model to use")
    parser.add_argument("--output", type=Path, help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        # Run evaluations
        symbolica_results, symbolica_summary = run_symbolica_evaluation(
            args.symbolica_rules, args.test_cases, args.verbose
        )
        
        gpt_results, gpt_summary = run_gpt_evaluation(
            args.gpt_model, args.gpt_prompts, args.test_cases, args.verbose
        )
        
        # Analyze results
        suite_analysis = analyze_suite_performance(symbolica_results, gpt_results)
        
        # Print comparison report
        print_comparison_report(symbolica_summary, gpt_summary, suite_analysis)
        
        # Save results if requested
        if args.output:
            save_results(symbolica_results, gpt_results, symbolica_summary, 
                        gpt_summary, suite_analysis, args.output)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 