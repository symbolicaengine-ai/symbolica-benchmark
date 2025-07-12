# Symbolica Benchmark Suite

A comprehensive evaluation framework comparing Symbolica's hybrid rule engine against pure LLM reasoning across multiple dimensions.

## Directory Structure

```
benchmark/
├── README.md                    # This file
├── BENCHMARK_PLAN.md           # Detailed project plan and specifications
├── test_cases/                 # Test case definitions
│   ├── s1_symbolic/           # Pure symbolic reasoning cases
│   ├── s2_hybrid/             # Hybrid AI-rule cases with PROMPT()
│   ├── s3_temporal/           # Temporal pattern detection cases
│   └── s4_workflow/           # Multi-step workflow cases
├── rules/                      # Rule definitions
│   ├── s1_symbolic_rules.yaml # Pure symbolic rules
│   ├── s2_hybrid_rules.yaml   # Hybrid rules with LLM integration
│   ├── s3_temporal_rules.yaml # Temporal pattern rules
│   └── s4_workflow_rules.yaml # Multi-step workflow rules
├── harness/                    # Evaluation harness
│   ├── run_eval.py            # Main evaluation script
│   ├── __init__.py            # Package initialization
│   └── baselines/             # Baseline implementations
└── results/                    # Evaluation results
    ├── *.csv                  # Result files
    └── archive/               # Historical results
```

## Quick Start

### Prerequisites

1. **Python 3.8+** with required packages:
   ```bash
   pip install symbolica openai pyyaml python-dotenv
   ```

2. **OpenAI API Key** (for S2 hybrid cases):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   # or create a .env file in the project root
   ```

### Running the Benchmark

```bash
cd harness/
python run_eval.py --verbose
```

### Command Line Options

```bash
python run_eval.py --help

Options:
  --benchmark-dir PATH    Directory containing test cases (default: ../test_cases)
  --rules PATH           Rules directory (default: ../rules/)
  --runner {symbolica,gpt}  Which runner to use (default: symbolica)
  --output PATH          Output CSV file for detailed results
  --verbose              Show detailed progress
```

### Example Usage

```bash
# Run all tests with Symbolica
python run_eval.py --verbose

# Run with GPT baseline
python run_eval.py --runner gpt --output gpt_results.csv

# Run specific test suite
python run_eval.py --benchmark-dir ../test_cases/s2_hybrid --verbose
```

## Test Suites

### S1: Pure Symbolic Reasoning (30%)
- **Focus**: Deterministic logic (age checks, credit thresholds, regex patterns)
- **Examples**: Loan eligibility, age verification, credit score filtering
- **Strengths**: Perfect accuracy, sub-millisecond latency, zero cost

### S2: Hybrid AI-Rule Integration (30%)
- **Focus**: Combining deterministic rules with LLM reasoning
- **Examples**: Sentiment analysis + eligibility checks, risk assessment with NLP
- **Strengths**: Structured reasoning with AI flexibility

### S3: Temporal Pattern Detection (25%)
- **Focus**: Time-based patterns and sustained conditions
- **Examples**: Fraud detection, spending pattern analysis, behavioral monitoring
- **Strengths**: Complex temporal logic with rule chaining

### S4: Multi-Step Workflows (15%)
- **Focus**: Complex approval pipelines and backward chaining
- **Examples**: Loan approval workflows, risk assessment pipelines
- **Strengths**: Orchestrated decision-making with multiple checkpoints

## Performance Metrics

The benchmark evaluates systems across four key dimensions:

1. **Accuracy**: Percentage of correct decisions
2. **Latency**: Average response time in milliseconds
3. **Cost**: Token usage and API costs (for LLM-based systems)
4. **Explainability**: Quality and presence of reasoning traces

## Recent Results

```
==== SYMBOLICA Benchmark Results ====
Total Cases: 5
Accuracy: 100.0%
Avg Latency: 158.0ms
Total Cost: $0.0007
Errors: 0

Suite Breakdown:
  s1_symbolic: 2/2 (100.0%)
  s2_hybrid: 1/1 (100.0%)
  s3_temporal: 1/1 (100.0%)
  s4_workflow: 1/1 (100.0%)
```

## Adding New Test Cases

### Test Case Format

Each test case is a YAML file with the following structure:

```yaml
id: "unique_test_id"
description: "Human-readable description"
category: "test_category"
scenario: |
  Detailed scenario description
facts:
  field1: value1
  field2: value2
expected_decision:
  approved: true
  reason: "meets_criteria"
ground_truth: |
  Explanation of why this decision is correct
difficulty: "simple|medium|complex"
```

### Adding a New Case

1. Create a new YAML file in the appropriate `test_cases/` subdirectory
2. Follow the naming convention: `{suite_id}_{case_number}_{description}.yaml`
3. Ensure the `expected_decision` matches what the rules should produce
4. Test with: `python run_eval.py --benchmark-dir ../test_cases/your_suite --verbose`

## Rule Development

### Rule Format

Rules are defined in YAML files with the following structure:

```yaml
rules:
  - id: "unique_rule_id"
    priority: 100  # Higher priority = evaluated first
    condition: "age >= 18 and credit_score >= 650"
    actions:
      approved: true
      reason: "meets_criteria"
    tags: ["category", "type"]
```

### Best Practices

1. **Null Safety**: Always check for field existence: `field is not None and field > threshold`
2. **Priority Ordering**: Higher priority rules should be more specific
3. **Clear Conditions**: Use readable expressions that match business logic
4. **Comprehensive Actions**: Set all expected output fields

## Contributing

1. **Test Cases**: Add realistic scenarios that test edge cases
2. **Rules**: Ensure rules handle all expected input combinations
3. **Baselines**: Implement additional baseline systems for comparison
4. **Documentation**: Update this README when adding new features

## License

This benchmark suite is released under CC-BY-4.0 license for academic and research use.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{symbolica-benchmark-2024,
  title={Symbolica Benchmark: Evaluating Hybrid AI-Rule Systems},
  author={Symbolica Team},
  year={2024},
  url={https://github.com/symbolica/benchmark}
}
``` 