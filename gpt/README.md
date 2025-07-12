# GPT Baseline Runner

Sophisticated GPT-based runner for the Symbolica benchmark with advanced prompt engineering.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Quick Test
```bash
cd symbolica-benchmark
python run_gpt_benchmark.py
```

### Full Evaluation
```bash
cd gpt/runners
python gpt_runner.py --verbose --output results.json
```

## Features

- **Suite-specific prompts**: Specialized prompts for each test suite
- **Sophisticated parsing**: Robust response parsing with multiple fallback strategies
- **Cost tracking**: Detailed token usage and cost analysis
- **Error handling**: Comprehensive error handling and reporting
- **Template system**: YAML-based prompt templates for easy customization

## Prompt Templates

- `generic_prompt.yaml`: General-purpose banking decisions
- `s1_symbolic_prompt.yaml`: Pure symbolic/deterministic reasoning
- `s2_hybrid_prompt.yaml`: Sentiment analysis + eligibility rules
- `s3_temporal_prompt.yaml`: Fraud detection and temporal patterns
- `s4_workflow_prompt.yaml`: Multi-step workflow decisions

## Test Suites

- **S1 Symbolic**: Age verification, credit score thresholds, income-based approvals
- **S2 Hybrid**: Sentiment analysis combined with eligibility rules
- **S3 Temporal**: Fraud detection based on spending patterns and geographic velocity
- **S4 Workflow**: Multi-step approval workflows with risk assessment

## Results

The runner provides detailed metrics including:
- Accuracy per suite and overall
- Latency measurements
- Token usage and costs
- Error analysis
- Decision comparison (expected vs actual) 