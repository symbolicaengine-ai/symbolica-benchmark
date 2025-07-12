#!/usr/bin/env python3
"""
GPT Baseline Runner - Multi-Record Support
==========================================

Enhanced runner for evaluating GPT performance consistency across multiple similar records,
comparing against Symbolica's deterministic reliability.
"""

import time
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml


class GPTMultiRecordRunner:
    """GPT runner with multi-record support for consistency testing."""
    
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
            print(f"✓ GPT multi-record runner initialized with model: {self.model}")
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
        """Execute a single test case with multiple records through GPT."""
        start_time = time.perf_counter()
        
        try:
            # Extract case information
            case_id = case_data["id"]
            suite = self._extract_suite(case_data)
            expected_decision = case_data["expected_decision"]
            customer_records = case_data["customer_records"]
            
            # Process each record
            record_results = []
            total_tokens_input = 0
            total_tokens_output = 0
            total_cost = 0.0
            
            for i, record in enumerate(customer_records):
                record_start_time = time.perf_counter()
                
                # Build prompt for this record
                prompt = self._build_prompt_for_record(case_data, record, suite)
                
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,  # Deterministic
                    max_tokens=1000
                )
                
                # Calculate record metrics
                record_latency_ms = (time.perf_counter() - record_start_time) * 1000
                
                # Extract response
                response_text = response.choices[0].message.content
                decision_actual = self._parse_response(response_text, suite, case_id)
                
                # Extract usage stats
                usage = response.usage
                tokens_input = usage.prompt_tokens
                tokens_output = usage.completion_tokens
                cost_usd = self._calculate_cost(tokens_input, tokens_output)
                
                # Update totals
                total_tokens_input += tokens_input
                total_tokens_output += tokens_output
                total_cost += cost_usd
                
                # Check correctness
                decision_correct = self._compare_decisions(expected_decision, decision_actual)
                
                record_results.append({
                    "record_index": i + 1,
                    "decision_correct": decision_correct,
                    "expected": expected_decision,
                    "actual": decision_actual,
                    "latency_ms": record_latency_ms,
                    "tokens_input": tokens_input,
                    "tokens_output": tokens_output,
                    "cost_usd": cost_usd,
                    "reasoning": response_text
                })
            
            # Calculate overall metrics
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            consistency_metrics = self._calculate_consistency_metrics(record_results)
            
            # Update global stats
            self.stats['total_calls'] += len(customer_records)
            self.stats['total_tokens'] += total_tokens_input + total_tokens_output
            self.stats['total_cost'] += total_cost
            
            return {
                "case_id": case_id,
                "suite": suite,
                "total_records": len(customer_records),
                "record_results": record_results,
                "consistency_metrics": consistency_metrics,
                "total_latency_ms": total_latency_ms,
                "avg_latency_ms": total_latency_ms / len(customer_records),
                "total_tokens_input": total_tokens_input,
                "total_tokens_output": total_tokens_output,
                "total_cost_usd": total_cost,
                "avg_cost_per_record": total_cost / len(customer_records),
                "error": None
            }
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return {
                "case_id": case_data.get("id", "unknown"),
                "suite": self._extract_suite(case_data),
                "total_records": len(case_data.get("customer_records", [])),
                "record_results": [],
                "consistency_metrics": {},
                "total_latency_ms": latency_ms,
                "avg_latency_ms": 0,
                "total_tokens_input": 0,
                "total_tokens_output": 0,
                "total_cost_usd": 0.0,
                "avg_cost_per_record": 0.0,
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
        elif "ambiguous" in case_id or "sentiment_analysis" in case_id or "risk_assessment" in case_id:
            return "generic"
        else:
            return "generic"
    
    def _build_prompt_for_record(self, case_data: Dict[str, Any], record: Dict[str, Any], suite: str) -> str:
        """Build prompt for a single record using templates."""
        # Try to use suite-specific template
        template_name = f"{suite}_prompt"
        if template_name in self.templates:
            template = self.templates[template_name]
            
            # Format the record data for the template
            customer_data_formatted = self._format_customer_data(record)
            facts_formatted = self._format_facts(record)
            
            return template.format(
                business_problem=case_data.get("business_problem", ""),
                scenario=case_data.get("business_problem", ""),
                description=case_data.get("description", ""),
                facts=record,
                facts_formatted=facts_formatted,
                customer_data=record,
                customer_data_formatted=customer_data_formatted,
                case_id=case_data["id"]
            )
        
        # Fall back to generic template
        if "generic_prompt" in self.templates:
            template = self.templates["generic_prompt"]
            return template.format(
                business_problem=case_data.get("business_problem", ""),
                scenario=case_data.get("business_problem", ""),
                description=case_data.get("description", ""),
                facts=record,
                facts_formatted=self._format_facts(record),
                customer_data=record,
                customer_data_formatted=self._format_customer_data(record),
                case_id=case_data["id"]
            )
        
        # Ultimate fallback: enhanced prompt with business context
        return self._build_enhanced_prompt(case_data, record, suite)
    
    def _format_facts(self, facts: Dict[str, Any]) -> str:
        """Format facts for display in prompts."""
        return "\n".join(f"- {k}: {v}" for k, v in facts.items())
    
    def _format_customer_data(self, customer_data: Dict[str, Any]) -> str:
        """Format customer data for display in prompts."""
        if not customer_data:
            return ""
        
        formatted = []
        for key, value in customer_data.items():
            formatted.append(f"{key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted)
    
    def _build_enhanced_prompt(self, case_data: Dict[str, Any], record: Dict[str, Any], suite: str) -> str:
        """Build enhanced prompt with full business context when no templates are available."""
        business_problem = case_data.get("business_problem", "")
        customer_data_str = self._format_customer_data(record)
        expected_decision = case_data.get("expected_decision", {})
        
        # Get the expected output format from the actual expected decision
        expected_format = json.dumps(expected_decision, indent=2)
        
        return f"""You are an expert banking decision system. Read the complete business problem and make a decision based on the full policy framework.

BUSINESS PROBLEM & POLICY:
{business_problem}

CUSTOMER APPLICATION:
{customer_data_str}

INSTRUCTIONS:
1. Read the complete business policy above carefully
2. Apply ALL the rules, criteria, and requirements to this customer application
3. Consider all tiers, regulatory requirements, and rejection criteria
4. Provide your decision in the EXACT same format as this example:

EXPECTED OUTPUT FORMAT:
{expected_format}

CRITICAL: Your response must be ONLY a valid JSON object matching the format above. No explanations, just JSON."""
    
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
        
        # Try to extract JSON first
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # Fallback parsing
        result = {}
        
        # Extract basic approval/eligibility
        if "approved" in text_lower:
            if "false" in text_lower or "not approved" in text_lower:
                result["approved"] = False
            else:
                result["approved"] = True
        
        if "eligible" in text_lower:
            if "false" in text_lower or "not eligible" in text_lower:
                result["eligible"] = False
            else:
                result["eligible"] = True
        
        # Extract tier information
        if "premium" in text_lower:
            result["tier"] = "premium"
        elif "standard" in text_lower:
            result["tier"] = "standard"
        elif "basic" in text_lower:
            result["tier"] = "basic"
        
        # Extract reason
        if "underage" in text_lower:
            result["reason"] = "underage"
        elif "credit" in text_lower and "low" in text_lower:
            result["reason"] = "credit_score_too_low"
        elif "income" in text_lower and "low" in text_lower:
            result["reason"] = "income_too_low"
        
        return result
    
    def _parse_hybrid_response(self, text: str, case_id: str) -> Dict[str, Any]:
        """Parse response for hybrid reasoning cases."""
        text_lower = text.lower()
        
        # Try to extract JSON first
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
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
            if "false" in text_lower or "not approved" in text_lower:
                result["approved"] = False
            else:
                result["approved"] = True
        
        # Extract approval type
        if "expedited" in text_lower:
            result["approval_type"] = "expedited"
        elif "standard" in text_lower:
            result["approval_type"] = "standard"
        
        return result
    
    def _parse_temporal_response(self, text: str, case_id: str) -> Dict[str, Any]:
        """Parse response for temporal reasoning cases."""
        text_lower = text.lower()
        
        # Try to extract JSON first
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        result = {}
        
        # Extract fraud alert
        if "fraud_alert" in text_lower:
            if "true" in text_lower:
                result["fraud_alert"] = True
            else:
                result["fraud_alert"] = False
        elif "fraud" in text_lower:
            result["fraud_alert"] = True
        else:
            result["fraud_alert"] = False
        
        # Extract alert level
        if "critical" in text_lower:
            result["alert_level"] = "critical"
        elif "high" in text_lower:
            result["alert_level"] = "high"
        elif "medium" in text_lower:
            result["alert_level"] = "medium"
        elif "low" in text_lower:
            result["alert_level"] = "low"
        elif "none" in text_lower:
            result["alert_level"] = "none"
        
        # Extract pattern
        if "sustained" in text_lower and "spending" in text_lower:
            result["pattern"] = "sustained_high_spending"
        elif "velocity" in text_lower or "impossible" in text_lower:
            result["pattern"] = "impossible_velocity"
        elif "escalation" in text_lower:
            result["pattern"] = "gradual_escalation"
        elif "moderate" in text_lower:
            result["pattern"] = "moderate_risk"
        elif "normal" in text_lower:
            result["pattern"] = "normal"
        
        return result
    
    def _parse_workflow_response(self, text: str, case_id: str) -> Dict[str, Any]:
        """Parse response for workflow cases."""
        text_lower = text.lower()
        
        # Try to extract JSON first
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        result = {}
        
        # Extract approval status
        if "approved" in text_lower:
            if "false" in text_lower or "not approved" in text_lower:
                result["approved"] = False
            else:
                result["approved"] = True
        
        # Extract approval level
        if "standard" in text_lower:
            result["approval_level"] = "standard"
        elif "premium" in text_lower:
            result["approval_level"] = "premium"
        
        return result
    
    def _parse_generic_response(self, text: str) -> Dict[str, Any]:
        """Generic response parsing fallback."""
        text_lower = text.lower()
        
        # Try to extract JSON first
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # Basic fallback
        if "approved" in text_lower and ("true" in text_lower or "yes" in text_lower):
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
    
    def _calculate_consistency_metrics(self, record_results: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        
        # Calculate variance in decisions (how many different outputs we got)
        unique_decisions = set(json.dumps(r["actual"], sort_keys=True) for r in record_results)
        decision_variance = len(unique_decisions)
        
        # Calculate latency statistics
        latencies = [r["latency_ms"] for r in record_results]
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        latency_variance = sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)
        
        # Calculate cost statistics
        costs = [r["cost_usd"] for r in record_results]
        avg_cost = sum(costs) / len(costs)
        total_cost = sum(costs)
        
        return {
            "all_correct": all_correct,
            "all_identical": all_identical,
            "correct_count": correct_count,
            "correctness_percentage": correctness_percentage,
            "decision_variance": decision_variance,
            "unique_decisions": len(unique_decisions),
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "latency_variance": latency_variance,
            "avg_cost_usd": avg_cost,
            "total_cost_usd": total_cost,
            "deterministic_performance": all_correct and all_identical,
            "consistency_score": correctness_percentage * (1 - (decision_variance - 1) / len(record_results))
        }
    
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

def main():
    """Main GPT benchmark execution."""
    print("🤖 GPT Multi-Record Benchmark - Consistency Testing")
    print("=" * 60)
    
    # Setup runner
    runner = GPTMultiRecordRunner()
    runner.setup()
    
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
        
        result = runner.run_case(case_data)
        results.append(result)
        total_records += result["total_records"]
        
        # Print result
        metrics = result["consistency_metrics"]
        if metrics.get("deterministic_performance", False):
            print(f"    ✓ DETERMINISTIC ({metrics['correct_count']}/{result['total_records']} correct, {metrics['unique_decisions']} unique, ${metrics['total_cost_usd']:.4f})")
        elif metrics.get("all_correct", False):
            print(f"    ✓ ALL CORRECT ({metrics['correct_count']}/{result['total_records']} correct, {metrics['unique_decisions']} unique, ${metrics['total_cost_usd']:.4f})")
        else:
            print(f"    ✗ INCONSISTENT ({metrics['correct_count']}/{result['total_records']} correct, {metrics['unique_decisions']} unique, ${metrics['total_cost_usd']:.4f})")
            
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
    total_cost = sum(r["consistency_metrics"].get("total_cost_usd", 0) for r in results)
    
    # Calculate variance metrics
    total_unique_decisions = sum(r["consistency_metrics"].get("unique_decisions", 1) for r in results)
    avg_decision_variance = total_unique_decisions / len(results)
    
    print(f"\n" + "=" * 60)
    print(f"🤖 GPT MULTI-RECORD BENCHMARK RESULTS")
    print(f"=" * 60)
    print(f"Total Cases:          {len(results)}")
    print(f"Total Records:        {total_records}")
    print(f"Cases All Correct:    {total_case_correct}/{len(results)} ({case_accuracy:.1f}%)")
    print(f"Records Correct:      {total_record_correct}/{total_records} ({record_accuracy:.1f}%)")
    print(f"Deterministic Cases:  {total_deterministic}/{len(results)} ({deterministic_rate:.1f}%)")
    print(f"Avg Decision Variance: {avg_decision_variance:.1f} unique outputs per case")
    print(f"Avg Latency:          {avg_latency:.1f}ms")
    print(f"Total Cost:           ${total_cost:.4f}")
    print(f"Total Time:           {total_time:.2f}s")
    print(f"Errors:               {sum(1 for r in results if r['error'])}")
    
    # Variance analysis
    high_variance_cases = [r for r in results if r["consistency_metrics"].get("unique_decisions", 1) > 1]
    print(f"\nVariance Analysis:")
    print(f"Cases with multiple outputs: {len(high_variance_cases)}/{len(results)} ({len(high_variance_cases)/len(results)*100:.1f}%)")
    
    if high_variance_cases:
        print(f"High variance cases:")
        for result in high_variance_cases:
            metrics = result["consistency_metrics"]
            print(f"  {result['case_id']}: {metrics['unique_decisions']} different outputs, {metrics['correctness_percentage']:.1f}% correct")
    
    print(f"\nGlobal Stats:")
    print(f"Total API calls: {runner.stats['total_calls']}")
    print(f"Total tokens: {runner.stats['total_tokens']:,}")
    print(f"Total cost: ${runner.stats['total_cost']:.4f}")

if __name__ == "__main__":
    main() 