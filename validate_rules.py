#!/usr/bin/env python3
"""
Rule-Test Case Validation Script (TXT Format)
=============================================

Validates that Symbolica rules produce expected outputs for TXT format test cases.
"""

import os
import sys
import re
from pathlib import Path

# Add symbolica to path
sys.path.insert(0, str(Path(__file__).parent / "symbolica"))

def load_test_cases():
    """Load all test cases from TXT files."""
    test_cases = []
    test_cases_dir = Path("shared/test_cases")
    
    for txt_file in test_cases_dir.rglob("*.txt"):
        case_data = parse_txt_file(txt_file)
        if case_data:
            test_cases.append(case_data)
    
    return sorted(test_cases, key=lambda x: x["id"])

def parse_txt_file(txt_file):
    """Parse a TXT test case file and extract the necessary data."""
    with txt_file.open("r") as f:
        content = f.read()
    
    # Extract test case ID
    id_match = re.search(r'TEST CASE: (\w+)', content)
    if not id_match:
        return None
    
    case_id = id_match.group(1)
    
    # Extract customer data section
    customer_data_match = re.search(r'CUSTOMER APPLICATION DATA:\s*-+\s*\n(.*?)\n\nEXPECTED SYSTEM DECISION:', content, re.DOTALL)
    if not customer_data_match:
        return None
    
    customer_data_text = customer_data_match.group(1)
    customer_data = parse_customer_data(customer_data_text)
    
    # Extract expected decision
    expected_match = re.search(r'EXPECTED SYSTEM DECISION:\s*-+\s*\n(.*?)\n\nGROUND TRUTH', content, re.DOTALL)
    if not expected_match:
        return None
    
    expected_text = expected_match.group(1)
    expected_decision = parse_expected_decision(expected_text)
    
    return {
        "id": case_id,
        "customer_data": customer_data,
        "expected_decision": expected_decision
    }

def parse_customer_data(customer_data_text):
    """Parse customer data from TXT format."""
    data = {}
    current_section = None
    
    for line in customer_data_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers
        if line.endswith(':') and not line.startswith('  '):
            current_section = line[:-1].lower().replace(' ', '_')
            if current_section not in data:
                data[current_section] = {}
        elif line.startswith('  ') and current_section:
            # Parse key-value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert to appropriate type
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.lower() == 'none':
                    value = None
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                elif value.startswith('[') and value.endswith(']'):
                    # Handle lists - more robust parsing
                    try:
                        value = eval(value)  # Safe for our controlled data
                    except:
                        # Fallback: parse manually
                        list_content = value[1:-1]  # Remove brackets
                        if list_content.strip():
                            items = [item.strip() for item in list_content.split(',')]
                            value = [int(item) if item.isdigit() else float(item) if item.replace('.', '').isdigit() else item for item in items]
                        else:
                            value = []
                    
                data[current_section][key] = value
        elif ':' in line and not line.startswith('  '):
            # Handle top-level key-value pairs
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            
            # Convert types
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
                
            data[key] = value
    
    return data

def parse_expected_decision(expected_text):
    """Parse expected decision from TXT format."""
    decision = {}
    
    for line in expected_text.split('\n'):
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Convert types
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
                
            decision[key] = value
    
    return decision

def flatten_customer_data(customer_data):
    """Flatten structured customer_data to flat facts dictionary."""
    facts = {}
    
    # Handle personal_info
    if "personal_info" in customer_data:
        for key, value in customer_data["personal_info"].items():
            facts[key] = value
    
    # Handle financial_info
    if "financial_info" in customer_data:
        for key, value in customer_data["financial_info"].items():
            facts[key] = value
    
    # Handle loan_details
    if "loan_details" in customer_data:
        facts["loan_amount"] = customer_data["loan_details"].get("requested_amount")
    
    # Handle transaction_history (for S3 temporal)
    if "transaction_history" in customer_data:
        for key, value in customer_data["transaction_history"].items():
            facts[key] = value
    
    # Handle application_feedback (for S2 hybrid)
    if "application_feedback" in customer_data:
        facts["feedback"] = customer_data["application_feedback"]
    
    # Handle direct fields
    for key, value in customer_data.items():
        if key not in ["personal_info", "financial_info", "loan_details", "transaction_history", "application_feedback"]:
            # Convert string lists to actual lists
            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                try:
                    value = eval(value)  # Safe for our controlled data
                except:
                    # Fallback: parse manually
                    list_content = value[1:-1]  # Remove brackets
                    if list_content.strip():
                        items = [item.strip() for item in list_content.split(',')]
                        value = [int(item) if item.isdigit() else float(item) if item.replace('.', '').isdigit() else item for item in items]
                    else:
                        value = []
            facts[key] = value
    
    return facts

def validate_case(case_data):
    """Validate a single test case against Symbolica rules."""
    try:
        from symbolica import Engine, facts
        
        # Determine which rules to use based on case ID patterns
        case_id = case_data["id"]
        # GPT-favorable cases (new category)
        if any(x in case_id for x in ["ambiguous", "nuanced", "contextual"]):
            rules_file = "symbolica/rules/gpt_favorable_rules.yaml"
        # S4 workflow cases (check first to avoid conflicts)
        elif any(x in case_id for x in ["pipeline", "workflow", "manager", "review", "rejection", "approval_pipeline", "escalation", "conditional"]):
            rules_file = "symbolica/rules/s4_workflow_rules.yaml"
        # S3 temporal cases  
        elif any(x in case_id for x in ["spending", "velocity", "fraud", "geographic", "sustained", "normal", "gradual", "escalation"]):
            rules_file = "symbolica/rules/s3_temporal_rules.yaml"
        # S2 hybrid cases
        elif any(x in case_id for x in ["sentiment", "positive", "negative", "expedited", "ineligible", "neutral"]):
            rules_file = "symbolica/rules/s2_hybrid_rules.yaml"
        # S1 symbolic cases (including edge cases)
        elif any(x in case_id for x in ["age", "credit_score", "income", "verification", "threshold", "high_income", "edge_case", "boundary", "exactly"]):
            rules_file = "symbolica/rules/s1_symbolic_rules.yaml"
        else:
            return {"error": f"Could not determine rule file for case: {case_id}"}
        
        # Create engine
        engine = Engine.from_file(rules_file)
        
        # For S2 cases, register a mock PROMPT function for validation
        if "sentiment" in case_id or "positive" in case_id or "negative" in case_id:
            # Use lambda and allow_unsafe for the mock function
            mock_prompt = lambda *args: case_data["expected_decision"].get("sentiment", "neutral")
            engine.register_function("PROMPT", mock_prompt, allow_unsafe=True)
        
        # Flatten customer data to facts
        fact_data = flatten_customer_data(case_data.get("customer_data", {}))
        
        # Execute rules
        result = engine.reason(facts(**fact_data))
        
        # Compare with expected
        expected = case_data["expected_decision"]
        actual = result.verdict
        
        # Check alignment
        issues = []
        for key, expected_value in expected.items():
            if key not in actual:
                issues.append(f"Missing key: {key}")
            elif actual[key] != expected_value:
                issues.append(f"Mismatch {key}: expected {expected_value}, got {actual[key]}")
        
        return {
            "case_id": case_id,
            "aligned": len(issues) == 0,
            "issues": issues,
            "expected": expected,
            "actual": actual,
            "reasoning": result.reasoning
        }
        
    except Exception as e:
        return {
            "case_id": case_data.get("id", "unknown"),
            "aligned": False,
            "issues": [f"Execution error: {str(e)}"],
            "expected": case_data.get("expected_decision", {}),
            "actual": {},
            "reasoning": ""
        }

def main():
    """Main validation function."""
    print("🔍 Validating Symbolica Rules vs TXT Test Cases")
    print("=" * 50)
    
    # Load test cases
    test_cases = load_test_cases()
    print(f"Found {len(test_cases)} test cases")
    
    # Validate each case
    results = []
    for case_data in test_cases:
        print(f"\nValidating: {case_data['id']}")
        result = validate_case(case_data)
        results.append(result)
        
        if result["aligned"]:
            print(f"  ✅ ALIGNED")
        else:
            print(f"  ❌ MISALIGNED")
            for issue in result["issues"]:
                print(f"    - {issue}")
    
    # Summary
    aligned_count = sum(1 for r in results if r["aligned"])
    total_count = len(results)
    
    print(f"\n" + "=" * 50)
    print(f"📊 VALIDATION SUMMARY")
    print(f"=" * 50)
    print(f"Aligned:     {aligned_count}/{total_count}")
    print(f"Misaligned:  {total_count - aligned_count}/{total_count}")
    print(f"Success Rate: {aligned_count/total_count*100:.1f}%")
    
    if aligned_count == total_count:
        print(f"\n🎉 All test cases are properly aligned!")
    else:
        print(f"\n⚠️  Found {total_count - aligned_count} misaligned cases:")
        for result in results:
            if not result["aligned"]:
                print(f"  - {result['case_id']}: {', '.join(result['issues'])}")

if __name__ == "__main__":
    main() 