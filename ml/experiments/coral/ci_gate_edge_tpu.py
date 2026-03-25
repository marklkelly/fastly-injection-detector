#!/usr/bin/env python3
"""
CI/CD gate script for Edge TPU compilation.
Parses compiler logs and enforces quality gates.

Usage: python ci_gate_edge_tpu.py <compiler_log_file>

Exit codes:
  0 - All gates passed
  1 - Gate failure (mapping < 90% or subgraphs != 1)
  2 - Parse error
"""

import re
import sys
from pathlib import Path
from typing import Dict, Optional


def parse_compiler_log(log_path: str) -> Optional[Dict]:
    """
    Parse Edge TPU compiler log to extract metrics.

    Returns:
        Dict with compilation metrics or None if parse fails
    """
    try:
        with open(log_path, 'r') as f:
            content = f.read()

        metrics = {}

        # Extract number of subgraphs
        subgraph_match = re.search(r'Number of Edge TPU subgraphs:\s*(\d+)', content)
        if subgraph_match:
            metrics['subgraphs'] = int(subgraph_match.group(1))

        # Extract total operations and TPU operations
        total_match = re.search(r'Total number of operations:\s*(\d+)', content)
        tpu_match = re.search(r'Number of operations that will run on Edge TPU:\s*(\d+)', content)

        if total_match and tpu_match:
            total_ops = int(total_match.group(1))
            tpu_ops = int(tpu_match.group(1))
            metrics['total_ops'] = total_ops
            metrics['tpu_ops'] = tpu_ops
            metrics['mapping_percent'] = (tpu_ops / total_ops * 100) if total_ops > 0 else 0

        # Extract compilation time
        time_match = re.search(r'Model successfully compiled in (\d+) seconds', content)
        if time_match:
            metrics['compile_time'] = int(time_match.group(1))

        # Check for errors
        if 'ERROR' in content or 'Failed' in content:
            metrics['has_errors'] = True

        return metrics

    except Exception as e:
        print(f"❌ Failed to parse log: {e}")
        return None


def check_gates(metrics: Dict) -> bool:
    """
    Check if compilation metrics pass quality gates.

    Gates:
    - subgraphs == 1
    - mapping >= 90%
    - no errors

    Returns:
        True if all gates pass
    """
    passed = True
    print("\n📊 Quality Gates:")
    print("-" * 40)

    # Gate 1: Single subgraph
    if 'subgraphs' in metrics:
        subgraph_pass = metrics['subgraphs'] == 1
        status = "✅ PASS" if subgraph_pass else "❌ FAIL"
        print(f"Subgraphs:    {metrics['subgraphs']} (required: 1) {status}")
        if not subgraph_pass:
            passed = False
    else:
        print(f"Subgraphs:    ⚠️ NOT FOUND")
        passed = False

    # Gate 2: TPU mapping >= 90%
    if 'mapping_percent' in metrics:
        mapping_pass = metrics['mapping_percent'] >= 90.0
        status = "✅ PASS" if mapping_pass else "❌ FAIL"
        print(f"TPU Mapping:  {metrics['mapping_percent']:.1f}% (required: ≥90%) {status}")
        if not mapping_pass:
            passed = False
    else:
        print(f"TPU Mapping:  ⚠️ NOT FOUND")
        passed = False

    # Gate 3: No compilation errors
    has_errors = metrics.get('has_errors', False)
    error_status = "❌ FAIL" if has_errors else "✅ PASS"
    print(f"No Errors:    {error_status}")
    if has_errors:
        passed = False

    print("-" * 40)

    return passed


def process_log_file(log_path: str) -> int:
    """
    Process a single compiler log file.

    Returns:
        Exit code (0 for pass, 1 for fail, 2 for error)
    """
    print(f"\n🔍 Processing: {log_path}")

    if not Path(log_path).exists():
        print(f"❌ Log file not found: {log_path}")
        return 2

    # Parse log
    metrics = parse_compiler_log(log_path)
    if not metrics:
        print("❌ Failed to parse compiler log")
        return 2

    # Display metrics
    print("\n📈 Compilation Metrics:")
    print("-" * 40)
    if 'total_ops' in metrics:
        print(f"Total operations:  {metrics['total_ops']}")
        print(f"TPU operations:    {metrics['tpu_ops']}")
    if 'compile_time' in metrics:
        print(f"Compile time:      {metrics['compile_time']}s")

    # Check gates
    if check_gates(metrics):
        print("\n✅ ALL GATES PASSED")
        return 0
    else:
        print("\n❌ GATE FAILURE - Model not suitable for production")
        return 1


def main():
    """Main entry point for CI gate."""

    if len(sys.argv) < 2:
        print("Usage: python ci_gate_edge_tpu.py <compiler_log_file> [additional_logs...]")
        sys.exit(2)

    print("="*60)
    print("EDGE TPU COMPILATION CI GATE")
    print("="*60)

    exit_codes = []

    # Process each log file
    for log_path in sys.argv[1:]:
        exit_code = process_log_file(log_path)
        exit_codes.append(exit_code)

    # Overall result
    print("\n" + "="*60)
    if all(code == 0 for code in exit_codes):
        print("✅ CI GATE: PASSED - All models meet quality requirements")
        sys.exit(0)
    else:
        print("❌ CI GATE: FAILED - One or more models failed quality gates")
        sys.exit(1)


if __name__ == "__main__":
    main()