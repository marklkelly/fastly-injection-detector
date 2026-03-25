#!/usr/bin/env python3
"""
Parse Edge TPU compiler output and enforce quality gates.
Gates: ≥85% mapping to TPU, ≤3 segments
"""

import argparse
import re
import sys
import pathlib


def parse_report(log_text):
    """
    Parse Edge TPU compiler log for mapping percentage and segments.

    Returns:
        tuple: (mapping_percentage, num_segments)
    """
    # Look for mapping percentage
    # Format: "XX% of the operations will run on the Edge TPU"
    # Or: "Model successfully compiled but not all operations are supported"
    # Or: "X ops are run on Edge TPU and Y ops are run on CPU"

    mapping_pct = 0.0
    num_segments = 0

    # Try different patterns
    patterns = [
        r'(\d+\.?\d*)%\s+of\s+the\s+operations?\s+will\s+run\s+on\s+the\s+Edge\s+TPU',
        r'(\d+\.?\d*)%\s+of\s+model.*?mapped\s+to\s+Edge\s+TPU',
        r'Operators\s+fully\s+assigned\s+to\s+EdgeTPU:\s+(\d+\.?\d*)%',
    ]

    for pattern in patterns:
        match = re.search(pattern, log_text, re.IGNORECASE)
        if match:
            mapping_pct = float(match.group(1)) / 100.0
            break

    # Alternative: count ops
    ops_match = re.search(
        r'(\d+)\s+ops?\s+are\s+run\s+on\s+Edge\s+TPU\s+and\s+(\d+)\s+ops?\s+are\s+run\s+on\s+CPU',
        log_text
    )
    if ops_match and mapping_pct == 0.0:
        tpu_ops = int(ops_match.group(1))
        cpu_ops = int(ops_match.group(2))
        total_ops = tpu_ops + cpu_ops
        if total_ops > 0:
            mapping_pct = tpu_ops / total_ops

    # Count segments
    # Look for "Segment X" patterns
    segment_matches = re.findall(r'Segment\s+\d+', log_text)
    num_segments = len(set(segment_matches))  # Unique segments

    # Alternative segment counting
    if num_segments == 0:
        # Look for delegation patterns
        delegation_matches = re.findall(r'EDGE_TPU_DELEGATE', log_text)
        if delegation_matches:
            num_segments = len(delegation_matches)

    return mapping_pct, num_segments


def check_gates(mapping_pct, num_segments, min_map=0.85, max_segments=3):
    """
    Check if compilation meets quality gates.

    Args:
        mapping_pct: Fraction of ops mapped to TPU (0.0-1.0)
        num_segments: Number of compiled segments
        min_map: Minimum mapping requirement (default 0.85)
        max_segments: Maximum segments allowed (default 3)

    Returns:
        tuple: (passed, reasons)
    """
    passed = True
    reasons = []

    # Check mapping percentage
    if mapping_pct < min_map:
        passed = False
        reasons.append(f"Mapping {mapping_pct:.1%} < {min_map:.0%} requirement")
    else:
        reasons.append(f"✓ Mapping {mapping_pct:.1%} ≥ {min_map:.0%}")

    # Check segments
    if num_segments > max_segments:
        passed = False
        reasons.append(f"Segments {num_segments} > {max_segments} maximum")
    elif num_segments > 0:
        reasons.append(f"✓ Segments {num_segments} ≤ {max_segments}")

    # If no segments detected, it might mean full CPU fallback
    if num_segments == 0 and mapping_pct < 0.5:
        passed = False
        reasons.append("No TPU segments detected (full CPU fallback?)")

    return passed, reasons


def provide_recommendations(mapping_pct, num_segments):
    """Provide optimization recommendations based on results."""
    recommendations = []

    if mapping_pct < 0.85:
        recommendations.append("\nOptimization suggestions for low mapping:")
        recommendations.append("• Replace Softmax with approximation or keep on CPU")
        recommendations.append("• Ensure all activations are ReLU6 (not GELU/Swish)")
        recommendations.append("• Replace LayerNormalization with simpler normalization")
        recommendations.append("• Convert Dense layers to Conv1D (kernel_size=1)")
        recommendations.append("• Ensure static shapes (batch=1, seq_len=128)")

    if num_segments > 3:
        recommendations.append("\nOptimization suggestions for fragmentation:")
        recommendations.append("• Use post-embedding cut strategy")
        recommendations.append("• Reduce model complexity")
        recommendations.append("• Split wide layers (>256 channels) into parallel paths")
        recommendations.append("• Ensure channel counts are multiples of 8")
        recommendations.append("• Minimize reshape/transpose operations")

    if mapping_pct > 0.5 and mapping_pct < 0.85:
        recommendations.append("\nYou're close! Consider:")
        recommendations.append("• Moving attention to CPU, keep FFN on TPU")
        recommendations.append("• Simplifying just 1-2 problematic layers")

    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="Parse Edge TPU compiler output and enforce gates"
    )
    parser.add_argument("--report", required=True, help="Path to compiler log file")
    parser.add_argument("--min_map", type=float, default=0.85,
                        help="Minimum mapping percentage (0.0-1.0)")
    parser.add_argument("--max_segments", type=int, default=3,
                        help="Maximum number of segments")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed output")

    args = parser.parse_args()

    # Read log file
    try:
        log_text = pathlib.Path(args.report).read_text()
    except FileNotFoundError:
        print(f"❌ Error: Log file not found: {args.report}")
        sys.exit(1)

    # Parse report
    mapping_pct, num_segments = parse_report(log_text)

    # Display results
    print("\n" + "="*60)
    print("EDGE TPU COMPILATION RESULTS")
    print("="*60)
    print(f"Mapping to TPU: {mapping_pct:.1%}")
    print(f"Number of segments: {num_segments}")

    # Check gates
    passed, reasons = check_gates(
        mapping_pct, num_segments,
        args.min_map, args.max_segments
    )

    print("\n" + "="*60)
    print("QUALITY GATES")
    print("="*60)
    for reason in reasons:
        print(reason)

    # Overall result
    print("\n" + "="*60)
    if passed:
        print("✅ GATES PASSED - Model is suitable for Edge TPU deployment")
        print("="*60)
        return 0
    else:
        print("❌ GATES FAILED - Model needs optimization")
        print("="*60)

        # Provide recommendations
        recommendations = provide_recommendations(mapping_pct, num_segments)
        if recommendations:
            for rec in recommendations:
                print(rec)

        # Show next steps
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        if mapping_pct < 0.5:
            print("→ Move to Phase B: Post-embedding hybrid approach")
            print("  Run: make export compile gate")
        elif mapping_pct < args.min_map:
            print("→ Optimize model architecture:")
            print("  - Replace problematic operations")
            print("  - Use Conv1D instead of Dense")
            print("  - Switch to ReLU6 activation")
        else:
            print("→ Reduce fragmentation:")
            print("  - Simplify model structure")
            print("  - Reduce number of operations")

        return 2  # Exit code 2 for failed gates


if __name__ == "__main__":
    sys.exit(main())