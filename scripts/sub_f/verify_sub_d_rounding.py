"""Verify sub-D's actual round() / quantization mechanism.

Per spec §5.2 + feedback_verify_before_lock_not_after: lock pending until
verified. Assumed default: Python round() round-half-to-even per PEP 3141.
Cascade per §9.6.1 if mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# Test inputs at exact bin-edge values per BP5 plan-write refinement: constructed,
# not real-data-derived. round() round-half-to-even should produce specific outputs.
TEST_CASES = [
    (0.5, 0),  # banker's rounding rounds 0.5 to 0
    (1.5, 2),  # 1.5 to 2 (even)
    (2.5, 2),  # 2.5 to 2 (even)
    (3.5, 4),  # 3.5 to 4 (even)
    (-0.5, 0),
    (-1.5, -2),
]


def main() -> int:
    # Test Python round() in this process; document for halt report.
    python_round_results = [(x, round(x)) for x, _ in TEST_CASES]
    expected_banker = [(x, e) for x, e in TEST_CASES]
    is_banker = python_round_results == expected_banker

    # Read sub-D source for actual usage pattern.
    sub_d_io = (ROOT / "src" / "cfm" / "data" / "sub_d" / "io.py").read_text(encoding="utf-8")
    uses_round = "round(" in sub_d_io
    uses_int_cast = "int(" in sub_d_io

    report = {
        "python_round_is_banker": is_banker,
        "sub_d_io_uses_round": uses_round,
        "sub_d_io_uses_int_cast": uses_int_cast,
        "test_cases": [
            {"input": x, "round_output": round(x), "expected_banker": e} for x, e in TEST_CASES
        ],
        "recommendation": (
            "LOCK Python round() round-half-to-even (PEP 3141 default) for sub-F"
            if is_banker
            else "ESCALATE: Python round() does not match banker's expectation in this env"
        ),
        "_status": "PROPOSED — pending Halt 5 reviewer approval per spec §10.3.",
    }
    out = ROOT / "reports" / "sub_f_task_5a_rounding.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(report, sort_keys=True), encoding="utf-8")
    print(f"[rounding] wrote {out}; banker={is_banker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
