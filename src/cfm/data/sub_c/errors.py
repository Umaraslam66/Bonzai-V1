"""Sub-C exception types.

PolicyError: unknown missing-value policy type encountered in YAML.
TileValidationError: inline + cross-tile validator failures, with structured
payload for diagnostic determinism (spec §12.4 + §13.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class PolicyError(ValueError):
    """Raised by apply_missing_value_policy when an unknown policy type is encountered."""


@dataclass
class TileValidationError(Exception):
    """Structured payload for inline + cross-tile validator failures.

    The fields (tile, invariant, failed_row, detail) form the canonical
    diagnostic payload that must be byte-deterministic across runs given
    identical input — tested by test_validator_diagnostic_payloads_byte_deterministic.
    """

    tile: str
    invariant: str
    failed_row: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(
            f"TileValidationError on {self.tile}: invariant={self.invariant}, "
            f"row={self.failed_row}, detail={self.detail}"
        )
