# Single-cell positive fixture

A hand-built 250 m × 250 m cell anchored at (0, 0) in cell-local metres. Contains exactly one feature of each kind the Phase-0 tokenizer must handle: one road exiting the east edge, one rectangular building, one POI, one axis-aligned land-use polygon. Small enough to read in your head.

Used by:
- `tests/tokenizer/test_round_trip.py`
- `tests/tokenizer/test_determinism.py`
- `scripts/smoke.py`

If you change `input.geojson`, update `expected.yaml` to match.
