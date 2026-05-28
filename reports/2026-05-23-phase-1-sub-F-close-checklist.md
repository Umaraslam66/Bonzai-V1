# Phase 1 Sub-F Close Checklist

Running obligations to verify before the Task 15 sub-F handoff commit.

- [ ] After Task 7 locks BP7 boundary-reference vocab, add the boundary-ref vocab source to the sub-F region manifest `vocab_sources`, recompute final complete `manifest_sha256`, and clear/update `vocab_sources_status` from `partial_pending_bp7`.
- [ ] In the sub-F-close handoff, document that sub-F SOURCE derivation hard-depends on sub-C manifest field names `sub_c_schema_version` and `initial_extraction.commit_sha`; future sub-C schema bumps or renames must update `src/cfm/data/sub_f/versions.py`.
- [ ] In the sub-F-close handoff, record SOURCE `VersionRef.value` canonical format: `overture=<release>;subc_schema=<ver>;subc_commit=<full_sha>`.
- [ ] In the sub-F-close handoff, document sub-E-v2 candidate: refine highway tiering in sub-E grouping map (`motorway` -> MAJOR; decide non-vehicular way handling). sub-F BP7 inherits this through architecture (b).
- [ ] In the sub-F-close handoff, document sub-E-v2 candidate: evaluate whether same-edge MultiLineString / multi-part road crossings need richer than one-class-per-edge representation.
- [ ] In the sub-F-close handoff, document cross-sub-project coupling: sub-F BP7 classes are 100% sub-E-derived; sub-E boundary_contract schema or semantic changes propagate directly to sub-F BP7 output and training-scaffold must know this.
- [ ] When sub-E output is regenerated or restored, spot-check actual `boundary_contract.parquet` emission for motorway-only and same-edge MultiLineString edges against Task 7's code-inferred behavior. If actual output diverges, update BP7 limitation docs; the BP7 lock remains faithful-passthrough.
- [ ] OPEN SCHEDULE DECISION (Phase-1 level, not sub-F-internal): sub-E MINOR-default under-tiers motorway as MINOR. Accepted for sub-F-v1 faithful-passthrough. Decision needed: does sub-E tiering fix (`motorway` -> MAJOR) land before first Leonardo training run, or is degraded arterial tiering acceptable for first-run validation? Owner: human reviewer. Inherited by training-scaffold sub-project.
