# Phase 1 Sub-F Close Checklist

Running obligations to verify before the Task 15 sub-F handoff commit.

- [ ] After Task 7 locks BP7 boundary-reference vocab, add the boundary-ref vocab source to the sub-F region manifest `vocab_sources`, recompute final complete `manifest_sha256`, and clear/update `vocab_sources_status` from `partial_pending_bp7`.
- [ ] In the sub-F-close handoff, document that sub-F SOURCE derivation hard-depends on sub-C manifest field names `sub_c_schema_version` and `initial_extraction.commit_sha`; future sub-C schema bumps or renames must update `src/cfm/data/sub_f/versions.py`.
- [ ] In the sub-F-close handoff, record SOURCE `VersionRef.value` canonical format: `overture=<release>;subc_schema=<ver>;subc_commit=<full_sha>`.
