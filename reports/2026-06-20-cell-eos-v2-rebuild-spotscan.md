# cell-EOS: v2 shard-cache rebuild + spot-scan (2026-06-20)

Branch `phase-2-cell-eos` (HEAD `8d94ed8`). After the cell-EOS change appended `<cell_end>`=260 to every
non-empty cell and bumped `SHARD_CACHE_DERIVATION_VERSION` 1→2, the EU-train-union shard cache was rebuilt
on Leonardo and verified. Scripts: `scripts/build_shard_cache_v2.sbatch`, `scripts/_spot_scan_cell_end.py`,
`scripts/spot_scan_cell_end.sbatch`.

## v2 rebuild — job 47462761 (`lrd_all_serial`, COMPLETED, exit 0)
- **Wall: 50:21** (`WALL_SECONDS=3019`); script logged `SEALED … in 49.9 min (38 cities, 22,019 tiles)`.
- `derivation_version: '2'`, `_SHARD_CACHE_VALID` sealed, **762 MB**, built from HEAD `8d94ed8`.
- Ran as a **serial Slurm job** after a login-node attempt was SIGKILL'd at ~26 min by the login
  watchdog (multi-city cold builds must use Slurm, not the login node — see memory
  `leonardo_shard_cache_rebuild_ops`). The v1 cache was moved aside (not deleted) before rebuild.
- NOTE: the spec's "~7 min" was the cache-**hit readback**; the cold build is ~50 min.

## Spot-scan — job 47464489 (`lrd_all_serial`, COMPLETED, 5:44)
Over every cell of the sealed v2 cache via the canonical `read_city_cache` (the live half of Tooth 2):
```
total cells             : 1409216
  non-empty             : 1046162
  empty (())            : 363054
non-empty ending 260    : 1046162/1046162 (100.000000%)
non-empty ending(510,260): 1046162/1046162 (100.000000%)
non-empty w/ exactly 1 260: 1046162/1046162
VIOLATIONS: non-empty NOT ending 260: 0 | 260 count != 1: 0 | interior 260: 0 |
            non-empty NOT ending (510,260): 0 | empty carrying 260: 0 | cell==(260,): 0
RESULT: ALL CLEAN
```
**Every one of 1,046,162 non-empty rebuilt cells ends exactly `(…,510,260)`, zero violations of any kind.**
This is the full-scale live confirmation of build-side Tooth 2 (`tests/data/training/test_build_shards_cell_end.py`).
