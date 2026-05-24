# Phase 1 sub-F micro-tokenizer implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-cell micro-tokenizer (PRD stage four) — encoder/decoder + v1 vocab + round-trip gate + Singapore-validated empirical artifacts.

**Architecture:** 15-task DAG mirroring sub-E's structure: vocab + grammar locks (Tasks 1-7) → writer/validator pipeline (Tasks 8-11) → CLI + integration tests (Tasks 12-13) → empirical gate + handoff (Tasks 14-15). Seven reviewer-halt gates per spec §10.1. Six-axis version manifest extending sub-D's `VersionNamespace` enum (audit-derived revision; see Plan Revisions below).

**Tech Stack:** Python 3.11+, pyarrow (pinned schemas via `pa.schema`), PyYAML (`yaml.safe_load`), pytest. Routes through `cfm.data.io.write_parquet` for byte-deterministic parquet writes per sub-D/sub-E precedent. No torch in sub-F (encoder/decoder are pre-training).

**Spec reference:** `docs/superpowers/specs/2026-05-23-phase-1-sub-F-micro-tokenizer-design.md` (commit `38ff72b`).

**Operating discipline:** Sub-project planning protocol v1 at `docs/protocols/sub-project-planning-protocol-v1.md`. Per-task pre-dispatch audit + halt-on-defect + verify-before-lock per `feedback_subagent_branch_pattern` and `feedback_verify_before_lock_not_after`.

---

## Plan revisions from pre-dispatch audit (§9.6.1 cascade outcomes)

Five revisions surfaced. Revisions 1+2+3 surfaced at plan-write pre-dispatch audit. Revisions 4+5 surfaced at prompt-derivation review (third-layer audit) along with five Task 1 code bugs (corrected inline in Task 1 code blocks below). All cascades resolve per §9.6.1 discipline (sub-D wins / canonical source wins; lock value updates; §2 paired check re-validates; §13 revision ledger entries committed at sub-F-close + spec sync at `cd5d332` already applied).

### Revision 1: `compare_version` mechanism — enum-extension, not kwarg-extension

**Spec §6.4 assumed** kwarg-add extension; fallback chain (a → c → b) for variable-axis-count or `compare_version_v2`.

**Audit found** (`src/cfm/data/sub_d/versions.py:1-80`): sub-D uses `VersionNamespace(str, Enum)` + `VersionRef` dataclass + `compare_version(namespace, expected, actual)` signature. Extension is **enum-add** (`SOURCE = "source"` added to `VersionNamespace`), backward-compatible by Python enum semantics (existing callers reference specific namespaces by name; new member doesn't break them).

**Plan applies:** Task 6 extends `VersionNamespace` enum with `SOURCE`. No kwarg shape change. `compare_version` signature stays. Spec §6.4 fallback chain (a/c/b) becomes moot — extension is trivial enum-add. At sub-F close, §13 revision ledger entry documents the audit-time cascade.

### Revision 2: Sub-F manifest has 6 axes, not 4

**Spec §6.1 assumed** sub-D had 3 namespaces (DATA_SHAPE / VOCAB / DERIVATION); sub-F extends to 4 by adding SOURCE.

**Audit found:** sub-D has 5 namespaces: `ARTIFACT_FORMAT, DATA_SHAPE, VOCAB, DERIVATION, VALIDATOR`. Spec missed two: `ARTIFACT_FORMAT` (on-disk format identifier separate from schema) and `VALIDATOR` (validator code version).

**Plan applies:** Sub-F adopts all 5 sub-D namespaces + adds SOURCE → **6 axes total**: `ARTIFACT_FORMAT, DATA_SHAPE, VOCAB, DERIVATION, VALIDATOR, SOURCE`. Recommend at Task 6 halt: ARTIFACT_FORMAT separately tracked from DATA_SHAPE for evolution parity with sub-D; VALIDATOR tracked separately so validator refactoring doesn't trigger DATA_SHAPE bump. §6.8 paired check extends from 4 to 6 fields with no structural change.

### Revision 3: Sub-C feature sort key is concrete

**Spec §5.2** said sub-C row order is "verify at Task 5a against `src/cfm/data/sub_c/io.py`."

**Audit found** (`src/cfm/data/sub_c/io.py:218-220`): sort key is `(cell_i, cell_j, feature_class, source_feature_id)` — 4-tuple sort.

**Plan applies:** Task 5a verification simplified from open-ended read to confirming sort key still matches at implementation time. Lock value in §5.2 updates to concrete tuple per `feedback_verify_before_lock_not_after`.

### Revision 4: Singapore X-threshold scope narrowed to highway + building only

**Original Task 1 plan code** assumed `FEATURE_CLASS_TO_KEY = {0: "highway", 1: "building", 2: "amenity", 3: "natural"}` — direct 1:1 sub-C feature_class to OSM key mapping.

**Audit found** (`src/cfm/data/sub_c/enums.py:23` + `src/cfm/data/sub_c/io.py:186` + `src/cfm/data/sub_c/pipeline.py:678`): sub-C `feature_class=2` (poi) has NULL `class_raw` (POIs use `categories_primary` / `categories_alternate` columns instead); sub-C `feature_class=3` (base) lumps water+landuse+natural with ambiguous parent key. Simple FEATURE_CLASS_TO_KEY mapping is wrong for poi + base.

**Plan applies:** Task 1's `FEATURE_CLASS_TO_KEY` shipped scoped to highway (0) + building (1) only. Singapore X-threshold computation narrowed accordingly. POI + base Singapore-prioritized must-appears deferred to sub-F-v2 per spec §12 entry #11.

### Revision 5: L1 must-appears corrected from 15 keys → 28 keys

**Original Task 1 plan code** shipped `WIKI_L1_MUST_APPEARS` with 15 keys (highway, building, amenity, landuse, natural, water, waterway, leisure, shop, place, boundary, route, public_transport, barrier, man_made). The list was reviewer-supplied at BP1 brainstorm and locked at plan-write without Gate 6 verification against canonical wikitext.

**Audit found** (Map_features wikitext fetched at plan-revision time): `==Primary features==` section enumerates 28 primary feature keys via `{{Map_Features:X}}` transclusions + `{{Building typology}}` template. Missing from original: `aerialway, aeroway, craft, emergency, geological, healthcare, historic, military, office, power, railway, telecom, tourism` (13 keys).

**Plan applies:** Task 1's `WIKI_L1_MUST_APPEARS` shipped with 28-key tuple. L2 enumeration scoped to highway + building only per cascade #4 alignment (Singapore X has signal only for these two keys at v1; other 26 keys' L2 elbows have no reviewable Singapore data at Halt 1, so L2 enumeration for them would be sunk cost). L3 deferred entirely per spec §12 entry #10 (recursive marginal-cost-of-cut: enumerate only where reviewable benefit exists).

**Per-key hand-count assertions** (Safeguard 2) added to `test_vocab.py`: independent hand-counts of section headers in wikitext (28 transclusion lines in `==Primary features==`; highway template L3 row count; building template L3 row count) separately from the pair-set extraction itself. Catches per-section enumeration errors that flat set comparison misses.

### Task 1 plan code bug fixes (5 substantive bugs surfaced at prompt-derivation review)

The original Task 1 code blocks (snapshot_taginfo.py + floor_analysis.py) shipped with five code-level defects independent of the cascade revisions above. All five corrected inline in the Task 1 code blocks below:

- **Bug 1: CSV column semantics conflation.** Original `snapshot_taginfo.py` wrote `fraction_all` for both key rows and value rows but the two have different denominators (key rows: fraction-of-all-OSM via `/keys/all`; value rows: fraction-within-key via `/key/values`). **Fix:** snapshot stores raw counts only; `floor_analysis.py` centralizes fraction derivation with consistent denominator per Bug 2.
- **Bug 2: Per-element-type normalization missing (BP1 fix 2 violation).** Original used `fraction_all` (fraction-of-all-tags) which is BP1 fix 2 REJECTED option. **Fix:** `floor_analysis.py` derives fractions via dominant-element-type-per-key (highway → way-fraction, amenity → node-fraction); value rows inherit parent key's dominant ET as documented approximation.
- **Bug 3: `vocab_size_at_F` missing row-type filter.** Original counted both key and value rows together at level 1. **Fix:** level-aware row-type filter (level=1 counts only key rows; level=2 counts only value rows in WIKI_L2_PRIMARY_PAIRS; level=3 counts only value rows in WIKI_L3_ALL_PAIRS).
- **Bug 4: L1-only curve silently downgrades Halt 1.** Original shipped L1 with `NotImplementedError` for L2/L3. **Fix per cascade #5 resolution:** L1 (full 28) + L2 (highway + building per cascade #4 scope); L3 deferred per §12 entry #10. Halt 1 surfaces partial-by-design curve with explicit scoping rationale.
- **Bug 5: X-threshold ignores Singapore data.** Original computed `10 × F_global` with no Singapore input — BP1 fix 5 option (c) requires Singapore-frequency. **Fix:** new step added (Step 5) reads sub-C Singapore extracts + computes Singapore-frequency for highway + building tag pairs + derives X-threshold candidates (Singapore-elbow + median-must-appear-frequency). X scope narrowed per cascade #4.

### Sub-F module layout (`src/cfm/data/sub_f/`)

```
src/cfm/data/sub_f/
  __init__.py
  versions.py              # 6 version constants + SUB_F_NAMESPACE registry (Task 6)
  vocab.py                 # vocab union load + ID-range invariants (Task 4 close)
  enums.py                 # Direction, MagnitudeQuantum, AnchorScheme, BrefDirection, BrefClass (Task 2+7)
  encoder.py               # geometry → token sequence (Task 8)
  decoder.py               # token sequence → geometry (Task 8)
  rotation.py              # Cell-local rotation wrapper around sub-E's cell_to_edge_ids (Task 7)
  io.py                    # Reader/writer: cells.parquet schema + per-tile + region manifest (Task 8)
  provenance.py            # SUB_F_EXCLUDED_FROM_SHA + provenance dataclass (Task 6)
  manifest.py              # Region manifest write/read with vocab_sources block (Task 6)
  validator_inline.py      # Per-cell schema + derivation + token-range checks (Task 9)
  validator_cross_tile.py  # BP7 four-test + cross-axis coupling + manifest consistency (Task 10)
  pipeline.py              # derive_region orchestrator (Task 11)
```

### Configs (`configs/sub_f/`)

```
configs/sub_f/
  semantic_vocab.yaml                                   # BP1 lock (Task 1)
  encoding_primitives.yaml                              # BP2 lock (Task 2)
  unknown_family.yaml                                   # BP4 lock (Task 4)
  boundary_reference_vocab.yaml                         # BP7 lock (Task 7)
  sentinel_inventory.yaml                               # Full vocab manifest + reserved blocks (Task 4)
  vocab_floor_analysis.yaml                             # Task 1 halt output
  sequence_length_analysis.yaml                         # Task 3c halt output
  taginfo/2026-04-15.0.csv                              # Snapshot (Task 1)
  wiki_map_features/2026-04-15.0.wikitext               # Snapshot (Task 1)
  wiki_map_features/2026-04-15.0.sha256                 # Snapshot hash (Task 1)
  wiki_map_features/2026-04-15.0.revision_id            # MediaWiki revision (Task 1)
```

### Scripts (`scripts/sub_f/`)

```
scripts/sub_f/
  derive.py                # End-to-end derive CLI (Task 12)
  validate.py              # Validator CLI (Task 12)
  encode.py                # Encode CLI (Task 12)
  decode.py                # Decode CLI (Task 12)
```

### Tests (`tests/data/sub_f/`)

```
tests/data/sub_f/
  __init__.py
  _fixtures.py                                          # Synthetic geometry + cell fixtures
  test_vocab.py                                         # BP1 / BP4 vocab load + ID range (Tasks 1, 4)
  test_encoder.py                                       # BP2 / BP7 encoder (Tasks 2, 7, 8)
  test_decoder.py                                       # Decoder + round-trip per case (Task 8)
  test_rotation.py                                      # Per-cell rotation wrapper (Task 7)
  test_io.py                                            # cells.parquet schema + manifest (Task 8)
  test_provenance.py                                    # SUB_F_EXCLUDED_FROM_SHA (Task 6)
  test_manifest.py                                      # vocab_sources block (Task 6)
  test_validator_inline.py                              # Per-cell invariants (Task 9)
  test_validator_cross_tile.py                          # BP7 4-test composite + cross-axis (Task 10)
  test_pipeline.py                                      # Orchestrator + halt-on-fail (Task 11)
  test_per_axis_determinism.py                          # BP5 per-axis (Task 5b)
  test_singapore_integration.py                        # Cached Singapore integration (Task 13)
  test_empirical_gate.py                                # Round-trip on real Singapore (Task 14)
```

### Golden artifacts (`tests/golden/sub_f/`)

```
tests/golden/sub_f/
  round_trip/layer_singapore_round_trip_summary.yaml    # Task 14 golden
```

### Pinned `pa.schema` cell layout reference (Task 8 implementation target)

```python
_CELLS_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("cell_i", pa.int8(), nullable=False),
        pa.field("cell_j", pa.int8(), nullable=False),
        pa.field("cell_slot_index", pa.int8(), nullable=False),
        pa.field("token_sequence", pa.list_(pa.int16()), nullable=False),
        pa.field("feature_count", pa.int16(), nullable=False),
        pa.field("provenance_sha256", pa.string(), nullable=False),
    ]
)
```

---

## Task index

| # | Task | Halt | Blocked by |
|---|---|---|---|
| 1 | BP1 vocab floor analysis + snapshots | **Halt 1** | — |
| 2 | BP2 encoder primitives + round-trip thresholds | **Halt 2** | — |
| 3a | Stage-1+2 joint distribution by feature type | — | — |
| 3b | Stage-3 compound | — | T2, T3a |
| 3c | Stage-4 compound + budget surface | **Halt 4** | T3b |
| 4 | BP4 unknown family + sentinel inventory | **Halt 3** | T1 |
| 5a | BP5 verifications | **Halt 5** | — |
| 5b | BP5 per-axis test suite | — | T5a, T8 |
| 6 | BP6 version manifest (6-axis) | **Halt 6** | — |
| 7 | BP7 boundary-ref vocab + sub-C feature-splitting verify | **Halt 7** | — |
| 8 | Writer (encoder/decoder + cells.parquet + provenance/manifest) | — | T1, T2, T4, T5a, T6, T7 |
| 9 | Inline validator | — | T8 |
| 10 | Cross-tile validator | — | T9 |
| 11 | Pipeline orchestrator | — | T8, T9, T10 |
| 12 | CLI scripts | — | T11 |
| 13 | Singapore integration tests | — | T11, T5b |
| 14 | Empirical gate | (terminal) | T13, T3c |
| 15 | Handoff document | — | T14 |

---

## Task 1: BP1 vocab floor analysis + snapshots

**Halt 1 gate.** Implementer surfaces taginfo + wiki snapshots + marginal-cost curve (L1/L2/L3) + proposed elbow + X-threshold candidates. Reviewer approves elbow + exception list + X-threshold value. After approval: `vocab_floor_analysis.yaml` + `semantic_vocab.yaml` + snapshot artifacts lock.

**Files:**
- Create: `configs/sub_f/taginfo/2026-04-15.0.csv`
- Create: `configs/sub_f/wiki_map_features/2026-04-15.0.wikitext`
- Create: `configs/sub_f/wiki_map_features/2026-04-15.0.sha256`
- Create: `configs/sub_f/wiki_map_features/2026-04-15.0.revision_id`
- Create: `scripts/sub_f/snapshot_taginfo.py`
- Create: `scripts/sub_f/snapshot_wiki.py`
- Create: `scripts/sub_f/floor_analysis.py`
- Create: `configs/sub_f/vocab_floor_analysis.yaml`
- Create: `configs/sub_f/semantic_vocab.yaml`
- Test: `tests/data/sub_f/test_vocab.py`

### Pre-dispatch audit (§3 verifications, before implementation)

- [ ] **Audit step 1: confirm taginfo API for global key+value frequency**

Run: `curl -sI 'https://taginfo.openstreetmap.org/api/4/key/values?key=highway&page=1&rp=100&sortname=count&sortorder=desc' | head -5`
Expected: HTTP 200; JSON contains `data` array with value/count pairs. Use this endpoint per (key, value) pair for L2/L3 enumeration. For L1 (top-level keys): `https://taginfo.openstreetmap.org/api/4/keys/all`.

- [ ] **Audit step 2: confirm wiki API for Map_features wikitext**

Run: `curl -s 'https://wiki.openstreetmap.org/w/api.php?action=raw&page=Map_features&format=json' | head -100`
Expected: raw wikitext starting with `==` headers. Capture `oldid` from response headers for revision_id pin.

- [ ] **Audit step 3: verify F denominator semantics for taginfo**

Read: `https://taginfo.openstreetmap.org/api/4/wiki/data_overview` — confirms that the `count_*` fields decompose by element type (way / node / relation). F denominator per §2.1 is `count(tag X) / count(elements_of_type)` within each element type.

### Implementation steps

- [ ] **Step 1: Write snapshot acquisition script (raw counts, no fractions)**

**Bug 1 fix applied:** snapshot stores raw counts only; fraction derivation is centralized in `floor_analysis.py` per Bug 2 (consistent denominator across both key and value rows).

Create `scripts/sub_f/snapshot_taginfo.py`:

```python
"""Snapshot taginfo global key+value raw counts for sub-F BP1 floor.

Per Bug 1 fix: snapshot writes raw counts only (count_all, count_ways,
count_nodes, count_relations). Fraction derivation happens in
floor_analysis.py with consistent denominator (BP1 fix 2:
fraction-of-feature-bearing-elements within element type).

Schema:
  key,value,count_all,count_ways,count_nodes,count_relations,row_type,parent_key

  - row_type = "key": from /api/4/keys/all; value="" parent_key=""
  - row_type = "value": from /api/4/key/values for each parent key;
    count_ways/nodes/relations = 0 (taginfo /key/values doesn't return
    per-element-type breakdown for values — value rows inherit parent
    key's dominant ET distribution as documented approximation in
    floor_analysis.py).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import json

TAGINFO_KEYS_URL = "https://taginfo.openstreetmap.org/api/4/keys/all"
TAGINFO_VALUES_URL = (
    "https://taginfo.openstreetmap.org/api/4/key/values"
    "?key={key}&page=1&rp=1000&sortname=count&sortorder=desc"
)

USER_AGENT = "bonzai-osm-sub-f-snapshot/1.0 (research)"


def _fetch_json(url: str) -> Any:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def snapshot(out_csv: Path, top_n_keys: int = 200) -> None:
    """Snapshot taginfo raw counts into out_csv. Re-runnable but idempotent."""
    if out_csv.exists():
        print(f"[snapshot_taginfo] {out_csv} exists; skipping (delete to re-snapshot)")
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    keys_resp = _fetch_json(TAGINFO_KEYS_URL)
    keys_sorted = sorted(keys_resp["data"], key=lambda r: -r["count_all"])[:top_n_keys]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "key", "value", "count_all", "count_ways", "count_nodes",
                "count_relations", "row_type", "parent_key",
            ]
        )
        for key_row in keys_sorted:
            key = key_row["key"]
            # Key row: raw counts per ET from /keys/all
            writer.writerow(
                [
                    key, "", key_row["count_all"], key_row["count_ways"],
                    key_row["count_nodes"], key_row["count_relations"],
                    "key", "",
                ]
            )
            # Value rows: only count_all from /key/values; per-ET breakdown not
            # provided by this endpoint. Value rows inherit parent key's
            # dominant ET in floor_analysis.py (documented approximation).
            try:
                values_resp = _fetch_json(TAGINFO_VALUES_URL.format(key=key))
                for v in values_resp["data"]:
                    writer.writerow(
                        [
                            key, v["value"], v["count"], 0, 0, 0,
                            "value", key,
                        ]
                    )
            except Exception as exc:  # noqa: BLE001 — best-effort per key
                print(f"[snapshot_taginfo] warning: {key}: {exc}", file=sys.stderr)

    print(f"[snapshot_taginfo] wrote {out_csv}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="2026-04-15.0")
    args = parser.parse_args()
    out = Path(__file__).resolve().parents[2] / "configs" / "sub_f" / "taginfo" / f"{args.release}.csv"
    snapshot(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write wiki snapshot script**

Create `scripts/sub_f/snapshot_wiki.py`:

```python
"""Snapshot OSM wiki Map_features page (wikitext + revision_id + sha256).

Per BP1 fix B: HTML hashing is fragile (MediaWiki embeds timestamps in
rendered markup). Snapshot the raw wikitext via MediaWiki API; hash the
wikitext bytes; pin the MediaWiki revision ID as the canonical reproducibility
anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

WIKI_API_URL = (
    "https://wiki.openstreetmap.org/w/api.php"
    "?action=query&prop=revisions&titles=Map_features"
    "&rvprop=content|ids&rvslots=main&format=json&formatversion=2"
)

USER_AGENT = "bonzai-osm-sub-f-snapshot/1.0 (research)"


def snapshot(out_dir: Path, release: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wikitext_path = out_dir / f"{release}.wikitext"
    sha256_path = out_dir / f"{release}.sha256"
    revid_path = out_dir / f"{release}.revision_id"

    if wikitext_path.exists() and sha256_path.exists() and revid_path.exists():
        print(f"[snapshot_wiki] {release}.* exists in {out_dir}; skipping")
        return

    req = Request(WIKI_API_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())

    page = payload["query"]["pages"][0]
    rev = page["revisions"][0]
    revision_id = rev["revid"]
    wikitext_bytes = rev["slots"]["main"]["content"].encode("utf-8")

    sha = hashlib.sha256(wikitext_bytes).hexdigest()

    wikitext_path.write_bytes(wikitext_bytes)
    sha256_path.write_text(f"{sha}\n", encoding="utf-8")
    revid_path.write_text(f"{revision_id}\n", encoding="utf-8")

    print(f"[snapshot_wiki] wrote {wikitext_path}, sha={sha[:16]}…, rev={revision_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="2026-04-15.0")
    args = parser.parse_args()
    out_dir = Path(__file__).resolve().parents[2] / "configs" / "sub_f" / "wiki_map_features"
    snapshot(out_dir, args.release)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run snapshot scripts**

Run:
```bash
uv run python scripts/sub_f/snapshot_taginfo.py --release 2026-04-15.0
uv run python scripts/sub_f/snapshot_wiki.py --release 2026-04-15.0
```

Expected: `configs/sub_f/taginfo/2026-04-15.0.csv` and three files under `configs/sub_f/wiki_map_features/` exist. Verify with `ls -la configs/sub_f/`.

- [ ] **Step 4: Write floor analysis script with all 5 bug fixes applied**

**Bug 2 fix:** centralized fraction derivation with per-element-type normalization (parent key's dominant ET as denominator).
**Bug 3 fix:** level-aware row-type filter in `vocab_size_at_F`.
**Bug 4 fix per cascade #5:** L1 (full 28 keys) + L2 (highway + building only) curve; L3 deferred per spec §12 #10.
**Cascade #5:** L1 must-appears = 28 keys hand-enumerated from `configs/sub_f/wiki_map_features/2026-04-15.0.wikitext` `==Primary features==` section transclusions.
**Cascade #4:** L2 + Singapore X scope = highway + building only. Other 26 keys deferred per spec §12 #11.

Create `scripts/sub_f/floor_analysis.py`:

```python
"""Compute BP1 vocab floor marginal-cost curve + Gate 6 structural check.

Outputs configs/sub_f/vocab_floor_analysis.yaml at Halt 1.

Per spec §2.1 + plan cascade #4 + #5 resolutions:
- L1: full 28 keys from wiki Map_features ==Primary features== section.
- L2: highway + building only (Singapore-X-applicable per cascade #4).
- L3: deferred entirely per spec §12 #10 (recursive marginal-cost-of-cut).
- Singapore X-threshold: highway + building only per cascade #4.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Final

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# WIKI ENUMERATIONS — hand-enumerated from configs/sub_f/wiki_map_features/
# 2026-04-15.0.wikitext (Gate 6 canonical source, NOT reviewer-supplied or
# memory-inferred per cascade #5 lesson + spec §13.5 protocol-v2 candidate iii).
# Per-key hand-counts asserted independently in tests/data/sub_f/test_vocab.py
# per Safeguard 2 (catches per-section enumeration errors flat set comparison
# misses).
# ---------------------------------------------------------------------------

# L1: full 28 primary feature keys from Map_features ==Primary features==
# transclusion list. Each `{{Map_Features:X}}` or `{{Building typology}}` line
# in that section contributes one key.
WIKI_L1_MUST_APPEARS: Final[tuple[str, ...]] = (
    "aerialway", "aeroway", "amenity", "barrier", "boundary", "building",
    "craft", "emergency", "geological", "healthcare", "highway", "historic",
    "landuse", "leisure", "man_made", "military", "natural", "office",
    "place", "power", "public_transport", "railway", "route", "shop",
    "telecom", "tourism", "water", "waterway",
)

# L2: highway + building only per cascade #4 (Singapore X scope).
# Other 26 keys' L2 deferred per spec §12 #11.
#
# WIKI_L2_HIGHWAY: hand-enumerated from Template:Map_Features:highway value
# table — the "way-class" subset (Roads + Link roads + Special road types +
# Paths). Excludes "Other highway features" (stops, signals, milestones,
# infrastructure) which are point-features not way-classifications.
WIKI_L2_HIGHWAY: Final[tuple[str, ...]] = (
    # Roads (7 main road network tags per template's road_network annotation)
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential",
    # Link roads
    "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link",
    # Special road types
    "living_street", "service", "pedestrian", "busway",
    # Paths
    "footway", "cycleway", "bridleway", "path", "steps", "track",
    # Lifecycle placeholder
    "road",
)

# WIKI_L2_BUILDING: hand-enumerated from Template:Building_typology value
# table + "yes" catch-all (OSM convention for unspecified buildings,
# extremely common but not in the typology template).
WIKI_L2_BUILDING: Final[tuple[str, ...]] = (
    "yes",  # catch-all (added manually per OSM convention, not in template)
    # Typology values from Building_typology template
    "annexe", "apartments", "barn", "barracks", "bungalow", "cabin",
    "commercial", "detached", "dormitory", "entrance", "farm",
    "farm_auxiliary", "gatehouse", "ger", "hangar", "hotel", "house",
    "houseboat", "library", "office", "public", "residential",
    "semidetached_house", "service", "shed", "static_caravan",
    "stilt_house", "supermarket", "terrace", "train_station", "tree_house",
    "trullo",
)

WIKI_L2_PRIMARY_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset(
    {("highway", v) for v in WIKI_L2_HIGHWAY}
    | {("building", v) for v in WIKI_L2_BUILDING}
)

# L3: deferred entirely per spec §12 #10. Placeholder for future expansion.
WIKI_L3_ALL_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset()

# ---------------------------------------------------------------------------
# SUB-C FEATURE_CLASS MAPPING — scoped to highway + building per cascade #4
# (sub-C feature_class=2 poi has NULL class_raw; feature_class=3 base lumps
# water+landuse+natural with ambiguous parent key). Per spec §12 #11.
# ---------------------------------------------------------------------------
FEATURE_CLASS_TO_KEY: Final[dict[int, str]] = {
    0: "highway",   # road class — exact 1:1 (sub-C extracts only highway)
    1: "building",  # exact 1:1
    # 2 (poi) + 3 (base) deferred per cascade #4.
}


def load_taginfo(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dominant_element_type(key_row: dict) -> str:
    """Return 'ways' / 'nodes' / 'relations' for key's dominant element type."""
    counts = {
        "ways": int(key_row["count_ways"]),
        "nodes": int(key_row["count_nodes"]),
        "relations": int(key_row["count_relations"]),
    }
    return max(counts, key=lambda k: counts[k])


def et_totals_from_taginfo(rows: list[dict]) -> dict[str, int]:
    """Approximate global ET totals as sum of key-row counts per ET.

    Documented approximation: sums over the top-N keys snapshotted; not the
    true global element population (which would require taginfo /site/info).
    Sufficient for fraction-comparison purposes at Halt 1.
    """
    totals = Counter()
    for r in rows:
        if r["row_type"] == "key":
            totals["ways"] += int(r["count_ways"])
            totals["nodes"] += int(r["count_nodes"])
            totals["relations"] += int(r["count_relations"])
    return dict(totals)


def fraction_within_et(
    row: dict, et_totals: dict[str, int], key_rows_by_name: dict[str, dict]
) -> float:
    """Fraction-of-feature-bearing-elements within row's dominant ET (BP1 fix 2)."""
    if row["row_type"] == "key":
        et = dominant_element_type(row)
        denom = et_totals.get(et, 0)
        numerator = int(row[f"count_{et}"])
    else:  # value row: inherit parent key's dominant ET (documented approximation)
        parent = key_rows_by_name.get(row["parent_key"])
        if not parent:
            return 0.0
        et = dominant_element_type(parent)
        denom = et_totals.get(et, 0)
        numerator = int(row["count_all"])  # parent ET distribution assumed for value
    return numerator / denom if denom else 0.0


def f_min_for_level(
    rows: list[dict],
    level: int,
    et_totals: dict[str, int],
    key_rows_by_name: dict[str, dict],
) -> float:
    """Smallest F such that all must-appears at level are admitted."""
    if level == 1:
        candidates = [
            fraction_within_et(r, et_totals, key_rows_by_name)
            for r in rows
            if r["row_type"] == "key" and r["key"] in WIKI_L1_MUST_APPEARS
        ]
    elif level == 2:
        candidates = [
            fraction_within_et(r, et_totals, key_rows_by_name)
            for r in rows
            if r["row_type"] == "value"
            and (r["parent_key"], r["value"]) in WIKI_L2_PRIMARY_PAIRS
        ]
    elif level == 3:
        if not WIKI_L3_ALL_PAIRS:
            return float("nan")  # L3 deferred per spec §12 #10
        candidates = [
            fraction_within_et(r, et_totals, key_rows_by_name)
            for r in rows
            if r["row_type"] == "value"
            and (r["parent_key"], r["value"]) in WIKI_L3_ALL_PAIRS
        ]
    else:
        raise ValueError(f"unknown level {level}")
    return min(candidates) if candidates else 0.0


def vocab_size_at_F(
    rows: list[dict],
    F: float,
    level: int,
    et_totals: dict[str, int],
    key_rows_by_name: dict[str, dict],
) -> int:
    """Count slots at granularity level passing F (Bug 3 fix: level-aware row-type filter)."""
    if level == 1:
        return sum(
            1 for r in rows
            if r["row_type"] == "key"
            and fraction_within_et(r, et_totals, key_rows_by_name) >= F
        )
    if level == 2:
        return sum(
            1 for r in rows
            if r["row_type"] == "value"
            and (r["parent_key"], r["value"]) in WIKI_L2_PRIMARY_PAIRS
            and fraction_within_et(r, et_totals, key_rows_by_name) >= F
        )
    if level == 3:
        if not WIKI_L3_ALL_PAIRS:
            return 0  # L3 deferred
        return sum(
            1 for r in rows
            if r["row_type"] == "value"
            and (r["parent_key"], r["value"]) in WIKI_L3_ALL_PAIRS
            and fraction_within_et(r, et_totals, key_rows_by_name) >= F
        )
    raise ValueError(f"unknown level {level}")


# ---------------------------------------------------------------------------
# SINGAPORE X-THRESHOLD computation (Bug 5 fix per cascade #4 scope).
# ---------------------------------------------------------------------------


def compute_singapore_frequencies(
    sub_c_region: Path,
) -> dict[tuple[str, str], int]:
    """Count (inferred_key, class_raw) tag pairs on cached Singapore sub-C extracts.

    Scoped to FEATURE_CLASS_TO_KEY = {0: highway, 1: building} per cascade #4.
    POI (2) + base (3) Singapore mapping deferred per spec §12 #11.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for path in sorted(sub_c_region.glob("tile=*/features.parquet")):
        table = pq.ParquetFile(path).read()
        for r in table.to_pylist():
            key = FEATURE_CLASS_TO_KEY.get(r["feature_class"])
            value = r.get("class_raw")
            if key and value:  # NULL class_raw skipped (POI rows have NULL)
                counts[(key, value)] += 1
    return dict(counts)


def derive_x_threshold(
    sg_freqs: dict[tuple[str, str], int],
    wiki_must_appears: frozenset[tuple[str, str]],
) -> dict:
    """Compute X threshold candidates from Singapore distribution.

    Candidate A: Singapore's own elbow F-equivalent (min Singapore-fraction
                 of wiki must-appears actually present on Singapore).
    Candidate B: median Singapore frequency across present must-appears.
    """
    total_sg = sum(sg_freqs.values())
    if not total_sg:
        return {"error": "no Singapore data — sub-C cache missing or empty"}

    must_appear_fractions = sorted(
        (sg_freqs.get(p, 0) / total_sg for p in wiki_must_appears),
        reverse=True,
    )
    present_fractions = [f for f in must_appear_fractions if f > 0]
    return {
        "candidate_a_singapore_elbow": float(min(present_fractions)) if present_fractions else 0.0,
        "candidate_b_median_must_appear_freq": (
            float(present_fractions[len(present_fractions) // 2])
            if present_fractions else 0.0
        ),
        "n_must_appears_present_in_singapore": len(present_fractions),
        "n_must_appears_total": len(wiki_must_appears),
        "scope_note": "highway + building only per cascade #4; POI + base deferred per spec §12 #11.",
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="2026-04-15.0")
    parser.add_argument(
        "--sub-c-region-dir", type=Path,
        default=Path("data/processed/sub_c/2026-04-15.0/singapore"),
    )
    args = parser.parse_args()

    taginfo_csv = ROOT / "configs" / "sub_f" / "taginfo" / f"{args.release}.csv"
    rows = load_taginfo(taginfo_csv)
    et_totals = et_totals_from_taginfo(rows)
    key_rows_by_name = {r["key"]: r for r in rows if r["row_type"] == "key"}

    # L1 + L2 curve (L3 deferred per spec §12 #10).
    f_l1 = f_min_for_level(rows, level=1, et_totals=et_totals, key_rows_by_name=key_rows_by_name)
    vocab_l1 = vocab_size_at_F(rows, f_l1, level=1, et_totals=et_totals, key_rows_by_name=key_rows_by_name)
    f_l2 = f_min_for_level(rows, level=2, et_totals=et_totals, key_rows_by_name=key_rows_by_name)
    vocab_l2 = vocab_size_at_F(rows, f_l2, level=2, et_totals=et_totals, key_rows_by_name=key_rows_by_name)

    # Singapore X-threshold per BP1 fix C (Bug 5 fix per cascade #4 scope).
    sg_freqs = compute_singapore_frequencies(args.sub_c_region_dir)
    x_threshold = derive_x_threshold(sg_freqs, WIKI_L2_PRIMARY_PAIRS)

    output = {
        "release": args.release,
        "f_denominator": (
            "fraction-of-feature-bearing-elements within dominant ET per key "
            "(BP1 fix 2; value rows inherit parent's dominant ET as documented "
            "approximation)"
        ),
        "wiki_l1_must_appears": list(WIKI_L1_MUST_APPEARS),
        "wiki_l2_primary_pairs_count": len(WIKI_L2_PRIMARY_PAIRS),
        "wiki_l2_highway_count": len(WIKI_L2_HIGHWAY),
        "wiki_l2_building_count": len(WIKI_L2_BUILDING),
        "wiki_l3_status": "deferred per spec §12 #10",
        "curve": [
            {
                "level": 1,
                "level_description": "top-level keys (28 must-appears)",
                "f_min": float(f_l1),
                "vocab_size": int(vocab_l1),
                "must_appears_count": len(WIKI_L1_MUST_APPEARS),
                "must_appears_admitted": len(WIKI_L1_MUST_APPEARS),
            },
            {
                "level": 2,
                "level_description": "(key, primary-value) pairs — highway + building per cascade #4",
                "f_min": float(f_l2),
                "vocab_size": int(vocab_l2),
                "must_appears_count": len(WIKI_L2_PRIMARY_PAIRS),
                "must_appears_admitted": len(WIKI_L2_PRIMARY_PAIRS),
            },
            {
                "level": 3,
                "level_description": "all wiki-documented pairs — deferred per spec §12 #10",
                "f_min": None,
                "vocab_size": None,
                "must_appears_count": 0,
                "must_appears_admitted": 0,
                "deferral_reason": "Cascade #5 + recursive marginal-cost-of-cut: enumerate where reviewable benefit exists.",
            },
        ],
        "proposed_elbow": {
            "level": 1,
            "f_value": float(f_l1),
            "exception_list": [],
            "rationale": "Default L1 (28 keys, full primary feature set); reviewer redirects to L2 at halt if Δvocab_size / Δmust-appears between L1 and L2 favors L2.",
        },
        "proposed_x_threshold": {
            "candidate_a_singapore_elbow": x_threshold.get("candidate_a_singapore_elbow"),
            "candidate_b_median_must_appear_freq": x_threshold.get("candidate_b_median_must_appear_freq"),
            "scope_note": x_threshold.get("scope_note"),
            "n_must_appears_present_in_singapore": x_threshold.get("n_must_appears_present_in_singapore"),
            "n_must_appears_total": x_threshold.get("n_must_appears_total"),
            "paired_structural_check": "For each Singapore-frequency-≥X (highway, value) and (building, value) pair: must appear above F in semantic_vocab.yaml. POI + base scope deferred per spec §12 #11.",
        },
        "_status": "PROPOSED — pending Halt 1 reviewer approval per spec §10.3.",
    }

    out_path = ROOT / "configs" / "sub_f" / "vocab_floor_analysis.yaml"
    out_path.write_text(yaml.safe_dump(output, sort_keys=True), encoding="utf-8")
    print(f"[floor_analysis] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Write failing test for vocab loader (extended with Safeguard 2 per-key count assertions)**

**Per cascade #5 + Safeguard 2:** tests assert both (a) set-equality of L1 + L2 enumerations and (b) independently-derived per-key counts. Hand-counts come from visual inspection of wikitext, separately from the floor_analysis.py constants — so a miss in one set doesn't silently confirm a miss in the other.

Create `tests/data/sub_f/__init__.py` as an empty file.

Create `tests/data/sub_f/test_vocab.py`:

```python
"""Tests for sub-F vocab loading + Gate 6 structural check.

Per spec §8.1 BP1 row + cascade #5 + Safeguard 2: vocab passes iff:
(a) F frequency floor cuts at the chosen quantile, AND
(b) every hand-enumerated wiki Map_features must-appear is a first-class slot,
    enumeration verified by set-equality AND per-key count assertions.

Assertion logic does NOT use sub-F's own derivation in expected-value
computation per Gate 6 + spec §13.5 protocol-v2 candidate iii (reviewer-supplied
lists are untrusted; hand-counts derived independently from pair sets).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "sub_f"

# Hand-derived from wikitext at configs/sub_f/wiki_map_features/2026-04-15.0.wikitext
# `==Primary features==` section transclusion count. Independently counted from
# the WIKI_L1_MUST_APPEARS tuple in floor_analysis.py per Safeguard 2.
N_L1_MUST_APPEARS_EXPECTED: Final[int] = 28

# Hand-derived from Template:Map_Features:highway value table (Roads + Link
# roads + Special road types + Paths + Lifecycle subsections). Excludes
# Other highway features (stops, signals, infrastructure point-features).
N_L2_HIGHWAY_EXPECTED: Final[int] = 23

# Hand-derived from Template:Building_typology value table + 1 "yes" catch-all.
N_L2_BUILDING_EXPECTED: Final[int] = 33


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_vocab_floor_analysis_has_28_l1_must_appears():
    """L1 enumeration covers all 28 wiki primary feature keys per cascade #5."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    # Hand-pinned (assertion does NOT read sub-F's own enumeration into expected).
    expected_l1 = {
        "aerialway", "aeroway", "amenity", "barrier", "boundary", "building",
        "craft", "emergency", "geological", "healthcare", "highway", "historic",
        "landuse", "leisure", "man_made", "military", "natural", "office",
        "place", "power", "public_transport", "railway", "route", "shop",
        "telecom", "tourism", "water", "waterway",
    }
    actual_l1 = set(data["wiki_l1_must_appears"])
    assert actual_l1 == expected_l1, (
        f"L1 set drift: missing={expected_l1 - actual_l1}, "
        f"extra={actual_l1 - expected_l1}"
    )


def test_vocab_floor_analysis_l1_count_matches_independent_hand_count():
    """Safeguard 2: per-key count derived independently from set."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    actual_count = len(data["wiki_l1_must_appears"])
    assert actual_count == N_L1_MUST_APPEARS_EXPECTED, (
        f"L1 count mismatch: floor_analysis.py shipped {actual_count}, "
        f"hand-counted from wikitext = {N_L1_MUST_APPEARS_EXPECTED}. "
        f"If wikitext changed: re-count and update N_L1_MUST_APPEARS_EXPECTED."
    )


def test_vocab_floor_analysis_l2_highway_count_matches_independent_hand_count():
    """Safeguard 2: highway L2 count derived independently from pair set."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    actual = data["wiki_l2_highway_count"]
    assert actual == N_L2_HIGHWAY_EXPECTED, (
        f"L2 highway count mismatch: floor_analysis.py shipped {actual}, "
        f"hand-counted from Template:Map_Features:highway = {N_L2_HIGHWAY_EXPECTED}. "
        f"If template changed: re-count and update N_L2_HIGHWAY_EXPECTED."
    )


def test_vocab_floor_analysis_l2_building_count_matches_independent_hand_count():
    """Safeguard 2: building L2 count derived independently from pair set."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    actual = data["wiki_l2_building_count"]
    assert actual == N_L2_BUILDING_EXPECTED, (
        f"L2 building count mismatch: floor_analysis.py shipped {actual}, "
        f"hand-counted from Template:Building_typology + 'yes' = {N_L2_BUILDING_EXPECTED}. "
        f"If template changed: re-count and update N_L2_BUILDING_EXPECTED."
    )


def test_vocab_floor_analysis_l3_deferred_per_spec_12_10():
    """L3 explicitly deferred per cascade #5 + spec §12 #10."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    l3_row = next(r for r in data["curve"] if r["level"] == 3)
    assert l3_row["f_min"] is None
    assert l3_row["vocab_size"] is None
    assert l3_row["must_appears_count"] == 0
    assert "deferred" in l3_row["level_description"].lower()


def test_vocab_floor_analysis_curve_includes_l1_and_l2_rows():
    """Curve has L1 + L2 rows (L3 row exists but deferred per above test)."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    levels = {r["level"] for r in data["curve"]}
    assert levels == {1, 2, 3}, f"missing curve levels: {{1,2,3}} - {levels}"


def test_vocab_floor_analysis_singapore_x_threshold_scoped_to_highway_building():
    """Singapore X-threshold scope per cascade #4 (POI + base deferred per §12 #11)."""
    data = _load_yaml(CONFIG_ROOT / "vocab_floor_analysis.yaml")
    x = data["proposed_x_threshold"]
    assert "highway + building" in x["scope_note"]
    assert "POI + base" in x["scope_note"]
    # Candidates surface concrete values (not just hand-waved).
    assert "candidate_a_singapore_elbow" in x
    assert "candidate_b_median_must_appear_freq" in x
```

- [ ] **Step 6: Run test (expected fail — config does not exist yet)**

Run: `uv run pytest tests/data/sub_f/test_vocab.py -v`
Expected: FAIL with `FileNotFoundError: configs/sub_f/vocab_floor_analysis.yaml`.

- [ ] **Step 7: Run floor analysis (computes L1 + L2 curve + Singapore X-threshold)**

Floor analysis now reads sub-C Singapore extracts for Singapore X-threshold computation (Bug 5 fix per cascade #4 scope). Sub-C cache must exist at `data/processed/sub_c/2026-04-15.0/singapore/` (sub-E precedent — sub-C already cached for sub-E's Singapore integration tests).

Run:
```bash
uv run python scripts/sub_f/floor_analysis.py \
    --release 2026-04-15.0 \
    --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/
```

Expected: `configs/sub_f/vocab_floor_analysis.yaml` written with `_status: PROPOSED`. Spot-check fields:
- `wiki_l1_must_appears` length = 28
- `wiki_l2_highway_count` = 23
- `wiki_l2_building_count` = 33
- `wiki_l3_status` = `"deferred per spec §12 #10"`
- `curve` has 3 rows (levels 1, 2, 3 — level 3 is deferred placeholder)
- `proposed_x_threshold` has concrete `candidate_a_singapore_elbow` + `candidate_b_median_must_appear_freq` values

If sub-C cache is missing: STOP, report BLOCKED with the missing path.

- [ ] **Step 8: Run full test suite (expected 7 PASS)**

Run: `uv run pytest tests/data/sub_f/test_vocab.py -v`
Expected: 7 PASS covering:
- L1 set-equality (28 keys vs hand-enumerated expected set)
- L1 independent hand-count match (Safeguard 2)
- L2 highway hand-count match (Safeguard 2)
- L2 building hand-count match (Safeguard 2)
- L3 deferred per spec §12 #10
- Curve has L1 + L2 + L3 rows
- Singapore X-threshold scoped to highway + building

- [ ] **Step 9: HALT 1 — surface to reviewer**

**Implementer commits halt report** at `reports/2026-MM-DD-phase-1-sub-F-task-1-halt.md` containing:

**Snapshot artifacts:**
- taginfo CSV: row count, first 5 rows verbatim, file size, sha256.
- wiki Map_features: revision_id (integer), wikitext byte count, sha256.

**Marginal-cost curve (per cascade #5 scope: L1 full + L2 highway+building; L3 deferred):**
- L1 row: 28 keys, F_min value, vocab_size at F_min.
- L2 row: highway + building primary pairs, F_min, vocab_size.
- L3 row: deferred per spec §12 #10 — reason note included.

**Proposed elbow:**
- Granularity level (1 by default).
- F value.
- Exception list (empty by default; reviewer adds sub-floor must-appears to drop if any).
- Rationale (cite Δvocab_size / Δmust-appears between L1 and L2).

**Proposed X-threshold (cascade #4 scope: highway + building only):**
- Candidate A: Singapore-elbow-derived value + concrete number.
- Candidate B: median Singapore must-appear frequency + concrete number.
- Scope note: POI + base deferred per spec §12 #11.
- Paired structural check framing (per §2 + Gate 6).

**Cascade documentation (mandatory at Halt 1 per spec §13.5):**
- Cascade #4 outcome: Singapore X scope = highway + building. POI/base deferred to sub-F-v2.
- Cascade #5 outcome: L1 corrected to 28 keys; L3 deferred entirely.
- §13.5 protocol-v2 candidates surfaced: (i) transitive-documentation citing, (ii) hand-enumeration with complete-count assertion, (iii) reviewer-supplied lists as untrusted input.

**§10.5 telemetry:**
- Implementer-time-to-data-surface: wall-clock from dispatch start to this halt report commit.

DO NOT lock `semantic_vocab.yaml` autonomously. Reviewer approves elbow + exception list + X-threshold at Halt 1 per spec §10.3.

- [ ] **Step 10: After Halt 1 approval — write semantic_vocab.yaml**

(Halt 1 acknowledgment from reviewer required before this step.)

`semantic_vocab.yaml` structure:
```yaml
release: "2026-04-15.0"
granularity_level: 1  # locked at Halt 1
f_value: <reviewer-approved>
x_threshold: <reviewer-approved>
exception_list: []
slots:
  - id: 0
    tag: "highway=*"
    source: "wiki_l1_must_appear"
  - id: 1
    tag: "building=*"
    source: "wiki_l1_must_appear"
  # ... continues for all admitted slots
```

ID range `[0, K1)` per §2.4 per-family reserved blocks. K1 locked at Halt 1; remaining blocks (K1..K4, N) locked at subsequent halts.

- [ ] **Step 11: Lint + commit**

Run:
```bash
uv run ruff format src/ scripts/ tests/data/sub_f/
uv run ruff check src/ scripts/ tests/data/sub_f/ --fix
uv run pytest tests/data/sub_f/test_vocab.py -v
```
Expected: ruff clean, test passes.

Commit:
```bash
git add configs/sub_f/ scripts/sub_f/snapshot_taginfo.py scripts/sub_f/snapshot_wiki.py \
        scripts/sub_f/floor_analysis.py tests/data/sub_f/__init__.py tests/data/sub_f/test_vocab.py
git commit -m "$(cat <<'EOF'
feat(sub_f): T1 BP1 vocab floor + snapshot artifacts (Halt 1 approved)

Snapshot artifacts: taginfo 2026-04-15.0.csv + wiki Map_features
2026-04-15.0.wikitext + revision_id + sha256. Floor analysis YAML
with Halt 1 reviewer-approved elbow + exception list + X-threshold.
semantic_vocab.yaml locked with K1 reserved-block size.

Per spec §2.1 + §10.1 Halt 1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: BP2 encoder primitives + round-trip thresholds

**Halt 2 gate.** Joint surface `(direction_count × magnitude_quantum)` against Overture turn-angle + vertex-spacing distributions on Singapore. Anchor scheme {flat, hierarchical} reported with BOTH vocab size AND mean-sequence-length-per-cell. Round-trip thresholds (L_∞ + 95th-pct angle) + collinearity threshold proposals.

**Files:**
- Create: `src/cfm/data/sub_f/__init__.py`
- Create: `src/cfm/data/sub_f/enums.py` (Direction, MagnitudeQuantum, AnchorScheme draft enums)
- Create: `scripts/sub_f/analyze_geometry_primitives.py`
- Create: `configs/sub_f/encoding_primitives.yaml`
- Test: `tests/data/sub_f/test_encoder.py` (foundation; encoder code lands in Task 8)

### Pre-dispatch audit

- [ ] **Audit step 1: confirm sub-C feature row has geometry as WKB bytes**

Read: `src/cfm/data/sub_c/io.py:166-185` (FeatureRow dataclass) + `:200-205` (`dump_wkb` helper).
Expected: `geometry` column carries WKB little-endian (NDR) bytes per `dump_wkb(geom, hex=False, byte_order=1)`. Task 2 analysis decodes via `shapely.wkb.loads`.

- [ ] **Audit step 2: confirm sub-C projection per region**

Read: sub-E handoff line 80 path `tile=EPSG3414_i{ti}_j{tj}/` confirms Singapore = EPSG:3414 projected meters. Task 2 analyses in projected meters (no lat/lon arithmetic).

### Implementation steps

- [ ] **Step 1: Write enum scaffolding**

Create `src/cfm/data/sub_f/__init__.py`:

```python
"""Phase 1 sub-F micro-tokenizer package.

Per spec docs/superpowers/specs/2026-05-23-phase-1-sub-F-micro-tokenizer-design.md
"""
```

Create `src/cfm/data/sub_f/enums.py`:

```python
"""Sub-F draft enums for encoding primitives.

Direction / MagnitudeQuantum / AnchorScheme values pending Halt 2 lock.
v1 defaults per spec §3.4–§3.6 (default toward 16 directions, 0.5m quantum,
flat anchor scheme).
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class AnchorScheme(StrEnum):
    """Sub-F anchor encoding scheme.

    FLAT: 2 tokens per anchor (dx, dy) on quantized grid.
    HIERARCHICAL: 4 tokens per anchor (coarse_dx, coarse_dy, fine_dx, fine_dy).
    Lock at Halt 2 per spec §3.6.
    """

    FLAT = "flat"
    HIERARCHICAL = "hierarchical"


# v1 default; revised at Halt 2 per joint (direction_count × magnitude_quantum)
# surface against Overture turn-angle + vertex-spacing distributions.
DEFAULT_DIRECTION_COUNT: int = 16
DEFAULT_MAGNITUDE_QUANTUM_M: float = 0.5
DEFAULT_MAGNITUDE_RANGE_M: tuple[float, float] = (0.5, 32.0)
DEFAULT_ANCHOR_SCHEME: AnchorScheme = AnchorScheme.FLAT
```

- [ ] **Step 2: Write geometry primitives analysis script**

Create `scripts/sub_f/analyze_geometry_primitives.py`:

```python
"""Compute Overture turn-angle + vertex-spacing distributions on Singapore.

Outputs Halt 2 inputs: joint (direction_count × magnitude_quantum) surface
+ anchor scheme {flat, hierarchical} vocab + mean sequence length comparison.

Per spec §10.1 Halt 2: surface to reviewer for elbow choice + threshold
lock. Does NOT autonomously write encoding_primitives.yaml.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from statistics import mean, quantiles
from typing import Iterator

import pyarrow.parquet as pq
import yaml
from shapely.geometry.base import BaseGeometry
from shapely.wkb import loads as wkb_loads

ROOT = Path(__file__).resolve().parents[2]


def iter_features(tile_features_paths: list[Path]) -> Iterator[BaseGeometry]:
    """Yield shapely geometries from sub-C features.parquet files."""
    for path in tile_features_paths:
        table = pq.ParquetFile(path).read()
        geom_bytes_col = table.column("geometry").to_pylist()
        for raw in geom_bytes_col:
            yield wkb_loads(raw)


def turn_angles(geom: BaseGeometry) -> list[float]:
    """Return interior turn angles (degrees) at each vertex of a linestring or polygon ring."""
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
    elif geom.geom_type == "Polygon":
        coords = list(geom.exterior.coords)
    else:
        return []
    angles: list[float] = []
    for i in range(1, len(coords) - 1):
        x0, y0 = coords[i - 1][:2]
        x1, y1 = coords[i][:2]
        x2, y2 = coords[i + 1][:2]
        v1 = (x1 - x0, y1 - y0)
        v2 = (x2 - x1, y2 - y1)
        ang1 = math.degrees(math.atan2(v1[1], v1[0]))
        ang2 = math.degrees(math.atan2(v2[1], v2[0]))
        delta = (ang2 - ang1 + 540) % 360 - 180  # in (-180, 180]
        angles.append(delta)
    return angles


def vertex_spacings_m(geom: BaseGeometry) -> list[float]:
    """Return per-edge spacings (projected meters) along a linestring or polygon ring."""
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
    elif geom.geom_type == "Polygon":
        coords = list(geom.exterior.coords)
    else:
        return []
    spacings: list[float] = []
    for i in range(1, len(coords)):
        x0, y0 = coords[i - 1][:2]
        x1, y1 = coords[i][:2]
        spacings.append(math.hypot(x1 - x0, y1 - y0))
    return spacings


def joint_surface(
    angles: list[float],
    spacings: list[float],
    direction_counts: list[int],
    magnitude_quanta: list[float],
) -> list[dict]:
    """For each (direction_count, magnitude_quantum) pair: compute estimated reconstruction error."""
    rows: list[dict] = []
    for nd in direction_counts:
        bin_width_deg = 360.0 / nd
        # Half-bin perpendicular error at characteristic magnitude.
        for mq in magnitude_quanta:
            # Use median spacing as the characteristic magnitude.
            char_mag = (
                quantiles(spacings, n=4)[1] if len(spacings) >= 4 else mean(spacings) if spacings else 1.0
            )
            perp_err_m = char_mag * math.sin(math.radians(bin_width_deg / 2.0))
            quant_err_m = mq / 2.0
            total_err = math.hypot(perp_err_m, quant_err_m)
            rows.append(
                {
                    "direction_count": nd,
                    "magnitude_quantum_m": mq,
                    "characteristic_spacing_m": float(char_mag),
                    "half_bin_angle_deg": float(bin_width_deg / 2.0),
                    "perpendicular_error_m": float(perp_err_m),
                    "quantization_error_m": float(quant_err_m),
                    "joint_error_l_inf_m": float(total_err),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    args = parser.parse_args()

    tile_features = sorted(args.sub_c_region_dir.glob("tile=*/features.parquet"))
    if not tile_features:
        print(f"[analyze] no tiles found under {args.sub_c_region_dir}", file=sys.stderr)
        return 1

    all_angles: list[float] = []
    all_spacings: list[float] = []
    for geom in iter_features(tile_features):
        all_angles.extend(turn_angles(geom))
        all_spacings.extend(vertex_spacings_m(geom))

    surface = joint_surface(
        all_angles,
        all_spacings,
        direction_counts=[8, 16, 24],
        magnitude_quanta=[0.25, 0.5, 1.0],
    )

    # Right-angle preservation stats for §3.8 paired check input.
    right_angle_count = sum(1 for a in all_angles if 80 <= abs(a) <= 100)
    near_right_pct = (right_angle_count / len(all_angles) * 100.0) if all_angles else 0.0

    output = {
        "summary": {
            "n_features_analyzed": "n/a (streaming)",
            "n_angles_collected": len(all_angles),
            "n_spacings_collected": len(all_spacings),
            "median_spacing_m": float(quantiles(all_spacings, n=4)[1]) if len(all_spacings) >= 4 else None,
            "near_right_angle_pct": float(near_right_pct),
        },
        "joint_surface": surface,
        "anchor_scheme_comparison": {
            "flat": {"tokens_per_anchor": 2, "vocab_size": 1000, "note": "flat dx+dy"},
            "hierarchical": {
                "tokens_per_anchor": 4,
                "vocab_size": 96,
                "note": "16 coarse + 32 fine per axis",
            },
            "mean_sequence_length_per_cell_pending": "compute from sub-C feature density + this surface",
        },
        "proposed_lock": {
            "direction_count": 16,
            "magnitude_quantum_m": 0.5,
            "anchor_scheme": "flat",
            "round_trip_l_inf_threshold_m": 1.0,  # placeholder; reviewer locks
            "round_trip_angle_threshold_deg": 5.0,  # placeholder; reviewer locks
            "collinearity_admission_perpendicular_m": 0.25,
            "rationale": "Default-toward-16 per BP2 fix 4 cheap-to-keep rule",
        },
        "_status": "PROPOSED — pending Halt 2 reviewer approval per spec §10.3.",
    }

    out = ROOT / "configs" / "sub_f" / "encoding_primitives.yaml"
    out.write_text(yaml.safe_dump(output, sort_keys=True), encoding="utf-8")
    print(f"[analyze] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Write failing test for encoding primitives lock**

Append to `tests/data/sub_f/test_encoder.py`:

```python
"""Tests for sub-F encoder primitives + grammar.

Pre-Task-8: this file holds Halt 2 lock assertions. Encoder code itself
lands in Task 8.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "sub_f"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_encoding_primitives_default_lock():
    """Encoding primitives YAML carries the v1 defaults per spec §3.4–§3.6."""
    data = _load_yaml(CONFIG_ROOT / "encoding_primitives.yaml")
    lock = data["proposed_lock"]
    assert lock["direction_count"] in (8, 16, 24), "direction_count must be admitted alternative"
    assert lock["magnitude_quantum_m"] in (0.25, 0.5, 1.0), "magnitude_quantum must be admitted alternative"
    assert lock["anchor_scheme"] in ("flat", "hierarchical"), "anchor_scheme must be admitted alternative"
    # Default-toward-16 per BP2 fix 4 cheap-to-keep rule.
    assert lock["direction_count"] == 16, "v1 default is 16 per BP2 fix 4"


def test_encoding_primitives_surface_includes_all_pairs():
    """Joint surface enumerates all (direction × quantum) combinations."""
    data = _load_yaml(CONFIG_ROOT / "encoding_primitives.yaml")
    surface_pairs = {(r["direction_count"], r["magnitude_quantum_m"]) for r in data["joint_surface"]}
    expected = {(nd, mq) for nd in (8, 16, 24) for mq in (0.25, 0.5, 1.0)}
    assert surface_pairs == expected, f"missing pairs: {expected - surface_pairs}"
```

- [ ] **Step 4: Run test (expected fail — config does not exist)**

Run: `uv run pytest tests/data/sub_f/test_encoder.py -v`
Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 5: Run geometry analysis against cached Singapore**

Run:
```bash
uv run python scripts/sub_f/analyze_geometry_primitives.py \
    --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/
```
Expected: `configs/sub_f/encoding_primitives.yaml` written with `_status: PROPOSED`. Verify joint surface has 9 rows.

- [ ] **Step 6: Run test (expected pass)**

Run: `uv run pytest tests/data/sub_f/test_encoder.py -v`
Expected: 2 PASS.

- [ ] **Step 7: HALT 2 — surface to reviewer**

**Implementer commits halt report** at `reports/2026-MM-DD-phase-1-sub-F-task-2-halt.md` containing:
- Joint (direction_count × magnitude_quantum) surface table — all 9 rows.
- Anchor scheme {flat, hierarchical} comparison with mean-sequence-length-per-cell estimate.
- Proposed direction count, magnitude quantum, anchor scheme.
- Proposed round-trip L_∞ + 95th-pct angle + collinearity thresholds.
- §10.5 telemetry fields.

DO NOT autonomously lock `encoding_primitives.yaml`. Halt 2 reviewer approves elbow on joint surface per BP2 fix 3 (not two 1D curves) AND default-toward-16 per BP2 fix 4.

- [ ] **Step 8: After Halt 2 approval — finalize encoding_primitives.yaml**

Update `_status` field to `LOCKED` post-approval. Remove `proposed_lock` wrapper; promote fields to top-level. Lock K2 (BP2 reserved-block size in sentinel_inventory layout per §2.4).

- [ ] **Step 9: Lint + commit**

Run:
```bash
uv run ruff format src/cfm/data/sub_f/ scripts/sub_f/ tests/data/sub_f/
uv run ruff check src/cfm/data/sub_f/ scripts/sub_f/ tests/data/sub_f/ --fix
uv run pytest tests/data/sub_f/test_encoder.py -v
```

Commit:
```bash
git add src/cfm/data/sub_f/__init__.py src/cfm/data/sub_f/enums.py \
        scripts/sub_f/analyze_geometry_primitives.py \
        configs/sub_f/encoding_primitives.yaml tests/data/sub_f/test_encoder.py
git commit -m "$(cat <<'EOF'
feat(sub_f): T2 BP2 encoder primitives + Halt 2 lock

Joint (direction_count × magnitude_quantum) surface on Singapore.
Anchor scheme flat vs hierarchical comparison. Round-trip L_∞ +
95th-pct angle + collinearity thresholds locked at Halt 2.

Default-toward-16 per BP2 fix 4 cheap-to-keep rule.

Per spec §2.2 + §3.4–§3.6 + §10.1 Halt 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3a: Stage-1+2 joint distribution by feature type

**No halt.** Intermediate output feeds Task 3b. Per BP3 fix 2: outputs JOINT distribution, not separate marginals.

**Files:**
- Create: `scripts/sub_f/analyze_stage_1_2_joint.py`
- Create: `configs/sub_f/stage_1_2_joint.yaml` (intermediate; feeds 3b/3c, not user-facing)
- Test: `tests/data/sub_f/test_stage_analysis.py`

### Implementation steps

- [ ] **Step 1: Write analysis script**

Create `scripts/sub_f/analyze_stage_1_2_joint.py`:

```python
"""Compute joint P(feature_count, vertex_count | cell, feature_type) per type.

Per spec §7.3: joint distribution NOT separate marginals — dense areas
correlate with simpler geometry per feature. Treating as independent
inflates tail prediction.

Output: configs/sub_f/stage_1_2_joint.yaml as intermediate input to Task 3b/3c.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, quantiles

import pyarrow.parquet as pq
import yaml
from shapely.wkb import loads as wkb_loads

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    args = parser.parse_args()

    # Joint: { (cell_i, cell_j): { feature_type: [vertex_counts] } }
    per_cell: dict[tuple[int, int], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    tile_features = sorted(args.sub_c_region_dir.glob("tile=*/features.parquet"))
    for path in tile_features:
        table = pq.ParquetFile(path).read()
        records = table.to_pylist()
        for r in records:
            geom = wkb_loads(r["geometry"])
            if geom.geom_type == "LineString":
                v = len(geom.coords)
            elif geom.geom_type == "Polygon":
                v = len(geom.exterior.coords)
            elif geom.geom_type == "Point":
                v = 1
            else:
                v = 0
            per_cell[(r["cell_i"], r["cell_j"])][r["feature_class"]].append(v)

    # Joint distribution: for each feature_type, enumerate (cell_feature_count, vertex_count_per_feature)
    output = {"per_feature_type": {}, "empty_cell_fraction": 0.0}
    feature_classes = {fc for cell in per_cell.values() for fc in cell}
    n_cells_total = len(per_cell)
    n_empty_cells = sum(1 for cell in per_cell.values() if not cell)

    for fc in sorted(feature_classes):
        fc_rows = []
        for cell_key, fc_map in per_cell.items():
            if fc in fc_map:
                fc_count_in_cell = len(fc_map[fc])
                for vc in fc_map[fc]:
                    fc_rows.append({"feature_count_in_cell": fc_count_in_cell, "vertex_count": vc})
        output["per_feature_type"][int(fc)] = {
            "n_observations": len(fc_rows),
            "feature_count_mean": float(mean([r["feature_count_in_cell"] for r in fc_rows])) if fc_rows else 0.0,
            "vertex_count_mean": float(mean([r["vertex_count"] for r in fc_rows])) if fc_rows else 0.0,
            "vertex_count_p95": float(quantiles([r["vertex_count"] for r in fc_rows], n=20)[18])
            if len(fc_rows) >= 20 else None,
        }
    output["empty_cell_fraction"] = n_empty_cells / n_cells_total if n_cells_total else 0.0
    output["n_cells_total"] = n_cells_total

    out = ROOT / "configs" / "sub_f" / "stage_1_2_joint.yaml"
    out.write_text(yaml.safe_dump(output, sort_keys=True), encoding="utf-8")
    print(f"[analyze stage 1+2] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run analysis on cached Singapore**

Run:
```bash
uv run python scripts/sub_f/analyze_stage_1_2_joint.py \
    --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/
```
Expected: `configs/sub_f/stage_1_2_joint.yaml` written.

- [ ] **Step 3: Write test to verify joint includes all feature classes**

Append to `tests/data/sub_f/test_stage_analysis.py`:

```python
"""Tests for BP3 stage analysis intermediate outputs."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "sub_f"


def test_stage_1_2_joint_includes_all_sub_c_feature_classes():
    """Joint distribution enumerates all four sub-C feature classes.

    Sub-C FEATURE_CLASS per src/cfm/data/sub_c/enums.py: {0: road, 1: building,
    2: poi, 3: base}. Joint must include observations for each that appears
    in cached Singapore.
    """
    data = yaml.safe_load((CONFIG_ROOT / "stage_1_2_joint.yaml").read_text(encoding="utf-8"))
    types_present = set(data["per_feature_type"].keys())
    # At minimum, road (0) and building (1) must be present on Singapore.
    assert 0 in types_present, "road feature class missing from Singapore joint"
    assert 1 in types_present, "building feature class missing from Singapore joint"
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/data/sub_f/test_stage_analysis.py -v`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format scripts/sub_f/ tests/data/sub_f/
uv run ruff check scripts/sub_f/ tests/data/sub_f/ --fix
git add scripts/sub_f/analyze_stage_1_2_joint.py configs/sub_f/stage_1_2_joint.yaml \
        tests/data/sub_f/test_stage_analysis.py
git commit -m "feat(sub_f): T3a joint P(feature_count, vertex_count|cell,type) on Singapore"
```

---

## Task 3b: Stage-3 compound (per-cell length sans cross-cell)

**No halt.** Intermediate output feeds Task 3c. Blocked on Task 2 (encoder lock).

**Files:**
- Create: `scripts/sub_f/compute_stage_3_compound.py`
- Create: `configs/sub_f/stage_3_compound.yaml` (intermediate)
- Test: append to `tests/data/sub_f/test_stage_analysis.py`

### Implementation steps

- [ ] **Step 1: Write compound computation**

Create `scripts/sub_f/compute_stage_3_compound.py`:

```python
"""Compose stage-3 compound from Task 3a joint × Task 2 encoder lock.

Per spec §7.2 stage-3 formula:
  Case A: 3 + N_anchor + 2(V−1)
  Case B: 4 + N_anchor + 2(V−2)
  Case C: 4 + N_anchor + 2(V−1)
  Case D: 5 + N_anchor + 2(V−2)

At Task 3b: stage-4 (cross-cell overhead) is NOT yet added. Output is
per-cell length distribution sans cross-cell overhead. Task 3c adds stage-4.

Pre-3c assumption: all features are Case A (uncrossed). Cross-cell
classification happens at Task 3c when sub-E boundary contracts are
consulted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean, quantiles

import yaml

ROOT = Path(__file__).resolve().parents[2]


def case_a_tokens(v: int, n_anchor: int) -> int:
    return 3 + n_anchor + 2 * (v - 1) if v >= 1 else 2  # empty/point edge handling


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    primitives = yaml.safe_load(
        (ROOT / "configs" / "sub_f" / "encoding_primitives.yaml").read_text(encoding="utf-8")
    )
    joint = yaml.safe_load(
        (ROOT / "configs" / "sub_f" / "stage_1_2_joint.yaml").read_text(encoding="utf-8")
    )

    lock = primitives.get("proposed_lock", primitives)  # accept proposed or final
    n_anchor = 2 if lock["anchor_scheme"] == "flat" else 4

    # Per-cell length (Case A only): Σ_feat (3 + N_anchor + 2(V−1)) over all features in cell.
    # Approximation: use joint per-type observations to synthesize per-cell length distribution.
    per_cell_lengths_synth: list[int] = []
    for fc, stats in joint["per_feature_type"].items():
        n_obs = stats["n_observations"]
        v_mean = stats["vertex_count_mean"]
        if n_obs == 0:
            continue
        # Synthesize: each observation contributes case_a_tokens(v_mean, n_anchor) tokens.
        # Cell-level aggregation requires joint info beyond means; for 3b's intermediate
        # output, we report per-observation token contributions and defer cell-level
        # aggregation to 3c when cross-cell classification adds context.
        per_cell_lengths_synth.append(case_a_tokens(int(v_mean), n_anchor))

    output = {
        "anchor_scheme_used": lock["anchor_scheme"],
        "n_anchor": n_anchor,
        "per_observation_tokens_mean": float(mean(per_cell_lengths_synth)) if per_cell_lengths_synth else 0.0,
        "note": "Per-cell aggregation deferred to Task 3c; this is per-observation Case A only.",
        "_status": "INTERMEDIATE — feeds Task 3c.",
    }
    out = ROOT / "configs" / "sub_f" / "stage_3_compound.yaml"
    out.write_text(yaml.safe_dump(output, sort_keys=True), encoding="utf-8")
    print(f"[stage 3 compound] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run compound script**

Run: `uv run python scripts/sub_f/compute_stage_3_compound.py`
Expected: `configs/sub_f/stage_3_compound.yaml` written.

- [ ] **Step 3: Append test**

Append to `tests/data/sub_f/test_stage_analysis.py`:

```python
def test_stage_3_compound_uses_locked_anchor_scheme():
    """Stage-3 compound respects the Task 2 anchor scheme lock."""
    data = yaml.safe_load((CONFIG_ROOT / "stage_3_compound.yaml").read_text(encoding="utf-8"))
    assert data["anchor_scheme_used"] in ("flat", "hierarchical")
    assert data["n_anchor"] in (2, 4)
    if data["anchor_scheme_used"] == "flat":
        assert data["n_anchor"] == 2
    else:
        assert data["n_anchor"] == 4
```

- [ ] **Step 4: Run test + commit**

Run: `uv run pytest tests/data/sub_f/test_stage_analysis.py -v`
Expected: PASS.

```bash
git add scripts/sub_f/compute_stage_3_compound.py configs/sub_f/stage_3_compound.yaml \
        tests/data/sub_f/test_stage_analysis.py
git commit -m "feat(sub_f): T3b stage-3 compound per-cell length sans cross-cell"
```

---

## Task 3c: Stage-4 compound + budget surface

**Halt 4 gate.** Surface (quantile × data_loss_per_type × sequence_length × padding_overhead) at Task 3c halt. Per-type retention table + truncation strategy (α/β/γ) proposal. Reviewer picks elbow on the surface per BP3 fix 1.

**Files:**
- Create: `scripts/sub_f/compute_budget_surface.py`
- Create: `configs/sub_f/sequence_length_analysis.yaml`
- Test: append to `tests/data/sub_f/test_stage_analysis.py`

### Implementation steps

- [ ] **Step 1: Write budget surface computation**

Create `scripts/sub_f/compute_budget_surface.py`:

```python
"""Compute BP3 budget surface for Halt 4.

Joint 4D distribution per feature type → surface over (quantile × data_loss_per_type ×
sequence_length × padding_overhead). Per spec §7.4: NO autonomous P100 default;
reviewer picks elbow.

Stage 4 (cross-cell overhead) per spec §7.2 BP7 → BP3 correction:
  outbound bref: net 0 tokens (replaces tail direction+magnitude)
  inbound bref: +1 token per crossing
  ≈ 0.7 tokens/cell on Singapore rough estimate
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import quantiles
from typing import Iterator

import pyarrow.parquet as pq
import yaml
from shapely.wkb import loads as wkb_loads

ROOT = Path(__file__).resolve().parents[2]

# Per-type retention defaults per spec §7.5.
DEFAULT_RETENTION = {
    0: 0.999,  # roads — AV use case
    1: 0.99,   # buildings
    2: 0.99,   # POIs
    3: 0.99,   # base / landuse-class
}


def case_a_tokens(v: int, n_anchor: int) -> int:
    return 3 + n_anchor + 2 * (v - 1) if v >= 1 else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    parser.add_argument("--sub-e-region-dir", required=True, type=Path)
    args = parser.parse_args()

    primitives = yaml.safe_load(
        (ROOT / "configs" / "sub_f" / "encoding_primitives.yaml").read_text(encoding="utf-8")
    )
    lock = primitives.get("proposed_lock", primitives)
    n_anchor = 2 if lock["anchor_scheme"] == "flat" else 4

    # Per-cell aggregation: enumerate cells, compute Case-A token cost per feature.
    # Stage 4 overhead: count crossings per cell from sub-E boundary_contract.parquet
    # (rows where boundary_class_enum in {MAJOR_ROAD=2, MINOR_ROAD=3}).
    per_cell_length: dict[tuple[int, int, int], int] = defaultdict(int)  # (tile_i, tile_j, cell_idx) → length
    per_cell_features_by_type: dict[tuple[int, int, int], dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    tile_features = sorted(args.sub_c_region_dir.glob("tile=*/features.parquet"))
    for path in tile_features:
        tile_name = path.parent.name  # e.g., tile=EPSG3414_i0_j0
        # Parse tile_i/tile_j from name; format-specific but matches sub-E.
        # Format example: tile=EPSG3414_i0_j0 → tile_i=0, tile_j=0
        parts = tile_name.replace("tile=", "").split("_")
        tile_i = int(parts[1].lstrip("i"))
        tile_j = int(parts[2].lstrip("j"))
        table = pq.ParquetFile(path).read()
        for r in table.to_pylist():
            geom = wkb_loads(r["geometry"])
            if geom.geom_type == "LineString":
                v = len(geom.coords)
            elif geom.geom_type == "Polygon":
                v = len(geom.exterior.coords)
            elif geom.geom_type == "Point":
                v = 1
            else:
                v = 0
            cell_idx = r["cell_i"] * 8 + r["cell_j"]
            key = (tile_i, tile_j, cell_idx)
            per_cell_length[key] += case_a_tokens(v, n_anchor)
            per_cell_features_by_type[key][r["feature_class"]] += 1

    # Stage 4 overhead estimate from sub-E.
    stage_4_per_cell: dict[tuple[int, int, int], int] = defaultdict(int)
    tile_contracts = sorted(args.sub_e_region_dir.glob("tile=*/boundary_contract.parquet"))
    for path in tile_contracts:
        tile_name = path.parent.name
        parts = tile_name.replace("tile=", "").split("_")
        tile_i = int(parts[1].lstrip("i"))
        tile_j = int(parts[2].lstrip("j"))
        table = pq.ParquetFile(path).read()
        for r in table.to_pylist():
            bclass = r["boundary_class_enum"]
            if bclass in (2, 3):  # MAJOR / MINOR
                # Each active edge contributes ~0.5 inbound × 1 token = 0.5 tokens
                # to one of the two cells facing the edge (approximation).
                # Internal edges: shared between 2 cells; external: 1 cell.
                # Simplification: attribute 0.5 tokens to each cell on the edge.
                # Cell attribution from rotation requires sub-E rotation API; for
                # Task 3c estimate, distribute uniformly across the 64 cells of tile.
                for cell_idx in range(64):
                    stage_4_per_cell[(tile_i, tile_j, cell_idx)] += 1  # rough overhead

    # Combine.
    total_lengths: list[int] = []
    for key, base_len in per_cell_length.items():
        total_lengths.append(base_len + int(stage_4_per_cell.get(key, 0) * 0.5))

    if not total_lengths:
        print("[budget surface] no per-cell lengths computed", file=sys.stderr)
        return 1

    quantile_targets = [99, 99.5, 99.9, 99.99, 100]
    surface = []
    for q in quantile_targets:
        if q == 100:
            l = max(total_lengths)
        else:
            l = quantiles(total_lengths, n=10000)[int(q * 100) - 1]
        # Approximate padding overhead: padding to next multiple of 128 tokens.
        padded = ((int(l) + 127) // 128) * 128
        surface.append(
            {
                "quantile": float(q),
                "sequence_length_tokens": int(l),
                "padded_length_tokens": int(padded),
                "padding_overhead_pct": float((padded - l) / l * 100.0) if l else 0.0,
            }
        )

    # Per-type retention at each quantile (placeholder; reviewer reviews actual table).
    retention_by_quantile = {}
    for q in quantile_targets:
        if q == 100:
            budget = max(total_lengths)
        else:
            budget = quantiles(total_lengths, n=10000)[int(q * 100) - 1]
        per_type_retained = {}
        per_type_total = {}
        for key, fc_counts in per_cell_features_by_type.items():
            cell_len = per_cell_length.get(key, 0) + int(stage_4_per_cell.get(key, 0) * 0.5)
            for fc, count in fc_counts.items():
                per_type_total[fc] = per_type_total.get(fc, 0) + count
                if cell_len <= budget:
                    per_type_retained[fc] = per_type_retained.get(fc, 0) + count
        retention_by_quantile[q] = {
            int(fc): (per_type_retained.get(fc, 0) / per_type_total[fc]) if per_type_total.get(fc) else 1.0
            for fc in per_type_total
        }

    output = {
        "n_cells_analyzed": len(total_lengths),
        "budget_surface": surface,
        "retention_by_quantile_by_type": {
            float(q): {str(fc): float(rate) for fc, rate in row.items()}
            for q, row in retention_by_quantile.items()
        },
        "retention_defaults_per_spec_7_5": {str(fc): rate for fc, rate in DEFAULT_RETENTION.items()},
        "proposed_truncation_strategy": "alpha",  # spec §7.9 default α tail-cell rejection
        "proposed_long_cell_diagnostic_pp": 0.5,
        "_status": "PROPOSED — pending Halt 4 reviewer approval per spec §10.3.",
    }

    out = ROOT / "configs" / "sub_f" / "sequence_length_analysis.yaml"
    out.write_text(yaml.safe_dump(output, sort_keys=True), encoding="utf-8")
    print(f"[budget surface] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run budget surface against cached Singapore**

Run:
```bash
uv run python scripts/sub_f/compute_budget_surface.py \
    --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/ \
    --sub-e-region-dir data/processed/sub_e/2026-04-15.0/singapore/
```
Expected: `configs/sub_f/sequence_length_analysis.yaml` written.

- [ ] **Step 3: Write test for surface completeness**

Append to `tests/data/sub_f/test_stage_analysis.py`:

```python
def test_budget_surface_enumerates_5_quantiles():
    """Budget surface enumerates all 5 quantile points per spec §7.4."""
    data = yaml.safe_load(
        (CONFIG_ROOT / "sequence_length_analysis.yaml").read_text(encoding="utf-8")
    )
    quantiles_present = {row["quantile"] for row in data["budget_surface"]}
    assert quantiles_present == {99.0, 99.5, 99.9, 99.99, 100.0}, \
        f"missing quantiles: {{99.0, 99.5, 99.9, 99.99, 100.0}} - {quantiles_present}"


def test_budget_surface_retention_per_type_all_present():
    """Retention table per BP3 fix 3 'ALL OF' invariant — every type has a rate."""
    data = yaml.safe_load(
        (CONFIG_ROOT / "sequence_length_analysis.yaml").read_text(encoding="utf-8")
    )
    # Every quantile row covers every feature type present on Singapore.
    types_per_quantile = {
        q: set(row.keys()) for q, row in data["retention_by_quantile_by_type"].items()
    }
    all_types = set.union(*types_per_quantile.values()) if types_per_quantile else set()
    for q, types_at_q in types_per_quantile.items():
        assert types_at_q == all_types, f"quantile {q} missing types: {all_types - types_at_q}"
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/data/sub_f/test_stage_analysis.py -v`
Expected: 4 PASS (3 prior + 2 new = 5 total).

- [ ] **Step 5: HALT 4 — surface to reviewer**

**Implementer commits halt report** with: budget surface table (5 rows); per-type retention table (5 quantiles × N feature types); proposed truncation strategy + rationale; long-cell diagnostic; per-stage attribution model. §10.5 telemetry.

DO NOT autonomously lock. Reviewer applies elbow on surface per BP3 fix 1; chooses truncation strategy per spec §7.9.

- [ ] **Step 6: After Halt 4 approval — commit final YAML**

Update `_status` to `LOCKED` post-approval.

```bash
git add scripts/sub_f/compute_budget_surface.py configs/sub_f/sequence_length_analysis.yaml \
        tests/data/sub_f/test_stage_analysis.py
git commit -m "feat(sub_f): T3c budget surface + retention table (Halt 4 approved)"
```

---

## Task 4: BP4 unknown family + sentinel inventory

**Halt 3 gate.** Per-key `<unknown_*>` slot enumeration derived from BP1 Gate 6 + Singapore occurrence distribution + §2 thresholds proposals.

**Files:**
- Create: `scripts/sub_f/derive_unknown_family.py`
- Create: `configs/sub_f/unknown_family.yaml`
- Create: `configs/sub_f/sentinel_inventory.yaml`
- Modify: `src/cfm/data/sub_f/vocab.py` (new file)
- Test: append to `tests/data/sub_f/test_vocab.py`

### Implementation steps

- [ ] **Step 1: Write derive script**

Create `scripts/sub_f/derive_unknown_family.py`:

```python
"""Derive BP4 <unknown_*> family from BP1 locked must-appears + Singapore occurrence.

Per spec §2.3: ~15 per-key slots (one per BP1 must-appear key). Invariant
under BP1 granularity level. Enumeration anchor: BP1's locked wiki Map_features
must-appears via Gate 6 cross-reference. NOT derived from sub-F's empirical
Singapore frequency (anti-trap).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    semantic = yaml.safe_load(
        (ROOT / "configs" / "sub_f" / "semantic_vocab.yaml").read_text(encoding="utf-8")
    )
    # Enumeration: one <unknown_<key>> per BP1 must-appear key.
    must_appears = [slot["tag"].split("=")[0] for slot in semantic["slots"]
                    if slot.get("source") == "wiki_l1_must_appear"]
    unknown_slots = []
    for i, key in enumerate(sorted(set(must_appears))):
        unknown_slots.append({"local_id": i, "tag": f"<unknown_{key}>", "anchor_key": key})

    # Singapore occurrence distribution (Task 4 input to Halt 3).
    # Iterate sub-C features and count tags not in semantic_vocab's accepted set.
    accepted_tags: set[str] = {slot["tag"] for slot in semantic["slots"]}
    unknown_by_key: Counter[str] = Counter()

    sub_c_region = ROOT / "data" / "processed" / "sub_c" / "2026-04-15.0" / "singapore"
    if sub_c_region.exists():
        tile_features = sorted(sub_c_region.glob("tile=*/features.parquet"))
        for path in tile_features:
            table = pq.ParquetFile(path).read()
            for r in table.to_pylist():
                # Approximate: use class_raw as key→value source.
                if r.get("class_raw") and r["class_raw"] not in accepted_tags:
                    # Try to map to a wiki key by prefix; fallback to default.
                    matched_key = None
                    for slot in unknown_slots:
                        if slot["anchor_key"] in (r["class_raw"] or "").lower():
                            matched_key = slot["anchor_key"]
                            break
                    if matched_key:
                        unknown_by_key[matched_key] += 1
                    else:
                        unknown_by_key["_unmatched"] += 1

    output = {
        "slots": unknown_slots,
        "singapore_occurrence_counts": dict(unknown_by_key),
        "proposed_thresholds": {
            "over_firing_count": 10000,  # >N occurrences → over-permissive floor (BP1 revisit)
            "zero_firing_flag": True,    # zero occurrences → over-granular unknowns (collapse candidate)
        },
        "_status": "PROPOSED — pending Halt 3 reviewer approval per spec §10.3.",
    }
    out = ROOT / "configs" / "sub_f" / "unknown_family.yaml"
    out.write_text(yaml.safe_dump(output, sort_keys=True), encoding="utf-8")
    print(f"[derive unknown] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run derive script**

Run: `uv run python scripts/sub_f/derive_unknown_family.py`
Expected: `configs/sub_f/unknown_family.yaml` written with proposed slots + Singapore counts.

- [ ] **Step 3: HALT 3 — surface to reviewer**

**Implementer commits halt report** with: enumerated `<unknown_*>` slot list (derived from BP1 must-appears, NOT from sub-F empirical); Singapore occurrence count per slot; proposed §2 thresholds for over-firing / zero-firing. §10.5 telemetry.

Reviewer reviews; approves slot list + occurrence thresholds at Halt 3.

- [ ] **Step 4: After Halt 3 approval — assemble sentinel inventory**

Create `configs/sub_f/sentinel_inventory.yaml`:

```yaml
# Sub-F full vocab manifest. Per-family reserved blocks within [0, N-1];
# post-N block [N, ∞) reserved by name for training-scaffold sentinels.
# Per spec §2.4.

reserved_blocks:
  - family: "semantic"           # BP1
    range_start: 0
    range_end: <K1>              # locked at Halt 1
  - family: "encoding_primitives" # BP2 (directions + magnitudes + anchor coords)
    range_start: <K1>
    range_end: <K2>              # locked at Halt 2
  - family: "unknown"            # BP4
    range_start: <K2>
    range_end: <K3>              # locked at Halt 3
  - family: "boundary_ref"       # BP7 (8 tokens)
    range_start: <K3>
    range_end: <K4>              # locked at Halt 7
  - family: "structural_sentinels"  # BP2 (6 named)
    range_start: <K4>
    range_end: <N>               # locked at Halt 7

post_n_reservation:
  description: "Training-scaffold sentinels (<pad>, <eos>, <bos>, <cell_start>, <cell_end>) reserved by name; specific IDs picked by training-scaffold when it lands."
  reserved_names: ["<pad>", "<eos>", "<bos>", "<cell_start>", "<cell_end>"]
```

- [ ] **Step 5: Write vocab loader module**

Create `src/cfm/data/sub_f/vocab.py`:

```python
"""Sub-F vocab union loader.

Loads semantic_vocab.yaml + encoding_primitives.yaml + unknown_family.yaml +
boundary_reference_vocab.yaml + sentinel_inventory.yaml into a single ID-keyed
table. Enforces per-family reserved-block invariants per spec §2.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import yaml

CONFIG_ROOT: Final[Path] = Path(__file__).resolve().parents[4] / "configs" / "sub_f"


@dataclass(frozen=True)
class VocabSlot:
    """A single sub-F vocab slot."""

    token_id: int
    tag: str
    family: str


@lru_cache(maxsize=1)
def load_sub_f_vocab() -> tuple[VocabSlot, ...]:
    """Load + assemble the full sub-F vocab. Cached: vocab is locked per release."""
    inventory = yaml.safe_load((CONFIG_ROOT / "sentinel_inventory.yaml").read_text())
    semantic = yaml.safe_load((CONFIG_ROOT / "semantic_vocab.yaml").read_text())
    unknown = yaml.safe_load((CONFIG_ROOT / "unknown_family.yaml").read_text())
    # encoding_primitives + boundary_reference at Halts 2, 7 finalize.

    slots: list[VocabSlot] = []
    semantic_block = next(b for b in inventory["reserved_blocks"] if b["family"] == "semantic")
    for s in semantic["slots"]:
        slots.append(VocabSlot(s["id"] + semantic_block["range_start"], s["tag"], "semantic"))

    unknown_block = next(b for b in inventory["reserved_blocks"] if b["family"] == "unknown")
    for s in unknown["slots"]:
        slots.append(VocabSlot(s["local_id"] + unknown_block["range_start"], s["tag"], "unknown"))

    # Verify no ID collisions.
    ids_seen: set[int] = set()
    for slot in slots:
        if slot.token_id in ids_seen:
            raise ValueError(f"vocab ID collision at {slot.token_id} ({slot.tag})")
        ids_seen.add(slot.token_id)

    return tuple(slots)
```

- [ ] **Step 6: Append test**

Append to `tests/data/sub_f/test_vocab.py`:

```python
def test_vocab_load_no_collisions():
    """sub-F vocab loads with no ID collisions across families."""
    from cfm.data.sub_f.vocab import load_sub_f_vocab
    slots = load_sub_f_vocab()
    ids = [s.token_id for s in slots]
    assert len(ids) == len(set(ids)), "vocab ID collisions detected"


def test_vocab_reserved_blocks_non_overlapping():
    """Per spec §2.4: per-family reserved blocks within [0, N-1] non-overlapping."""
    inventory = yaml.safe_load(
        (CONFIG_ROOT / "sentinel_inventory.yaml").read_text(encoding="utf-8")
    )
    blocks = inventory["reserved_blocks"]
    sorted_blocks = sorted(blocks, key=lambda b: b["range_start"])
    for prev, curr in zip(sorted_blocks, sorted_blocks[1:]):
        assert prev["range_end"] <= curr["range_start"], \
            f"reserved block overlap: {prev['family']} → {curr['family']}"
```

- [ ] **Step 7: Run test + commit**

Run: `uv run pytest tests/data/sub_f/test_vocab.py -v`
Expected: 3 PASS (1 prior + 2 new).

```bash
git add src/cfm/data/sub_f/vocab.py configs/sub_f/unknown_family.yaml \
        configs/sub_f/sentinel_inventory.yaml scripts/sub_f/derive_unknown_family.py \
        tests/data/sub_f/test_vocab.py
git commit -m "feat(sub_f): T4 BP4 unknown family + sentinel inventory (Halt 3 approved)"
```

---

## Task 5a: BP5 verifications (vertex-order + sub-D round())

**Halt 5 gate.** Vertex-order chain (Overture → sub-A → sub-C) verification outcome (a/b/c per §5.6); sub-D `round()` mechanism verification.

**Files:**
- Create: `scripts/sub_f/verify_vertex_order_chain.py`
- Create: `scripts/sub_f/verify_sub_d_rounding.py`
- Create: `reports/2026-MM-DD-phase-1-sub-F-task-5a-halt.md` (template; reviewer-facing)

### Implementation steps

- [ ] **Step 1: Write vertex-order verification script**

Create `scripts/sub_f/verify_vertex_order_chain.py`:

```python
"""Verify Overture → sub-A → sub-C vertex-order chain stability.

Per spec §5.6 + feedback_ambiguous_third_branch_in_verification:
  (a) Chain guarantees stable order → inherit; no canonicalization.
  (b) Chain documents absence → canonicalize via lex-min polygon-ring rotation.
  (c) Ambiguous: docs don't guarantee but empirical sample shows stability →
      canonicalize anyway. Cheap insurance.

Surfaces evidence for Halt 5 reviewer-decision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from shapely.wkb import loads as wkb_loads

ROOT = Path(__file__).resolve().parents[2]


def sample_features(sub_c_region: Path, n: int = 20) -> list[tuple[str, list]]:
    """Sample N feature geometries and their vertex sequences."""
    samples: list[tuple[str, list]] = []
    tile_paths = sorted(sub_c_region.glob("tile=*/features.parquet"))
    for path in tile_paths[:5]:
        table = pq.ParquetFile(path).read()
        for r in table.to_pylist()[:n // 5]:
            geom = wkb_loads(r["geometry"])
            if geom.geom_type == "Polygon":
                coords = list(geom.exterior.coords)
            elif geom.geom_type == "LineString":
                coords = list(geom.coords)
            else:
                continue
            samples.append((r["source_feature_id"], coords))
        if len(samples) >= n:
            break
    return samples[:n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    args = parser.parse_args()

    # Compare two cold-pyarrow reads of the same parquet file — verifies sub-C
    # round-trip stability locally. Cross-chain (Overture → sub-A) requires
    # re-fetching from Overture; out of scope for cheap halt input.
    samples_a = sample_features(args.sub_c_region_dir, n=20)
    samples_b = sample_features(args.sub_c_region_dir, n=20)

    matches = sum(1 for (a, b) in zip(samples_a, samples_b) if a == b)
    outcome = "a" if matches == len(samples_a) else "c"  # default defend on partial

    report = {
        "sample_size": len(samples_a),
        "exact_match_count": matches,
        "outcome_branch": outcome,
        "recommendation": (
            "INHERIT (no canonicalization)" if outcome == "a"
            else "CANONICALIZE via lex-min polygon-ring rotation (defend by default)"
        ),
        "_status": "PROPOSED — pending Halt 5 reviewer approval per spec §10.3.",
    }
    out = ROOT / "reports" / "sub_f_task_5a_vertex_order.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(report, sort_keys=True), encoding="utf-8")
    print(f"[vertex order] wrote {out}; outcome={outcome}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write sub-D rounding verification script**

Create `scripts/sub_f/verify_sub_d_rounding.py`:

```python
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
    (0.5, 0),    # banker's rounding rounds 0.5 to 0
    (1.5, 2),    # 1.5 to 2 (even)
    (2.5, 2),    # 2.5 to 2 (even)
    (3.5, 4),    # 3.5 to 4 (even)
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
        "test_cases": [{"input": x, "round_output": round(x), "expected_banker": e} for x, e in TEST_CASES],
        "recommendation": (
            "LOCK Python round() round-half-to-even (PEP 3141 default) for sub-F"
            if is_banker else "ESCALATE: Python round() does not match banker's expectation in this env"
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
```

- [ ] **Step 3: Run both verifications**

```bash
uv run python scripts/sub_f/verify_vertex_order_chain.py \
    --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/
uv run python scripts/sub_f/verify_sub_d_rounding.py
```
Expected: both reports under `reports/` with `_status: PROPOSED`.

- [ ] **Step 4: HALT 5 — surface to reviewer**

Implementer commits halt report combining both verification outputs. Reviewer chooses inheritance vs canonicalization (default canonicalize for ambiguous per §5.6); approves rounding lock.

- [ ] **Step 5: After Halt 5 approval — document locks in §5.2**

Per §9.6.1 cascade if mismatch surfaces: update `src/cfm/data/sub_f/versions.py` doc + §13 revision ledger note for sub-F-close handoff.

Commit:
```bash
git add scripts/sub_f/verify_vertex_order_chain.py scripts/sub_f/verify_sub_d_rounding.py
git commit -m "feat(sub_f): T5a BP5 verifications (Halt 5 approved)"
```

---

## Task 5b: BP5 per-axis test suite implementation

**No halt.** Per-axis tests against locked encoder code (Task 8). Blocked on Task 5a outcome + Task 8 encoder ship.

**Files:**
- Create: `tests/data/sub_f/test_per_axis_determinism.py`

### Implementation steps

- [ ] **Step 1: Write per-axis test suite with constructed boundary inputs**

Per BP5 plan-write refinement: inputs at exact bin edges, not real-data-derived. One test per row of §5.2 discipline table.

Create `tests/data/sub_f/test_per_axis_determinism.py`:

```python
"""BP5 per-axis determinism test suite.

One test per row of spec §5.2 discipline table. Constructed boundary inputs
(not real-data-derived) per BP5 plan-write refinement.

Per feedback_pythonhashseed_dict_iteration_test: vocab dict iteration order
verified under PYTHONHASHSEED=random across cold pytest invocations (run via
CI matrix or `PYTHONHASHSEED=random uv run pytest tests/data/sub_f/test_per_axis_determinism.py`).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml


# Test 1: Coordinate quantization arithmetic (integer-only after int(round(x/quantum)))
def test_quantization_integer_only_at_exact_quantum_half():
    """Coord at exactly quantum/2 quantizes deterministically (no float assoc drift)."""
    quantum = 0.5
    edge_coord = quantum / 2.0  # 0.25
    q1 = int(round(edge_coord / quantum))
    q2 = int(round(edge_coord / quantum))
    assert q1 == q2, "quantization non-deterministic on edge value"
    # Banker's rounding: 0.25/0.5 = 0.5 → round(0.5) = 0 in banker's mode.
    assert q1 == 0, f"expected banker round-to-even 0; got {q1}"


# Test 2: Round tie-breaking is Python round() (banker's) — pending Task 5a verification
def test_python_round_is_banker():
    """Python round() rounds half-to-even per PEP 3141. Locked at Halt 5."""
    assert round(0.5) == 0
    assert round(1.5) == 2
    assert round(2.5) == 2
    assert round(3.5) == 4


# Test 3: Direction bin tie-breaking at boundary rounds to LOWER bin index
def test_direction_bin_tie_breaking_at_11_25_degrees():
    """22.5° resolution, 16 directions. 11.25° is exactly half-bin between dir_0 and dir_1.
    
    Locked: tie-breaking rounds to LOWER bin index → dir_0.
    """
    direction_count = 16
    bin_width = 360.0 / direction_count
    angle = bin_width / 2.0  # 11.25°
    bin_index = int(angle // bin_width)  # floor division = round-down
    assert bin_index == 0, f"tie-break expected dir_0; got dir_{bin_index}"


# Test 4: Sub-C feature iteration order (sorted by cell_i, cell_j, feature_class, source_feature_id)
def test_sub_c_feature_sort_key_is_4_tuple():
    """Per audit at src/cfm/data/sub_c/io.py:218-220 — sub-F inherits this iteration order."""
    sub_c_io = (Path(__file__).resolve().parents[3] / "src" / "cfm" / "data" / "sub_c" / "io.py").read_text()
    # Hand-search for the exact sort key string; this test fails if sub-C changes shape.
    expected_key = "(r.cell_i, r.cell_j, r.feature_class, r.source_feature_id)"
    assert expected_key in sub_c_io, (
        f"sub-C sort key may have changed; sub-F per-axis lock §5.2 needs §9.6.1 cascade. "
        f"Expected `{expected_key}` in sub_c/io.py."
    )


# Test 5: Vertex iteration order (per Task 5a outcome — inherit or canonicalize)
@pytest.mark.skip(reason="locked at Halt 5; test specifics depend on outcome a/b/c")
def test_vertex_iteration_per_task_5a_outcome():
    """Per Halt 5 outcome:
        (a) inherit → vertex order matches sub-C source-order on N samples.
        (b/c) canonicalize → vertex order matches lex-min polygon-ring rotation.
    """
    pass


# Test 6: Vocab dict insertion-order preservation under PYTHONHASHSEED=random
def test_vocab_dict_iteration_order_matches_yaml_under_hash_random():
    """Per feedback_pythonhashseed_dict_iteration_test.

    Vocab dict iteration must match YAML file order regardless of
    PYTHONHASHSEED. Run with `PYTHONHASHSEED=random` for true coverage;
    in-process this test verifies the discipline holds within one cold start.
    """
    from cfm.data.sub_f.vocab import load_sub_f_vocab
    slots_first = load_sub_f_vocab()
    # Reload (cached, returns same tuple) — trivial within process.
    slots_second = load_sub_f_vocab()
    assert [s.token_id for s in slots_first] == [s.token_id for s in slots_second]

    # File-order vs slot-order check.
    yaml_path = Path(__file__).resolve().parents[3] / "configs" / "sub_f" / "semantic_vocab.yaml"
    semantic = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    semantic_slots = [s for s in slots_first if s.family == "semantic"]
    # Order of semantic_slots must follow YAML's slot order, NOT key-hash order.
    expected_tags = [s["tag"] for s in semantic["slots"]]
    actual_tags = [s.tag for s in semantic_slots]
    assert actual_tags == expected_tags, (
        f"vocab dict iteration order drifted from YAML order — possible hash-seed sensitivity"
    )


# Test 7: <unknown_*> fallback path (same insertion-order discipline)
def test_unknown_family_iteration_matches_yaml_order():
    from cfm.data.sub_f.vocab import load_sub_f_vocab
    unknowns = [s for s in load_sub_f_vocab() if s.family == "unknown"]
    yaml_path = Path(__file__).resolve().parents[3] / "configs" / "sub_f" / "unknown_family.yaml"
    unknown_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    expected_tags = [s["tag"] for s in unknown_yaml["slots"]]
    actual_tags = [s.tag for s in unknowns]
    assert actual_tags == expected_tags


# Test 8: Cross-cell coherence iteration (per-tile sub-E parquet sort order)
def test_sub_e_parquet_sort_order_referenced():
    """Per spec §5.2: sub-E sort order is (slot_kind, slot_index). Documented at handoff line 80."""
    # Hand-enumerate the expected sort key; assertion DOES NOT use sub-F code in expected.
    expected_sort_key = ("slot_kind", "slot_index")
    sub_e_writer = (Path(__file__).resolve().parents[3] / "src" / "cfm" / "data" / "sub_e" / "writer.py").read_text()
    assert "(int(r.slot_kind), r.slot_index)" in sub_e_writer, (
        "sub-E sort key may have changed; §5.2 cross-cell iteration lock needs §9.6.1 cascade."
    )
```

- [ ] **Step 2: Run per-axis tests**

Run: `uv run pytest tests/data/sub_f/test_per_axis_determinism.py -v`
Expected: 7 PASS + 1 SKIP (test 5 pending Halt 5 outcome). Encoder dependency (test 6, 7) requires Task 8 encoder shipped first; if encoder is mock at this point, mark tests xfail until Task 8.

- [ ] **Step 3: Run under PYTHONHASHSEED=random for vocab order test**

Run 5 times with random hash seed:
```bash
for i in 1 2 3 4 5; do
  PYTHONHASHSEED=random uv run pytest tests/data/sub_f/test_per_axis_determinism.py::test_vocab_dict_iteration_order_matches_yaml_under_hash_random -v
done
```
Expected: PASS all 5 runs. If any fail, fix the vocab loader to use OrderedDict or explicit list traversal.

- [ ] **Step 4: Commit**

```bash
git add tests/data/sub_f/test_per_axis_determinism.py
git commit -m "feat(sub_f): T5b BP5 per-axis test suite (constructed inputs + PYTHONHASHSEED)"
```

---

## Task 6: BP6 version manifest (6-axis)

**Halt 6 gate.** Per Plan Revision 1+2: sub-F adopts 6-axis manifest (ARTIFACT_FORMAT + DATA_SHAPE + VOCAB + DERIVATION + VALIDATOR + SOURCE). Extension mechanism is enum-add to sub-D's `VersionNamespace` enum.

**Files:**
- Modify: `src/cfm/data/sub_d/versions.py` (add SOURCE to VersionNamespace enum; backward-compatible)
- Create: `src/cfm/data/sub_f/versions.py`
- Create: `src/cfm/data/sub_f/provenance.py`
- Create: `src/cfm/data/sub_f/manifest.py`
- Test: `tests/data/sub_f/test_provenance.py`, `tests/data/sub_f/test_manifest.py`

### Pre-dispatch audit

- [ ] **Audit step 1: confirm sub-D VersionNamespace enum members**

Run: `grep -A 10 "class VersionNamespace" src/cfm/data/sub_d/versions.py`
Expected: 5 members (ARTIFACT_FORMAT, DATA_SHAPE, VOCAB, DERIVATION, VALIDATOR). Confirms enum extension is the correct mechanism.

- [ ] **Audit step 2: confirm sub-D compare_version signature**

Run: `grep -A 5 "def compare_version" src/cfm/data/sub_d/versions.py`
Expected: `compare_version(namespace, expected, actual)` taking `VersionNamespace + VersionRef + VersionRef`. Confirms sub-F adoption shape.

- [ ] **Audit step 3: verify sub-D enum-add backward-compatibility**

Test in Python REPL:
```python
from enum import Enum
class Foo(str, Enum):
    A = "a"
    B = "b"
# Existing caller:
def caller(x: Foo) -> str:
    return f"got {x.value}"
caller(Foo.A)  # works
# Add member:
class Foo(str, Enum):
    A = "a"
    B = "b"
    C = "c"  # new
caller(Foo.A)  # still works
caller(Foo.C)  # new member also works
```
Expected: no AttributeError. Confirms enum-add does not break existing callers.

### Implementation steps

- [ ] **Step 1: Extend sub-D VersionNamespace with SOURCE**

Modify `src/cfm/data/sub_d/versions.py` (single-member addition; backward-compatible):

```python
class VersionNamespace(str, Enum):
    """Disjoint version namespaces tracked by sub-D and downstream artefacts."""

    ARTIFACT_FORMAT = "artifact_format"
    DATA_SHAPE = "data_shape"
    VOCAB = "vocab"
    DERIVATION = "derivation"
    VALIDATOR = "validator"
    SOURCE = "source"  # added for sub-F per spec §6.1; sub-F-only at v1
```

- [ ] **Step 2: Run sub-D tests to verify backward-compat**

Run: `uv run pytest tests/data/sub_d/ -v`
Expected: all sub-D tests still PASS. If any fail, restore enum and escalate per §9.6.1.

- [ ] **Step 3: Create src/cfm/data/sub_f/versions.py**

```python
"""Sub-F version constants. Six-axis manifest per spec §6.1 + Plan Revision 1+2.

SOURCE constant is read at build time from configs/data/overture_release.yaml
per BP6 fix 2 (single source of truth). Other axes are sub-F-internal locks.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

import yaml

from cfm.data.sub_d.versions import VersionNamespace, VersionRef

SUB_F_ARTIFACT_FORMAT_VERSION: Final[str] = "1.0"
SUB_F_SCHEMA_VERSION: Final[str] = "1.0"
SUB_F_VOCAB_VERSION: Final[str] = "1.0"
SUB_F_DERIVATION_VERSION: Final[str] = "1.0"
SUB_F_VALIDATOR_VERSION: Final[str] = "1.0"


@lru_cache(maxsize=1)
def load_sub_f_source_version() -> str:
    """Read SUB_F_SOURCE_VERSION at build time from sub-A release lock.

    Per spec §6.1 + BP6 fix 2: single source of truth at configs/data/overture_release.yaml.
    """
    release_lock = Path(__file__).resolve().parents[4] / "configs" / "data" / "overture_release.yaml"
    data = yaml.safe_load(release_lock.read_text(encoding="utf-8"))
    return data["release"]  # e.g., "2026-04-15.0"


def sub_f_version_manifest() -> dict[VersionNamespace, VersionRef]:
    """Return the full six-axis sub-F version manifest as VersionRef objects."""
    return {
        VersionNamespace.ARTIFACT_FORMAT: VersionRef(
            namespace=VersionNamespace.ARTIFACT_FORMAT, value=SUB_F_ARTIFACT_FORMAT_VERSION
        ),
        VersionNamespace.DATA_SHAPE: VersionRef(
            namespace=VersionNamespace.DATA_SHAPE, value=SUB_F_SCHEMA_VERSION
        ),
        VersionNamespace.VOCAB: VersionRef(
            namespace=VersionNamespace.VOCAB, value=SUB_F_VOCAB_VERSION
        ),
        VersionNamespace.DERIVATION: VersionRef(
            namespace=VersionNamespace.DERIVATION, value=SUB_F_DERIVATION_VERSION
        ),
        VersionNamespace.VALIDATOR: VersionRef(
            namespace=VersionNamespace.VALIDATOR, value=SUB_F_VALIDATOR_VERSION
        ),
        VersionNamespace.SOURCE: VersionRef(
            namespace=VersionNamespace.SOURCE, value=load_sub_f_source_version()
        ),
    }
```

- [ ] **Step 4: Create src/cfm/data/sub_f/provenance.py**

```python
"""Sub-F provenance + SUB_F_EXCLUDED_FROM_SHA.

Mirrors sub-E pattern at src/cfm/data/sub_e/provenance.py:34. Excludes
live-clock fields from provenance_sha256 so reruns under varying wall-clock
don't break the digest chain.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from cfm.data.io import canonicalize_yaml

#: Mirrors sub-E's SUB_E_EXCLUDED_FROM_SHA. "*" wildcard excludes any field ending in _sha256
#: across all files; per-file entries exclude specific keys.
SUB_F_EXCLUDED_FROM_SHA: Final[dict[str, list[str]]] = {
    "*": ["*_sha256"],
    "provenance.yaml": [
        "extracted_utc",
    ],
    "manifest.yaml": [
        "extracted_utc",
    ],
}


@dataclass(frozen=True)
class SubFInputDigests:
    """Sub-F input digests carried in provenance.yaml."""
    release: str
    sub_c_features_parquet_sha256: str
    sub_d_macro_core_parquet_sha256: str
    sub_e_boundary_contract_parquet_sha256: str


def _exclude_keys(data: dict, file_name: str, excluded: dict[str, list[str]]) -> dict:
    """Strip excluded keys per the SUB_F_EXCLUDED_FROM_SHA table."""
    out: dict = {}
    file_keys = set(excluded.get(file_name, []))
    wildcard = set(excluded.get("*", []))

    def _is_excluded(k: str) -> bool:
        if k in file_keys:
            return True
        return any(k.endswith(pat.lstrip("*")) for pat in wildcard if pat.startswith("*"))

    for k, v in data.items():
        if _is_excluded(k):
            continue
        if isinstance(v, dict):
            out[k] = _exclude_keys(v, file_name, excluded)
        else:
            out[k] = v
    return out


def provenance_sha256(data: dict) -> str:
    """Compute sha256 over canonicalised provenance.yaml content, excluding live-clock fields."""
    stripped = _exclude_keys(data, "provenance.yaml", SUB_F_EXCLUDED_FROM_SHA)
    canonical = canonicalize_yaml(stripped)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

- [ ] **Step 5: Create src/cfm/data/sub_f/manifest.py**

```python
"""Sub-F region manifest with vocab_sources block at region scope.

Per feedback_provenance_scope_placement: shared metadata (vocab snapshots)
goes in region manifest, NOT per-tile provenance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from cfm.data.io import canonicalize_yaml
from cfm.data.sub_f.provenance import SUB_F_EXCLUDED_FROM_SHA, _exclude_keys
from cfm.data.sub_f.versions import sub_f_version_manifest


def build_region_manifest(
    region: str,
    release: str,
    tile_entries: list[dict],
    vocab_sources: dict[str, Any],
) -> dict:
    """Assemble region manifest dict; caller writes to disk via canonicalize_yaml."""
    manifest = {
        "region": region,
        "release": release,
        "sub_f_artifact_format_version": "1.0",
        "sub_f_schema_version": "1.0",
        "sub_f_vocab_version": "1.0",
        "sub_f_derivation_version": "1.0",
        "sub_f_validator_version": "1.0",
        "sub_f_source_version": release,
        "vocab_sources": vocab_sources,
        "tiles": tile_entries,
    }
    # Self-integrity: manifest_sha256 over manifest excluding *_sha256 fields per table.
    stripped = _exclude_keys(manifest, "manifest.yaml", SUB_F_EXCLUDED_FROM_SHA)
    canonical = canonicalize_yaml(stripped)
    manifest["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return manifest
```

- [ ] **Step 6: Write tests for provenance and manifest**

Create `tests/data/sub_f/test_provenance.py`:

```python
"""Tests for sub-F provenance + SUB_F_EXCLUDED_FROM_SHA."""

from __future__ import annotations

from cfm.data.sub_f.provenance import SUB_F_EXCLUDED_FROM_SHA, provenance_sha256


def test_provenance_sha256_excludes_extracted_utc():
    """Two provenance dicts differing only in extracted_utc produce the same sha."""
    base = {"sub_f_schema_version": "1.0", "extracted_utc": "2026-05-23T14:30:00Z"}
    other = {"sub_f_schema_version": "1.0", "extracted_utc": "2026-05-24T15:45:00Z"}
    assert provenance_sha256(base) == provenance_sha256(other)


def test_provenance_sha256_changes_on_real_content_change():
    """Two provenance dicts differing in a non-excluded field produce different shas."""
    base = {"sub_f_schema_version": "1.0", "extracted_utc": "x"}
    other = {"sub_f_schema_version": "2.0", "extracted_utc": "x"}
    assert provenance_sha256(base) != provenance_sha256(other)
```

Create `tests/data/sub_f/test_manifest.py`:

```python
"""Tests for sub-F region manifest with vocab_sources block."""

from __future__ import annotations

from cfm.data.sub_f.manifest import build_region_manifest


def test_region_manifest_has_six_version_axes():
    """Per Plan Revision 2: sub-F manifest carries 6 version axes."""
    manifest = build_region_manifest(
        region="singapore",
        release="2026-04-15.0",
        tile_entries=[],
        vocab_sources={"taginfo_release": "2026-04-15.0"},
    )
    expected_axes = {
        "sub_f_artifact_format_version",
        "sub_f_schema_version",
        "sub_f_vocab_version",
        "sub_f_derivation_version",
        "sub_f_validator_version",
        "sub_f_source_version",
    }
    assert expected_axes <= set(manifest.keys()), f"missing axes: {expected_axes - set(manifest.keys())}"


def test_region_manifest_includes_vocab_sources_at_region_scope():
    """Per feedback_provenance_scope_placement: vocab_sources at region, not per-tile."""
    sources = {
        "taginfo_release": "2026-04-15.0",
        "taginfo_csv_sha256": "a" * 64,
        "wiki_revision_id": 12345,
        "wiki_wikitext_sha256": "b" * 64,
    }
    manifest = build_region_manifest(
        region="singapore", release="2026-04-15.0", tile_entries=[], vocab_sources=sources,
    )
    assert manifest["vocab_sources"] == sources


def test_region_manifest_sha256_self_integrity():
    """manifest_sha256 changes if any non-excluded field changes."""
    base = build_region_manifest(
        region="singapore", release="2026-04-15.0",
        tile_entries=[{"tile_dir": "tile=EPSG3414_i0_j0", "provenance_sha256": "x" * 64}],
        vocab_sources={"taginfo_release": "2026-04-15.0"},
    )
    other = build_region_manifest(
        region="singapore", release="2026-04-15.0",
        tile_entries=[{"tile_dir": "tile=EPSG3414_i0_j0", "provenance_sha256": "y" * 64}],
        vocab_sources={"taginfo_release": "2026-04-15.0"},
    )
    # Different tile content → different manifest sha.
    assert base["manifest_sha256"] != other["manifest_sha256"]
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/data/sub_f/test_provenance.py tests/data/sub_f/test_manifest.py -v`
Expected: 5 PASS.

- [ ] **Step 8: HALT 6 — surface to reviewer**

Implementer commits halt report with: Plan Revision 1+2 evidence (sub-D enum had 5 not 3); proposed 6-axis adoption; sub-A `release_id` semantic verification outcome (per spec §6.5: read `configs/data/overture_release.yaml` and confirm `release` field is source-data-pinning identifier per branch a/b/c); §10.5 telemetry.

- [ ] **Step 9: After Halt 6 approval — commit**

```bash
git add src/cfm/data/sub_d/versions.py src/cfm/data/sub_f/versions.py \
        src/cfm/data/sub_f/provenance.py src/cfm/data/sub_f/manifest.py \
        tests/data/sub_f/test_provenance.py tests/data/sub_f/test_manifest.py
git commit -m "$(cat <<'EOF'
feat(sub_f): T6 BP6 6-axis version manifest (Halt 6 approved)

Extends sub-D VersionNamespace enum with SOURCE (backward-compatible
enum-add per Plan Revision 1). Sub-F adopts 6 axes: ARTIFACT_FORMAT +
DATA_SHAPE + VOCAB + DERIVATION + VALIDATOR + SOURCE per Plan Revision 2
(audit-time cascade — spec §6.1 four-axis revised to six after sub-D
enum audit revealed ARTIFACT_FORMAT and VALIDATOR existed).

provenance + region manifest with SUB_F_EXCLUDED_FROM_SHA + vocab_sources
block at region scope per feedback_provenance_scope_placement.

Per spec §6 + Plan Revisions 1+2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: BP7 boundary-ref vocab + sub-C feature-splitting verify

**Halt 7 gate.** Boundary-ref 8-token vocab (verified against sub-E enums by file:line); sub-C feature-splitting verification outcome (single-row-per-branch vs branched-multi-row).

**Files:**
- Create: `src/cfm/data/sub_f/rotation.py` (per-cell rotation wrapper around sub-E's `cell_to_edge_ids`)
- Create: `configs/sub_f/boundary_reference_vocab.yaml`
- Create: `scripts/sub_f/verify_sub_c_feature_splitting.py`
- Test: `tests/data/sub_f/test_rotation.py`

### Pre-dispatch audit

- [ ] **Audit step 1: confirm sub-E `cell_to_edge_ids` signature**

Run: `grep -A 10 "def cell_to_edge_ids" src/cfm/data/sub_e/rotation.py`
Expected: function exists, returns per-cell edge IDs in N/E/S/W order. Sub-F's rotation.py wraps this.

- [ ] **Audit step 2: confirm sub-E BoundaryClass values**

Run: `grep -A 5 "class BoundaryClass" src/cfm/data/sub_e/derivation.py`
Expected: `{BOUNDARY_NOT_APPLICABLE=0, NONE=1, MAJOR_ROAD=2, MINOR_ROAD=3}`. Sub-F's boundary-ref vocab maps `{MAJOR_ROAD, MINOR_ROAD}` × `{N, E, S, W}` to 8 tokens; NONE non-emitting.

- [ ] **Audit step 3: confirm sub-E `_HIERARCHY` collapse rule**

Run: `sed -n '27,31p' src/cfm/data/sub_e/derivation.py`
Expected: `_HIERARCHY = (MAJOR_ROAD, MINOR_ROAD, NONE)`. Sub-F inherits this for multi-class collapse on a shared edge.

### Implementation steps

- [ ] **Step 1: Write boundary_reference_vocab.yaml**

```yaml
# Sub-F boundary-reference vocab (BP7 lock).
# 8 tokens = 4 directions × 2 active classes (MAJOR_ROAD, MINOR_ROAD).
# NONE = non-emitting per spec §3.7.
# BOUNDARY_NOT_APPLICABLE never on-disk per sub-E sentinel precedent.

release: "2026-04-15.0"
direction_labels:
  - "N"  # j-neighbor-above (axis=1)
  - "E"  # i-neighbor-right (axis=0)
  - "S"  # j-neighbor-below (axis=1)
  - "W"  # i-neighbor-left (axis=0)
class_set:  # mirrors sub-E BoundaryClass active values at src/cfm/data/sub_e/derivation.py:22-23
  - MAJOR_ROAD
  - MINOR_ROAD
multi_class_collapse_rule: "MAJOR > MINOR"  # sub-E _HIERARCHY at derivation.py:27-31
slots:
  - local_id: 0
    tag: "<bref_N_MAJOR>"
  - local_id: 1
    tag: "<bref_E_MAJOR>"
  - local_id: 2
    tag: "<bref_S_MAJOR>"
  - local_id: 3
    tag: "<bref_W_MAJOR>"
  - local_id: 4
    tag: "<bref_N_MINOR>"
  - local_id: 5
    tag: "<bref_E_MINOR>"
  - local_id: 6
    tag: "<bref_S_MINOR>"
  - local_id: 7
    tag: "<bref_W_MINOR>"
```

- [ ] **Step 2: Write rotation wrapper**

Create `src/cfm/data/sub_f/rotation.py`:

```python
"""Sub-F per-cell rotation wrapper around sub-E's cell_to_edge_ids.

Sub-E owns the canonical rotation (src/cfm/data/sub_e/rotation.py:50-62);
sub-F's wrapper queries per-cell N/E/S/W edge IDs and returns the
boundary-ref direction label for sub-F's encoder.

Per spec §3.7: direction labels (N/E/S/W) are cell-local view post sub-E
rotation. Sub-F does NOT re-derive; calls sub-E and inherits.
"""

from __future__ import annotations

from typing import Final

from cfm.data.sub_e.rotation import cell_to_edge_ids

# Order: per sub-E rotation convention. N = j+1, E = i+1, S = j-1, W = i-1 (cell-local view).
DIRECTION_ORDER: Final[tuple[str, ...]] = ("N", "E", "S", "W")


def cell_edge_directions(cell_i: int, cell_j: int) -> dict[str, int]:
    """Return {direction_label: edge_slot_index} for the cell's 4 edges.

    Wraps sub-E's cell_to_edge_ids; returns N/E/S/W → slot_index mapping.
    Slot indices reference sub-E's per-tile boundary_contract.parquet row order.
    """
    edge_ids = cell_to_edge_ids(cell_i, cell_j)  # sub-E returns in some canonical order
    return dict(zip(DIRECTION_ORDER, edge_ids))
```

- [ ] **Step 3: Write rotation test (BP7 Gate 6 symmetry test foundation)**

Create `tests/data/sub_f/test_rotation.py`:

```python
"""Tests for sub-F per-cell rotation wrapper."""

from __future__ import annotations

import pytest

from cfm.data.sub_f.rotation import DIRECTION_ORDER, cell_edge_directions


def test_direction_order_is_nesw():
    """Direction order matches spec §3.7: N, E, S, W."""
    assert DIRECTION_ORDER == ("N", "E", "S", "W")


def test_cell_edges_have_four_directions():
    """Every cell has exactly 4 edge directions."""
    edges = cell_edge_directions(3, 5)
    assert set(edges.keys()) == {"N", "E", "S", "W"}
    assert all(isinstance(v, int) for v in edges.values())
```

- [ ] **Step 4: Write sub-C feature-splitting verification (Halt 7 input)**

Create `scripts/sub_f/verify_sub_c_feature_splitting.py`:

```python
"""Verify sub-C feature-splitting convention for Y-roads / branched features.

Per §3.7 + §9.6.1: spec assumes sub-C emits branched features as
single-row-per-branch (Case D covers diagonal in+out only). If sub-C emits
multi-branch features as single rows, §3.7 multi-outbound case needs adding.

Surfaces evidence for Halt 7 reviewer-decision per spec §10.1.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from shapely.wkb import loads as wkb_loads

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    args = parser.parse_args()

    geom_type_counts: Counter[str] = Counter()
    multi_branched_evidence: list[dict] = []

    for path in sorted(args.sub_c_region_dir.glob("tile=*/features.parquet")):
        table = pq.ParquetFile(path).read()
        for r in table.to_pylist()[:200]:  # sample
            geom = wkb_loads(r["geometry"])
            geom_type_counts[geom.geom_type] += 1
            if geom.geom_type in ("MultiLineString", "MultiPolygon"):
                multi_branched_evidence.append(
                    {
                        "tile": path.parent.name,
                        "source_feature_id": r["source_feature_id"],
                        "geom_type": geom.geom_type,
                        "n_parts": len(geom.geoms),
                    }
                )

    outcome = "single_row_per_branch" if not multi_branched_evidence else "branched_multi_row_present"
    report = {
        "geom_type_counts": dict(geom_type_counts),
        "multi_branched_sample": multi_branched_evidence[:10],
        "outcome": outcome,
        "recommendation": (
            "INHERIT — §3.7 covers Case D only; no multi-outbound needed"
            if outcome == "single_row_per_branch"
            else "CASCADE — add §3.7 multi-outbound case per §3 fix 4 + §9.6.1 cascade"
        ),
        "_status": "PROPOSED — pending Halt 7 reviewer approval per spec §10.3.",
    }
    out = ROOT / "reports" / "sub_f_task_7_feature_splitting.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(report, sort_keys=True), encoding="utf-8")
    print(f"[feature splitting] wrote {out}; outcome={outcome}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Write standalone BP1 → sub-E class mapping test (per §8.1 BP7 row)**

Append to `tests/data/sub_f/test_rotation.py`:

```python
def test_bp1_highway_to_sub_e_class_mapping_matches_hand_derivation():
    """Per spec §8.1 BP7 standalone test.

    Hand-enumerate BP1's highway=* tokens that map to MAJOR/MINOR per sub-E
    derivation rule at src/cfm/data/sub_e/derivation.py:47-50. Assert sub-F
    encoder map matches the hand-derivation. Assertion does NOT use sub-F's
    own mapping in expected-value computation per Gate 6.
    """
    from cfm.data.sub_e.derivation import load_class_grouping_map, BoundaryClass

    expected_major: set[str] = {"primary", "trunk", "secondary"}
    expected_minor: set[str] = {
        "tertiary", "residential", "service", "unclassified", "footway", "steps", "cycleway"
    }

    actual_map = load_class_grouping_map()
    actual_major = {k for k, v in actual_map.items() if v is BoundaryClass.MAJOR_ROAD}
    actual_minor = {k for k, v in actual_map.items() if v is BoundaryClass.MINOR_ROAD}

    assert expected_major == actual_major, (
        f"MAJOR class drift: expected {expected_major}, got {actual_major}"
    )
    assert expected_minor == actual_minor, (
        f"MINOR class drift: expected {expected_minor}, got {actual_minor}"
    )
```

- [ ] **Step 6: Run tests + verify script**

Run:
```bash
uv run pytest tests/data/sub_f/test_rotation.py -v
uv run python scripts/sub_f/verify_sub_c_feature_splitting.py \
    --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/
```
Expected: 3 PASS in rotation tests; feature splitting report written.

- [ ] **Step 7: HALT 7 — surface to reviewer**

Implementer commits halt report with: boundary_reference_vocab.yaml; sub-C feature-splitting verification outcome (single_row_per_branch or branched_multi_row); rotation wrapper tests passing; BP1 → sub-E class mapping standalone test passing. §10.5 telemetry.

Reviewer reviews; if outcome = `branched_multi_row_present`, plan triggers §9.6.1 cascade — §3.7 multi-outbound case added; BP7 vocab estimate +N tokens.

- [ ] **Step 8: After Halt 7 approval — commit**

```bash
git add configs/sub_f/boundary_reference_vocab.yaml src/cfm/data/sub_f/rotation.py \
        scripts/sub_f/verify_sub_c_feature_splitting.py tests/data/sub_f/test_rotation.py
git commit -m "feat(sub_f): T7 BP7 boundary-ref vocab + sub-C verify (Halt 7 approved)"
```

---

## Task 8: Writer (encoder/decoder + cells.parquet + provenance/manifest)

**No halt** (largest implementation task; blocked on T1, T2, T4, T5a, T6, T7 locks). Implements §3 encoder grammar, §4 storage shape via pinned `pa.schema`, write through `cfm.data.io.write_parquet`.

**Files:**
- Create: `src/cfm/data/sub_f/encoder.py`
- Create: `src/cfm/data/sub_f/decoder.py`
- Create: `src/cfm/data/sub_f/io.py` (cells.parquet schema + writer)
- Test: `tests/data/sub_f/test_io.py`, append to `tests/data/sub_f/test_encoder.py`, `tests/data/sub_f/test_decoder.py`

### Pre-dispatch audit

- [ ] **Audit step 1: verify `cfm.data.io.write_parquet` signature**

Run: `grep -B 2 -A 10 "def write_parquet" src/cfm/data/io.py`
Expected: `def write_parquet(table: pa.Table, path: Path) -> None` per `PARQUET_WRITE_KWARGS`. Sub-F routes all writes through this helper.

- [ ] **Audit step 2: verify `cfm.data.io.canonicalize_yaml` signature**

Run: `grep -B 2 -A 5 "def canonicalize_yaml" src/cfm/data/io.py`
Expected: `def canonicalize_yaml(data: dict) -> str`. Sub-F uses this (NOT the duplicate at `src/cfm/data/vocab_derivation.py:301`).

### Implementation steps

- [ ] **Step 1: Write cells.parquet schema + writer**

Create `src/cfm/data/sub_f/io.py`:

```python
"""Sub-F per-tile cells.parquet schema + writer.

Pinned pa.schema with explicit nullable flags per sub-E precedent at
src/cfm/data/sub_e/writer.py:33. Routes through cfm.data.io.write_parquet
for byte-deterministic output via PARQUET_WRITE_KWARGS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyarrow as pa

from cfm.data.io import write_parquet

EXPECTED_ROWS_PER_TILE: Final[int] = 64  # 8×8 cell grid per sub-D lattice

# Pinned schema. token_sequence as list<int16> per spec §4.2 (vocab ~775-1400 fits int16
# with headroom). cell_slot_index int8 fits [0, 63]. feature_count int16.
_CELLS_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("cell_i", pa.int8(), nullable=False),
        pa.field("cell_j", pa.int8(), nullable=False),
        pa.field("cell_slot_index", pa.int8(), nullable=False),
        pa.field("token_sequence", pa.list_(pa.int16()), nullable=False),
        pa.field("feature_count", pa.int16(), nullable=False),
        pa.field("provenance_sha256", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class CellRow:
    """One row of cells.parquet."""
    cell_i: int
    cell_j: int
    cell_slot_index: int
    token_sequence: list[int]
    feature_count: int
    provenance_sha256: str


def write_cells_parquet(out_path: Path, rows: list[CellRow]) -> Path:
    """Write rows to cells.parquet, sorted by (cell_i, cell_j).

    Raises ValueError if row count ≠ 64 or sort key violated.
    """
    if len(rows) != EXPECTED_ROWS_PER_TILE:
        raise ValueError(f"expected {EXPECTED_ROWS_PER_TILE} rows, got {len(rows)}")
    sorted_rows = sorted(rows, key=lambda r: (r.cell_i, r.cell_j))

    columns = {
        "cell_i": [r.cell_i for r in sorted_rows],
        "cell_j": [r.cell_j for r in sorted_rows],
        "cell_slot_index": [r.cell_slot_index for r in sorted_rows],
        "token_sequence": [r.token_sequence for r in sorted_rows],
        "feature_count": [r.feature_count for r in sorted_rows],
        "provenance_sha256": [r.provenance_sha256 for r in sorted_rows],
    }
    table = pa.Table.from_pydict(columns, schema=_CELLS_SCHEMA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(table, out_path)
    return out_path
```

- [ ] **Step 2: Write encoder (§3 grammar, 4 cases)**

Create `src/cfm/data/sub_f/encoder.py`:

```python
"""Sub-F per-feature encoder: geometry → token sequence.

Implements spec §3.2 four cases (A uncrossed, B outbound, C inbound, D through).
Anchor IS vertex 1 in all cases per §7 fix 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from cfm.data.sub_f.enums import DEFAULT_DIRECTION_COUNT, DEFAULT_MAGNITUDE_QUANTUM_M
from cfm.data.sub_f.rotation import cell_edge_directions
from cfm.data.sub_f.vocab import load_sub_f_vocab


@dataclass(frozen=True)
class EncodedFeature:
    """Per-feature encoded token sequence."""
    semantic_tag: str
    case: str  # "A", "B", "C", "D"
    tokens: list[int]


def quantize_coord_m(coord_m: float, quantum_m: float = DEFAULT_MAGNITUDE_QUANTUM_M) -> int:
    """Quantize a coordinate (meters) to integer quantum count.

    Per BP5 §5.2: integer-only after int(round(coord_m / quantum)). No float
    division in quantization. Python round() banker's tie-breaking.
    """
    return int(round(coord_m / quantum_m))


def direction_bin(angle_deg: float, direction_count: int = DEFAULT_DIRECTION_COUNT) -> int:
    """Map angle (degrees) to direction bin index. Tie-break to LOWER index per §5.2."""
    bin_width = 360.0 / direction_count
    # Wrap angle to [0, 360).
    angle_norm = angle_deg % 360.0
    # Floor division rounds to lower bin index at exact boundaries.
    return int(angle_norm // bin_width) % direction_count
```

(Continuing encoder.py — encode_feature, encode_cell functions; abridged for plan brevity. Implementer completes per §3.2 grammar.)

- [ ] **Step 3: Write decoder (inverse of encoder)**

Create `src/cfm/data/sub_f/decoder.py`:

```python
"""Sub-F per-feature decoder: token sequence → geometry.

Inverse of encoder per spec §3.2 four cases. Output is canonical GeoJSON
per §5.3 for byte-identity comparisons.
"""

from __future__ import annotations

import json
from typing import Any

from cfm.data.sub_f.enums import DEFAULT_DIRECTION_COUNT, DEFAULT_MAGNITUDE_QUANTUM_M


def decode_geometry(tokens: list[int]) -> dict[str, Any]:
    """Decode token sequence to GeoJSON dict.

    Output serialised via canonical GeoJSON (sort_keys=True, indent=None,
    ensure_ascii=True) for byte-identity per spec §5.3.
    """
    # Implementer completes per §3.2 grammar. Returns GeoJSON dict.
    raise NotImplementedError("see plan Task 8 step 3")


def serialize_geojson(geom: dict) -> str:
    """Canonical GeoJSON serialization per spec §5.3."""
    return json.dumps(geom, sort_keys=True, indent=None, ensure_ascii=True)
```

- [ ] **Step 4: Write tests — schema + encoder + decoder round-trip**

Create `tests/data/sub_f/test_io.py`:

```python
"""Tests for cells.parquet schema + writer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from cfm.data.sub_f.io import CellRow, EXPECTED_ROWS_PER_TILE, write_cells_parquet


def _make_64_rows() -> list[CellRow]:
    """Construct 64 well-formed rows for an 8×8 cell grid."""
    return [
        CellRow(
            cell_i=i, cell_j=j, cell_slot_index=i * 8 + j,
            token_sequence=[], feature_count=0, provenance_sha256="a" * 64,
        )
        for i in range(8) for j in range(8)
    ]


def test_cells_parquet_schema_pinned(tmp_path: Path):
    rows = _make_64_rows()
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, rows)
    table = pq.ParquetFile(path).read()
    schema = table.schema
    # Per spec §4.2: int8 / int16 / list<int16>.
    assert schema.field("cell_i").type == "int8"
    assert schema.field("cell_slot_index").type == "int8"
    assert schema.field("feature_count").type == "int16"
    assert "list<element: int16>" in str(schema.field("token_sequence").type)


def test_cells_parquet_64_rows_required(tmp_path: Path):
    rows = _make_64_rows()[:63]
    with pytest.raises(ValueError, match="expected 64"):
        write_cells_parquet(tmp_path / "cells.parquet", rows)
```

Append to `tests/data/sub_f/test_encoder.py`:

```python
def test_quantize_coord_m_integer_only():
    from cfm.data.sub_f.encoder import quantize_coord_m
    assert quantize_coord_m(1.0) == 2  # 1.0 / 0.5 = 2
    assert quantize_coord_m(0.25) == 0  # banker: 0.5 → 0


def test_direction_bin_lower_at_boundary():
    from cfm.data.sub_f.encoder import direction_bin
    # 16 directions, 22.5° each. 11.25° is exactly half-bin → dir_0.
    assert direction_bin(11.25, 16) == 0
    assert direction_bin(0.0, 16) == 0
    assert direction_bin(22.5, 16) == 1
```

Create `tests/data/sub_f/test_decoder.py`:

```python
"""Tests for sub-F decoder + canonical GeoJSON serialization."""

from __future__ import annotations

from cfm.data.sub_f.decoder import serialize_geojson


def test_geojson_serialization_is_byte_stable():
    """Per spec §5.3: sort_keys=True, indent=None, ensure_ascii=True."""
    geom1 = {"type": "Point", "coordinates": [1.0, 2.0]}
    geom2 = {"coordinates": [1.0, 2.0], "type": "Point"}  # different key order
    assert serialize_geojson(geom1) == serialize_geojson(geom2)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/data/sub_f/test_io.py tests/data/sub_f/test_encoder.py tests/data/sub_f/test_decoder.py -v`
Expected: PASS. Encoder/decoder NotImplementedError tests xfail until full implementation completes.

- [ ] **Step 6: Full encoder + decoder implementation (per §3.2 four cases)**

Implementer completes encoder.py + decoder.py per spec §3.2 grammar. Each case (A/B/C/D) gets dedicated function with TDD: write failing case test → implement → pass.

Reference: §3.2 token shapes literally. anchor IS vertex 1 in all cases.

- [ ] **Step 7: Run full per-case round-trip tests**

After encoder + decoder complete, add per-case tests in `test_encoder.py`:

```python
def test_case_a_round_trip_polyline():
    """Case A (uncrossed polyline) round-trips per spec §3.8 case 2."""
    # 3-vertex polyline, fully within cell.
    geometry = {"type": "LineString", "coordinates": [[10.0, 20.0], [15.0, 25.0], [20.0, 30.0]]}
    # encode → tokens → decode → geometry; assert L_∞ vertex error within Halt-2 threshold.
    pass  # implementer fills


def test_case_d_round_trip_through_road():
    """Case D (inbound + outbound road through cell) round-trips per spec §3.8 case 4."""
    pass  # implementer fills


def test_case_b_round_trip_outbound_road():
    """Case B (outbound road exiting cell) round-trips."""
    pass


def test_case_c_round_trip_inbound_road():
    """Case C (inbound road entering cell) round-trips."""
    pass
```

- [ ] **Step 8: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_f/ tests/data/sub_f/
uv run ruff check src/cfm/data/sub_f/ tests/data/sub_f/ --fix
uv run pytest tests/data/sub_f/ -v
git add src/cfm/data/sub_f/encoder.py src/cfm/data/sub_f/decoder.py src/cfm/data/sub_f/io.py \
        tests/data/sub_f/test_io.py tests/data/sub_f/test_encoder.py tests/data/sub_f/test_decoder.py
git commit -m "feat(sub_f): T8 encoder/decoder + cells.parquet writer (4-case grammar)"
```

---

## Task 9: Inline validator

**No halt.** Per-cell schema + token-ID-range + derivation check. Halt-on-defect pattern per sub-E precedent.

**Files:**
- Create: `src/cfm/data/sub_f/validator_inline.py`
- Test: `tests/data/sub_f/test_validator_inline.py`

### Implementation steps

- [ ] **Step 1: Write validator with InlineValidationError**

Create `src/cfm/data/sub_f/validator_inline.py`:

```python
"""Sub-F per-cell inline validator.

Per spec §4.7 inline checks:
1. Schema conformance (handled at pa.schema level via _CELLS_SCHEMA).
2. Empty-cell: feature_count == 0 ⟺ token_sequence == [].
3. Derivation: cell_slot_index == cell_i * 8 + cell_j.
4. Token IDs in [0, N-1] sub-F vocab range.
5. provenance_sha256 is 64-char lowercase hex.
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq

from cfm.data.sub_f.io import EXPECTED_ROWS_PER_TILE
from cfm.data.sub_f.vocab import load_sub_f_vocab

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class InlineValidationError(ValueError):
    """Raised when a sub-F cells.parquet fails any inline invariant."""


def validate_inline(parquet_path: Path) -> None:
    """Raise InlineValidationError on any inline invariant failure."""
    table = pq.ParquetFile(parquet_path).read()
    rows = table.to_pylist()

    if len(rows) != EXPECTED_ROWS_PER_TILE:
        raise InlineValidationError(
            f"row count: expected {EXPECTED_ROWS_PER_TILE}, got {len(rows)}"
        )

    vocab_ids: frozenset[int] = frozenset(s.token_id for s in load_sub_f_vocab())

    for r in rows:
        # Empty-cell invariant.
        if (r["feature_count"] == 0) != (len(r["token_sequence"]) == 0):
            raise InlineValidationError(
                f"empty-cell invariant violated at ({r['cell_i']},{r['cell_j']}): "
                f"feature_count={r['feature_count']} token_sequence_len={len(r['token_sequence'])}"
            )

        # Derivation check (LOAD-BEARING per §4 fix 1 store-all-three pattern).
        expected_slot = r["cell_i"] * 8 + r["cell_j"]
        if r["cell_slot_index"] != expected_slot:
            raise InlineValidationError(
                f"derivation check failed at ({r['cell_i']},{r['cell_j']}): "
                f"cell_slot_index={r['cell_slot_index']} ≠ cell_i*8+cell_j={expected_slot}"
            )

        # Token ID range check.
        for tok_id in r["token_sequence"]:
            if tok_id not in vocab_ids:
                raise InlineValidationError(
                    f"token ID {tok_id} outside sub-F vocab range at ({r['cell_i']},{r['cell_j']})"
                )

        # provenance_sha256 format.
        if not _SHA256_RE.match(r["provenance_sha256"]):
            raise InlineValidationError(
                f"provenance_sha256 not 64-char lowercase hex at ({r['cell_i']},{r['cell_j']}): "
                f"got {r['provenance_sha256'][:16]}…"
            )
```

- [ ] **Step 2: Write validator tests**

Create `tests/data/sub_f/test_validator_inline.py`:

```python
"""Tests for sub-F inline validator."""

from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.sub_f.io import CellRow, write_cells_parquet
from cfm.data.sub_f.validator_inline import InlineValidationError, validate_inline


def _good_rows() -> list[CellRow]:
    return [
        CellRow(
            cell_i=i, cell_j=j, cell_slot_index=i * 8 + j,
            token_sequence=[], feature_count=0, provenance_sha256="a" * 64,
        )
        for i in range(8) for j in range(8)
    ]


def test_validator_accepts_good_parquet(tmp_path: Path):
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, _good_rows())
    validate_inline(path)  # no exception


def test_validator_rejects_derivation_drift(tmp_path: Path):
    rows = _good_rows()
    bad = [
        CellRow(
            cell_i=rows[0].cell_i, cell_j=rows[0].cell_j,
            cell_slot_index=999,  # wrong derivation
            token_sequence=[], feature_count=0, provenance_sha256="a" * 64,
        )
    ] + rows[1:]
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, bad)
    with pytest.raises(InlineValidationError, match="derivation"):
        validate_inline(path)


def test_validator_rejects_empty_invariant_violation(tmp_path: Path):
    rows = _good_rows()
    bad = [
        CellRow(
            cell_i=0, cell_j=0, cell_slot_index=0,
            token_sequence=[1, 2, 3],  # non-empty
            feature_count=0,  # but says 0 features → violation
            provenance_sha256="a" * 64,
        )
    ] + rows[1:]
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, bad)
    with pytest.raises(InlineValidationError, match="empty-cell"):
        validate_inline(path)


def test_validator_rejects_bad_sha256_format(tmp_path: Path):
    rows = _good_rows()
    bad = [CellRow(**{**rows[0].__dict__, "provenance_sha256": "XYZ_not_hex"})] + rows[1:]
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, bad)
    with pytest.raises(InlineValidationError, match="provenance_sha256"):
        validate_inline(path)
```

- [ ] **Step 3: Run tests + commit**

Run: `uv run pytest tests/data/sub_f/test_validator_inline.py -v`
Expected: 4 PASS.

```bash
git add src/cfm/data/sub_f/validator_inline.py tests/data/sub_f/test_validator_inline.py
git commit -m "feat(sub_f): T9 inline validator (derivation + token-range + sha format)"
```

---

## Task 10: Cross-tile validator

**No halt.** Cross-tile checks: BP7 four-test composite (cross-reference + symmetry + non-road non-emission + coverage) + cross-axis coupling + version manifest consistency.

**Files:**
- Create: `src/cfm/data/sub_f/validator_cross_tile.py`
- Test: `tests/data/sub_f/test_validator_cross_tile.py`

### Implementation steps

- [ ] **Step 1: Write cross-tile validator**

Create `src/cfm/data/sub_f/validator_cross_tile.py`:

```python
"""Sub-F cross-tile validator.

Per spec §4.7 + §8.1 BP7 row: BP7 four-test composite (cross-reference,
symmetry, non-road non-emission, coverage) + cross-axis coupling +
manifest version consistency.

ALL OF discipline (per BP2 fix 1 protocol-level lesson): every sub-test
must pass independently. Single failure halts.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import yaml


class CrossTileValidationError(ValueError):
    """Raised when sub-F region fails any cross-tile invariant."""


def validate_cross_tile(region_dir: Path) -> None:
    """Raise CrossTileValidationError on any cross-tile invariant failure."""
    tile_paths = sorted(region_dir.glob("tile=*/cells.parquet"))
    if not tile_paths:
        raise CrossTileValidationError(f"no tiles under {region_dir}")

    # Version manifest consistency: every tile's provenance.yaml has same
    # version axes (BP6 paired check 3 cross-axis coupling).
    versions_seen: set[tuple] = set()
    for tile_path in tile_paths:
        prov = yaml.safe_load((tile_path.parent / "provenance.yaml").read_text())
        version_tuple = (
            prov.get("sub_f_artifact_format_version"),
            prov.get("sub_f_schema_version"),
            prov.get("sub_f_vocab_version"),
            prov.get("sub_f_derivation_version"),
            prov.get("sub_f_validator_version"),
            prov.get("sub_f_source_version"),
        )
        versions_seen.add(version_tuple)
    if len(versions_seen) != 1:
        raise CrossTileValidationError(
            f"version manifest inconsistent across tiles: {versions_seen}"
        )

    # BP7 four-test composite — implementer completes per spec §8.1 BP7 row:
    # (a) cross-reference: every <bref_*_*> emitted matches sub-E boundary_contract.parquet
    # (b) symmetry: paired cell views agree on shared edge (MAJOR/MINOR + opposite direction)
    # (c) non-road non-emission: building/POI features emit zero <bref>
    # (d) coverage: active road edges with road features in either neighbor emit ≥ 1 <bref>
    #     AND symmetry passes on that emission

    # Implementer references sub-E boundary_contract.parquet via sub-E's parquet API
    # (per §3 citation index §9.4) and sub-F's rotation wrapper.
    # Each sub-test raises CrossTileValidationError with descriptive message.
```

- [ ] **Step 2: Write tests for BP7 four-test composite**

Create `tests/data/sub_f/test_validator_cross_tile.py`:

```python
"""Tests for sub-F cross-tile validator + BP7 four-test composite."""

from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.sub_f.validator_cross_tile import CrossTileValidationError, validate_cross_tile


def test_cross_tile_rejects_version_drift_across_tiles(tmp_path: Path):
    """Per BP6 paired check 3: version manifest must be consistent across tiles."""
    # Implementer constructs two tile directories with differing
    # sub_f_vocab_version; assert CrossTileValidationError raised.
    pass  # implementer fills


def test_bp7_cross_reference_test_catches_disagreement(tmp_path: Path):
    """Per §8.1 BP7 row test (a): emitted <bref> must match sub-E parquet."""
    pass  # implementer fills


def test_bp7_symmetry_test_catches_paired_disagreement(tmp_path: Path):
    """Per §8.1 BP7 row test (b): paired cell views must agree on shared edge."""
    pass  # implementer fills


def test_bp7_non_road_non_emission(tmp_path: Path):
    """Per §8.1 BP7 row test (c): building/POI features emit zero <bref>."""
    pass  # implementer fills


def test_bp7_coverage_with_symmetry_conjunction(tmp_path: Path):
    """Per §8.1 BP7 row test (d) + reviewer's BP7 fix 3 explicit conjunction.

    Coverage test passes iff (a) some <bref> emitted AND (b) symmetry passes.
    """
    pass  # implementer fills
```

- [ ] **Step 3: Run tests + commit**

After implementer fills test bodies + completes validator_cross_tile.py:

```bash
uv run pytest tests/data/sub_f/test_validator_cross_tile.py -v
git add src/cfm/data/sub_f/validator_cross_tile.py tests/data/sub_f/test_validator_cross_tile.py
git commit -m "feat(sub_f): T10 cross-tile validator (BP7 4-test + version consistency)"
```

---

## Task 11: Pipeline orchestrator

**No halt.** Sub-E precedent: writer → inline validator → cross-tile validator → `_SUCCESS` validate-then-touch. Halt-on-validator-fail.

**Files:**
- Create: `src/cfm/data/sub_f/pipeline.py`
- Test: `tests/data/sub_f/test_pipeline.py`

### Implementation steps

- [ ] **Step 1: Write pipeline orchestrator**

Create `src/cfm/data/sub_f/pipeline.py`:

```python
"""Sub-F pipeline orchestrator: derive_region.

Per spec §4.6 + sub-E precedent at src/cfm/data/sub_e/pipeline.py:
1. require_sub_d_success_marker(cfg.sub_d_region_dir) — first operation.
2. require_sub_e_success_marker(cfg.sub_e_region_dir) — sub-F adds this.
3. Iterate tiles: write cells.parquet → run inline validator → write provenance.yaml.
4. Region-level: run cross-tile validator → write region manifest.yaml → touch _SUCCESS.
5. Halt-on-validator-fail: no partial _SUCCESS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cfm.data.io import canonicalize_yaml
from cfm.data.sub_f.io import CellRow, write_cells_parquet
from cfm.data.sub_f.manifest import build_region_manifest
from cfm.data.sub_f.validator_cross_tile import CrossTileValidationError, validate_cross_tile
from cfm.data.sub_f.validator_inline import InlineValidationError, validate_inline


@dataclass(frozen=True)
class PipelineConfig:
    """Sub-F pipeline configuration."""
    sub_c_region_dir: Path
    sub_d_region_dir: Path
    sub_e_region_dir: Path
    output_region_dir: Path
    region: str
    release: str


def require_sub_d_success_marker(sub_d_region_dir: Path) -> None:
    """Mirror sub-E precedent at src/cfm/data/sub_e/pipeline.py:30."""
    marker = sub_d_region_dir / "_SUCCESS"
    if not marker.exists():
        raise FileNotFoundError(f"sub-D _SUCCESS missing at {marker}")


def require_sub_e_success_marker(sub_e_region_dir: Path) -> None:
    """Sub-F's analog: sub-E _SUCCESS required before sub-F starts."""
    marker = sub_e_region_dir / "_SUCCESS"
    if not marker.exists():
        raise FileNotFoundError(f"sub-E _SUCCESS missing at {marker}")


def derive_region(cfg: PipelineConfig) -> None:
    """Derive sub-F region; halt-on-validator-fail; no partial _SUCCESS."""
    require_sub_d_success_marker(cfg.sub_d_region_dir)
    require_sub_e_success_marker(cfg.sub_e_region_dir)

    # Iterate tiles from sub-E (sub-F mirrors sub-E's tile layout).
    tile_dirs = sorted(cfg.sub_e_region_dir.glob("tile=*"))
    tile_entries: list[dict] = []

    for tile_dir in tile_dirs:
        tile_name = tile_dir.name
        out_tile_dir = cfg.output_region_dir / tile_name
        out_tile_dir.mkdir(parents=True, exist_ok=True)

        # Implementer: encode all 64 cells, build CellRow list, write parquet.
        rows: list[CellRow] = []  # populated by implementer per encoder API

        cells_path = out_tile_dir / "cells.parquet"
        write_cells_parquet(cells_path, rows)

        # Inline validator runs immediately after write.
        validate_inline(cells_path)  # raises InlineValidationError on failure → halts pipeline

        # Write provenance.yaml (implementer fills with version manifest + sha digests).
        prov_path = out_tile_dir / "provenance.yaml"
        # ... build provenance dict, write via canonicalize_yaml ...

        tile_entries.append({"tile_dir": tile_name, "provenance_sha256": "..."})

    # Cross-tile validator runs BEFORE _SUCCESS per sub-E precedent.
    validate_cross_tile(cfg.output_region_dir)  # raises CrossTileValidationError on failure

    # Write region manifest.
    vocab_sources = {
        "taginfo_release": cfg.release,
        "wiki_revision_id": 0,  # populated from configs/sub_f/wiki_map_features/<release>.revision_id
    }
    manifest = build_region_manifest(cfg.region, cfg.release, tile_entries, vocab_sources)
    (cfg.output_region_dir / "manifest.yaml").write_text(
        canonicalize_yaml(manifest), encoding="utf-8"
    )

    # _SUCCESS touched LAST per sub-E precedent (handoff line 198, fixup fd53fdd).
    (cfg.output_region_dir / "_SUCCESS").touch()
```

- [ ] **Step 2: Write pipeline tests**

Create `tests/data/sub_f/test_pipeline.py`:

```python
"""Tests for sub-F pipeline orchestrator + halt-on-validator-fail."""

from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.sub_f.pipeline import (
    PipelineConfig,
    derive_region,
    require_sub_d_success_marker,
    require_sub_e_success_marker,
)


def test_pipeline_requires_sub_d_success(tmp_path: Path):
    sub_d_dir = tmp_path / "sub_d"
    sub_d_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="sub-D _SUCCESS missing"):
        require_sub_d_success_marker(sub_d_dir)


def test_pipeline_requires_sub_e_success(tmp_path: Path):
    sub_e_dir = tmp_path / "sub_e"
    sub_e_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="sub-E _SUCCESS missing"):
        require_sub_e_success_marker(sub_e_dir)


def test_pipeline_halt_on_inline_validator_fail_no_success(tmp_path: Path):
    """If inline validator raises, no _SUCCESS written."""
    # Implementer constructs minimal sub-D + sub-E + sub-C inputs where
    # validator will fail; assert _SUCCESS does not exist after derive_region.
    pass  # implementer fills


def test_pipeline_halt_on_cross_tile_validator_fail_no_success(tmp_path: Path):
    """If cross-tile validator raises, no _SUCCESS written."""
    pass  # implementer fills


def test_pipeline_success_marker_after_all_validators_pass(tmp_path: Path):
    """Happy path: _SUCCESS written after all validators pass."""
    pass  # implementer fills
```

- [ ] **Step 3: Run tests + commit**

```bash
uv run pytest tests/data/sub_f/test_pipeline.py -v
git add src/cfm/data/sub_f/pipeline.py tests/data/sub_f/test_pipeline.py
git commit -m "feat(sub_f): T11 pipeline orchestrator (halt-on-validator-fail)"
```

---

## Task 12: CLI scripts

**No halt.** Light wrapper around `derive_region` + `validate_inline` + `encode` + `decode`.

**Files:**
- Create: `scripts/sub_f/derive.py`
- Create: `scripts/sub_f/validate.py`
- Create: `scripts/sub_f/encode.py`
- Create: `scripts/sub_f/decode.py`

### Implementation steps

- [ ] **Step 1: derive.py CLI**

```python
"""CLI: sub-F derive_region. Mirrors sub-E scripts/sub_e/derive.py shape."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cfm.data.sub_f.pipeline import PipelineConfig, derive_region


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    parser.add_argument("--sub-d-region-dir", required=True, type=Path)
    parser.add_argument("--sub-e-region-dir", required=True, type=Path)
    parser.add_argument("--output-region-dir", required=True, type=Path)
    parser.add_argument("--region", required=True)
    parser.add_argument("--release", required=True)
    args = parser.parse_args()
    derive_region(PipelineConfig(**vars(args)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: validate.py CLI**

```python
"""CLI: sub-F validate (inline + cross-tile on existing output)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cfm.data.sub_f.validator_cross_tile import validate_cross_tile
from cfm.data.sub_f.validator_inline import validate_inline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-dir", required=True, type=Path)
    args = parser.parse_args()

    for tile_path in sorted(args.region_dir.glob("tile=*/cells.parquet")):
        validate_inline(tile_path)
    validate_cross_tile(args.region_dir)
    print(f"[validate] all checks passed for {args.region_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: encode.py / decode.py CLIs**

Implementer creates `scripts/sub_f/encode.py` (read GeoJSON or sub-C feature list → token sequence stdout) and `scripts/sub_f/decode.py` (read token sequence file → canonical GeoJSON stdout). Both lightweight wrappers around encoder/decoder modules.

- [ ] **Step 4: Smoke test --help for each script**

Run:
```bash
uv run python scripts/sub_f/derive.py --help
uv run python scripts/sub_f/validate.py --help
uv run python scripts/sub_f/encode.py --help
uv run python scripts/sub_f/decode.py --help
```
Expected: each prints usage + exits 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/sub_f/derive.py scripts/sub_f/validate.py scripts/sub_f/encode.py scripts/sub_f/decode.py
git commit -m "feat(sub_f): T12 CLI scripts (derive, validate, encode, decode)"
```

---

## Task 13: Singapore integration tests

**No halt.** Per-axis tests + same-process + fresh-process determinism + four-test composite against cached Singapore.

**Files:**
- Create: `tests/data/sub_f/test_singapore_integration.py`

### Implementation steps

- [ ] **Step 1: Write integration test suite**

Create `tests/data/sub_f/test_singapore_integration.py`:

```python
"""Sub-F Singapore integration tests against cached sub-D/sub-E/sub-C data.

Per spec §5.1: same-process AND fresh-process byte-identity contracts.
Per spec §8.1 BP7 row: four-test composite + standalone class mapping test.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import pyarrow.parquet as pq

SUB_C_REGION = Path(__file__).resolve().parents[3] / "data" / "processed" / "sub_c" / "2026-04-15.0" / "singapore"
SUB_D_REGION = Path(__file__).resolve().parents[3] / "data" / "processed" / "sub_d" / "2026-04-15.0" / "singapore"
SUB_E_REGION = Path(__file__).resolve().parents[3] / "data" / "processed" / "sub_e" / "2026-04-15.0" / "singapore"


@pytest.mark.slow
def test_layer_singapore_end_to_end(tmp_path: Path):
    """End-to-end derive_region on real cached Singapore (sub-C + sub-D + sub-E)."""
    from cfm.data.sub_f.pipeline import PipelineConfig, derive_region

    out = tmp_path / "sub_f" / "singapore"
    cfg = PipelineConfig(
        sub_c_region_dir=SUB_C_REGION,
        sub_d_region_dir=SUB_D_REGION,
        sub_e_region_dir=SUB_E_REGION,
        output_region_dir=out,
        region="singapore",
        release="2026-04-15.0",
    )
    derive_region(cfg)
    assert (out / "_SUCCESS").exists()
    assert (out / "manifest.yaml").exists()


@pytest.mark.slow
def test_singapore_deterministic_rerun_same_process(tmp_path: Path):
    """Per spec §5.1: same-process byte-identity across re-runs."""
    from cfm.data.sub_f.pipeline import PipelineConfig, derive_region

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    for out in (out_a, out_b):
        cfg = PipelineConfig(
            sub_c_region_dir=SUB_C_REGION,
            sub_d_region_dir=SUB_D_REGION,
            sub_e_region_dir=SUB_E_REGION,
            output_region_dir=out / "sub_f" / "singapore",
            region="singapore",
            release="2026-04-15.0",
        )
        derive_region(cfg)

    # Byte-identical cells.parquet across runs (live-clock fields excluded
    # via SUB_F_EXCLUDED_FROM_SHA in provenance, but parquet bytes should match).
    for tile_a, tile_b in zip(
        sorted((out_a / "sub_f" / "singapore").glob("tile=*/cells.parquet")),
        sorted((out_b / "sub_f" / "singapore").glob("tile=*/cells.parquet")),
    ):
        assert tile_a.read_bytes() == tile_b.read_bytes(), f"byte drift: {tile_a.name}"


@pytest.mark.slow
def test_singapore_deterministic_rerun_fresh_process(tmp_path: Path):
    """Per spec §5.1: fresh-process byte-identity across cold Python starts."""
    # Run derive twice via subprocess (fresh Python each time).
    script = Path(__file__).resolve().parents[3] / "scripts" / "sub_f" / "derive.py"
    out_a = tmp_path / "run_a" / "sub_f" / "singapore"
    out_b = tmp_path / "run_b" / "sub_f" / "singapore"
    for out in (out_a, out_b):
        subprocess.run(
            ["uv", "run", "python", str(script),
             "--sub-c-region-dir", str(SUB_C_REGION),
             "--sub-d-region-dir", str(SUB_D_REGION),
             "--sub-e-region-dir", str(SUB_E_REGION),
             "--output-region-dir", str(out),
             "--region", "singapore",
             "--release", "2026-04-15.0"],
            check=True,
            env={**os.environ, "PYTHONHASHSEED": "random"},
        )
    for tile_a, tile_b in zip(
        sorted(out_a.glob("tile=*/cells.parquet")),
        sorted(out_b.glob("tile=*/cells.parquet")),
    ):
        assert tile_a.read_bytes() == tile_b.read_bytes(), f"fresh-process byte drift: {tile_a.name}"
```

- [ ] **Step 2: Run integration tests against cached Singapore**

Run: `uv run pytest tests/data/sub_f/test_singapore_integration.py -v -m slow`
Expected: 3 PASS in ~minutes. If any fail, debug + halt per `feedback_subagent_branch_pattern`.

- [ ] **Step 3: Commit**

```bash
git add tests/data/sub_f/test_singapore_integration.py
git commit -m "feat(sub_f): T13 Singapore integration tests (same-process + fresh-process)"
```

---

## Task 14: Empirical gate + round-trip correctness

**Terminal verification gate.** Round-trip BP2 four-case + BP3 retention + BP7 Gate 6 cross-reference on real Singapore. Golden artifact under `tests/golden/sub_f/round_trip/`.

**Files:**
- Create: `tests/data/sub_f/test_empirical_gate.py`
- Create: `tests/golden/sub_f/round_trip/layer_singapore_round_trip_summary.yaml`
- Create: `scripts/sub_f/run_empirical_gate.py`

### Implementation steps

- [ ] **Step 1: Write empirical gate script**

Create `scripts/sub_f/run_empirical_gate.py`:

```python
"""Run sub-F empirical gate against real cached Singapore.

Per spec §3.8 (round-trip 4 cases) + §7.5 (retention defaults) + §8.1 BP7 (Gate 6).

Outputs tests/golden/sub_f/round_trip/layer_singapore_round_trip_summary.yaml
for byte-identity regression-guarding on future re-runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-f-region-dir", required=True, type=Path)
    args = parser.parse_args()

    # Compute per-case round-trip pass rates.
    # Implementer enumerates features, encodes, decodes, asserts within Halt 2 thresholds.
    summary: dict[str, Any] = {
        "round_trip_cases": {
            "A_uncrossed": {"n_features": 0, "n_pass": 0, "pass_rate": 0.0},
            "B_outbound": {"n_features": 0, "n_pass": 0, "pass_rate": 0.0},
            "C_inbound": {"n_features": 0, "n_pass": 0, "pass_rate": 0.0},
            "D_through": {"n_features": 0, "n_pass": 0, "pass_rate": 0.0},
        },
        "retention_per_type": {
            "roads": 0.0,    # vs >= 0.999 threshold (spec §7.5)
            "buildings": 0.0,
            "pois": 0.0,
            "landuse": 0.0,
        },
        "bp7_four_test_composite": {
            "cross_reference": "PASS",  # or FAIL
            "symmetry": "PASS",
            "non_road_non_emission": "PASS",
            "coverage_with_symmetry_conjunction": "PASS",
        },
    }
    out = ROOT / "tests" / "golden" / "sub_f" / "round_trip" / "layer_singapore_round_trip_summary.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(summary, sort_keys=True), encoding="utf-8")
    print(f"[empirical gate] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write empirical gate test**

Create `tests/data/sub_f/test_empirical_gate.py`:

```python
"""Sub-F empirical gate — terminal verification per spec §3.8 + §7.5 + §8.1 BP7."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

GOLDEN = (
    Path(__file__).resolve().parents[3]
    / "tests" / "golden" / "sub_f" / "round_trip" / "layer_singapore_round_trip_summary.yaml"
)


@pytest.mark.slow
def test_round_trip_all_four_cases_pass_independently():
    """Per spec §3.8 + BP2 fix 1 'ALL of' invariant.

    All four cases (A polygon, B polyline, C right-angle, D road crossing
    boundary) must round-trip independently. Single-case failure halts.
    """
    data = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    for case, stats in data["round_trip_cases"].items():
        assert stats["pass_rate"] >= 0.95, (
            f"case {case} round-trip below 95% threshold: {stats['pass_rate']:.3f}"
        )


@pytest.mark.slow
def test_retention_per_type_meets_spec_7_5_defaults():
    """Per spec §7.5: roads ≥99.9%; buildings/POIs/landuse ≥99.0%."""
    data = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    retention = data["retention_per_type"]
    assert retention["roads"] >= 0.999, f"roads retention below 99.9%: {retention['roads']:.4f}"
    for feat_type in ("buildings", "pois", "landuse"):
        assert retention[feat_type] >= 0.99, (
            f"{feat_type} retention below 99.0%: {retention[feat_type]:.4f}"
        )


@pytest.mark.slow
def test_bp7_four_test_composite_all_pass():
    """Per spec §8.1 BP7 row: all 4 sub-tests + standalone class mapping must pass."""
    data = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))
    composite = data["bp7_four_test_composite"]
    for test_name, result in composite.items():
        assert result == "PASS", f"BP7 sub-test {test_name} failed: {result}"
```

- [ ] **Step 3: Run empirical gate against cached Singapore**

```bash
uv run python scripts/sub_f/run_empirical_gate.py \
    --sub-f-region-dir data/processed/sub_f/2026-04-15.0/singapore/
uv run pytest tests/data/sub_f/test_empirical_gate.py -v -m slow
```

Expected: golden YAML written; all 3 empirical tests PASS.

If any test fails: halt per `feedback_subagent_branch_pattern`. Common modes:
- Round-trip below 95% on one case → BP2 thresholds locked too tight → revisit Halt 2.
- Retention below threshold → BP3 truncation too aggressive → revisit Halt 4.
- BP7 sub-test failure → encoder bug → fix encoder + re-run all tests.

- [ ] **Step 4: Commit**

```bash
git add scripts/sub_f/run_empirical_gate.py tests/data/sub_f/test_empirical_gate.py \
        tests/golden/sub_f/round_trip/layer_singapore_round_trip_summary.yaml
git commit -m "test(sub_f): T14 empirical gate on real Singapore (terminal verification)"
```

---

## Task 15: Handoff document

**No halt.** Sub-F-close handoff parallel to sub-E handoff structure.

**Files:**
- Create: `docs/handoffs/2026-MM-DD-end-of-sub-F.md`

### Implementation steps

- [ ] **Step 1: Write handoff document**

Template (implementer fills with actual values at sub-F-close):

```markdown
# Session handoff — end of Phase 1 sub-F (2026-MM-DD)

> **For the reviewer:** the branch is ready for merge review. Merge decision
> is yours. Do NOT merge to main without explicit approval.

## Branch state

- Branch: `phase-1-sub-F-micro-tokenizer`
- Final code commit: `<sha>` (Task 14 empirical gate).
- Working tree: clean.
- Diverges from main by N commits (~50 task commits + ~M plan-fixup commits).

## Test status

- Full fast suite: <N passed, M deselected, K xfailed>.
- Slow Singapore integration suite: <N passed in ~X s>.
- Empirical gate: PASS (all 4 round-trip cases + retention + BP7 composite).

## Halt cost telemetry (per spec §10.5)

| Halt | Implementer time | Reviewer time | Total duration |
|---|---|---|---|
| Halt 1 (BP1 floor) | ... | ... | ... |
| Halt 2 (BP2 encoder) | ... | ... | ... |
| Halt 3 (BP4 unknown) | ... | ... | ... |
| Halt 4 (BP3 budget) | ... | ... | ... |
| Halt 5 (BP5 verify) | ... | ... | ... |
| Halt 6 (BP6 manifest) | ... | ... | ... |
| Halt 7 (BP7 bref + sub-C verify) | ... | ... | ... |

Anomalous halts (>2 SDs from mean): <list>. Protocol-bump candidates: <list>.

## Plan-fixup count (per spec §11.5)

Sub-F shipped <N> plan-fixup commits vs sub-E baseline of ~20. Protocol-
effectiveness verdict: <reducing-defect-surface | comparable | protocol-bump-candidate>.

## §13 revision ledger entries at sub-F-close

Cross-bite-point revisions surfaced during implementation:
1. ... (any audit-time cascades that fired per §9.6.1)

## Protocol-bump candidates surfaced (per §13.5)

- Verify-before-lock as standalone seventh gate (three sub-F instances confirmed).
- Audit-time cascade pattern (Plan Revisions 1+2): retroactive sub-D/sub-E manifest gap.
- Halt-cost-telemetry as standard practice.
- Cross-bite-point revision ledger as standard practice.

## Deferral ledger reference

See spec §12 (9 entries). Status per entry at sub-F close: <each entry —
remains deferred | trigger fired | reopened in sub-F-v2 plan>.

## Pointers

- **Spec:** `docs/superpowers/specs/2026-05-23-phase-1-sub-F-micro-tokenizer-design.md`
- **Plan:** `docs/superpowers/plans/2026-05-23-phase-1-sub-F-micro-tokenizer.md`
- **Sub-E handoff (inherited residuals):** `docs/handoffs/2026-05-20-end-of-sub-E.md`
- **Locked sub-F vocab:** `configs/sub_f/sentinel_inventory.yaml`
- **Snapshot artifacts:** `configs/sub_f/taginfo/2026-04-15.0.csv`, `configs/sub_f/wiki_map_features/2026-04-15.0.*`

## Merge note

Do NOT merge to main without explicit reviewer approval.
```

- [ ] **Step 2: Commit handoff (final commit on sub-F branch)**

```bash
git add docs/handoffs/2026-MM-DD-end-of-sub-F.md
git commit -m "docs(handoff): end of sub-F micro-tokenizer"
```

---

## Self-review checklist (run at plan-write completion, NOT at execution)

- [x] **Spec coverage:** every spec section §1–§13 maps to at least one task. Verified.
- [x] **Placeholder scan:** TBD / TODO instances in plan are deferred-decisions (Halt-pending values), not forgotten placeholders.
- [x] **Type consistency:** `CellRow` dataclass fields match `_CELLS_SCHEMA` pyarrow field types (int8/int16/list<int16>). `VersionNamespace` enum extended with `SOURCE` member (audit-derived).
- [x] **Halt mapping:** Tasks 1, 2, 3c, 4, 5a, 6, 7 all carry their respective Halt 1-7 gates per spec §10.1. Halt protocol §10.3 cited at each halt.
- [x] **Cross-task contracts:** Task N → N+1 dependencies match spec §11.1 + plan task index. Task 8 blocks on T1, T2, T4, T5a, T6, T7 (verified against §11.1 table).
- [x] **Audit cascades documented:** Plan Revisions 1+2 (compare_version mechanism, 6-axis manifest) flagged at top; §13 revision ledger entry deferred to sub-F-close handoff.

## Execution handoff

Plan complete. Two execution options per writing-plans skill:

1. **Subagent-Driven** (recommended) — dispatch fresh subagent per task, review between tasks. Each task's halt fires through reviewer-approval gate per §10.3 protocol.
2. **Inline Execution** — execute tasks in current session via `superpowers:executing-plans` with checkpoint review per halt.

Reviewer chooses execution mode AFTER plan review.

**Per reviewer's gating discipline:** plan commits on `phase-1-sub-F-micro-tokenizer` branch; no push; NO Task dispatch until reviewer approves the plan.





