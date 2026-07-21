"""Task 2 — driver core: ordered generation over ``sharded_eval`` (torch-free).

Turns the Task-1 ``ConditionedCell`` list into decoded generation records by
dispatching a per-cell ``gen_fn`` through the golden-verified sharding primitive
:func:`cfm.eval.shard.sharded_eval`. The key contract is **global-index seed
keying**: cell ``i`` is generated with ``seed = base_seed + i`` regardless of which
rank happens to compute it, so a cell's tokens do not depend on the shard layout
(the rank-independence property golden tooth #1 verifies; A5). ``sharded_eval`` is
the ONLY sharding mechanism used here; its ``all_gather_object`` merge returns the
per-cell results in canonical global order on every rank.

Torch discipline (mirrors Task 1): this module must import WITHOUT torch. It only
delegates to ``sharded_eval`` (whose ``torch.distributed`` import is lazy, inside
the function body) and reads ``CELL_END_TOKEN_ID`` from the torch-free
``cfm.data.sub_f.vocab``. ``gen_fn`` is injected — Task 3 supplies a closure over
the GPU generate/decode path; the unit tests supply a pure deterministic fake — so
this core stays GPU-free.

Artifact I/O is write-once JSON (refuse an existing path), mirroring the eval-set
/ conditioning-floor write-once discipline: re-generating means deleting the old
artifact deliberately, so a stale run can never be silently half-overwritten.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# CELL_END_TOKEN_ID (=260) lives in the torch-free sub-F vocab (its canonical home;
# cfm.inference.generate re-imports it from here but pulls torch, so import direct).
from cfm.data.sub_f.vocab import CELL_END_TOKEN_ID
from cfm.eval.realism_driver.conditioning import ConditionedCell
from cfm.eval.shard import sharded_eval

logger = logging.getLogger(__name__)

#: Schema version stamped into the gen artifact. Bump on any incompatible layout
#: change (LOCK-AND-GUARDS-TRAVEL-TOGETHER: update the reader in the same commit).
GEN_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GenCellRecord:
    """One generated, decoded cell in canonical manifest order.

    ``blocks``/``geoms`` are the aligned kept decode results (same shape the
    downstream ``DecodedCell`` scorer consumes). ``self_terminated`` is DERIVED by
    the driver from ``tokens`` (ends in ``<cell_end>``=260), never trusted from
    ``gen_fn``, so it always reflects the actual token tail.
    """

    cell_key: tuple[str, int, int, int, int]
    density_bucket: int
    tokens: list[int]
    blocks: list[list[int]]
    geoms: list[dict]
    self_terminated: bool


def _derive_self_terminated(tokens: Sequence[int]) -> bool:
    """True iff the generated tail ends in ``<cell_end>`` (the model self-stopped).

    ``generate_cell_tokens`` KEEPS the emitted ``260`` and breaks; a run that hits
    the ``max_new`` cap instead ends on an ordinary token. So the last token alone
    distinguishes the two (an empty tail is not a self-termination)."""
    return bool(tokens) and tokens[-1] == CELL_END_TOKEN_ID


def run_generation(
    cells: Sequence[ConditionedCell],
    gen_fn: Callable[[ConditionedCell, int], dict],
    *,
    base_seed: int,
    rank: int | None = None,
    world_size: int | None = None,
) -> list[GenCellRecord]:
    """Generate every cell via ``gen_fn``, sharded through :func:`sharded_eval`.

    ``score_one(i) = gen_fn(cells[i], base_seed + i)`` — the seed is keyed on the
    GLOBAL index ``i`` (not a per-rank index), so a cell's result is independent of
    the shard layout (rank independence, A5). Dispatch goes through
    ``sharded_eval(len(cells), score_one, rank=rank, world_size=world_size)``, which
    is the ONLY sharding mechanism used and which returns the count-conserved list
    in canonical global order on every rank. ``rank``/``world_size`` default to the
    live ``torch.distributed`` group (resolved lazily inside ``sharded_eval``); a
    single-process (``world_size=1``) run yields the full ordered list.

    ``gen_fn`` returns a ``score_cell``-shaped dict with ``tokens`` (the generated
    tail), ``blocks`` and ``geoms`` (aligned decode results). ``self_terminated`` is
    derived here from ``tokens``.
    """

    def score_one(i: int) -> GenCellRecord:
        cell = cells[i]
        result = gen_fn(cell, base_seed + i)
        tokens = [int(t) for t in result["tokens"]]
        blocks = [list(b) for b in result["blocks"]]
        geoms = list(result["geoms"])
        return GenCellRecord(
            cell_key=cell.cell_key,
            density_bucket=cell.density_bucket,
            tokens=tokens,
            blocks=blocks,
            geoms=geoms,
            self_terminated=_derive_self_terminated(tokens),
        )

    records = sharded_eval(len(cells), score_one, rank=rank, world_size=world_size)
    n_terminated = sum(1 for r in records if r.self_terminated)
    logger.info(
        "generation complete: %d cells (base_seed=%d), %d self-terminated",
        len(records),
        base_seed,
        n_terminated,
    )
    return records


def _record_to_json(record: GenCellRecord) -> dict[str, Any]:
    return {
        "cell_key": list(record.cell_key),
        "density_bucket": record.density_bucket,
        "tokens": record.tokens,
        "blocks": record.blocks,
        "geoms": record.geoms,
        "self_terminated": record.self_terminated,
    }


def _record_from_json(obj: dict[str, Any]) -> GenCellRecord:
    key = obj["cell_key"]
    cell_key: tuple[str, int, int, int, int] = (
        str(key[0]),
        int(key[1]),
        int(key[2]),
        int(key[3]),
        int(key[4]),
    )
    return GenCellRecord(
        cell_key=cell_key,
        density_bucket=int(obj["density_bucket"]),
        tokens=[int(t) for t in obj["tokens"]],
        blocks=[[int(t) for t in b] for b in obj["blocks"]],
        geoms=list(obj["geoms"]),
        self_terminated=bool(obj["self_terminated"]),
    )


def write_gen_artifact(records: Sequence[GenCellRecord], path: str | Path, *, meta: dict) -> None:
    """Write ``records`` + ``meta`` to ``path`` as write-once JSON.

    Refuses to overwrite an existing file (``FileExistsError``): a re-run must delete
    the old artifact deliberately, so a stale generation is never silently clobbered.
    The parent directory is created if absent."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"gen artifact already exists at {path}; it is write-once — delete "
            "deliberately only to re-generate."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": GEN_ARTIFACT_SCHEMA_VERSION,
        "meta": meta,
        "records": [_record_to_json(r) for r in records],
    }
    # Write to a temp sibling then atomic-rename, so an interrupted write cannot leave
    # a half-file that a later read would mistake for a sealed artifact.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.info("wrote gen artifact: %d records -> %s", len(records), path)


def read_gen_artifact(path: str | Path) -> tuple[dict, list[GenCellRecord]]:
    """Read a gen artifact back into ``(meta, records)``.

    Reconstructs each ``GenCellRecord`` — notably restoring ``cell_key`` to a tuple
    (JSON round-trips it as a list). Raises ``ValueError`` on an unrecognized schema
    version so a layout drift fails loud rather than silently mis-parsing."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != GEN_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"gen artifact at {path} has schema_version={version!r}, expected "
            f"{GEN_ARTIFACT_SCHEMA_VERSION!r}; refusing to parse."
        )
    meta = payload["meta"]
    records = [_record_from_json(obj) for obj in payload["records"]]
    logger.info("read gen artifact: %d records <- %s", len(records), path)
    return meta, records
