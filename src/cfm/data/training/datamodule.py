"""CellDataModule (spec §6 trigger 1, §7) — the Lightning data layer for the slice.

``setup()`` runs the fail-closed holdout audit FIRST (on the on-disk manifest's
STAMPED lineage, so a tile lacking lineage fires G-F4) and raises BEFORE any
DataLoader/sampler is built — so a leak yields zero training steps on every rank.
Only after the audit passes does it build per-cell examples and the seeded split.

Sequence unit = CELL (spec §7). Each example is ``[conditioning prefix | cell
tokens]``: the prefix is the VALUE-BEARING conditioning id-block (F6 delivery —
``build_value_bearing_prefix`` of the shard's tile conditioning + the cell's own
density bucket); the body is the cell's sub-F token sequence. The legacy
value-agnostic field-slot block (``build_conditioning_prefix``) is retained for
reference but is no longer the training path.
Empty cells (nothing to learn) and cells exceeding the sub-F P99.9 length lock are
dropped, with counts LOGGED (no silent truncation).

Train/val split is TILE-LEVEL and seeded: a tile's cells never span train and val,
and the val split is disjoint from the 132 holdout tiles BY CONSTRUCTION (holdout
tiles were removed from the shards at build time). The DistributedSampler is seeded
from the config so a 4->4 resume continues at the same data position.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from cfm.data.training.build_shards import build_shards_in_memory, guard_city_identity
from cfm.data.training.conditioning import build_value_bearing_prefix, conditioning_field_to_id
from cfm.data.training.holdout_guard import (
    load_training_manifest,
    manifest_to_reachable,
    run_holdout_audit,
)
from cfm.data.training.shard_schema import TrainingShard

logger = logging.getLogger(__name__)

#: sub-F P99.9 cell-token length lock (spec §7). Longer cells (~0.1% of Singapore)
#: are dropped rather than truncated (truncation would cut a feature mid-grammar).
DEFAULT_MAX_CELL_TOKENS = 5760

#: id for right-padding (a valid sub-F id; padded targets are masked in the loss).
PAD_ID = 0

#: F13 action contract (readiness-closure Task 15): ceiling on the fraction of
#: NON-EMPTY cells dropped for exceeding max_cell_tokens, checked over the whole
#: manifest union in ``setup``. 0.005 = 5x the Singapore P99.9 design point behind
#: DEFAULT_MAX_CELL_TOKENS (~0.1% of SG cells exceed the lock). Above this the drop
#: is no longer a tail trim — the budget assumption itself has failed for the corpus.
MAX_TOO_LONG_DROP_RATE = 0.005


class DropRateExceeded(RuntimeError):
    """The over-length drop rate exceeded MAX_TOO_LONG_DROP_RATE (F13 action contract).

    Raised by ``CellDataModule.setup`` AFTER accumulating ``flatten_shards_to_cells``'s
    dropped counts across the whole manifest union — silently training on a corpus
    whose length tail was amputated would bias the model against exactly the dense
    cells the budget was sized to keep. Escalation: raise DEFAULT_MAX_CELL_TOKENS via
    a recorded decision or re-chunk; see readiness F13.
    """


def build_conditioning_prefix() -> list[int]:
    """LEGACY slice-v1 prefix: the field-slot id-block, in recorded order.

    Value-agnostic (one id per field) — see ``micro_ar`` DECISION. NO LONGER the
    training path: ``flatten_shards_to_cells`` builds the VALUE-BEARING prefix
    (F6 delivery). Retained as the documented slot layout (unit-tested as such)."""
    field_to_id = conditioning_field_to_id()
    return [field_to_id[f] for f in field_to_id]  # dict preserves recorded order


def _cell_prefix_ids(
    shard: TrainingShard,
    cell_density_bucket: int | None,
    seed: int,
    ablation: str = "full",
) -> tuple[int, ...]:
    """The VALUE-BEARING prefix for one (shard, cell).

    EXPLICIT field mapping (spec §3.1): never ``**tile_conditioning`` — a splat would
    let wrong-keyed constants ride in silently; strict ``tc[...]`` indexing makes key
    drift fail LOUD as KeyError. Note the deliberate renames at this seam:
    dominant_zoning_class -> zoning_class, modal_road_skeleton_class ->
    road_skeleton_class, admin_region -> region (the admin DIVISION, never the city —
    see build_shards._tile_conditioning_dict). cell_density_bucket is PER-CELL (from
    the cell payload; None allowed — the builder buckets absence to 0). ``seed`` is
    constant-bucketed by the builder, so its value is inert in the ids.
    ``city_identity`` (Task 24a) is the shard's CITY name (``TrainingShard.region``),
    registry-encoded; ``ablation`` ("no_city" = Lane S) threads from the config."""
    tc = shard.tile_conditioning
    return tuple(
        build_value_bearing_prefix(
            population_density_bucket=tc["population_density_bucket"],
            zoning_class=tc["dominant_zoning_class"],
            road_skeleton_class=tc["modal_road_skeleton_class"],
            cell_density_bucket=cell_density_bucket,
            region=tc["admin_region"],
            coastal_inland_river=tc["coastal_inland_river"],
            sub_c_morphology_class=tc["sub_c_morphology_class"],
            seed=seed,
            city_identity=shard.region,
            ablation=ablation,
        )
    )


@dataclass(frozen=True)
class CellExample:
    region: str
    tile_i: int
    tile_j: int
    cell_i: int
    cell_j: int
    prefix_ids: tuple[int, ...]
    tokens: tuple[int, ...]
    cell_density_bucket: int | None

    @property
    def key(self) -> tuple[str, int, int, int, int]:
        # region-keyed so a multi-region union is unambiguous (two cities can share a
        # (tile_i, tile_j)); the single-region split keys are a stable subset of this.
        return (self.region, self.tile_i, self.tile_j, self.cell_i, self.cell_j)

    @property
    def ids(self) -> list[int]:
        return [*self.prefix_ids, *self.tokens]

    @property
    def prefix_len(self) -> int:
        return len(self.prefix_ids)

    @property
    def seq_len(self) -> int:
        return len(self.prefix_ids) + len(self.tokens)

    @property
    def stratum(self) -> int:
        """Bucket for stratified reporting (-1 when density is unknown)."""
        return self.cell_density_bucket if self.cell_density_bucket is not None else -1


def flatten_shards_to_cells(
    shards: list[TrainingShard],
    *,
    max_cell_tokens: int = DEFAULT_MAX_CELL_TOKENS,
    seed: int = 0,
    ablation: str = "full",
) -> tuple[list[CellExample], dict[str, int]]:
    """Flatten shards to per-cell examples, dropping empty + over-length cells.

    Each example's prefix is the VALUE-BEARING conditioning block of its (shard, cell)
    — see ``_cell_prefix_ids``. ``seed`` is wired from the datamodule's run seed but is
    constant-bucketed in the prefix (the value is inert in the ids; see conditioning).
    ``ablation`` (Task 24a) threads to every prefix built here — train AND val, and
    therefore the generation-side matched conditioning too (it consumes val prefixes).

    Returns ``(examples, dropped)`` where ``dropped`` counts ``{"empty", "too_long"}``
    (LOGGED, never silently truncated). Examples are ordered by (tile, cell) so the
    flatten itself is deterministic."""
    examples: list[CellExample] = []
    dropped = {"empty": 0, "too_long": 0}
    for shard in shards:
        for cell in shard.cells:
            n = len(cell.tokens)
            if n == 0:
                dropped["empty"] += 1
                continue
            if n > max_cell_tokens:
                dropped["too_long"] += 1
                continue
            examples.append(
                CellExample(
                    region=shard.region,
                    tile_i=shard.tile_i,
                    tile_j=shard.tile_j,
                    cell_i=cell.cell_i,
                    cell_j=cell.cell_j,
                    prefix_ids=_cell_prefix_ids(shard, cell.cell_density_bucket, seed, ablation),
                    tokens=tuple(cell.tokens),
                    cell_density_bucket=cell.cell_density_bucket,
                )
            )
    examples.sort(key=lambda e: e.key)
    logger.info(
        "flatten_shards_to_cells: %d examples (dropped %d empty, %d over-length > %d)",
        len(examples),
        dropped["empty"],
        dropped["too_long"],
        max_cell_tokens,
    )
    return examples, dropped


def split_train_val(
    examples: list[CellExample],
    *,
    seed: int,
    val_fraction: float = 0.1,
) -> tuple[list[CellExample], list[CellExample]]:
    """Deterministic TILE-LEVEL split: shuffle the unique tile ids with ``seed`` and
    assign a ``val_fraction`` slice to val. A tile's cells never span train and val,
    so the val set is disjoint from holdout by construction.

    The tile key is ``(region, tile_i, tile_j)`` — region-scoped so a multi-region
    union (where two cities may share a ``(tile_i, tile_j)``) splits cleanly; for a
    single region this is identical to the legacy ``(tile_i, tile_j)`` behavior.
    ``val_fraction == 0`` yields an empty val set (whole-union train), used by the
    union span/CRS checks; otherwise at least one tile goes to val (non-vacuous)."""
    tiles = sorted({(e.region, e.tile_i, e.tile_j) for e in examples})
    rng = random.Random(seed)
    rng.shuffle(tiles)
    if not tiles or val_fraction <= 0:
        n_val = 0
    else:
        n_val = max(1, round(len(tiles) * val_fraction))
    val_tiles = set(tiles[:n_val])
    train = [e for e in examples if (e.region, e.tile_i, e.tile_j) not in val_tiles]
    val = [e for e in examples if (e.region, e.tile_i, e.tile_j) in val_tiles]
    return train, val


def _as_item(example: CellExample) -> dict[str, Any]:
    return {"ids": example.ids, "prefix_len": example.prefix_len, "seq_len": example.seq_len}


def collate_cells(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Right-pad ``ids`` to the batch max with ``PAD_ID``; carry per-example
    ``prefix_len`` + ``seq_len`` (the loss masks both the prefix and the padding)."""
    max_len = max(len(item["ids"]) for item in batch)
    ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    for row, item in enumerate(batch):
        ids[row, : len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
    return {
        "ids": ids,
        "prefix_len": torch.tensor([item["prefix_len"] for item in batch], dtype=torch.long),
        "seq_len": torch.tensor([item["seq_len"] for item in batch], dtype=torch.long),
    }


class CellDataset(Dataset):
    def __init__(self, examples: list[CellExample]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return _as_item(self._examples[idx])


def _distributed_sampler(dataset: Dataset, *, seed: int, shuffle: bool) -> DistributedSampler:
    """A DistributedSampler that constructs whether or not a process group exists:
    inside DDP it reads num_replicas/rank from the group; on CPU/1-rank it falls
    back to a single replica (so the same code path is testable off-GPU)."""
    if dist.is_available() and dist.is_initialized():
        return DistributedSampler(dataset, seed=seed, shuffle=shuffle)
    return DistributedSampler(dataset, num_replicas=1, rank=0, seed=seed, shuffle=shuffle)


class CellDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        training_manifest: Path | None = None,
        training_manifests: list[Path] | None = None,
        holdout_manifest: Path,
        seed: int = 7,
        val_fraction: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 0,
        max_cell_tokens: int = DEFAULT_MAX_CELL_TOKENS,
        expected_holdout_schema: str = "2.0",
        conditioning_ablation: str = "full",
    ) -> None:
        """Two construction modes, ONE audit-then-build-then-split pipeline:

          - SINGLE-REGION (legacy, Singapore): pass ``training_manifest`` — one per-region
            manifest is loaded, its tiles built, split.
          - MULTI-REGION UNION (bake-off EU corpus): pass ``training_manifests`` — a LIST
            of per-city schema-1.0 manifests; their cells are CONCATENATED (the union),
            audited as one reachable set against the (schema-2.0) holdout, then split.

        Exactly one of ``training_manifest`` / ``training_manifests`` must be given. The
        per-city manifests stay schema 1.0 — the union lives in this loader, NOT a new
        manifest schema."""
        super().__init__()
        if (training_manifest is None) == (training_manifests is None):
            raise ValueError(
                "CellDataModule: pass exactly one of training_manifest (single-region) "
                "or training_manifests (multi-region union)"
            )
        if training_manifest is not None:
            self._train_manifests = [Path(training_manifest)]
        else:
            self._train_manifests = [Path(p) for p in training_manifests]  # type: ignore[union-attr]
        self._holdout_manifest = Path(holdout_manifest)
        self.seed = seed
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_cell_tokens = max_cell_tokens
        self._expected_holdout_schema = expected_holdout_schema
        # Task 24a (spec §8): "full" | "no_city" | "no_character"; applied to EVERY
        # prefix this module builds (train + val => generation-side too, by construction)
        self._conditioning_ablation = conditioning_ablation
        self._train: list[CellExample] = []
        self._val: list[CellExample] = []
        self._batches_yielded = 0

    def setup(self, stage: str | None = None) -> None:
        # Runs on ALL ranks. The audit raises (halts) BEFORE any DataLoader/sampler
        # is constructed -> zero training steps execute on a leak. The audit covers the
        # UNION of all per-city manifests' reachable artifacts against the one holdout
        # manifest (single-region is the 1-manifest case of the same code path).
        manifests = [load_training_manifest(p) for p in self._train_manifests]
        holdout = yaml.safe_load(self._holdout_manifest.read_text(encoding="utf-8"))
        reachable = [a for m in manifests for a in manifest_to_reachable(m)]
        run_holdout_audit(
            holdout,
            reachable,
            expected_schema_version=self._expected_holdout_schema,  # default "2.0"
            manifest_path=self._holdout_manifest,  # F9: sha + _EVAL_SET_LOCKED verified at read
        )  # raises on any leak across the union

        # Audit passed: build per-city shards from the SAME tile set each manifest
        # lists (the manifest is the authoritative per-city training inventory), UNION
        # the cells across cities, then split. Per-city manifests stay schema 1.0; the
        # union lives here, not in a new manifest schema.
        examples: list[CellExample] = []
        total_dropped = {"empty": 0, "too_long": 0}
        carried_by_source: dict[str, list[str | None]] = {}
        for manifest in manifests:
            tile_ids = [(int(t["tile_i"]), int(t["tile_j"])) for t in manifest.get("tiles", [])]
            shards = build_shards_in_memory(
                manifest["release"], manifest["region"], tile_ids=tile_ids
            )
            # Task-24a city-identity guard input: the carried TrainingShard.region per
            # DECLARED manifest region (the value the city_identity field will encode).
            carried_by_source.setdefault(manifest["region"], []).extend(s.region for s in shards)
            city_examples, dropped = flatten_shards_to_cells(
                # seed: the run seed, constant-bucketed in the prefix (value inert in ids)
                shards,
                max_cell_tokens=self.max_cell_tokens,
                seed=self.seed,
                ablation=self._conditioning_ablation,
            )
            total_dropped["empty"] += dropped["empty"]
            total_dropped["too_long"] += dropped["too_long"]
            examples.extend(city_examples)

        # Task 24a: halt on city_identity wiring bugs (all-None city / the same city
        # value across distinct manifests) BEFORE any split/loader exists.
        guard_city_identity(carried_by_source)

        # F13 action contract: accumulate-then-check over the WHOLE union (never
        # per-city — a small city's tail must not halt a corpus it cannot skew).
        # Denominator = NON-EMPTY cells (kept + too_long): empty cells are a
        # different defect class and would dilute the length signal.
        # DECISION: the contract is ENFORCED only at/above the DEFAULT_MAX_CELL_TOKENS
        # design point. The 0.005 threshold is calibrated to that budget (5x SG
        # P99.9). Sub-design budgets are not hypothetical — they are KNOWN, accepted
        # opt-downs: three Leonardo entry points deliberately pass --max-len 2048 for
        # memory headroom / smoke (scripts/bakeoff_diagnostic.sbatch,
        # scripts/scaffold_ddp_short.sbatch, scripts/scaffold_scaleup_probe.sbatch),
        # and run_smoke's max_len=256 tiny-budget loop drops ~64% of real SG cells by
        # design. In that regime the rate measures the opt-down, not a corpus defect,
        # and the F13 escalation ('raise DEFAULT_MAX_CELL_TOKENS') is the wrong action
        # for a conscious opt-down — so the enforcement boundary is deliberate.
        # Visibility there comes from the unconditional INFO log below (counts + rate
        # + budget land in every job log). Revisit if a sub-design budget ever appears
        # in a SCORED bake-off run (those run at the 5760 default).
        too_long = total_dropped["too_long"]
        total_nonempty = len(examples) + too_long
        logger.info(
            "setup: union over-length drops %d/%d non-empty (rate %.6f) at "
            "max_cell_tokens=%d; %d empty dropped",
            too_long,
            total_nonempty,
            (too_long / total_nonempty) if total_nonempty else 0.0,
            self.max_cell_tokens,
            total_dropped["empty"],
        )
        if (
            self.max_cell_tokens >= DEFAULT_MAX_CELL_TOKENS
            and total_nonempty
            and too_long / total_nonempty > MAX_TOO_LONG_DROP_RATE
        ):
            raise DropRateExceeded(
                f"over-length drop rate {too_long}/{total_nonempty} = "
                f"{too_long / total_nonempty:.6f} exceeds threshold "
                f"{MAX_TOO_LONG_DROP_RATE} (5x the SG P99.9 design point): "
                f"raise DEFAULT_MAX_CELL_TOKENS via a recorded decision or "
                f"re-chunk; see readiness F13"
            )

        examples.sort(key=lambda e: e.key)  # deterministic union order across cities
        self._train, self._val = split_train_val(
            examples, seed=self.seed, val_fraction=self.val_fraction
        )

    @property
    def train_cells(self) -> list[CellExample]:
        return self._train

    @property
    def val_cells(self) -> list[CellExample]:
        return self._val

    def train_order(self) -> list[tuple[str, int, int, int, int]]:
        """The seeded training order (epoch 0) as example keys — same seed yields the
        same order, so a checkpoint resumes at the same data position."""
        sampler = _distributed_sampler(CellDataset(self._train), seed=self.seed, shuffle=True)
        sampler.set_epoch(0)
        return [self._train[i].key for i in sampler]

    def train_dataloader(self) -> DataLoader:
        dataset = CellDataset(self._train)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=_distributed_sampler(dataset, seed=self.seed, shuffle=True),
            num_workers=self.num_workers,
            collate_fn=collate_cells,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = CellDataset(self._val)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=_distributed_sampler(dataset, seed=self.seed, shuffle=False),
            num_workers=self.num_workers,
            collate_fn=collate_cells,
        )
