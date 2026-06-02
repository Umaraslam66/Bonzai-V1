"""CellDataModule (spec §6 trigger 1, §7) — the Lightning data layer for the slice.

``setup()`` runs the fail-closed holdout audit FIRST (on the on-disk manifest's
STAMPED lineage, so a tile lacking lineage fires G-F4) and raises BEFORE any
DataLoader/sampler is built — so a leak yields zero training steps on every rank.
Only after the audit passes does it build per-cell examples and the seeded split.

Sequence unit = CELL (spec §7). Each example is ``[conditioning prefix | cell
tokens]``: the prefix is the field-slot conditioning id-block (value-agnostic in
slice v1 — see ``micro_ar`` DECISION); the body is the cell's sub-F token sequence.
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

from cfm.data.training.build_shards import build_shards_in_memory
from cfm.data.training.conditioning import conditioning_field_to_id
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


def build_conditioning_prefix() -> list[int]:
    """The slice-v1 conditioning prefix: the field-slot id-block, in recorded order.

    Value-agnostic (one id per field) — see ``micro_ar`` DECISION. The tier-1
    conditioning VALUES live on the shard / ``conditioning_prefix_ids`` for trigger-2
    identity + future compliance scoring (out-of-slice)."""
    field_to_id = conditioning_field_to_id()
    return [field_to_id[f] for f in field_to_id]  # dict preserves recorded order


@dataclass(frozen=True)
class CellExample:
    tile_i: int
    tile_j: int
    cell_i: int
    cell_j: int
    prefix_ids: tuple[int, ...]
    tokens: tuple[int, ...]
    cell_density_bucket: int | None

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.tile_i, self.tile_j, self.cell_i, self.cell_j)

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
) -> tuple[list[CellExample], dict[str, int]]:
    """Flatten shards to per-cell examples, dropping empty + over-length cells.

    Returns ``(examples, dropped)`` where ``dropped`` counts ``{"empty", "too_long"}``
    (LOGGED, never silently truncated). Examples are ordered by (tile, cell) so the
    flatten itself is deterministic."""
    prefix = tuple(build_conditioning_prefix())
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
                    tile_i=shard.tile_i,
                    tile_j=shard.tile_j,
                    cell_i=cell.cell_i,
                    cell_j=cell.cell_j,
                    prefix_ids=prefix,
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
    so the val set is disjoint from holdout by construction."""
    tiles = sorted({(e.tile_i, e.tile_j) for e in examples})
    rng = random.Random(seed)
    rng.shuffle(tiles)
    n_val = max(1, round(len(tiles) * val_fraction)) if tiles else 0
    val_tiles = set(tiles[:n_val])
    train = [e for e in examples if (e.tile_i, e.tile_j) not in val_tiles]
    val = [e for e in examples if (e.tile_i, e.tile_j) in val_tiles]
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
        training_manifest: Path,
        holdout_manifest: Path,
        seed: int = 7,
        val_fraction: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 0,
        max_cell_tokens: int = DEFAULT_MAX_CELL_TOKENS,
    ) -> None:
        super().__init__()
        self._train_manifest = Path(training_manifest)
        self._holdout_manifest = Path(holdout_manifest)
        self.seed = seed
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_cell_tokens = max_cell_tokens
        self._train: list[CellExample] = []
        self._val: list[CellExample] = []
        self._batches_yielded = 0

    def setup(self, stage: str | None = None) -> None:
        # Runs on ALL ranks. The audit raises (halts) BEFORE any DataLoader/sampler
        # is constructed -> zero training steps execute on a leak.
        manifest = load_training_manifest(self._train_manifest)
        holdout = yaml.safe_load(self._holdout_manifest.read_text(encoding="utf-8"))
        run_holdout_audit(holdout, manifest_to_reachable(manifest))  # raises on any leak

        # Audit passed: build per-cell examples from the SAME tile set the manifest
        # lists (the manifest is the authoritative training inventory), then split.
        tile_ids = [(int(t["tile_i"]), int(t["tile_j"])) for t in manifest.get("tiles", [])]
        shards = build_shards_in_memory(manifest["release"], manifest["region"], tile_ids=tile_ids)
        examples, _ = flatten_shards_to_cells(shards, max_cell_tokens=self.max_cell_tokens)
        self._train, self._val = split_train_val(
            examples, seed=self.seed, val_fraction=self.val_fraction
        )

    @property
    def train_cells(self) -> list[CellExample]:
        return self._train

    @property
    def val_cells(self) -> list[CellExample]:
        return self._val

    def train_order(self) -> list[tuple[int, int, int, int]]:
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
