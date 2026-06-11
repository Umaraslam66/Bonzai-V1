"""ScaffoldLit — the LightningModule (model + step + optim) for the slice.

Wires the toy ``MicroAR`` to the REAL vocab: the head predicts the sealed sub-F
range (``max sub-F id + 1``) and the embedding additionally covers the appended
conditioning id-block (``len(conditioning_field_to_id())``). The model's positional
capacity is the cell-token budget PLUS the conditioning prefix so a full-length cell
fits after prepending conditioning.

Next-token CE masks the conditioning prefix AND right-padding (``seq_len``). The
validation metric is ``val_loss`` on the INTERNAL val split — never the frozen
holdout (that eval is run once, out-of-loop; see ``cfm.eval.slice_metrics``).
"""

from __future__ import annotations

import lightning as L
import torch

from cfm.models.backbone import build_backbone
from cfm.training.config import ScaffoldConfig


class ScaffoldLit(L.LightningModule):
    def __init__(self, cfg: ScaffoldConfig) -> None:
        super().__init__()
        # Seed BEFORE constructing the model so weight init is reproducible across
        # separate processes (the experiment = config + commit + snapshot). Without
        # this, a fresh process inits from nondeterministic startup RNG, so two runs
        # of the same config diverge from step 0 — and a 4->4 resume can never be
        # bit-identical to an uninterrupted reference (the init differs).
        L.seed_everything(cfg.seed, workers=True)
        self.save_hyperparameters(cfg.model_dump())
        self.cfg = cfg
        # Swappable backbone (§9). transformer-ar today; mamba-hybrid / discrete-diffusion
        # are gated behind Task 5's mamba-ssm verify-before-lock. The shared embedding spans
        # the value-bearing conditioning id span; the head is the sealed sub-F range.
        self.model = build_backbone(cfg.backbone, cfg)

    def _loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model.training_loss(
            batch["ids"],
            prefix_len=batch["prefix_len"],
            seq_len=batch.get("seq_len"),
            # Task 24b: the continuous character carrier; the char-built model's
            # forward REFUSES None, so a batch missing the key fails loud here.
            char_stats=batch.get("char_stats"),
        )
        return out.loss

    def training_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)  # selection signal
        return loss

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.cfg.max_steps))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
