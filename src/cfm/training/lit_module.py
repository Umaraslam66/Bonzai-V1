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

from cfm.data.sub_f.vocab import vocab_tag_to_id
from cfm.data.training.conditioning import conditioning_field_to_id
from cfm.models.micro_ar import MicroAR, MicroARConfig
from cfm.training.config import ScaffoldConfig


class ScaffoldLit(L.LightningModule):
    def __init__(self, cfg: ScaffoldConfig) -> None:
        super().__init__()
        self.save_hyperparameters(cfg.model_dump())
        self.cfg = cfg
        n_subf = max(vocab_tag_to_id().values()) + 1
        n_cond = len(conditioning_field_to_id())
        self.model = MicroAR(
            MicroARConfig(
                d_model=cfg.d_model,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                n_subf_vocab=n_subf,
                n_cond=n_cond,
                max_len=cfg.max_len + n_cond,  # positions cover prefix + cell-token budget
            )
        )

    def _loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model.training_loss(
            batch["ids"], prefix_len=batch["prefix_len"], seq_len=batch.get("seq_len")
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
