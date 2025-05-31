from typing import Any

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import LRSchedulerConfig
from torch.optim.lr_scheduler import LambdaLR


class LRWarmup(Callback):
    def __init__(self, warmup_steps: int):
        assert warmup_steps > 0, "warmup_steps should be greater than 0"
        self.warmup_steps = warmup_steps

    def on_train_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self.original_scheduler_configs = trainer.lr_scheduler_configs
        self.warmup_scheduler = LambdaLR(
            trainer.optimizers[0], lambda step: min(1.0, (step + 1) / self.warmup_steps)
        )

        trainer.strategy.lr_scheduler_configs = [
            LRSchedulerConfig(
                scheduler=self.warmup_scheduler, interval="step", frequency=1
            )
        ]

    def on_train_batch_start(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        # switch to the original scheduler after warmup_steps
        if trainer.global_step == self.warmup_steps:
            trainer.strategy.lr_scheduler_configs = self.original_scheduler_configs
