import torch
from lightning import Callback
from torchmetrics import Metric

from ftchd.data import FP_NAMES, CHDDataItem, CHDImageDataModule, Task
from ftchd.modules.metrics import (
    CHDMetricCLSBinary,
    CHDMetricCLSMulticlass,
    CHDMetricFPMultilabel,
)


class CLSMetricCaculator(Callback):
    def __init__(self):
        super().__init__()

    def setup(self, trainer, pl_module, stage):
        datamodule: CHDImageDataModule = getattr(trainer, "datamodule")
        self.name_cls = datamodule.cls_name
        self.task = datamodule.task

    def on_fit_start(self, trainer, pl_module):
        if self.task == Task.BINARY:
            self.metric_train = CHDMetricCLSBinary(stage="Train").to(pl_module.device)
            self.metric_val = CHDMetricCLSBinary(stage="Val").to(pl_module.device)
        elif self.task == Task.MULTICLASS:
            self.metric_train = CHDMetricCLSMulticlass(self.name_cls, stage="Train").to(
                pl_module.device
            )
            self.metric_val = CHDMetricCLSMulticlass(self.name_cls, stage="Val").to(
                pl_module.device
            )

    def on_test_start(self, trainer, pl_module):
        datamodule: CHDImageDataModule = getattr(trainer, "datamodule")
        self.metric_test_by_idx: dict[int, Metric] = {}
        for idx, data_name in enumerate(datamodule.test_dataloader().keys()):
            if self.task == Task.BINARY:
                self.metric_test_by_idx[idx] = CHDMetricCLSBinary(stage=data_name).to(
                    pl_module.device
                )
            elif self.task == Task.MULTICLASS:
                self.metric_test_by_idx[idx] = CHDMetricCLSMulticlass(
                    self.name_cls, stage=data_name
                ).to(pl_module.device)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: dict[str, torch.Tensor],
        batch: CHDDataItem,
        batch_idx,
    ):
        self.metric_train.update(outputs["logits"], batch.target)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch: CHDDataItem, batch_idx
    ):
        self.metric_val.update(outputs["logits"], batch.target)

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs: dict[str, torch.Tensor],
        batch: CHDDataItem,
        batch_idx,
        dataloader_idx=0,
    ):
        self.metric_test_by_idx[dataloader_idx].update(outputs["logits"], batch.target)

    def on_train_epoch_end(self, trainer, pl_module):
        train_metrics = self.metric_train.compute()
        val_metrics = self.metric_val.compute()
        self.metric_train.reset()
        self.metric_val.reset()

        pl_module.log_dict(train_metrics, sync_dist=True)
        pl_module.log_dict(val_metrics, sync_dist=True)

    def on_test_epoch_end(self, trainer, pl_module):
        for _, metric in self.metric_test_by_idx.items():
            test_metrics = metric.compute()
            metric.reset()
            pl_module.log_dict(test_metrics, sync_dist=True)


class FPMultilabelMetricCalculator(Callback):
    def __init__(self):
        super().__init__()
        self.fp_name = FP_NAMES

    def on_fit_start(self, trainer, pl_module):
        self.metric_train = CHDMetricFPMultilabel(self.fp_name, stage="Train").to(
            pl_module.device
        )
        self.metric_val = CHDMetricFPMultilabel(self.fp_name, stage="Val").to(
            pl_module.device
        )

    def on_test_start(self, trainer, pl_module):
        datamodule: CHDImageDataModule = getattr(trainer, "datamodule")
        self.metric_test_by_idx: dict[int, Metric] = {}
        for idx, domain_name in enumerate(datamodule.test_dataloader().keys()):
            self.metric_test_by_idx[idx] = CHDMetricFPMultilabel(
                self.fp_name, stage=domain_name
            ).to(pl_module.device)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: dict[str, torch.Tensor],
        batch: CHDDataItem,
        batch_idx,
    ):
        self.metric_train.update(outputs["flow_patterns"], batch.flow_pattern)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch: CHDDataItem, batch_idx
    ):
        self.metric_val.update(outputs["flow_patterns"], batch.flow_pattern)

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs: dict[str, torch.Tensor],
        batch: CHDDataItem,
        batch_idx,
        dataloader_idx=0,
    ):
        self.metric_test_by_idx[dataloader_idx].update(
            outputs["flow_patterns"], batch.flow_pattern
        )

    def on_train_epoch_end(self, trainer, pl_module):
        train_metrics = self.metric_train.compute()
        val_metrics = self.metric_val.compute()
        self.metric_train.reset()
        self.metric_val.reset()

        pl_module.log_dict(train_metrics, sync_dist=True)
        pl_module.log_dict(val_metrics, sync_dist=True)

    def on_test_epoch_end(self, trainer, pl_module):
        for _, metric in self.metric_test_by_idx.items():
            test_metrics = metric.compute()
            metric.reset()
            pl_module.log_dict(test_metrics, sync_dist=True)
