from dataclasses import dataclass, field

import pandas as pd
import torch
import wandb
from lightning import Callback
from wandb.sdk.wandb_run import Run

from common.logging import get_logger
from common.pl_helper import resolve_pl_dir
from ftchd.data import FP_NAMES, CHDDataItem, CHDImageDataModule, Task
from ftchd.modules.metrics import (
    CHDMetricCLSBinary,
    CHDMetricCLSMulticlass,
    CHDMetricFPMultilabel,
    bootstrap_ci,
)

log = get_logger(__name__)


@dataclass
class MidState:
    logits: list[torch.Tensor] = field(default_factory=list)
    targets: list[torch.Tensor] = field(default_factory=list)
    pred_fps: list[torch.Tensor] = field(default_factory=list)
    flow_patterns: list[torch.Tensor] = field(default_factory=list)


class Bootstrap(Callback):
    def __init__(self, num_bootstrap: int = 1000):
        super().__init__()
        self.num_bootstrap = num_bootstrap

    def on_test_start(self, trainer, pl_module):
        self.dirpath = resolve_pl_dir(trainer)
        datamodule: CHDImageDataModule = getattr(trainer, "datamodule")
        self.name_cls = datamodule.cls_name
        self.task = datamodule.task
        self.mid_state_by_domain: dict[str, MidState] = {}
        self.domain_list = list(datamodule.test_dataloader().keys())
        for domain_name in self.domain_list:
            self.mid_state_by_domain[domain_name] = MidState()

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs: dict[str, torch.Tensor],
        batch: CHDDataItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        logits = outputs["logits"]
        pred_fps = outputs["flow_patterns"]
        targets = batch.target
        flow_patterns = batch.flow_pattern
        domain_name = self.domain_list[dataloader_idx]
        self.mid_state_by_domain[domain_name].logits.append(logits)
        self.mid_state_by_domain[domain_name].pred_fps.append(pred_fps)
        self.mid_state_by_domain[domain_name].targets.append(targets)
        self.mid_state_by_domain[domain_name].flow_patterns.append(flow_patterns)

    def on_test_epoch_end(self, trainer, pl_module):
        dfs_cls, dfs_fp = [], []
        wandb_run: Run = getattr(trainer.logger, "experiment")
        for domain_name, mid_state in self.mid_state_by_domain.items():
            logits = torch.cat(mid_state.logits, dim=0)
            pred_fps = torch.cat(mid_state.pred_fps, dim=0)
            targets = torch.cat(mid_state.targets, dim=0)
            flow_patterns = torch.cat(mid_state.flow_patterns, dim=0)

            if self.task == Task.BINARY:
                metric = CHDMetricCLSBinary("BootstrapCLS")
            elif self.task == Task.MULTICLASS:
                metric = CHDMetricCLSMulticlass(self.name_cls, "BootstrapCLS")
            else:
                raise ValueError(f"Invalid task: {self.task}")
            fp_metric = CHDMetricFPMultilabel(FP_NAMES, "BootstrapFP")

            log.info(f"Bootstrap {domain_name} CLS metrics...")
            result_cls = bootstrap_ci(
                metric, logits, targets, self.num_bootstrap, pl_module.device
            )

            log.info(f"Bootstrap {domain_name} FP metrics...")
            result_fp = bootstrap_ci(
                fp_metric,
                pred_fps,
                flow_patterns,
                self.num_bootstrap,
                pl_module.device,
            )

            df_cls = pd.DataFrame.from_dict(
                result_cls, orient="index", columns=[domain_name]
            ).rename_axis(columns="Metric")
            df_fp = pd.DataFrame.from_dict(
                result_fp, orient="index", columns=[domain_name]
            ).rename_axis(columns="Metric")
            dfs_cls.append(df_cls)
            dfs_fp.append(df_fp)

            wandb_run.log(
                {
                    f"Bootstrap/CLS/{domain_name}": wandb.Table(
                        dataframe=reshape_df(df_cls)
                    ),
                    f"Bootstrap/FP/{domain_name}": wandb.Table(
                        dataframe=reshape_df(df_fp)
                    ),
                }
            )

        result_df_cls = merge_metric_dataframes(dfs_cls)
        result_df_fp = merge_metric_dataframes(dfs_fp)

        log.info(f"CLS Results:\n{result_df_cls}")
        log.info(f"FP Results:\n{result_df_fp}")

        save_dir = self.dirpath / "bootstrap"
        save_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving bootstrap results to {save_dir}")
        result_df_cls.to_excel(save_dir / f"bootstrap={self.num_bootstrap}_cls.xlsx")
        result_df_fp.to_excel(save_dir / f"bootstrap={self.num_bootstrap}_fp.xlsx")


def merge_metric_dataframes(df_list: list[pd.DataFrame]):
    merge = pd.concat(df_list, axis=1)
    merge.index = pd.MultiIndex.from_tuples(
        [(metric.split("/")[1], metric.split("/")[2]) for metric in merge.index]
    )
    merge = merge.sort_index(level=[0, 1], ascending=[True, True])
    merge = merge.reindex(["AUC", "SENS", "SPEC", "F1", "ACC"], level=1)
    return merge


def reshape_df(df: pd.DataFrame):
    df = df.copy()
    df["Class"] = df.index.str.split("/").str[-2]
    df["Metric"] = df.index.str.split("/").str[-1]

    pivot_df = df.pivot(index="Class", columns="Metric", values=df.columns[0])
    pivot_df.columns.name = None
    pivot_df.index.name = None
    return pivot_df.reindex(columns=["AUC", "SENS", "SPEC", "F1", "ACC"]).reset_index(
        names="Class"
    )
