from abc import ABC, abstractmethod
from pathlib import Path

import torch
from lightning import Callback, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from rich import traceback
from wandb.sdk.wandb_run import Run

from common.config import (
    add_configure_optimier_method_to_model,
    instantiate,
    load_config,
)
from common.logging import get_logger
from ftchd.data import CHDImageDataModule

torch.set_float32_matmul_precision("high")
traceback.install()
log = get_logger(__name__)


class Pipeline(ABC):
    def __init__(self, config_file: str):
        self.config_file = config_file

        self.config: DictConfig
        self.wandb_logger: WandbLogger
        self.wandb_run: Run
        self.datamodule: CHDImageDataModule
        self.model: LightningModule
        self.callbacks: list[Callback] | None = None
        self.trainer: Trainer

        self.load_config()
        self.init_wandb()
        self.init_data()
        self.init_model()
        self.setup_optim_scheduler()
        self.setup_callbacks()
        self.setup_trainer()

    @abstractmethod
    def __call__(self): ...

    @classmethod
    def start(cls, config_file: str):
        pipeline = cls(config_file)
        try:
            log.info(f"Starting {cls.__name__} with {pipeline.config_file}")
            pipeline()
        except Exception as e:
            log.exception(f"Error: {e}")
        finally:
            pipeline.finish()

    def load_config(self):
        self.config = load_config()
        seed_everything(self.config.get("seed"))
        Path(self.config["wandb"]["save_dir"]).mkdir(parents=True, exist_ok=True)

    def init_wandb(self):
        self.wandb_logger = instantiate(self.config.get("wandb"))
        cfg_dict = OmegaConf.to_container(self.config, resolve=True)
        self.wandb_logger.log_hyperparams(cfg_dict)  # type: ignore
        self.wandb_run = self.wandb_logger.experiment
        log.info(f"Run: {self.wandb_run}")

    def init_data(self):
        self.datamodule = instantiate(self.config.get("data"))
        log.info(f"Datamodule: {self.datamodule}")

    def init_model(self):
        self.model = instantiate(self.config.get("model"))
        log.info(f"Model: {type(self.model)}")
        if self.config.get("compile") and hasattr(self.model, "setup_compile"):
            self.model.setup_compile()
            log.info("Applied torch.compile")

    def setup_optim_scheduler(self):
        assert self.model is not None, (
            "Model must be initialized before setting up optimizer and scheduler"
        )
        optimizer, scheduler = None, None
        if self.config.get("optimizer") is not None:
            optimizer = instantiate(
                self.config.get("optimizer"),
                params=filter(lambda p: p.requires_grad, self.model.parameters()),
            )
            log.info(f"Optim: {type(optimizer)}, lr: {optimizer.param_groups[0]['lr']}")

            if self.config.get("scheduler") is not None:
                scheduler = instantiate(
                    self.config.get("scheduler"), optimizer=optimizer
                )
                log.info(f"Scheduler: {type(scheduler)}")
            add_configure_optimier_method_to_model(self.model, optimizer, scheduler)

    def setup_callbacks(self):
        if self.config.get("callbacks") is not None:
            self.callbacks = [
                instantiate(cb_cfg)
                for _, cb_cfg in self.config.get("callbacks").items()
                if cb_cfg is not None
            ]
            log.info(f"Callbacks: {[type(cb) for cb in self.callbacks]}")

    def setup_trainer(self):
        assert self.wandb_logger is not None, (
            "WandbLogger must be initialized before setting up trainer"
        )
        self.trainer = instantiate(
            self.config.get("trainer"),
            callbacks=self.callbacks,
            logger=self.wandb_logger,
        )

    def finish(self):
        run_cfg = self.wandb_run.config.as_dict()
        run_dir = Path(self.wandb_run.dir)
        run_dir.joinpath("exp_cfg.yaml").write_text(OmegaConf.to_yaml(run_cfg))
        self.wandb_run.save("exp_cfg.yaml")
        self.datamodule.label_stats.to_excel(run_dir.joinpath("label_stats.xlsx"))
        self.datamodule.annotation.to_csv(run_dir.joinpath("annotations.csv"))
        self.wandb_run.save("label_stats.xlsx")
        self.wandb_run.save("annotations.csv")
        self.wandb_run.finish()
