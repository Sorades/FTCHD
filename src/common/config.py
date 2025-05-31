import importlib
from functools import partial, update_wrapper
from pathlib import Path
from types import MethodType
from typing import Any

from lightning import LightningModule
from lightning.pytorch.cli import ReduceLROnPlateau
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from common.logging import get_logger

log = get_logger(__name__)


class InstantiationError(Exception):
    pass


def load_config() -> DictConfig:
    root_dir = Path("config")
    base_cfg = OmegaConf.load(root_dir / "base.yaml")

    cli_cfg = OmegaConf.from_cli()
    assert isinstance(cli_cfg, DictConfig), "CLI config must be a DictConfig"

    exp = cli_cfg.get("exp")
    assert exp is not None, (
        "You must specify an experiment name with exp=<binary|subtype>"
    )
    exp_cfg = OmegaConf.load(root_dir / f"{exp}.yaml")
    final_cfg = OmegaConf.merge(base_cfg, exp_cfg, cli_cfg)
    OmegaConf.resolve(final_cfg)

    return final_cfg  # type: ignore


def str_to_class(class_path: str) -> Any:
    """Convert string path to class"""
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise InstantiationError(f"Error loading class {class_path}: {e}")


def instantiate(init_cfg: dict | DictConfig, *args, **kwargs) -> Any:
    """Instantiate an object from config"""
    if isinstance(init_cfg, DictConfig):
        init_cfg = OmegaConf.to_container(init_cfg, resolve=True)  # type: ignore

    if init_cfg is None:
        return None

    if not isinstance(init_cfg, dict):
        raise InstantiationError("`Config` must be a dictionary")

    try:
        target = init_cfg.pop("_target_")
        cls = str_to_class(target)

        init_cfg.update(kwargs)

        return cls(*args, **init_cfg)
    finally:
        if target is not None:
            init_cfg["_target_"] = target


def add_configure_optimier_method_to_model(
    model: LightningModule,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler | None = None,
):
    def configure_optimizers(
        lightning_module: LightningModule,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None = None,
    ):
        """Override to customize the :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers` method.

        Args:
            lightning_module: A reference to the model.
            optimizer: The optimizer.
            lr_scheduler: The learning rate scheduler (if used).

        """
        if lr_scheduler is None:
            return optimizer
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": lr_scheduler.monitor,
                },
            }
        return [optimizer], [lr_scheduler]

    fn = partial(configure_optimizers, optimizer=optimizer, lr_scheduler=lr_scheduler)
    update_wrapper(fn, configure_optimizers)  # necessary for `is_overridden`
    # override the existing method
    model.configure_optimizers = MethodType(fn, model)
