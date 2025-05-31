from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from torchmetrics.functional.segmentation import mean_iou

from common.logging import get_logger
from common.pl_helper import resolve_pl_dir
from ftchd.data import CHDDataItem, CHDImageDataModule
from ftchd.misc import logits2cls_idx, process_heatmaps

log = get_logger(__name__)


class HeatmapDrawing(Callback):
    def __init__(
        self,
        dirpath: str | None = None,
        vis_type: Literal["cam_only", "cam_on_img", "all", None] = None,
        heatmap_size: int = 224,
        heatmap_ratio: float = 0.4,
        file_ext: str = "jpg",
        cam_prefix: str = "cam",
    ) -> None:
        """if 'vis_type = None', then only miou to be logged"""
        super().__init__()
        self.cls_name: list[str]

        self.dirpath = Path(dirpath) if dirpath is not None else None
        self.vis_type = vis_type
        self.heatmap_size = heatmap_size
        self.heatmap_ratio = heatmap_ratio
        self.line = np.ones((heatmap_size, 3, 3)) * 255
        self.cam_prefix = cam_prefix
        self.file_ext = file_ext

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.dirpath is None:
            self.dirpath = resolve_pl_dir(trainer)
        datamodule: CHDImageDataModule = getattr(trainer, "datamodule")
        self.cam_dir_by_idx: dict[int, Path] = {}
        self.miou_input_by_idx: dict[int, dict] = {}
        self.cls_name = datamodule.cls_name
        self.task = datamodule.task
        for idx, domain_name in enumerate(datamodule.test_dataloader().keys()):
            self.miou_input_by_idx[idx] = {"preds": [], "target": []}
            cam_dir = self.dirpath / f"{self.cam_prefix}_{domain_name}"
            self.cam_dir_by_idx[idx] = cam_dir
            if self.vis_type is None:
                continue
            (cam_dir / "correct").mkdir(parents=True, exist_ok=True)
            (cam_dir / "wrong").mkdir(parents=True, exist_ok=True)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: CHDDataItem,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        targets: torch.Tensor = batch.target
        masks: torch.Tensor = batch.mask
        paths: list[str] = batch.img_path
        ids: list[str] = batch.id
        logits: torch.Tensor = outputs["logits"]
        heatmaps: torch.Tensor = outputs["heatmaps"]

        cls_idxs = logits2cls_idx(self.task, logits)
        heatmaps = process_heatmaps(heatmaps, self.heatmap_size)

        heatmaps = heatmaps.detach().cpu()
        masks = masks.detach().cpu()

        self.miou_input_by_idx[dataloader_idx]["preds"].append(
            (heatmaps.unsqueeze(1) > 0.5).int()
        )
        self.miou_input_by_idx[dataloader_idx]["target"].append(
            (masks.unsqueeze(1) > 127).int()
        )

        if self.vis_type is None:
            return

        cam_dir = self.cam_dir_by_idx[dataloader_idx]
        correct_dir = cam_dir / "correct"
        wrong_dir = cam_dir / "wrong"
        _g = zip(heatmaps.numpy(), masks.numpy(), paths, targets, cls_idxs, ids)
        for heatmap, mask, path, target, cls_idx, data_id in _g:
            target, cls_idx = target.item(), cls_idx.item()
            heatmap = cv2.applyColorMap(
                (heatmap * 255.0).astype(np.uint8), cv2.COLORMAP_JET
            )
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            raw_img = cv2.imread(path)
            raw_img = cv2.resize(raw_img, (self.heatmap_size, self.heatmap_size))
            map2save = self.final_map2save(raw_img, heatmap, mask)
            file_name = f"[lbl={self.cls_name[int(target)]}][pred={self.cls_name[int(cls_idx)]}][id={data_id}].{self.file_ext}"
            cam_path = (
                correct_dir / file_name if target == cls_idx else wrong_dir / file_name
            )
            cv2.imwrite(str(cam_path), map2save)

    def on_test_epoch_end(self, trainer, pl_module):
        datamodule: CHDImageDataModule = getattr(trainer, "datamodule")
        for idx, domain_name in enumerate(datamodule.test_dataloader().keys()):
            miou_input = {
                k: torch.cat(v) for k, v in self.miou_input_by_idx[idx].items()
            }
            miou = mean_iou(
                miou_input["preds"], miou_input["target"], num_classes=1
            ).mean()
            log.info(f"Mean IoU: {miou:.4f}")
            pl_module.log(f"{domain_name}/Test/mIoU", miou)

    def final_map2save(self, raw_img, heatmap, mask_img):
        match self.vis_type:
            case "cam_only":
                return heatmap
            case "cam_on_img":
                return (1 - self.heatmap_ratio) * raw_img + self.heatmap_ratio * heatmap
            case "all":
                mix = (1 - self.heatmap_ratio) * raw_img + self.heatmap_ratio * heatmap
                return np.concatenate(
                    [mix, self.line, raw_img, self.line, mask_img, self.line, heatmap],
                    axis=1,
                )
            case _:
                raise ValueError(f"Invalid vis_type: {self.vis_type}")
