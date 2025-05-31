from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Literal

import torch
from lightning import LightningModule
from torch import nn

from common.logging import get_logger
from ftchd.data import FP_NAMES, FP_WEIGHTS, CHDDataItem, Setting, Task
from ftchd.misc import process_heatmaps
from ftchd.modules.base_models import ModelFactory, OutputFeatExtractor
from ftchd.modules.fp_layers import Predictor, Projector

BinaryOutput = namedtuple(
    "BinaryOutput", ["logits", "flow_patterns", "fp_embeds", "heatmaps", "feat"]
)
SubtypeOutput = namedtuple(
    "SubtypeOutput", ["logits", "flow_patterns", "attn_maps", "heatmaps", "feat"]
)

log = get_logger(__name__)


class _Mixin(ABC):
    @abstractmethod
    def shared_step(
        self, batch: CHDDataItem, batch_idx: int, stage: Literal["Train", "Val", "Test"]
    ):
        pass

    def training_step(self, batch: CHDDataItem, batch_idx: int = 0):
        return self.shared_step(batch, batch_idx, "Train")

    def validation_step(self, batch: CHDDataItem, batch_idx: int = 0):
        return self.shared_step(batch, batch_idx, "Val")

    def test_step(self, batch: CHDDataItem, batch_idx: int = 0, dataloader_idx=0):
        return self.shared_step(batch, batch_idx, "Test")


class BinaryModel(_Mixin, LightningModule):
    def __init__(
        self,
        base_model_name: str,
        setting: str,
        apply_fp_reweight: bool = False,
        img_size: int = 224,
        base_model_pretrain: bool = True,
        fp_proj_type: Literal["Linear", "FC", "MLP"] = "FC",
        fp_hidden_dim: int = 512,
        fp_predictor_dropout: float | None = None,
        fp_share_params: bool = False,
        logit_norm_t: float | None = None,
    ):
        super().__init__()
        self.setting = Setting(setting)
        self.task = self.setting.task
        if self.task != Task.BINARY:
            raise ValueError(f"Only support binary task, but got {self.task}")
        self.cls_num = self.setting.cls_num
        self.fp_num = len(FP_NAMES)
        self.cls_name = self.setting.cls_name
        self.fp_name = FP_NAMES

        self.base_model_name = base_model_name
        self.apply_fp_reweight = apply_fp_reweight
        self.img_size = img_size
        self.fp_proj_type = fp_proj_type
        self.fp_hidden_dim = fp_hidden_dim
        self.fp_share_params = fp_share_params
        self.logit_norm_t = logit_norm_t

        hf_model = ModelFactory[self.base_model_name]
        self.base_model = hf_model.get(
            num_classes=self.cls_num, base_model=True, pretrained=base_model_pretrain
        )
        self.feat_extractor = hf_model.feat_extractor(self.base_model)
        feat_dim = hf_model.feat_dim

        self.fp_projector = Projector(
            in_feat=feat_dim,
            out_feat=self.fp_hidden_dim,
            fp_num=self.fp_num,
            proj_type=self.fp_proj_type,
            share_params=self.fp_share_params,
        )

        self.predictor = Predictor(
            self.fp_hidden_dim,
            self.fp_num,
            nn.LeakyReLU(),
            self.fp_share_params,
            fp_predictor_dropout,
        )

        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_guide = MeanIoULoss()

    def on_train_start(self):
        if self.apply_fp_reweight:
            pos_weight = torch.Tensor(FP_WEIGHTS).to(self.device)
            log.info(f"cls pos_weight: {pos_weight}")
            self.loss_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def setup_compile(self):
        self.feat_extractor = torch.compile(self.feat_extractor)

    def forward(self, x: torch.Tensor):
        featex_output: OutputFeatExtractor = self.feat_extractor(x, output_heatmap=True)

        fp_embeds: torch.Tensor = self.fp_projector(featex_output.feat)
        flow_patterns: torch.Tensor = self.predictor(fp_embeds)

        if self.logit_norm_t is not None:
            flow_patterns = logit_norm(flow_patterns, self.logit_norm_t)

        logits, _ = torch.min(flow_patterns, dim=1)
        logits = (1 - logits.sigmoid()).unsqueeze(1)

        return BinaryOutput(
            heatmaps=featex_output.heatmaps,
            feat=featex_output.feat,
            fp_embeds=fp_embeds,
            logits=logits,
            flow_patterns=flow_patterns,
        )

    def shared_step(
        self, batch: CHDDataItem, batch_idx: int, stage: Literal["Train", "Val", "Test"]
    ):
        images: torch.Tensor = batch.image
        flow_patterns: torch.Tensor = batch.flow_pattern
        masks: torch.Tensor = batch.mask
        bs = batch.image.size(0)

        output: BinaryOutput = self(images)

        loss_fp = self.loss_bce(output.flow_patterns, flow_patterns)

        heatmaps = process_heatmaps(output.heatmaps, self.img_size)
        loss_guide = self.loss_guide(heatmaps, masks)

        loss = loss_fp + loss_guide

        loss_dict = {f"Loss/{stage}/FP": loss_fp, f"Loss/{stage}/Guide": loss_guide}

        self.log(f"Loss/{stage}", loss, batch_size=bs, prog_bar=True, sync_dist=True)
        self.log_dict(loss_dict, batch_size=bs, sync_dist=True)

        ret_dict = {"loss": loss, **output._asdict()}

        return ret_dict


class SubtypeModel(_Mixin, LightningModule):
    def __init__(
        self,
        setting: str,
        img_size: int = 224,
        logit_norm_t: float | None = None,
        enable_prior: bool = False,
        cls_dim: int = 512,
        attn_type: Literal["cross_attn", "self_attn"] = "cross_attn",
        pretrained: bool = True,
    ):
        super().__init__()
        self.cls_setting = Setting(setting)
        self.task = self.cls_setting.task
        if self.task != Task.MULTICLASS:
            raise ValueError(f"Only support multiclass task, but got {self.task}")
        self.cls_num = self.cls_setting.cls_num
        self.fp_num = len(FP_NAMES)
        self.cls_name = self.cls_setting.cls_name
        self.fp_name = FP_NAMES
        self.num_heads = 8
        self.img_size = img_size
        self.logit_norm_t = logit_norm_t
        self.enable_prior = enable_prior
        self.attn_type = attn_type
        self.pretrained = pretrained

        self.bin_model = load_binary_model(pretrained=pretrained)

        self.fp_hidden_dim = self.bin_model.fp_hidden_dim
        self.hidden_proj = nn.Sequential(
            nn.Linear(self.fp_hidden_dim, cls_dim),
            nn.ReLU(),
        )

        self.cls_token = nn.Parameter(torch.randn(1, self.cls_num - 1, cls_dim))

        self.attn = nn.MultiheadAttention(
            cls_dim, self.num_heads, dropout=0, batch_first=True
        )

        if pretrained:
            for _, param in self.bin_model.named_parameters():
                param.requires_grad = False

        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_guide = MeanIoULoss()

    @torch.inference_mode()
    def binary_forward(self, x: torch.Tensor) -> BinaryOutput:
        return self.bin_model(x)

    def setup_compile(self):
        self.bin_model.setup_compile()

    def forward(self, x: torch.Tensor) -> SubtypeOutput:
        bs = x.size(0)
        bin_output = self.binary_forward(x)

        fp_embeds = bin_output.fp_embeds.clone()
        fp_probs = bin_output.flow_patterns.sigmoid()
        if self.enable_prior:
            fp_embeds: torch.Tensor = fp_embeds * torch.unsqueeze(fp_probs, -1)
        fp_embeds = self.hidden_proj(fp_embeds)

        logits = torch.zeros((bs, self.cls_num), device=self.device)
        if self.attn_type == "cross_attn":
            init_attn_maps = torch.zeros(
                (bs, self.num_heads, self.cls_num - 1, self.fp_num),
                device=self.device,
            )
        elif self.attn_type == "self_attn":
            init_attn_maps = torch.zeros(
                (
                    bs,
                    self.num_heads,
                    self.cls_num - 1 + self.fp_num,
                    self.cls_num - 1 + self.fp_num,
                ),
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown attention type: {self.attn_type}")
        if self.training:
            abnormal_mask = torch.ones(bs).bool()
        else:
            abnormal_mask = torch.any(fp_probs < 0.5, dim=-1).bool().squeeze()
            logits[~abnormal_mask, 0] = 1
            if (~abnormal_mask).all():
                return SubtypeOutput(
                    logits=logits,
                    flow_patterns=bin_output.flow_patterns,
                    attn_maps=init_attn_maps,
                    heatmaps=bin_output.heatmaps,
                    feat=bin_output.feat,
                )

        abnormal_bs: torch.Tensor = abnormal_mask.sum()
        cls_tokens = self.cls_token.repeat(abnormal_bs, 1, 1)  # type: ignore
        abnormal_fp_embeds = fp_embeds[abnormal_mask]
        if self.attn_type == "cross_attn":
            out_values, attn_maps = self.attn(
                cls_tokens,
                abnormal_fp_embeds,
                abnormal_fp_embeds,
                average_attn_weights=False,
            )
        elif self.attn_type == "self_attn":
            inp_emb = torch.cat([cls_tokens, abnormal_fp_embeds], dim=1)
            out_values, attn_maps = self.attn(
                inp_emb, inp_emb, inp_emb, average_attn_weights=False
            )

        logits[abnormal_mask, 1:] += out_values[:, : self.cls_num - 1, :].mean(-1)
        init_attn_maps[abnormal_mask, :] += attn_maps

        if self.logit_norm_t is not None:
            logits = logit_norm(logits, self.logit_norm_t)

        output = SubtypeOutput(
            logits=logits,
            flow_patterns=bin_output.flow_patterns,
            attn_maps=init_attn_maps,
            heatmaps=bin_output.heatmaps,
            feat=bin_output.feat,
        )
        return output

    def shared_step(
        self, batch: CHDDataItem, batch_idx: int, stage: Literal["Train", "Val", "Test"]
    ):
        images: torch.Tensor = batch.image
        targets: torch.Tensor = batch.target
        flow_patterns: torch.Tensor = batch.flow_pattern
        masks: torch.Tensor = batch.mask
        bs = batch.image.size(0)

        output: SubtypeOutput = self(images)

        loss_cls = self.loss_ce(output.logits, targets)

        loss_fp = self.loss_bce(output.flow_patterns, flow_patterns)

        heatmaps = process_heatmaps(output.heatmaps, self.img_size)
        loss_guide = self.loss_guide(heatmaps, masks)

        loss_dict = {
            "FP": loss_fp,
            "CLS": loss_cls,
            "Guide": loss_guide,
        }

        loss = loss_fp + loss_cls + loss_guide

        self.log(f"Loss/{stage}", loss, batch_size=bs, prog_bar=True, sync_dist=True)
        self.log_dict(loss_dict, batch_size=bs, sync_dist=True)

        ret_dict = {"loss": loss, **output._asdict()}

        return ret_dict


def load_binary_model(pretrained: bool = True) -> BinaryModel:
    bin_model = BinaryModel(
        setting="binary",
        base_model_name="vit_base",
        base_model_pretrain=False,
        logit_norm_t=0.01,
    )
    if pretrained:
        state_dict = torch.load(
            "data/weights/ftchd_binary.ckpt", map_location="cpu", weights_only=True
        )["state_dict"]
        bin_model.load_state_dict(state_dict, strict=True)
        bin_model.freeze()
    return bin_model


def logit_norm(logits: torch.Tensor, temp: float = 1.0):
    norms = torch.norm(logits, p=2, dim=-1, keepdim=True)
    return torch.div(logits, norms) / temp


class MeanIoULoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.epsilon = 1e-6

    def forward(self, heatmap: torch.Tensor, mask: torch.Tensor):
        if mask.dtype == torch.uint8:
            mask = mask.float() / 255.0

        intersection = (heatmap * mask).sum(dim=(1, 2))
        union = (heatmap + mask).sum(dim=(1, 2)) - intersection

        iou = (intersection + self.epsilon) / (union + self.epsilon)

        loss = 1 - iou.mean()

        return loss
