from copy import deepcopy
from typing import Literal

import torch
from torch import nn


def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Sequential):
        for layer in m:
            init_linear_weights(layer)


class Projector(nn.Module):
    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        fp_num: int,
        proj_type: Literal["Linear", "FC", "MLP"],
        share_params: bool,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_feat
        if proj_type == "Linear":
            proj = nn.Linear(in_feat, out_feat)
        elif proj_type == "FC":
            proj = nn.Sequential(
                nn.Linear(in_feat, out_feat),
                nn.ReLU(),
            )
        elif proj_type == "MLP":
            proj = nn.Sequential(
                nn.Linear(in_feat, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_feat),
            )

        proj = nn.Sequential(nn.LayerNorm(in_feat), proj)

        if not share_params:
            proj = nn.ModuleList([deepcopy(proj) for _ in range(fp_num)])

        init_linear_weights(proj)
        self.proj = proj
        self.fp_num = fp_num

    def forward(self, x) -> torch.Tensor:
        if isinstance(self.proj, nn.ModuleList):
            return torch.stack([proj(x) for proj in self.proj], dim=1)
        elif isinstance(self.proj, nn.Module):
            return torch.stack([self.proj(x) for _ in range(self.fp_num)], dim=1)
        else:
            raise ValueError("proj should be nn.Module or nn.ModuleList")


class Predictor(nn.Module):
    def __init__(
        self,
        in_feat: int,
        fp_num: int,
        activation: nn.Module | None = None,
        share_params: bool = False,
        dropout: float | None = 0.0,
    ):
        super().__init__()
        self.share_params = share_params
        layers: list[nn.Module] = [nn.Linear(in_feat, 1)]
        if activation is not None:
            layers.append(activation)
        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))
        head = nn.Sequential(*layers)
        if not share_params:
            head = nn.ModuleList([deepcopy(head) for _ in range(fp_num)])
        init_linear_weights(head)
        self.head = head
        self.fp_num = fp_num

    def forward(self, x) -> torch.Tensor:
        if isinstance(self.head, nn.ModuleList):
            return torch.cat(
                [self.head[i](x[:, i, :]) for i in range(self.fp_num)], dim=1
            )
        elif isinstance(self.head, nn.Module):
            return torch.cat([self.head(x[:, i, :]) for i in range(self.fp_num)], dim=1)
        else:
            raise ValueError("proj should be nn.Module or nn.ModuleList")
