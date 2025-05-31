import torch
from torch import nn

from ftchd.data import Task


def process_heatmaps(heatmaps: torch.Tensor, img_size: int):
    min_vals = heatmaps.amin(dim=(1, 2), keepdim=True)
    max_vals = heatmaps.amax(dim=(1, 2), keepdim=True)
    heatmaps = (heatmaps - min_vals) / (max_vals - min_vals)
    heatmaps = nn.functional.interpolate(
        heatmaps.unsqueeze(1),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    heatmaps = heatmaps[:, 0]

    return heatmaps


def logits2probs(task: str, logits: torch.Tensor):
    """Turn logits into probabilities"""
    if task == Task.BINARY:
        return logits
    elif task == Task.MULTICLASS:
        return torch.softmax(logits, dim=-1)
    else:
        raise ValueError(f"Unknown task: {task}")


def probs2cls_idx(task: str, outputs: torch.Tensor):
    """Get the index of the predicted class"""
    if task == Task.BINARY:
        ret = outputs > 0.5
    elif task == Task.MULTICLASS:
        ret = torch.argmax(outputs, dim=-1)
    else:
        raise ValueError(f"Unknown task: {task}")
    return ret.int().detach()


def logits2cls_idx(task: str, logits: torch.Tensor):
    """Turn logits into class index"""
    return probs2cls_idx(task, logits2probs(task, logits))
