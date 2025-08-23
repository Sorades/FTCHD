from collections import defaultdict
from pathlib import Path

import torch
from torchmetrics import (
    AUROC,
    ROC,
    F1Score,
    Metric,
    MetricCollection,
    Recall,
    Specificity,
)
from tqdm import tqdm


@torch.no_grad()
def bootstrap_ci(
    base_metric: Metric,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_bootstraps: int = 1000,
    device: str | torch.device = "cpu",
    save_path: str | None = None,
):
    num_class = preds.shape[1] if len(preds.shape) > 1 else 1
    if isinstance(base_metric, CHDMetricCLSBinary):
        roc_curve = ROC(task="binary")
    elif isinstance(base_metric, CHDMetricFPMultilabel):
        roc_curve = None
    elif isinstance(base_metric, CHDMetricCLSMulticlass):
        roc_curve = ROC(task="multiclass", num_classes=num_class)
        name_cls = base_metric.name_cls
    else:
        raise ValueError(f"base_metric {type(base_metric)} is not supported")
    base_metric = base_metric.to(device)
    preds = preds.to(device)
    targets = targets.to(device)

    results = defaultdict(list)

    for _ in tqdm(range(num_bootstraps)):
        indices = torch.randint(0, len(preds), (len(preds),))
        pred_sample = preds[indices]
        target_sample = targets[indices]

        base_metric.update(pred_sample, target_sample)
        metrics = base_metric.compute()
        base_metric.reset()

        if roc_curve is not None:
            roc_curve.update(
                pred_sample.squeeze(), target_sample.squeeze().to(torch.int)
            )
            fpr, tpr, _ = roc_curve.compute()
            roc_curve.reset()

            if isinstance(fpr, torch.Tensor):
                results["fpr"].append(fpr.cpu())
                results["tpr"].append(tpr.cpu())
            elif isinstance(fpr, list):
                for i, nc in enumerate(name_cls):
                    results[f"fpr_{nc}"].append(fpr[i].cpu())
                    results[f"tpr_{nc}"].append(tpr[i].cpu())
        for key, value in metrics.items():
            results[key].append(value.cpu())

    ci_results = {}
    for key, values in results.items():
        if "fpr" in key or "tpr" in key:
            continue
        values = torch.stack(values)
        lower = torch.quantile(values, 0.025).item()
        upper = torch.quantile(values, 0.975).item()
        value = torch.mean(values).item()
        ci_results[key] = f"{value:.3f} ({lower:.3f}-{upper:.3f})"

    if save_path:
        _save_path: Path = Path(save_path)
        _save_path.parent.mkdir(parents=True, exist_ok=True)
        _save_path = _save_path.with_suffix(".pth")
        torch.save(results, _save_path)

    return ci_results


class CHDMetricCLSBinary(Metric):
    def __init__(self, stage: str):
        super().__init__()
        self.metric = MetricCollection(
            {
                "F1": F1Score(task="binary"),
                "SENS": Recall(task="binary"),
                "SPEC": Specificity(task="binary"),
                "AUC": AUROC(task="binary"),
            },
            prefix=f"{stage}/CLS/",
        )

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.metric.update(pred, target.to(torch.int))

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()


class CHDMetricCLSMulticlass(Metric):
    def __init__(self, name_cls: list[str], stage: str, **kwargs):
        super().__init__()
        self.name_cls = name_cls
        num_cls = len(name_cls)
        self.stage = stage
        self.metric = MetricCollection(
            {
                "F1": F1Score(
                    task="multiclass", num_classes=num_cls, average="none", **kwargs
                ),
                "SENS": Recall(
                    task="multiclass", num_classes=num_cls, average="none", **kwargs
                ),
                "SPEC": Specificity(
                    task="multiclass", num_classes=num_cls, average="none", **kwargs
                ),
                "AUC": AUROC(
                    task="multiclass", num_classes=num_cls, average="none", **kwargs
                ),
                "CLS/F1": F1Score(
                    task="multiclass", num_classes=num_cls, average="macro", **kwargs
                ),
            }
        )

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.metric.update(pred, target.to(torch.int))

    def compute(self):
        metric_out = self.metric.compute()
        cls_f1 = metric_out.pop("CLS/F1")

        ret = {
            f"{self.stage}/{self.name_cls[idx]}/{key}": v
            for key, value in metric_out.items()
            for idx, v in enumerate(value)
        }
        ret[f"{self.stage}/CLS/F1"] = cls_f1
        return ret

    def reset(self):
        self.metric.reset()


class CHDMetricFPMultilabel(Metric):
    def __init__(self, name_fp: list[str], stage: str):
        super().__init__()
        self.name_fp = name_fp
        num_labels = len(name_fp)
        self.stage = stage
        self.metric = F1Score(task="multilabel", num_labels=num_labels, average="none")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.metric.update(pred, target.to(torch.int))

    def compute(self):
        metric_out = self.metric.compute()
        ret = {
            f"{self.stage}/{self.name_fp[idx]}/F1": v
            for idx, v in enumerate(metric_out)
        }
        return ret

    def reset(self):
        self.metric.reset()
