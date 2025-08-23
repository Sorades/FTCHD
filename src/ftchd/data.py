import os
from collections import namedtuple
from enum import StrEnum
from functools import partial

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common.logging import get_logger

log = get_logger(__name__)

CHDDataItem = namedtuple(
    "CHDDataItem",
    [
        "image",
        "flow_pattern",
        "mask",
        "target",
        "id",
        "case_id",
        "img_path",
        "mask_path",
    ],
)

FP_VALUE = {
    "ChamberNum": ["<2", ">=2"],
    "FlowNum": ["Single", "Two"],
    "FlowSymmetry": ["Unequal", "Equal"],
}
FP_NAMES = list(FP_VALUE.keys())
FP_WEIGHTS = [0.006, 0.23, 0.23]

CHD_MEAN = [0.325, 0.325, 0.325]
CHD_STD = [0.229, 0.229, 0.229]


class CHDImageDataModule(LightningDataModule):
    def __init__(
        self,
        setting: str,
        annotation_file: str = "data/binary_label.csv",
        apply_cls_resample: bool = False,
        apply_fp_reweight: bool = False,
        root_dir: str = "data",
        img_size: int = 224,
        mixed_img: bool = False,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.setting = Setting(setting)
        self.task = self.setting.task
        self.cls_num = self.setting.cls_num
        self.cls_name = self.setting.cls_name
        self.num_workers = num_workers
        self.fp_values = FP_VALUE

        self.apply_cls_resample = apply_cls_resample
        self.apply_fp_reweight = apply_fp_reweight
        self.annotation_file = annotation_file
        self.root_dir = root_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.mixed_img = mixed_img

        self.annotation = pd.read_csv(annotation_file)

    def setup(self, stage):
        _dataset = partial(
            CHDImageDataset,
            task=self.task,
            root_dir=self.root_dir,
            cls_name=self.cls_name,
            fp_values=self.fp_values,
            mixed_img=self.mixed_img,
            img_size=self.img_size,
        )
        train_trans = img_train_aug(self.img_size)
        eval_trans = img_eval_aug(self.img_size)
        annt = self.annotation

        if stage == "fit":
            train_df = annt[annt["Stage"] == "Train"]
            val_df = annt[annt["Stage"] == "Val"]
            self.trainset: CHDImageDataset = _dataset(
                annotation=train_df, transform=train_trans
            )
            self.valset = _dataset(annotation=val_df, transform=eval_trans)
        elif stage == "validate":
            val_df = annt[annt["Stage"] == "Val"]
            self.valset = _dataset(annotation=val_df, transform=eval_trans)
        elif stage == "test":
            test_dfs = {
                test_name: annt[annt["Stage"] == test_name]
                for test_name in annt[annt["Source"] == "External"]["Stage"]
                .value_counts()
                .keys()
                .to_list()
            }
            self.testsets = {
                dm: _dataset(annotation=test_dfs[dm], transform=eval_trans)
                for dm in test_dfs
            }
            log.info(f"Test Data: {list(self.testsets.keys())}")
        else:
            raise ValueError(f"Invalid stage: {stage}")

    @property
    def label_stats(self):
        int_label = self.annotation[self.annotation["Source"] == "Internal"]
        ext_label = self.annotation[self.annotation["Source"] == "External"]

        def format_stats(group):
            mean_age = group.mean()
            std_age = group.std()
            min_age = group.min()
            max_age = group.max()
            return f"{int(mean_age)} ({int(std_age)}; {int(min_age)}-{int(max_age)})"

        int_age_stats = int_label.groupby("Stage")["Age"].agg(format_stats).to_frame().T
        intall_age_stats = (
            int_label.groupby("Source")["Age"]
            .agg(format_stats)
            .to_frame()
            .T.rename(columns={"Internal": "IntAll"})
        )
        int_age_stats = pd.concat([int_age_stats, intall_age_stats], axis=1)
        ext_age_stats = (
            ext_label.groupby("Domain")["Age"].agg(format_stats).to_frame().T
        )
        extall_age_stats = (
            ext_label.groupby("Source")["Age"]
            .agg(format_stats)
            .to_frame()
            .T.rename(columns={"External": "ExtAll"})
        )
        ext_age_stats = pd.concat([ext_age_stats, extall_age_stats], axis=1)
        age_stats = pd.concat([int_age_stats, ext_age_stats], axis=1)

        int_stats = pd.crosstab(
            int_label["Label"], int_label["Stage"], margins=True
        ).rename(columns={"All": "IntAll"})
        ext_stats = pd.crosstab(
            ext_label["Label"], ext_label["Domain"], margins=True
        ).rename(columns={"All": "ExtAll"})

        cls_stats = pd.concat([int_stats, ext_stats], axis=1)
        cls_stats = cls_stats.reindex(self.cls_name + ["All"], axis=0)

        fp_stats = []
        for fp in FP_NAMES:
            int_fp_stats = (
                pd.crosstab(int_label[fp], int_label["Stage"], margins=True)
                .rename(columns={"All": "IntAll"})
                .drop("All")
            )
            ext_fp_stats = (
                pd.crosstab(ext_label[fp], ext_label["Domain"], margins=True)
                .rename(columns={"All": "ExtAll"})
                .drop("All")
            )

            fp_stats.append(pd.concat([int_fp_stats, ext_fp_stats], axis=1))

        fp_stats = pd.concat(fp_stats, axis=0, keys=FP_NAMES)
        cls_stats.index = pd.MultiIndex.from_product([["Diagnosis"], cls_stats.index])
        stats = pd.concat([cls_stats, fp_stats], axis=0).fillna(0).astype(int)

        age_stats.index = pd.MultiIndex.from_product(
            [age_stats.index, ["Mean (SD; Range)"]]
        )
        stats = pd.concat([age_stats, stats], axis=0)

        return stats

    def train_dataloader(self) -> DataLoader:
        if self.apply_cls_resample:
            sample_weights, _ = self.trainset.resample_cls_weights()
            log.info("WeightedRandomSampler setup.")
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            return DataLoader(
                self.trainset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                sampler=sampler,
            )
        else:
            return DataLoader(
                self.trainset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> dict[str, DataLoader]:
        return {
            dm: DataLoader(
                self.testsets[dm],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            for dm in self.testsets
        }

    def __str__(self):
        return f"<{self.__class__.__name__}>:\n{self.label_stats}"


class CHDImageDataset(Dataset):
    def __init__(
        self,
        task: str,
        root_dir: str,
        annotation: pd.DataFrame | str,
        cls_name: list[str],
        fp_values: dict[str, list[str]],
        transform: A.Compose,
        mixed_img: bool = False,
        img_size: int = 224,
    ):
        super().__init__()
        self.root_dir = root_dir
        if isinstance(annotation, str):
            self.annotation = pd.read_csv(annotation)
        elif isinstance(annotation, pd.DataFrame):
            self.annotation = annotation
        else:
            raise ValueError(
                "annotation_file must be a DataFrame or a path to a CSV file"
            )
        self.cls_name = cls_name
        self.fp_values = fp_values
        self.task = task
        self.mixed_img = mixed_img
        self.img_size = img_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, idx: int):
        item = self.annotation.iloc[idx]
        item_id = item["ID"]
        case_id = item["CaseID"]
        target = self.cls_name.index(item["Label"])
        if self.task == Task.BINARY:
            target = torch.FloatTensor([target])

        concept = []
        for fp_name in self.fp_values:
            fp_value = item[fp_name]
            if fp_name == "ChamberNum":
                fp_value = ">=2" if fp_value >= 2 else "<2"
            fp_idx = self.fp_values[fp_name].index(fp_value)
            concept.append(fp_idx)
        concept = torch.FloatTensor(concept)

        img_path = os.path.join(self.root_dir, "images", f"{item_id}.png")
        mask_path = os.path.join(self.root_dir, "masks", f"{item_id}.png")

        bgr_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.mixed_img:
            _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            img = img * binary_mask[:, :, np.newaxis]

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        return CHDDataItem(
            img,
            concept,
            mask,
            target,
            item_id,
            case_id,
            img_path,
            mask_path,
        )

    def resample_cls_weights(self) -> tuple[list[float], list[int]]:
        target_df = self.annotation["Label"]
        target_counts = target_df.value_counts()
        class_weights = {cls: 1.0 / count for cls, count in target_counts.items()}
        sample_weights = target_df.map(class_weights).tolist()
        group_counts = target_counts.tolist()

        return sample_weights, group_counts


def img_train_aug(img_size):
    return A.Compose(
        [
            A.ToGray(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussianBlur(p=0.2),
            A.RandomResizedCrop(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(CHD_MEAN, CHD_STD),
            ToTensorV2(),
        ]
    )


def img_eval_aug(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(CHD_MEAN, CHD_STD),
            ToTensorV2(),
        ]
    )


class Task(StrEnum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


class Setting(StrEnum):
    BINARY = "binary"
    SUBTYPE = "subtype"

    @property
    def task(self) -> "Task":
        if self == self.BINARY:
            return Task.BINARY
        else:
            return Task.MULTICLASS

    @property
    def cls_num(self) -> int:
        if self == self.BINARY:
            return 1
        else:
            return len(self.cls_name)

    @property
    def cls_name(self) -> list[str]:
        if self == self.BINARY:
            return ["Normal", "Abnormal"]
        elif self == self.SUBTYPE:
            return ["Normal", "AVSD", "HV", "FSV"]
        else:
            raise ValueError(f"Invalid CLSSetting: {self}")
