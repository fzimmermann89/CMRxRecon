from typing import Literal
from .cine_ds import CineDataDS
from torch.utils.data import DataLoader, Dataset
import torch
from pathlib import Path
import pytorch_lightning as pl
from typing import *
from collections import defaultdict
from .utils import MultiDataSets, MultiDataSetsSampler


class CineData(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "files/MultiCoil/Cine/ProcessedTrainingSet",
        axis: Literal["sax", "lax"] = "sax",
        batch_size: int = 4,
        augments: bool = False,
        singleslice: bool = True,
        center_lines: int = 24,
        random_acceleration: bool = True,
        acceleration: Tuple[int, ...] = (4,),
    ):
        super().__init__()

        self.data_dir = data_dir
        self.axis = axis
        self.batch_size = batch_size
        self.singleslice = singleslice
        self.kwargs = dict(
            center_lines=center_lines,
            random_acceleration=random_acceleration,
            acceleration=acceleration,
        )

        if augments:
            raise NotImplementedError("Augments not implemented yet")

        different_sizes = list((Path(self.data_dir) / self.axis).glob("*_*_*"))

        if not different_sizes:
            raise ValueError(f"No data found in {Path(self.data_dir).absolute()}")

        paths = defaultdict(list)
        for sizepath in different_sizes:
            name = sizepath.name
            if self.singleslice:  # ignore number of slices
                name = "_".join(name.split("_")[:-1])
            paths[name].append(sizepath)

        val_size = list(paths.keys())[0]
        val_ds = paths[val_size][0]
        if len(paths[val_size]) > 1:
            paths[val_size] = paths[val_size][1:]
        else:
            del paths[val_size]

        datasets = [CineDataDS(path, singleslice=self.singleslice, **self.kwargs) for path in paths.values()]

        self.train_multidatasets = MultiDataSets(datasets)
        self.val_dataset = CineDataDS(val_ds, singleslice=self.singleslice, **self.kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_multidatasets,
            batch_sampler=MultiDataSetsSampler(self.train_multidatasets.lenghts(), self.batch_size),
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=4,
        )

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
