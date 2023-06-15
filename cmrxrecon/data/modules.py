from typing import Literal
from .cine_ds import CineDataDS
from torch.utils.data import DataLoader, Dataset
import torch
from pathlib import Path
import pytorch_lightning as pl
from typing import *
from collections import defaultdict


class MultiDataSets(Dataset):
    def __init__(self, datasets: tuple[Dataset]):
        super().__init__()
        self.datasets = datasets

    def __getitem__(self, index: tuple[int, tuple[int, ...]]) -> Any:
        return self.datasets[index[0]][index[1]]

    def __len__(self) -> int:
        return len(self.datasets)

    def lenghts(self):
        return tuple([len(d) for d in self.datasets])


class MultiDataSetsSampler(torch.utils.data.Sampler):
    def __init__(
        self, lengths: tuple[int, ...], batch_size: int, droplast: bool = True
    ):
        self.batch_size = batch_size
        self._lengths = lengths
        self.droplast = droplast

    def __iter__(self) -> Iterator[int]:

        batches = []
        for ds_id, lenghts_of_ds in enumerate(self._lengths):
            perm = torch.randperm(lenghts_of_ds)
            for i in range(
                0,
                len(perm) - self.batch_size if self.droplast else len(perm),
                self.batch_size,
            ):
                sample_ids = perm[i : i + self.batch_size]
                ids = [(ds_id, int(sample_id)) for sample_id in sample_ids]
                batches.append(ids)

        for idx in torch.randperm(len(batches)):
            yield batches[idx]

    def __len__(self) -> int:
        if self.droplast:
            samplesperds = [i // self.batch_size for i in self._lengths]
        else:
            samplesperds = [
                (i + self.batch_size - 1) // self.batch_size for i in self._lengths
            ]
        return sum(samplesperds)


class CineData(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "cmrxrecon/data/files/MultiCoil/Cine/ProcessedTrainingSet",
        axis: Literal["sax", "lax"] = "sax",
        batch_size: int = 4,
        augments: bool = False,
        singleslice: bool = True,
        center_lines: int = 24,
        random_acceleration: bool = False,
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

        different_sizes = (Path(self.data_dir) / self.axis).glob("*_*_*")
        paths = defaultdict(list)
        for sizepath in different_sizes:
            name = sizepath.name
            if self.singleslice:  # ignore number of slices
                name = "_".join(name.split("_")[:-1])
            paths[name].append(sizepath)
        
        val_size=list(paths.keys())[0]
        val_ds=paths[val_size][0]
        if len(paths[val_size])>1:
            paths[val_size]=paths[val_size][1:]
        else:
            del paths[val_size]
        
        datasets = [
            CineDataDS(path, singleslice=self.singleslice, **self.kwargs)
            for path in paths.values()
        ]

        self.train_multidatasets = MultiDataSets(datasets)
        self.val_dataset = CineDataDS(val_ds, singleslice=self.singleslice, **self.kwargs)

    # %%

    def train_dataloader(self):
        return DataLoader(
            self.train_multidatasets,
            batch_sampler=MultiDataSetsSampler(
                self.train_multidatasets.lenghts(), self.batch_size
            ),
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=1,
        )


    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
