from typing import Literal
from .cine_ds import CineDataDS, CineTestDataDS
from .mapping_ds import MappingDataDS
from torch.utils.data import DataLoader, Dataset
import torch
from pathlib import Path
from typing import Optional, Literal
import pytorch_lightning as pl
from collections import defaultdict
from .utils import MultiDataSets, MultiDataSetsSampler
from .augments import (
    RandomShuffleAlongDimensions,
    RandomKFlipUndersampled,
    RandomPhase,
    RandomFlipAlongDimensions,
    AugmentDataset,
)
from functools import partial


class CineData(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Optional[str] = "files/MultiCoil/Cine/ProcessedTrainingSet",
        axis: tuple[Literal["sax", "lax"], ...] | Literal["sax", "lax"] = "sax",
        batch_size: int = 4,
        augments: bool = False,
        singleslice: bool = True,
        center_lines: int = 24,
        random_acceleration: bool = True,
        acceleration: tuple[int, ...] = (4, 8, 10),
        return_csm: bool = False,
        test_data_dir: Optional[str] = "files/MultiCoil/Cine/ValidationSet",
    ):
        """
        A Cine Datamodule
        data_dir: Path(s) to h5 files
        axis: sax or lax
        batch_size: batch size
        augments: use data augments
        singleslice: if true, return a single z-slice/view, otherwise return all for one subject
        center_lines: ACS lines
        random_acceleration: randomly choose offset for undersampling mask
        acceleration: tupe of acceleration factors to randomly choose from
        return_csm: return coil sensitivity maps

        A sample is a dict with
            - k: undersampled k-data (shifted, k=0 is on the corner)
            - mask: mask
            - gt: RSS ground truth reconstruction
            - csm: coil sensitivity maps (optional)


        Order of Axes:
         (Coils, Slice/view, Time, Phase Enc. (undersampled), Frequency Enc. (fully sampled))
        """
        super().__init__()
        if isinstance(axis, str):
            axis = (axis,)
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.axis = axis
        self.batch_size = batch_size
        self.singleslice = singleslice
        self.return_csm = return_csm
        self.kwargs = dict(
            center_lines=center_lines,
            random_acceleration=random_acceleration,
            acceleration=acceleration,
        )

        if data_dir is not None and data_dir != "":
            different_sizes = sum([list((Path(self.data_dir) / ax).glob("*_*_*")) for ax in self.axis], [])

            if not different_sizes:
                raise ValueError(f"No data found in {Path(self.data_dir).absolute()}")

            paths = defaultdict(list)
            for sizepath in different_sizes:
                name = sizepath.name
                if self.singleslice:  # ignore number of slices
                    name = "_".join(name.split("_")[:-1])
                paths[name].append(sizepath)

            # use first dataset as validation set
            val_size = list(paths.keys())[0]
            val_ds = paths[val_size][0]
            if len(paths[val_size]) > 1:
                paths[val_size] = paths[val_size][1:]
            else:
                del paths[val_size]

            datasets = [
                CineDataDS(path, singleslice=self.singleslice, return_csm=return_csm, augments=augments, **self.kwargs)
                for path in paths.values()
            ]

            self.train_multidatasets = MultiDataSets(datasets)
            self.val_dataset = CineDataDS(val_ds, singleslice=self.singleslice, return_csm=return_csm, **self.kwargs)
        else:
            self.train_multidatasets = None
            self.val_dataset = None
        if test_data_dir is not None and test_data_dir != "":
            self.test_dataset = CineTestDataDS(
                test_data_dir, axis=self.axis, singleslice=self.singleslice, return_csm=self.return_csm
            )

    def train_dataloader(self):
        if self.train_multidatasets is None:
            raise ValueError("No training data available")
        return DataLoader(
            self.train_multidatasets,
            batch_sampler=MultiDataSetsSampler(self.train_multidatasets.lenghts(), self.batch_size),
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("No validation data available")
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=8,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("No test data available")
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=8,
            collate_fn=lambda batch: torch.utils.data._utils.collate.collate(
                batch,
                collate_fn_map={**torch.utils.data._utils.collate.default_collate_fn_map, tuple: lambda x, *args, **kwargs: x},
            ),
        )

    def predict_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("No test data available")
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=8,
            collate_fn=lambda batch: torch.utils.data._utils.collate.collate(
                batch,
                collate_fn_map={**torch.utils.data._utils.collate.default_collate_fn_map, tuple: lambda x, *args, **kwargs: x},
            ),
        )

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...


class MappingData(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "files/MultiCoil/Mapping/ProcessedTrainingSet",
        job: Literal["t1", "t2"] | tuple[Literal["t1", "t2"], ...] = "t1",
        batch_size: int = 4,
        augments: bool = False,
        singleslice: bool = True,
        center_lines: int = 24,
        random_acceleration: bool = True,
        acceleration: tuple[int, ...] = (4,),
        return_csm: bool = False,
        return_roi: bool = False,
    ):
        """
        A Cine Datamodule
        data_dir: Path(s) to h5 files
        job: t1 or t2
        batch_size: batch size
        augments: use data augments
        singleslice: if true, return a single z-slice/view, otherwise return all for one subject
        center_lines: ACS lines
        random_acceleration: randomly choose offset for undersampling mask
        acceleration: tupe of acceleration factors to randomly choose from
        return_csm: return coil sensitivity maps
        return_roi: return ROI mask

        A sample is a dict with
            - k: undersampled k-data (shifted, k=0 is on the corner)
            - mask: mask
            - gt: RSS ground truth reconstruction
            - times: time stamps
            - csm: coil sensitivity maps (optional)
            - roi: ROI mask (optional)



        Order of Axes:
         (Coils, Slice/view, Time, Phase Enc. (undersampled), Frequency Enc. (fully sampled))
        """
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.axis = job
        self.batch_size = batch_size
        self.singleslice = singleslice
        self.kwargs = dict(
            center_lines=center_lines,
            random_acceleration=random_acceleration,
            acceleration=acceleration,
        )

        if augments:
            raise NotImplementedError("Augments not implemented yet")

        different_sizes = sum([list((Path(self.data_dir) / ax).glob("*_*_*")) for ax in self.axis], [])

        if not different_sizes:
            raise ValueError(f"No data found in {Path(self.data_dir).absolute()}")

        paths = defaultdict(list)
        for sizepath in different_sizes:
            name = sizepath.name
            if self.singleslice:  # ignore number of slices
                name = "_".join(name.split("_")[:-1])
            paths[name].append(sizepath)

        # use first dataset as validation set
        val_size = list(paths.keys())[0]
        val_ds = paths[val_size][0]
        if len(paths[val_size]) > 1:
            paths[val_size] = paths[val_size][1:]
        else:
            del paths[val_size]

        datasets = [
            MappingDataDS(path, singleslice=self.singleslice, return_csm=return_csm, **self.kwargs) for path in paths.values()
        ]

        self.train_multidatasets = MultiDataSets(datasets)
        self.val_dataset = MappingDataDS(val_ds, singleslice=self.singleslice, return_csm=return_csm, **self.kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_multidatasets,
            batch_sampler=MultiDataSetsSampler(self.train_multidatasets.lenghts(), self.batch_size),
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=8,
        )

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
