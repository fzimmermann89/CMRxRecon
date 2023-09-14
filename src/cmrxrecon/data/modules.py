from typing import Literal
from .cine_ds import CineDataDS, CineSelfSupervisedDataDS, CineTestDataDS
from .mapping_ds import MappingDataDS, MappingTestDataDS
from torch.utils.data import DataLoader, Dataset
import torch
from pathlib import Path
from typing import Optional, Literal
import pytorch_lightning as pl
from collections import defaultdict
from .utils import MultiDataSets, MultiDataSetsSampler
from functools import partial


class CineData(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Optional[str] = "files/MultiCoil/Cine/ProcessedTrainingSet",
        selfsupervised_data_dir: Optional[str] = "files/MultiCoil/Cine/ValidationSelfSupervised",
        mode: Literal["supervised", "selfsupervised", "both"] = "supervised",
        selfsupervised_acceleration: tuple[tuple[int, int], ...] = ((4, 0), (8, 0), (8, 4), (10, 0)),
        axis: tuple[Literal["sax", "lax"], ...] | Literal["sax", "lax"] = "sax",
        batch_size: int = 4,
        augments: bool = False,
        singleslice: bool = True,
        center_lines: int = 24,
        random_acceleration: bool | float = True,
        acceleration: tuple[int, ...] = (4, 8, 10),
        return_csm: bool = False,
        return_kfull: bool = False,
        return_kfull_ift_fs: bool = False,
        test_data_dir: Optional[str] = "files/MultiCoil/Cine/ValidationSet",
        val_acceleration: tuple[int, ...] | None = None,
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
        self.return_kfull = return_kfull
        self.return_kfull_ift_fs = return_kfull_ift_fs
        self.kwargs = dict(
            center_lines=center_lines,
            singleslice=singleslice,
        )

        unique_accelerations = sorted(set(acceleration))

        if mode in ("both", "selfsupervised") and selfsupervised_data_dir is not None and selfsupervised_data_dir != "":
            different_sizes = sorted(sum([list((Path(selfsupervised_data_dir) / ax).glob("*_*_*")) for ax in self.axis], []))
            if not different_sizes:
                raise ValueError(f"No data found in {Path(selfsupervised_data_dir).absolute()}")

            paths = defaultdict(list)
            for sizepath in different_sizes:
                name = sizepath.name
                if self.singleslice:  # ignore number of slices
                    name = "_".join(name.split("_")[:-1])
                paths[name].append(sizepath)

            train_self_supervised_datasets = [
                CineSelfSupervisedDataDS(
                    path,
                    acceleration=selfsupervised_acceleration,
                    augments=augments,
                    return_ktarget_ift_fs=return_kfull_ift_fs,
                    **self.kwargs,
                )
                for path in paths.values()
            ]
        else:
            train_self_supervised_datasets = []

        if data_dir is not None and data_dir != "":
            different_sizes = sorted(sum([list((Path(self.data_dir) / ax).glob("*_*_*")) for ax in self.axis], []))

            if not different_sizes:
                print(f"No data found in {Path(self.data_dir).absolute()}")

            val_paths = defaultdict(list)
            paths = defaultdict(list)
            for sizepath in different_sizes:
                name = sizepath.name
                if self.singleslice:  # ignore number of slices
                    name = "_".join(name.split("_")[:-1])
                if "val" in name:
                    val_paths[name].append(sizepath)
                else:
                    paths[name].append(sizepath)
            if mode in ("both", "supervised"):
                train_supervised_datasets = [
                    CineDataDS(
                        path,
                        return_csm=return_csm,
                        return_kfull=return_kfull,
                        return_kfull_ift_fs=return_kfull_ift_fs,
                        random_acceleration=random_acceleration,
                        augments=augments,
                        acceleration=acceleration,
                        **self.kwargs,
                    )
                    for path in paths.values()
                ]
            else:
                train_supervised_datasets = []

            if val_acceleration is None:
                val_acceleration = unique_accelerations

            val_datasets = [
                CineDataDS(
                    val_path, return_csm=return_csm, acceleration=acc, random_acceleration=False, augments=False, **self.kwargs
                )
                for val_path in sorted(list(val_paths.values()))
                for acc in val_acceleration
            ]

            self.val_multidatasets = MultiDataSets(val_datasets)
        else:
            train_supervised_datasets = []
            train_self_supervised_datasets = []
            self.val_multidatasets = None

        train_datasets = train_self_supervised_datasets + train_supervised_datasets
        self.train_multidatasets = MultiDataSets(train_datasets) if len(train_datasets) > 0 else None

        if test_data_dir is not None and test_data_dir != "":
            self.test_dataset = CineTestDataDS(test_data_dir, axis=self.axis, return_csm=self.return_csm, slicepersample=1)

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
        if self.val_multidatasets is None:
            raise ValueError("No validation data available")
        return DataLoader(
            self.val_multidatasets,
            batch_sampler=MultiDataSetsSampler(self.val_multidatasets.lenghts(), batch_size=1, droplast=False, shuffle=False),
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
            raise ValueError("No predict data available")
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
        job: Literal["t1", "t2"] | tuple[Literal["t1", "t2"], ...] = ["t1", "t2"],
        batch_size: int = 1,
        augments: bool = True,
        singleslice: bool = True,
        center_lines: int = 24,
        random_acceleration: bool | float = 0.5,
        acceleration: tuple[int, ...] = (4, 8, 10),
        return_csm: bool = False,
        return_roi: bool = False,
        return_roi_dilated: bool = False,
        return_kfull_ift_fs: bool = True,
        test_data_dir: Optional[str] = "files/MultiCoil/Mapping/ValidationSet",
        val_acceleration: tuple[int, ...] | None = None,
        mode: Literal["supervised", "selfsupervised", "both"] = "supervised",
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
        return_roi_dilated: return dilated ROI mask
        return_kfull_ift_fs: return fully sampled hybrid-space data, i.e. k along undersampled and image along fully sampled (last)


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
        self.jobs = job
        self.batch_size = batch_size
        self.singleslice = singleslice
        self.return_kfull_ift_fs = return_kfull_ift_fs
        self.return_csm = return_csm
        self.return_roi = return_roi
        self.return_roi_dilated = return_roi_dilated

        self.kwargs = dict(
            center_lines=center_lines,
        )

        unique_accelerations = sorted(set(acceleration))

        if mode in ("both", "selfsupervised"):
            raise ValueError("Self-supervised mode not supported for MappingData")
        else:
            train_self_supervised_datasets = []

        if data_dir is not None and data_dir != "":
            different_sizes = sum(
                [list((Path(self.data_dir) / f"{job.capitalize()}map").glob("*_*_*")) for job in self.jobs], []
            )

            if not different_sizes:
                print(f"No data found in {Path(self.data_dir).absolute()}")

            val_paths = defaultdict(list)
            paths = defaultdict(list)
            for sizepath in different_sizes:
                name = sizepath.name
                if self.singleslice:  # ignore number of slices
                    name = "_".join(name.split("_")[:-1])
                if "val" in name:
                    val_paths[name].append(sizepath)
                else:
                    paths[name].append(sizepath)

            if mode in ("both", "supervised"):
                train_supervised_datasets = [
                    MappingDataDS(
                        path,
                        return_csm=return_csm,
                        return_kfull_ift_fs=return_kfull_ift_fs,
                        random_acceleration=random_acceleration,
                        augments=augments,
                        acceleration=acceleration,
                        return_roi=return_roi,
                        return_roi_dilated=return_roi_dilated,
                        **self.kwargs,
                    )
                    for path in paths.values()
                ]
            else:
                train_supervised_datasets = []

            if val_acceleration is None:
                val_acceleration = unique_accelerations

            val_datasets = [
                MappingDataDS(
                    val_path,
                    return_csm=return_csm,
                    acceleration=acc,
                    random_acceleration=False,
                    return_roi_dilated=True,
                    return_kfull_ift_fs=False,
                    augments=False,
                    **self.kwargs,
                )
                for val_path in sorted(list(val_paths.values()))
                for acc in val_acceleration
            ]

            self.val_multidatasets = MultiDataSets(val_datasets)

        else:
            train_supervised_datasets = []
            train_self_supervised_datasets = []
            self.val_multidatasets = None

        train_datasets = train_self_supervised_datasets + train_supervised_datasets
        self.train_multidatasets = MultiDataSets(train_datasets) if len(train_datasets) > 0 else None

        if test_data_dir is not None and test_data_dir != "":
            self.test_dataset = MappingTestDataDS(test_data_dir, job=self.jobs, return_csm=self.return_csm)

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
        if self.val_multidatasets is None:
            raise ValueError("No validation data available")
        return DataLoader(
            self.val_multidatasets,
            batch_sampler=MultiDataSetsSampler(self.val_multidatasets.lenghts(), batch_size=1, droplast=False, shuffle=False),
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
            raise ValueError("No predict data available")
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
