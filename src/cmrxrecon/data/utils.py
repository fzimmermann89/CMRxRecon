from typing import Literal
from torch.utils.data import DataLoader, Dataset
import torch
from pathlib import Path
from typing import *
from collections import defaultdict
import numpy as np


def create_mask(lines, center_lines, acceleration, offset):
    center = lines // 2
    mask = np.zeros(lines, dtype=bool)
    mask[offset::acceleration] = 1
    mask[center - center_lines // 2 : center + center_lines // 2] = 1
    mask = np.fft.fftshift(mask)
    return mask


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
    def __init__(self, lengths: tuple[int, ...], batch_size: int, droplast: bool = True):
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
            samplesperds = [(i + self.batch_size - 1) // self.batch_size for i in self._lengths]
        return sum(samplesperds)
