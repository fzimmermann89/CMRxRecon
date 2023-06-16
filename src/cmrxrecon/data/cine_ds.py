from torch.utils.data import Dataset
import h5py
from pathlib import Path
import numpy as np
import torch
from typing import Tuple
from .utils import create_mask


class CineDataDS(Dataset):
    def __init__(
        self,
        path,
        acceleration: Tuple[int,] = (4,),
        singleslice: bool = True,
        random_acceleration: bool = False,
        center_lines: int = 24,
    ):
        """
        A Cine Dataset
        path: Path(s) to h5 files
        acceleration: tupe of acceleration factors to randomly choose from
        single slice: if true, return a single z-slice/view, otherwise return all for one subject
        random_acceleration: randomly choose offset for undersampling mask
        center_lines: ACS lines

        A sample consists of
            - undersampled k-data (shifted, k=0 is on the corner)
            - mask
            - RSS ground truth reconstruction

        Order of Axes:
         (Coils, Slice/view, Time, Phase Enc. (undersampled), Frequency Enc. (fully sampled))
        """
        if isinstance(path, (str, Path)):
            path = [Path(path)]
        self.files = sum([list(Path(p).rglob(f"P*.h5")) for p in path], [])
        self.shapes = [(h5py.File(fn)["k"]).shape for fn in self.files]
        self.accumslices = np.cumsum(np.array([s[0] for s in self.shapes]))
        self.singleslice = singleslice
        self.acceleration = acceleration
        self.random_acceleration = random_acceleration
        self.center_lines = center_lines

    def __len__(self):
        if self.singleslice:
            return self.accumslices[-1]
        else:
            return len(self.files)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        if self.singleslice:
            filenr = np.argmax(self.accumslices > idx)
            slicenr = idx - self.accumslices[filenr - 1] if filenr > 0 else idx
            data = h5py.File(self.files[filenr])["k"][slicenr : slicenr + 1]
            gt = h5py.File(self.files[filenr])["sos"][slicenr : slicenr + 1]
        else:
            data = h5py.File(self.files[idx])["k"]
            gt = h5py.File(self.files[idx])["sos"]

        acceleration = self.acceleration[int(torch.randint(len(self.acceleration), size=(1,)))]
        if self.random_acceleration:
            offset = int(torch.randint(acceleration, size=(1,)))
        else:
            offset = 0
        lines = data.shape[-5]
        mask = create_mask(lines, self.center_lines, acceleration, offset)
        tmp = torch.view_as_complex(torch.as_tensor((data[:, mask]))).permute((3, 0, 2, 1, 4))
        k = torch.zeros(*tmp.shape[:3], lines, tmp.shape[-1], dtype=torch.complex64)
        k[:, :, :, mask, :] = tmp
        mask = torch.as_tensor(mask[None, None, None, :, None]).broadcast_to(k.shape)
        gt = torch.as_tensor(gt)
        return k, mask, gt
