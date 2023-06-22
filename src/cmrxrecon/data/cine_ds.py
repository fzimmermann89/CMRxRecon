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
        return_csm: bool = False,
    ):
        """
        A Cine Dataset
        path: Path(s) to h5 files
        acceleration: tupe of acceleration factors to randomly choose from
        single slice: if true, return a single z-slice/view, otherwise return all for one subject
        random_acceleration: randomly choose offset for undersampling mask
        center_lines: ACS lines
        return_csm: return coil sensitivity maps

        A sample consists of a dict with
            - k: undersampled k-data (shifted, k=0 is on the corner)
            - mask: mask (z, t, x, y)
            - csm: coil sensitivity maps (optional) (c, z, x, y)
            - gt: RSS ground truth reconstruction


        Order of Axes:
         (Coils , Slice/view, Time, Phase Enc. (undersampled), Frequency Enc. (fully sampled))
        """
        if isinstance(path, (str, Path)):
            path = [Path(path)]
        self.filenames = sum([list(Path(p).rglob(f"P*.h5")) for p in path], [])
        self.shapes = [(h5py.File(fn)["k"]).shape for fn in self.filenames]
        self.accumslices = np.cumsum(np.array([s[0] for s in self.shapes]))
        self.singleslice = singleslice
        self.acceleration = acceleration
        self.random_acceleration = random_acceleration
        self.center_lines = center_lines
        self.return_csm = return_csm

    def __len__(self):
        if self.singleslice:
            return self.accumslices[-1]
        else:
            return len(self.filenames)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError

        acceleration = self.acceleration[int(torch.randint(len(self.acceleration), size=(1,)))]
        if self.random_acceleration:
            offset = int(torch.randint(acceleration, size=(1,)))
        else:
            offset = 0
        filenr = np.argmax(self.accumslices > idx) if self.singleslice else idx
        if self.singleslice:
            # return a single slice for each subject
            slicenr = idx - self.accumslices[filenr - 1] if filenr > 0 else idx
            selection = slice(slicenr, slicenr + 1)
        else:
            # return all slices for each subject
            selection = slice(None)

        with h5py.File(self.filenames[filenr], "r") as file:
            lines = file["k"].shape[-5]
            mask = create_mask(lines, self.center_lines, acceleration, offset)
            k_data = file["k"][selection, mask]
            gt = file["sos"][selection]
            if self.return_csm:
                csm = file["csm"][selection]

        k_data = torch.view_as_complex(torch.as_tensor(k_data)).permute((3, 0, 2, 1, 4))
        k = torch.zeros(*k_data.shape[:3], lines, k_data.shape[-1], dtype=torch.complex64)
        k[:, :, :, mask, :] = k_data
        mask = torch.as_tensor(mask[None, None, :, None])
        gt = torch.as_tensor(gt)
        ret = {"k": k, "mask": mask, "gt": gt}
        if self.return_csm:
            csm = torch.view_as_complex(torch.as_tensor(csm)).swapaxes(0, 1) if self.return_csm else None
            ret["csm"] = csm
        return ret
