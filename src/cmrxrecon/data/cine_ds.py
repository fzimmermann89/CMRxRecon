from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Callable
from cmrxrecon.models.utils.csm import sigpy_espirit
from cmrxrecon.models.utils import rss
from cmrxrecon.data.utils import create_mask
from cmrxrecon.data.augments import CineAugment


class CineDataDS(Dataset):
    def __init__(
        self,
        path,
        acceleration: tuple[int, ...] = (4,),
        singleslice: bool = True,
        random_acceleration: bool | float = False,
        center_lines: int = 24,
        return_csm: bool = False,
        augments: bool = False,
        return_kfull: bool = False,
        return_kfull_ift_fs: bool = False,
    ):
        """
        A Cine Dataset
        path: Path(s) to h5 files
        acceleration: tupe of acceleration factors to randomly choose from
        single slice: if true, return a single z-slice/view, otherwise return all for one subject
        random_acceleration: randomly choose offset for undersampling mask
        center_lines: ACS lines
        return_csm: return coil sensitivity maps
        augments: augment data
        return_kfull: return fully sampled k space data
        return_kfull_ift_fs: return kfull as data after ifft along fully sampled dimension

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
        self.filenames = sum([list(Path(p).rglob("P*.h5")) for p in path], [])
        self.shapes = [(h5py.File(fn)["k"]).shape for fn in self.filenames]
        self.accumslices = np.cumsum(np.array([s[0] for s in self.shapes]))
        self.singleslice = singleslice
        if isinstance(acceleration, (int, float)):
            acceleration = (acceleration,)
        self.acceleration = acceleration
        self.random_acceleration = random_acceleration
        self.center_lines = center_lines
        self.return_csm = return_csm
        self.return_kfull = return_kfull
        self.return_kfull_ift_fs = return_kfull_ift_fs

        if augments:
            self.augments: Callable = CineAugment(
                p_flip_spatial=0.4,
                p_flip_temporal=0.2,
                p_shuffle_coils=0.2,
                p_phase=0.2,
                flip_view=False,
            )
        else:
            self.augments = lambda x: x

    def __len__(self):
        if self.singleslice:
            return self.accumslices[-1]
        else:
            return len(self.filenames)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError

        acceleration = self.acceleration[int(torch.randint(len(self.acceleration), size=(1,)))]
        if self.random_acceleration and torch.rand(1) < self.random_acceleration:
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
            slicenr = None

        with h5py.File(self.filenames[filenr], "r") as file:
            lines = file["k"].shape[-5]
            mask = create_mask(lines, self.center_lines, acceleration, offset)
            if self.return_kfull or self.return_kfull_ift_fs:
                k = torch.as_tensor(np.array(file["k"][selection]))
                gt = None
            else:
                k_tmp = torch.as_tensor(np.array(file["k"][selection, mask]))
                k = torch.zeros(k_tmp.shape[0], lines, *k_tmp.shape[2:], dtype=k_tmp.dtype)
                k[:, mask, :, :, :] = k_tmp
                gt = file["sos"][selection]
            if self.return_csm:
                csm = file["csm"][selection]
            if slicenr is None:
                slicevalue = np.arange(0, file["sos"].shape[0]) / (file["sos"].shape[0] - 1)
            else:
                slicevalue = slicenr / (file["sos"].shape[0] - 1)

        k = torch.view_as_complex(k).permute((3, 0, 2, 1, 4))
        mask = torch.as_tensor(mask[None, None, :, None])
        ret = {
            "k": k,
            "mask": mask,
            "gt": gt,
            "acceleration": float(acceleration),
            "offset": float(offset),
            "axis": float("sax" in self.filenames[filenr].parent.parent.stem),
            "slice": slicevalue,
        }
        if self.return_csm:
            csm = torch.view_as_complex(torch.as_tensor(csm)).swapaxes(0, 1) if self.return_csm else None
            ret["csm"] = csm

        ret = self.augments(ret)

        if self.return_kfull or self.return_kfull_ift_fs:
            kfull = ret.pop("k")
            ret["k"] = kfull * mask
            xfull = torch.fft.ifft2(kfull, norm="ortho")
            ret["gt"] = rss(xfull, 0)
        if self.return_kfull_ift_fs:
            ret["kfull_ift_fs"] = torch.fft.ifft(kfull, norm="ortho", dim=-1)
        if self.return_kfull:
            ret["kfull"] = kfull

        return ret


class CineTestDataDS(Dataset):
    def __init__(
        self,
        path: str | Path | tuple[str | Path, ...],
        axis: str | tuple[str, ...] = ("lax", "sax"),
        slicepersample: int = 1,
        return_csm: bool = False,
    ):
        """
        A Cine Validation Dataset
        path: Path(s) to h5 files
        single slice: if true, return a single z-slice/view, otherwise return all for one subject
        return_csm: return coil sensitivity maps via espirit

        A sample consists of a dict with
            - k: undersampled k-data (shifted, k=0 is on the corner)
            - mask: mask (z, t, x, y)
            - csm: coil sensitivity maps (optional) (c, z, x, y)


        Order of Axes:
         (Coils , Slice/view, Time, Phase Enc. (undersampled), Frequency Enc. (fully sampled))
        """
        if isinstance(path, (str, Path)):
            path = (Path(path),)
        if isinstance(axis, str):
            axis = (axis,)
        filenames = sorted(sum([list(Path(p).rglob(f"cine_{ax}.mat")) for p in path for ax in axis], []))

        self.filenames = [filename for filename in filenames if "AccFactor" in str(filename)]

        shapes = [self._getdata(fn).shape for fn in self.filenames]
        slices = np.array([s[1] for s in shapes])

        samples = []
        for filenumber, slicesinfile in enumerate(slices):
            for start in range(0, slicesinfile, slicepersample):
                stop = min(start + slicepersample, slicesinfile)
                samples.append((filenumber, start, stop))
        self.samples = samples
        self.return_csm = return_csm

    @staticmethod
    def _getdata(file: str | Path | h5py.File):
        if isinstance(file, (str, Path)):
            file = h5py.File(file, "r")
        key = next(iter(file.keys()))
        return file[key]

    @staticmethod
    def _shift(data: np.ndarray) -> np.ndarray:
        """
        shift k-space so that k=0 is in the corner and
        A=fft2 without any shifts, i.e.
        perform fft(fftshift(ifft(ifftshift(data))
        """
        data = np.fft.ifftshift(data, axes=(-1, -2))
        data = np.fft.ifft2(data)
        data = np.fft.fftshift(data, axes=(-1, -2))
        data = np.fft.fft2(data)
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> dict[str, Any]:
        if idx >= len(self):
            raise IndexError

        filenr, start, stop = self.samples[idx]
        filename = self.filenames[filenr]

        with h5py.File(filename, "r") as file:
            data = self._getdata(file)
            shape = [data.shape[i] for i in (2, 1, 0, 3, 4)]
            selection = slice(start, stop)
            slicenr = np.arange(start, stop)
            k_data_centered = np.array(data[:, selection]).view(np.complex64)  # (t,z,c,us,fs)
            slicevalue = slicenr / (data.shape[1] - 1)
            if len(slicevalue) == 1:
                slicevalue = slicevalue[0]

        k_data = self._shift(k_data_centered).transpose((2, 1, 0, 3, 4))  # (c,z,t,us,fs)
        k_data = k_data.astype(np.complex64)
        mask = (~np.isclose(k_data[0, ..., :, :1], 0)).astype(np.float32)
        axis = float("sax" in filename.stem.split("_")[-1])
        try:
            acc_idx = str(filename).find("AccFactor")
            acceleration = float(str(filename)[acc_idx + 9 : acc_idx + 11])
        except:
            acceleration = 8.0

        ret = {
            "k": k_data,
            "mask": mask,
            "sample": (filename, selection, shape),
            "axis": axis,
            "acceleration": acceleration,
            "offset": 0.0,
            "slice": slicevalue,
        }

        if self.return_csm:
            csm = sigpy_espirit(k_data_centered[0])
            ret["csm"] = csm.transpose((1, 0, 2, 3))  # (c,z,us,fs)

        return ret


class CineSelfSupervisedDataDS(Dataset):
    def __init__(
        self,
        path,
        acceleration: tuple[tuple[int, int], ...] = ((4, 0),),
        singleslice: bool = True,
        center_lines: int = 24,
        return_csm: bool = False,
        augments: bool = True,
        return_ktarget_ift_fs: bool = True,
    ):
        """
        A Cine Self Supervised Dataset
        path: Path(s) to prepared h5 files
        acceleration: tuple of tuples(acceleration factors,first line) to randomly choose from
        single slice: if true, return a single z-slice/view, otherwise return all for one subject
        random_acceleration: randomly choose offset for undersampling mask
        center_lines: ACS lines. shall match data.
        augments: augment data

        A sample consists of a dict with
            - k: undersampled k-data (shifted, k=0 is on the corner)
            - mask: mask (z, t, x, y)
            - k_target
            - mask_target


        Order of Axes:
         (Coils , Slice/view, Time, Phase Enc. (undersampled), Frequency Enc. (fully sampled))
        """

        if isinstance(path, (str, Path)):
            path = [Path(path)]
        if return_csm:
            raise NotImplementedError("return_csm not implemented")

        self.filenames = sorted(sum([list(Path(p).rglob("P*.h5")) for p in path], []))
        self.shapes = [(h5py.File(fn)["k"]).shape for fn in self.filenames]
        self.accumslices = np.cumsum(np.array([s[0] for s in self.shapes]))
        self.singleslice = singleslice
        self.acceleration = acceleration
        self.center_lines = center_lines
        self.return_ktarget_ift_fs = return_ktarget_ift_fs

        if augments:
            self.augments: Callable = CineAugment(
                p_flip_spatial=0.4,
                p_flip_temporal=0.2,
                p_shuffle_coils=0.2,
                p_phase=0.2,
                flip_view=False,
                std_amp=0.05,
            )
        else:
            self.augments = lambda x: x

    def __len__(self):
        if self.singleslice:
            return self.accumslices[-1]
        else:
            return len(self.filenames)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError

        acceleration, offset = self.acceleration[int(torch.randint(len(self.acceleration), size=(1,)))]

        filenr = np.argmax(self.accumslices > idx) if self.singleslice else idx
        if self.singleslice:
            # return a single slice for each subject
            slicenr = idx - self.accumslices[filenr - 1] if filenr > 0 else idx
            selection = slice(slicenr, slicenr + 1)

        else:
            # return all slices for each subject
            selection = slice(None)
            slicenr = None

        with h5py.File(self.filenames[filenr], "r") as file:
            lines = file["k"].shape[-5]
            mask_in = np.array(file["mask"])
            k_tmp = torch.as_tensor(np.array(file["k"][selection, mask_in]))
            k = torch.zeros(k_tmp.shape[0], lines, *k_tmp.shape[2:], dtype=k_tmp.dtype)
            k[:, mask_in, :, :, :] = k_tmp
            if slicenr is None:
                slicevalue = np.arange(0, file["k"].shape[0]) / (file["k"].shape[0] - 1)
            else:
                slicevalue = slicenr / (file["k"].shape[0] - 1)

        k = torch.view_as_complex(k).permute((3, 0, 2, 1, 4))
        ret = {"k": k}
        ret = self.augments(ret)
        k_target = ret.pop("k")

        mask = create_mask(lines, self.center_lines, acceleration, offset)
        mask &= mask_in
        mask_target = torch.as_tensor(mask_in[None, None, :, None])
        mask = torch.as_tensor(mask[None, None, :, None])

        k = k_target * mask
        ret = {
            **ret,
            "k": k,
            "mask": mask,
            "mask_target": mask_target,
            "acceleration": float(acceleration),
            "offset": float(offset),
            "axis": float("sax" in self.filenames[filenr].name),
            "slice": slicevalue,
        }
        if self.return_ktarget_ift_fs:
            ret["k_target_ift_fs"] = torch.fft.ifft(k_target, norm="ortho", dim=-1)
        else:
            ret["k_target"] = k_target
        return ret
