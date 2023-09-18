from torch.utils.data import Dataset
import h5py
from pathlib import Path
import numpy as np
import torch
from typing import Tuple
from cmrxrecon.data.utils import create_mask
from typing import Any, Callable
from cmrxrecon.models.utils import rss
from cmrxrecon.data.augments import MappingAugment


class MappingDataDS(Dataset):
    def __init__(
        self,
        path,
        acceleration: Tuple[int,] = (4,),
        singleslice: bool = True,
        random_acceleration: float | bool = 0.5,
        center_lines: int = 24,
        return_csm: bool = False,
        return_roi: bool = False,
        return_roi_dilated: bool = True,
        return_bbox: bool = True,
        return_kfull_ift_fs: bool = True,
        augments: bool = True,
    ):
        """
        A Mapping Dataset
        path: Path(s) to h5 files
        acceleration: tupe of acceleration factors to randomly choose from
        single slice: if true, return a single z-slice/view, otherwise return all for one subject
        random_acceleration: randomly choose offset for undersampling mask
        center_lines: ACS lines
        augment: perform data augmentation
        return_csm: return coil sensitivity maps
        return_roi: return segmentation ROI mask
        return_roi_dilated: return dilated binary mask of ROI
        return_bbox: return bounding box of ROI

        A sample consists of a dict with
            - k: undersampled k-data (shifted, k=0 is on the corner)
            - mask: mask (z, t, x, y)
            - csm: coil sensitivity maps (optional) (c, z, x, y)
            - roi: ROI mask (optional) (z, x, y)
            - mask_dilated: dilated ROI mask (optional) (z, x, y)
            - gt: RSS ground truth reconstruction
            - kfull_ift_fs: fully sampled hybrid-space data (z, t, k_us, y_fs)
            - times: time stamps


        Order of Axes:
         (Coils , Slice/view, Time, Phase Enc. (undersampled), Frequency Enc. (fully sampled))
        """
        if isinstance(path, (str, Path)):
            path = [Path(path)]
        self.filenames = sum([list(Path(p).rglob(f"P*.h5")) for p in path], [])
        self.shapes = [(h5py.File(fn)["k"]).shape for fn in self.filenames]
        self.accumslices = np.cumsum(np.array([s[0] for s in self.shapes]))
        self.singleslice = singleslice
        if isinstance(acceleration, (int, float)):
            acceleration = (acceleration,)
        self.acceleration = acceleration
        self.random_acceleration = random_acceleration
        self.center_lines = center_lines
        self.return_csm = return_csm
        self.return_roi = return_roi
        self.return_roi_dilated = return_roi_dilated
        self.return_kfull_ift_fs = return_kfull_ift_fs
        self.return_bbox = return_bbox
        if augments:
            self.augments: Callable = MappingAugment(
                p_flip_spatial=0.4,
                p_shuffle_coils=0.3,
                p_phase=0.3,
                p_amp=0.3,
                std_amp=0.05,
                std_phase=0.1,
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
            if self.return_kfull_ift_fs:
                k = torch.as_tensor(np.array(file["k"][selection]))
                gt = None
            else:
                k_tmp = torch.as_tensor(np.array(file["k"][selection, mask]))
                k = torch.zeros(k_tmp.shape[0], lines, *k_tmp.shape[2:], dtype=k_tmp.dtype)
                k[:, mask, :, :, :] = k_tmp
                gt = file["sos"][selection]
            if self.return_csm:
                csm = file["csm"][selection]
            if self.return_roi:
                roi = file["roi"][selection]
            if self.return_roi_dilated or self.return_bbox:
                roi_dilated = file["mask_dilated"][selection]
            times = np.array(file["times"])
            times = np.array((np.broadcast_to(times, (file["k"].shape[0], times.shape[1]))[selection]))

            if slicenr is None:
                slicevalue = np.arange(0, file["sos"].shape[0]) / (file["sos"].shape[0] - 1)
            else:
                slicevalue = slicenr / (file["sos"].shape[0] - 1)

        k = torch.view_as_complex(k).permute((3, 0, 2, 1, 4))
        mask = torch.as_tensor(mask[None, None, :, None])
        ret = {
            "acceleration": float(acceleration),
            "axis": 1.0,  # always SAX
            "gt": gt,
            "job": float("T1" in self.filenames[filenr].parent.parent.stem),
            "k": k,
            "mask": mask,
            "offset": float(offset),
            "slice": slicevalue,
            "times": times,
        }
        if self.return_csm:
            csm = torch.view_as_complex(torch.as_tensor(csm)).swapaxes(0, 1)
            ret["csm"] = csm

        if self.return_roi:
            ret["roi"] = torch.as_tensor(roi)
        if self.return_roi_dilated:
            ret["roi_dilated"] = torch.as_tensor(roi_dilated)
        if self.return_bbox:
            ids = np.argwhere(roi_dilated)
            if len(ids) == 0:
                ret["bbox"] = torch.tensor([0, 0, 0, 0])
            else:
                ret["bbox"] = torch.tensor([ids[:, -2].min(), ids[:, -2].max() + 1, ids[:, -1].min(), ids[:, -1].max() + 1])

        ret = self.augments(ret)

        if self.return_kfull_ift_fs:
            kfull = ret.pop("k")
            ret["k"] = kfull * mask
            xfull = torch.fft.ifft2(kfull, norm="ortho")
            ret["gt"] = rss(xfull, 0)
            ret["kfull_ift_fs"] = torch.fft.ifft(kfull, norm="ortho", dim=-1)
        return ret


class MappingTestDataDS(Dataset):
    def __init__(
        self,
        path: str | Path | tuple[str | Path, ...],
        job: str | tuple[str, ...] = ("t1", "t2"),
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
        if return_csm:
            raise NotImplementedError("return_csm not implemented yet")
        if slicepersample != 1:
            raise NotImplementedError("slicepersample not implemented yet")

        if isinstance(path, (str, Path)):
            path = (Path(path),)
        if isinstance(job, str):
            job = (job,)
        filenames = sorted(sum([list(Path(p).rglob(f"{j.capitalize()}map.mat")) for p in path for j in job], []))
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
        try:
            csvfilename = filename.parent / f"{filename.stem}.csv"
            times = np.genfromtxt(csvfilename, delimiter=",", skip_header=1)[:, 1:].T
            times = times[: np.where(np.all(np.isnan(times), axis=1))[0][0]]
            times = np.array(np.broadcast_to(times, (shape[1], times.shape[1]))[selection])
        except:
            job_T1 = float("T1" in filename.stem)
            if job_T1:
                times = np.array([[100.0, 180.0, 260.0, 1245.0, 1323.0, 1340.0, 2395.0, 2450.0, 3550.0]])
            else:
                times = np.array([[0.0, 35.0, 55.0]])

        k_data = self._shift(k_data_centered).transpose((2, 1, 0, 3, 4))  # (c,z,t,us,fs)
        k_data = k_data.astype(np.complex64)
        mask = (~np.isclose(k_data[0, ..., :, :1], 0)).astype(np.float32)
        try:
            acc_idx = str(filename).find("AccFactor")
            acceleration = float(str(filename)[acc_idx + 9 : acc_idx + 11])
        except:
            acceleration = 8

        ret = {
            "acceleration": acceleration,
            "axis": 1.0,  # always SAX
            "job": float("T1" in filename.stem),
            "k": k_data,
            "mask": mask,
            "offset": 0.0,
            "sample": (filename, selection, shape),
            "slice": slicevalue,
            "times": times,
        }

        return ret
