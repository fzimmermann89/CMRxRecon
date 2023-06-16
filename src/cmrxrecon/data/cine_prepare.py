import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

PATH = Path("files/MultiCoil/Cine/TrainingSet/")
OUTPATH = PATH /".."/"ProcessedTrainingSet"


def ft(data):
    data = np.fft.ifftshift(data, axes=(-1, -2))
    data = np.fft.fft2(data, norm="ortho")
    data = np.fft.fftshift(data, axes=(-1, -2))
    return data


def ift(data):
    data = np.fft.ifftshift(data, axes=(-1, -2))
    data = np.fft.ifft2(data, norm="ortho")
    data = np.fft.fftshift(data, axes=(-1, -2))
    return data


def sos(data):
    return ((np.abs(data) ** 2).sum(-3) ** 0.5).real

if True:
    files = list((PATH / "FullSample").rglob("*sax.mat"))
    for fn in tqdm(files):
        with h5py.File(fn) as f:
            datac = np.array(f["kspace_full"]["real"] + 1j * f["kspace_full"]["imag"], dtype=np.complex64)
            r = ift(datac)
            gt_sos = sos(r)
            k = np.fft.fft2(r, norm="ortho")
            gt_sos = np.moveaxis(gt_sos, 0, 1).astype(np.float32)
            datar = np.stack((k.real, k.imag), -1)
            datar = np.stack((k.real.astype(np.float32), k.imag.astype(np.float32)), -1)
            datar = np.transpose(datar, (1, 3, 0, 2, 4, 5))
        cpath = OUTPATH / "sax" / f"{datac.shape[-1]}_{datac.shape[-2]}_{datac.shape[1]}"
        cpath.mkdir(parents=True, exist_ok=True)
        outfilename = str(cpath / fn.parent.name) + ".h5"
        with h5py.File(outfilename, mode="w") as outfile:
            outfile.create_dataset("k", datar.shape, dtype=np.float32, data=datar)
            outfile.create_dataset("sos", gt_sos.shape, dtype=np.float32, data=gt_sos)
            
if True:
    files = list((PATH / "FullSample").rglob("*lax.mat"))
    for fn in tqdm(files):
        with h5py.File(fn) as f:
            datac = np.array(f["kspace_full"]["real"] + 1j * f["kspace_full"]["imag"], dtype=np.complex64)
            r = ift(datac)
            gt_sos = sos(r)
            k = np.fft.fft2(r, norm="ortho")
            gt_sos = np.moveaxis(gt_sos, 0, 1).astype(np.float32)
            datar = np.stack((k.real, k.imag), -1)
            datar = np.stack((k.real.astype(np.float32), k.imag.astype(np.float32)), -1)
            datar = np.transpose(datar, (1, 3, 0, 2, 4, 5))
        cpath = OUTPATH / "lax" / f"{datac.shape[-1]}_{datac.shape[-2]}_{datac.shape[1]}"
        cpath.mkdir(parents=True, exist_ok=True)
        outfilename = str(cpath / fn.parent.name) + ".h5"
        with h5py.File(outfilename, mode="w") as outfile:
            outfile.create_dataset("k", datar.shape, dtype=np.float32, data=datar)
            outfile.create_dataset("sos", gt_sos.shape, dtype=np.float32, data=gt_sos)