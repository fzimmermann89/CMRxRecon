# %%
import torch
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from tqdm import tqdm


val_path = Path("../../../../../files/MultiCoil/Cine/ValidationSet")
out_path = Path("../../../../../files/MultiCoil/Cine/ValidationSelfSupervised")


def shift(data: np.ndarray) -> np.ndarray:
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


# %%
filenames = list(val_path.rglob("*x.mat"))
ordered = defaultdict(list)
for fn in filenames:
    with h5py.File(fn, "r") as f:
        ds = list(f.values())[0]
        k = float(ds[0, 0, 0, 0, 0]["real"])
    ordered[k].append(fn)

# %%

for sample in tqdm(list(ordered.values())):
    names = sorted(sample)
    axis = names[0].stem.split("_")[-1]
    img = None
    for fn in names:
        if "AccFactor08" in str(fn):
            continue
        with h5py.File(fn, "r") as f:
            ds = list(f.values())[0]
            datac = np.array(ds["real"] + 1j * ds["imag"], dtype=np.complex64)
        if img is None:
            img = datac
            joint_mask = ~(np.isclose(datac.real, 0.0) & np.isclose(datac.imag, 0.0))
        else:
            current_mask = ~(np.isclose(datac.real, 0.0) & np.isclose(datac.imag, 0.0))
            new = current_mask & ~joint_mask
            img[new] = datac[new]
            joint_mask |= current_mask
    mask = joint_mask[0, 0, 0, :, 0]
    k = shift(datac)
    k = np.stack((k.real.astype(np.float32), k.imag.astype(np.float32)), -1)
    k = np.transpose(k, (1, 3, 0, 2, 4, 5))
    cpath = out_path / view / f"{datac.shape[-1]}_{datac.shape[-2]}_{datac.shape[1]}"
    cpath.mkdir(parents=True, exist_ok=True)
    outfilename = cpath / (fn.parent.name + ".h5")
    with h5py.File(outfilename, mode="w") as outfile:
        outfile.create_dataset("k", k.shape, dtype=np.float32, data=k)
        outfile.create_dataset("mask", mask.shape, dtype=bool, data=mask)

# %%
