## Extract k-space data from the MultiCoil dataset and save it in a format that can be used by the model
## The data is saved in a h5 file with the following structure:
##	- k: k-space data (z, undersampled, t, coils, fullysampled, real/imag)
##	- sos: sum-of-squares of the fully-sampled data  (z, t, undersampled, fullysampled)
##	- csm: coil sensitivity maps (z, coils, undersampled, fullysampled, real/imag)
##  - roi: region of interest (z, undersampled, fullysampled)
##  - mask_dilated: dilated region of interest (z, undersampled, fullysampled)
# %%
from joblib import Parallel, delayed
import scipy.ndimage as snd
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import nibabel
import sigpy.mri as sp_mri

PATH = Path("../../../../files/MultiCoil/Mapping/TrainingSet/")
OUTPATH = PATH / ".." / "ProcessedTrainingSet"
THRESHOLD = 0.00025
MAX_ITER = 250


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


def process(job, fn):
    with h5py.File(fn) as f:
        k_centered = np.array(f["kspace_full"]["real"] + 1j * f["kspace_full"]["imag"], dtype=np.complex64)

    img = ift(k_centered)
    gt_sos = sos(img)  # rss reconstruction

    # make shifted k-space, i.e. k-space with center at (0,0)
    k = np.fft.fft2(img, norm="ortho")

    # move z/view axis to the front
    gt_sos = np.moveaxis(gt_sos, 0, 1).astype(np.float32)

    # real view with shape (z, undersampled, t, coil, fullysampled, r/i)
    k_reshaped = np.stack((k.real, k.imag), -1)
    k_reshaped = np.stack((k.real.astype(np.float32), k.imag.astype(np.float32)), -1)
    k_reshaped = np.transpose(k_reshaped, (1, 3, 0, 2, 4, 5))

    # coil sensitivity maps
    csm = [
        sp_mri.app.EspiritCalib(z_slice, max_iter=MAX_ITER, thresh=THRESHOLD, show_pbar=False).run() for z_slice in k_centered[0]
    ]
    csm = np.array(csm)
    Nc = csm.shape[1]
    zeros = np.all(np.abs(csm) < 1e-6, axis=1)
    fills = np.sqrt(1 / Nc) * np.exp(1j * np.angle(csm).mean((-1, -2)))
    for c, f, m in zip(csm, fills, zeros):
        c[:, m] = f[:, None]
    csm = np.stack((csm.real.astype(np.float32), csm.imag.astype(np.float32)), -1)

    # segmentation rois
    segmentpath = fn.parent.parent.parent / "SegmentROI" / fn.parent.name / f"{job}_label.nii.gz"
    roi = nibabel.load(segmentpath).get_fdata().transpose(2, 1, 0).astype(np.int8)  # z,undersampled,fullysampled

    mask_dilated = snd.binary_dilation(snd.binary_fill_holes(roi > 0), iterations=15)

    # output path based on data shape
    cpath = OUTPATH / job / f"{k_centered.shape[-1]}_{k_centered.shape[-2]}_{k_centered.shape[1]}"
    cpath.mkdir(parents=True, exist_ok=True)
    outfilename = str(cpath / fn.parent.name) + ".h5"

    # read times from csv up to first nan-row
    times = np.genfromtxt(fn.parent / f"{job}.csv", delimiter=",", skip_header=1)[:, 1:].T
    times = times[: np.where(np.all(np.isnan(times), axis=1))[0][0]]

    with h5py.File(outfilename, mode="w") as outfile:
        outfile.create_dataset("k", k_reshaped.shape, dtype=np.float32, data=k_reshaped)
        outfile.create_dataset("sos", gt_sos.shape, dtype=np.float32, data=gt_sos)
        outfile.create_dataset("times", times.shape, dtype=np.float32, data=times)
        outfile.create_dataset("roi", roi.shape, dtype=np.int8, data=roi)
        outfile.create_dataset("mask_dilated", roi.shape, dtype=np.int8, data=roi)

        outfile.create_dataset("csm", csm.shape, dtype=np.float32, data=csm)


files = []
for job in ["T1map", "T2map"]:
    files.extend([(job, fn) for fn in list((PATH / "FullSample").rglob(f"*{job}.mat"))])
Parallel(n_jobs=32, verbose=10)(delayed(process)(fn, job) for fn, job in files)

# %%
