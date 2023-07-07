## Extract coil sensitivity maps using ESPIRiT
## and save them in the same h5 file as the k-space data
## The data is saved in a h5 file with the following structure:
##	- k: k-space data (z, undersampled, t, coils, fullysampled, real/imag)
##	- sos: sum-of-squares of the fully-sampled data  (z, t, undersampled, fullysampled)
##	- csm: coil sensitivity maps (z, coils, undersampled, fullysampled, real/imag)


import torch
import numpy as np
import sigpy.mri as sp_mri
from pathlib import Path
import h5py
import tqdm

PATH = "/data/cardiac/files/MultiCoil/Cine/"
THRESHOLD = 0.00025
MAX_ITER = 250
CROP = 0.95


for view in ["lax", "sax"]:
    for file in tqdm(Path(PATH) / "ProcessedTrainingSet" / view).rglob("P*.h5"):
        with h5py.File(file, "a") as f:
            # get k-data with (z, undersampled, t, coils, fully-sampled, real/imag)
            y = np.array(f["k"])
            y = torch.view_as_complex(torch.tensor(y)).mean(2)  # average over time

            # fftshift the data for the CSM estimation
            y = torch.fft.fftn(
                torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(y, dim=(-1, -3)), dim=(-1, -3)), dim=(-1, -3)),
                dim=(-1, -3),
            )

            y = y.permute(0, 2, 1, 3)  # z, c, y, x
            csm = [sp_mri.app.EspiritCalib(z_slice.numpy(), max_iter=MAX_ITER, thresh=THRESHOLD, crop=CROP).run() for z_slice in y]
            csm = np.stack(csm, axis=0)
            csm = np.stack([np.real(csm), np.imag(csm)], axis=-1)

            if "csm" in f:
                f["csm"] = csm
            else:
                f.create_dataset("csm", data=csm, dtype=np.float32)
