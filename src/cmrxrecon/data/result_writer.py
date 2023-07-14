import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import tempfile
from pathlib import Path
import h5py
import numpy as np
from math import ceil, floor
import shutil


def round(x):
    # round half up like matlab
    return floor(x + 0.5)


def cropslice(oldshape, newshape) -> slice:
    return np.r_[oldshape // 2 - newshape // 2 : oldshape // 2 + (newshape + 1) // 2]


def create_mat_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        h5py.File(path, "w", userblock_size=512).close()
    with open(path, "r+b") as f:
        f.seek(0)
        f.write("MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Sun Jan  1 00:00:00 2023 HDF5 schema 1.00 .".encode("ascii"))
        f.seek(124)
        f.write(b"\x00\x02IM")


class OnlineValidationWriter(Callback):
    def on_test_start(self, trainer, pl_module) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def on_predict_start(self, trainer, pl_module) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def on_predict_end(self, trainer, pl_module) -> None:
        print("results are under", self.tmpdir)
        shutil.make_archive("MultiCoil", "zip", self.tmpdir)

    def on_test_end(self, trainer, pl_module) -> None:
        print("results are under", self.tmpdir)
        shutil.make_archive("MultiCoil", "zip", self.tmpdir)
        print("done")

    def write_results(self, outputs, batch, resize=True):
        for output, sample in zip(outputs, batch["sample"]):
            filename, currentslices, shape = sample

            shape = [shape[i] for i in [2, 1, 3, 4]]
            data = output.detach().cpu().transpose(0, 1).numpy()
            if resize:  # resize to 1/2 in y and 1/3 in x, keep 2 slices in z and 3 in t
                sz, sy, sx = shape[1:]
                if not sz < 3:
                    to_keep_z = (round(sz / 2) - 2, round(sz / 2) - 1)  # slice indices to keep
                    to_keep_t = (0, 1, 2)  # slice indices to keep
                    idx_z = []
                    save_slices = []
                    for i, z in enumerate(to_keep_z):
                        # which of the current slices do we need to keep
                        match = list((np.arange(100)[currentslices] == z).nonzero()[0])
                        if match:
                            idx_z = idx_z + match
                            save_slices.append(i)
                    if not idx_z:  # none
                        continue
                    currentslices = tuple(save_slices)
                # take instead of indexing to avoid collapsing dimensions
                data = data.take(to_keep_t, 0).take(idx_z, 1).take(cropslice(sy, round(sy / 2)), 2).take(cropslice(sx, round(sx / 3)), 3)
                shape = [len(to_keep_t), len(to_keep_z), round(sy / 2), round(sx / 3)]

            # swap order of axis according to error messages of validation script
            shape = shape[::-1]
            data = data.transpose()

            path = Path(self.tmpdir, *filename.parts[-6:])
            create_mat_file(path)
            with h5py.File(path, "a") as file:
                if "img4ranking" not in file:
                    ds = file.create_dataset("img4ranking", shape=shape, dtype=np.float32)
                    ds.attrs["MATLAB_class"] = np.bytes_("single")
                file["img4ranking"][..., currentslices, :] = data

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self.write_results(outputs, batch, resize=True)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self.write_results(outputs, batch, resize=False)


# def cropslice(oldshape, newshape) -> slice:
#     return slice(oldshape // 2 - newshape // 2, oldshape // 2 + (newshape + 1) // 2)
# sz, sy, sx = data.shape[1:]
# if not sz < 3:
#     to_keep_z = (ceil(sz / 2) - 2, ceil(sz / 2) - 1)
# else:
#     to_keep_z = slice(sz)
# to_keep_y = cropslice(sy, ceil(sy / 2))
# to_keep_x = cropslice(sy, ceil(sy / 2))
# to_keep_t = (0, 1, 2)

# data = data[to_keep_t, to_keep_z, to_keep_y, to_keep_x]
