from collections import defaultdict
import time
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
    h5py.File(path, "w", userblock_size=512).close()
    with open(path, "r+b") as f:
        f.seek(0)
        f.write(
            "MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Sun Jan  1 00:00:00 2023 HDF5 schema 1.00 .".encode("ascii")
        )
        f.seek(124)
        f.write(b"\x00\x02IM")


class OnlineValidationWriter(Callback):
    def __init__(
        self,
        output_dir: str | None = None,
        resize: bool | None = None,
        zip: bool | None = None,
        swap: bool = True,
        resize_keep_alltimes: bool = False,
        zipstr=None,
    ):
        self.path = output_dir
        self.resize = resize
        self.resize_keep_alltimes = resize_keep_alltimes
        self.tmpdir = None
        self.zip = zip
        self.swap = swap
        self.cache = defaultdict(list)
        self.zipstr = zipstr

    def on_test_start(self, trainer, pl_module) -> None:
        if self.zip is None:
            self.zip = True
        if self.zip and self.tmpdir is None:
            self.tmpdir = tempfile.mkdtemp()

        if self.path is None:
            if trainer.checkpoint_callback.dirpath:
                self.path = Path(trainer.checkpoint_callback.dirpath).parent
            elif trainer.log_dir:
                self.path = Path(trainer.log_dir)
            else:
                self.path = Path(".")

    def on_predict_start(self, trainer, pl_module) -> None:
        if self.zip is None:
            self.zip = False
        if self.zip and self.tmpdir is None:
            self.tmpdir = tempfile.mkdtemp()
        if self.path is None:
            self.path = Path("./output")

    def create_zip(self, modelname, zipstr=""):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"MultiCoil_{zipstr}_{modelname}_{timestamp}"
        shutil.make_archive(str(self.path / filename), "zip", self.tmpdir)
        return (self.path / (filename + ".zip")).absolute()

    def on_predict_end(self, trainer, pl_module) -> None:
        modelname = pl_module.__class__.__name__
        if self.zip:
            f = self.create_zip(modelname, zipstr="predict" if self.zipstr is None else self.zipstr)
            print("Predictions written to", f)
            shutil.rmtree(self.tmpdir)
        else:
            print("Predictions written to", self.path)

    def on_test_end(self, trainer, pl_module) -> None:
        modelname = pl_module.__class__.__name__
        if self.zip:
            f = self.create_zip(modelname, zipstr="test" if self.zipstr is None else self.zipstr)
            print("Predictions written to", f)
            shutil.rmtree(self.tmpdir)
        else:
            print("Predictions written to", self.path)

    def write_results(self, outputs, batch, resize=True, swap=True, resize_keep_alltimes=False):
        for output, sample in zip(outputs, batch["sample"]):
            filename, currentslices, shape = sample

            shape = [shape[i] for i in [2, 1, 3, 4]]
            data = output.detach().cpu().transpose(0, 1).numpy()
            if resize:  # resize to 1/2 in y and 1/3 in x, keep 2 slices in z and 3 in t
                st, sz, sy, sx = shape
                if resize_keep_alltimes:
                    to_keep_t = tuple(range(st))
                else:
                    to_keep_t = (0, 1, 2)  # slice indices to keep

                if not sz < 3:
                    to_keep_z = (round(sz / 2) - 2, round(sz / 2) - 1)  # slice indices to keep
                else:
                    to_keep_z = list(range(sz))
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
                data = (
                    data.take(to_keep_t, 0)
                    .take(idx_z, 1)
                    .take(cropslice(sy, round(sy / 2)), 2)
                    .take(cropslice(sx, round(sx / 3)), 3)
                )
                shape = [len(to_keep_t), len(to_keep_z), round(sy / 2), round(sx / 3)]

            self.cache[filename].append((currentslices, data))

            # check if all slices are there and write to disk
            needed_slices = np.arange(shape[1])
            existing_slices = [np.atleast_1d(needed_slices[s]) for s, _ in self.cache[filename]]

            if set(np.concatenate(existing_slices)) == set(needed_slices):
                # all slices are there, write to disk

                data = np.empty(shape, dtype=np.float32)
                for s, current_data in self.cache[filename]:
                    data[:, s, ...] = current_data

                if swap:
                    data = data.transpose()
                    shape = shape[::-1]

                if self.zip:
                    path = self.tmpdir
                else:
                    path = self.path
                path = Path(path, *filename.parts[-6:])
                create_mat_file(path)
                with h5py.File(path, "a") as file:
                    ds = file.create_dataset("img4ranking", shape=shape, dtype=np.float32, data=data)
                    ds.attrs["MATLAB_class"] = np.bytes_("single")
                del self.cache[filename]

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if self.resize is None:
            resize = True
        else:
            resize = self.resize

        self.write_results(outputs, batch, resize=resize, swap=self.swap, resize_keep_alltimes=self.resize_keep_alltimes)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if self.resize is None:
            resize = False
        else:
            resize = self.resize
        self.write_results(outputs, batch, resize=resize, swap=self.swap, resize_keep_alltimes=self.resize_keep_alltimes)


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
