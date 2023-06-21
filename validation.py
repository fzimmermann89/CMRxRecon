from pathlib import Path
import tempfile

import numpy as np

## For validation we need to:
# 1. run the model on the validation data
# 2. save the output in the format expected by the evaluation script
# a) reorder the axes to (sx, sy, sz, t/w)
# b) crop the output
# c) save the output as a .mat file
# d) compress the files into .zip files


# Converted from matlab code
def crop(data: np.ndarray, newshape: list[int]) -> np.ndarray:
    """
    Crop a multi-dimensional array around its center.

    Args:
        data: Input array.
        newshape: Desired new shape for cropping.

    Returns:
        Cropped array.

    """
    oldshape = data.shape
    newshape.extend([1] * (len(oldshape) - len(newshape)))  # Extend newshape with 1s to match the dimensions
    idx = [slice((old - new) // 2, (old - new) // 2 + new) for old, new in zip(oldshape, newshape)]
    return data[tuple(idx)]


def run4Ranking(img: np.ndarray, filetype: str) -> np.ndarray:
    """
    Convert the input data for ranking.

    Args:
        img: Input complex images reconstructed with dimensions (sx, sy, sz, t/w).
            - sx: Matrix size in x-axis.
            - sy: Matrix size in y-axis.
            - sz: Slice number (short axis view); slice group (long axis view).
            - t/w: Time frame/weighting.
        filetype: File type indicating the type of data ('cine_lax' or 'cine_sax').

    Returns:
        Data used for ranking.

    """
    sx, sy, sz, t = img.shape

    if filetype == "cine_lax" or filetype == "cine_sax":
        reconImg = img[:, :, sz // 2 - 1 : sz // 2, :3]  # Select the first 3 time frames for ranking
        newshape = [sx // 3, sy // 2, 2, 3]  # Crop the middle 1/6 of the original image
    else:  # mapping
        reconImg = img[:, :, sz // 2 - 1 : sz // 2, :]  # Use all time frames for mapping
        newshape = [sx // 3, sy // 2, 2, t]  # Crop the middle 1/6 of the original image

    img4ranking = crop(np.abs(reconImg), newshape).astype(np.float32)
    return img4ranking
