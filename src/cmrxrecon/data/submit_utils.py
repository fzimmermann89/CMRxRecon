import numpy as np


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
        filetype: File type indicating the type of data ('cine' or 'map').

    Returns:
        Data used for ranking.

    """
    sx, sy, sz, t = img.shape

    if filetype == "cine":
        reconImg = img[:, :, sz // 2 - 1 : sz // 2, :3]  # Select the first 3 time frames for ranking
        newshape = [sx // 3, sy // 2, 2, 3]  # Crop the middle 1/6 of the original image
    elif filetype == "map":
        reconImg = img[:, :, sz // 2 - 1 : sz // 2, :]  # Use all time frames for mapping
        newshape = [sx // 3, sy // 2, 2, t]  # Crop the middle 1/6 of the original image
    else:
        raise ValueError(f"Invalid filetype: {filetype}")
    img4ranking = crop(np.abs(reconImg), newshape).astype(np.float32)
    return img4ranking
