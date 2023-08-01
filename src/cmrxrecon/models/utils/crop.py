import numpy as np
import torch


def crops_by_threshold(data: torch.Tensor, thresholds: tuple[float, ...]) -> tuple[slice, ...]:
    """
    Crop data by thresholded absolute maximum projection. Returns a tuple of slices.

    Usage:
    >>> data = np.zeros(10, 10, 10)
    >>> data[:2,2:5,2:5] = 1
    >>> cut = data[crops_by_threshold(data, (None, 0.5, None))]
    >>> print(cut.shape)
    <<< (10, 3, 10)

    Parameters
    ----------
    data:       tensor, n-dim
    thresholds: tuple of floats, of length m<=n. If None, no cut along that axis is performed.
                if m<n, cuts will be performed along the last m axes.

    """
    absdata = torch.abs(data.detach())
    cuts = []
    for i, threshold in enumerate(thresholds):
        if threshold is not None:
            axis = data.ndim - len(thresholds) + i
            mip = torch.amax(absdata, axis=tuple([j for j in range(data.ndim) if j != axis]))
            mask = (mip > threshold).cpu().numpy()
            low = np.argmax(mask)
            high = len(mask) - np.argmax(mask[::-1] > threshold)
            cuts.append(slice(low, high))
        else:
            cuts.append(slice(None))
    cuts = [slice(None)] * (data.ndim - len(cuts)) + cuts
    return tuple(cuts)


def uncrop(x, target_shape: tuple[int, ...], crops: tuple[slice, ...], **pad_kwargs) -> torch.Tensor:
    """Undo the cropping operation performed by crops_by_threshold.

    Performs a torch.nn.functional.pad operation.

    Parameters
    ----------
    x: tensor, n-dim
    target_shape: tuple of ints, of length n
    crops: crops (as returned by crops_by_threshol) that were used to crop x
    pad_kwargs: keyword arguments passed to torch.nn.functional.pad.
        default: mode="constant", value=0

    """
    if crops is None:
        return x
    if pad_kwargs is None:
        pad_kwargs = dict(mode="constant", value=0)
    paddings = []
    for crop, orig in zip(crops, target_shape):
        before = 0 if crop.start is None else crop.start
        paddings.append(before)
        after = 0 if crop.stop is None else orig - crop.stop
        paddings.append(after)
    return torch.nn.functional.pad(x, paddings[::-1], **pad_kwargs)
