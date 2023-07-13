import numpy as np


def crops_by_threshold(data, thresholds: tuple[float, ...]) -> tuple[slice, ...]:
    """
    Crop data by thresholded absolute maximum projection. Returns a tuple of slices.

    Usage:
    >>> data = np.zeros(10, 10, 10)
    >>> data[:2,2:5,2:5] = 1
    >>> cut = data[crops_by_threshold(data, (None, 0.5, None))]
    >>> print(cut.shape)
    <<< (10, 3, 10)

    Parameter
    ------
    data:       array-like, n-dim
    thresholds: tuple of floats, of length m<=n. If None, no cut along that axis is performed.
                if m<n, cuts will be performed along the last m axes.

    """
    absdata = np.abs(data)
    cuts = []
    for i, threshold in enumerate(thresholds):
        if threshold is not None:
            mip = np.max(absdata, axis=tuple([j for j in range(data.ndim) if j != i]))
            mask = mip > threshold
            low = np.argmax(mask)
            high = len(mask) - np.argmax(mask[::-1] > threshold)
            cuts.append(slice(low, high))
        else:
            cuts.append(slice(None))
    cuts = [slice(None)] * (data.ndim - len(cuts)) + cuts
    return tuple(cuts)
