import torch


def ssim(gt, pred):
    """Computes the Structural Similarity Index (SSIM) between two images.
    with K1 = 0.01 and K2 = 0.03 and uniform window size 7x7.
    Matches the defaults of skimage.
    All axis but the last two are first batched and averaged over.
    Returns the mean SSIM between `gt` and `pred`.
    """
    K1 = 0.01
    K2 = 0.03
    win_size = 7
    gt = gt.view(-1, 1, gt.shape[-2], gt.shape[-1])
    pred = pred.view(-1, 1, pred.shape[-2], pred.shape[-1])
    data_range = torch.amax(gt, dim=(-1, -2))
    data_range = torch.maximum(data_range, torch.tensor(1e-6))
    x = torch.nn.functional.unfold(gt, kernel_size=win_size, padding=0, stride=1)
    y = torch.nn.functional.unfold(pred, kernel_size=win_size, padding=0, stride=1)
    NP = win_size**2
    cov_norm = NP / (NP - 1)  # sample covariance
    ux = x.mean(dim=1)
    uy = y.mean(dim=1)
    uxx = x.square().mean(dim=1)
    uyy = y.square().mean(dim=1)
    uxy = (x * y).mean(dim=1)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    A1, A2, B1, B2 = (2 * ux * uy + C1, 2 * vxy + C2, ux**2 + uy**2 + C1, vx + vy + C2)
    D = B1 * B2
    S = (A1 * A2) / D
    ssim = S.mean()
    return ssim
