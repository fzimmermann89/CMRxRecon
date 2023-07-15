from math import e
from lightning_fabric.connector import _FSDP_ALIASES
import torch
from torch import nn


class CineAugment:
    def __init__(self, p_flip_spatial: float = 0.4, p_flip_temporal: float = 0.2, p_shuffle_coils: float = 0.2, p_phase: float = 0.2, flip_view: bool = False):
        """
        Augmentations for cine data
             (Coils , Slice/view, Time, Phase Enc. (undersampled), Frequency Enc. (fully sampled))
        """
        self.p_flip_spatial = p_flip_spatial
        self.p_flip_temporal = p_flip_temporal
        self.p_phase = p_phase
        self.p_shuffle_coils = p_shuffle_coils
        self.flip_view = flip_view

    def __call__(self, sample):
        k = sample["k"]
        gt = sample["gt"]
        csm = sample.get("csm", None)
        flippedx, flippedy, flippedz, flippedt = 0, 0, 0, 0
        shuffled = 0
        phase = 0.0
        if torch.rand(1) < self.p_flip_spatial:
            # flip along spatial dimensions
            r = torch.rand(1)
            if r < 1 / 3:  # flip along fully sampled dimension
                k = torch.fft.fft(torch.fft.fft(k, dim=-1), dim=-1, norm="forward")
                gt = torch.flip(gt, (-1,))
                if csm is not None:
                    csm = torch.flip(csm, (-1,))
                flippedx = 1
            elif r < 2 / 3:  # flip along y
                k = torch.fft.fft(torch.fft.fft(k, dim=-1), dim=-1, norm="forward").conj()
                gt = torch.flip(gt, (-2,))
                if csm is not None:
                    csm = torch.flip(csm, (-2,))
                flippedy = 1
            else:  # flip along x and y
                k = k.conj()
                gt = torch.flip(gt, (-2, -1))
                if csm is not None:
                    csm = torch.flip(csm, (-2, -1))
                flippedy = 1
                flippedx = 1
            if k.shape[-4] > 1 and self.flip_view and torch.rand(1) > 0.5:
                # flip along view dimension
                k = torch.flip(k, (-4,))
                gt = torch.flip(gt, (-4,))
                if csm is not None:
                    csm = torch.flip(csm, (-4,))
                flippedz = 1

        if torch.rand(1) < self.p_flip_temporal:
            k = torch.flip(k, (-3,))
            gt = torch.flip(gt, (-3,))
            if csm is not None:
                csm = torch.flip(csm, (-3,))
            flippedt = 1

        if torch.rand(1) < self.p_phase:
            phase = (2 * torch.pi + (0.1 * torch.randn(1))) % 2 * torch.pi
            k = k * torch.exp(1j * phase)
            if csm is not None:
                csm = csm * phase

        if torch.rand(1) < self.p_shuffle_coils:
            shuffle = torch.randperm(k.shape[-5])
            k = k[..., shuffle, :, :, :, :]
            if csm is not None:
                csm = csm[..., shuffle, :, :, :, :]
            shuffled = 1

        augmentinfo = torch.tensor([flippedx, flippedy, flippedz, flippedt, shuffled, phase])
        sample["k"] = k
        sample["gt"] = gt
        sample["augmentinfo"] = augmentinfo
        if csm is not None:
            sample["csm"] = csm
        return sample


class AugmentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, augments: None | nn.Module | tuple[nn.Module, ...] = None, getter=lambda x: x, setter=lambda x, y: y):
        self.dataset = dataset
        if not isinstance(augments, nn.Module):
            augments = nn.Sequential(*augments)
        self.augments = augments
        self.getter = getter
        self.setter = setter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]
        if self.augments is not None:
            x = self.setter(x, self.augments(self.getter(x)))
        return x

    def __getattribute__(self, name):
        if name in ("dataset", "augments", "getter", "setter"):
            return super().__getattribute__(name)
        else:
            return getattr(self.dataset, name)


class RandomFlipAlongDimensions(nn.Module):
    def __init__(self, p: float | tuple[float, ...] = 0.5, dim: int | tuple[int, ...] = (-1, -2)):
        """
        Randomly flip along dimensions.

        Parameters
        ----------
        p, default 0.5
            Propability of flip along a dimension, default 0.5 means 50% for all dimensions in dim)
        dim, default (-1,-2)
            Dimensions to flip in
        """
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        if isinstance(p, float):
            p = len(dim) * (p,)
        self.p = p
        self.dim = dim

    def forward(self, x):
        for d, p in zip(self.dim, self.p):
            if torch.rand(1) < p:
                x = torch.flip(x, dims=(d,))
        return x


class RandomKSpaceFlipAlongDimensions(nn.Module):
    def __init__(self, p: float | tuple[float, ...] = 0.5, dim: int | tuple[int, ...] = (-1, -2)):
        """
        Randomly flip along dimensions in reciprocal space.
        I.e. does ifft, flips, fft along an axis.

        Parameters
        ----------
        p, default 0.5
            Propability of flip along a dimension,  default 0.5 means 50% for all dimensions in dim)
        dim, default (-1,-2)
            Dimensions to flip in
        """
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        if isinstance(p, float):
            p = len(dim) * (p,)
        self.p = p
        self.dim = dim

    def forward(self, k):
        for d, p in zip(self.dim, self.p):
            if torch.rand(1) < p:
                k = torch.fft.fft(torch.fft.fft(k, dim=d), dim=d, norm="forward")
        return k


class RandomKSpaceFlipAlongDimensionsIndirect(nn.Module):
    def __init__(self, p: float | tuple[float, ...] = 0.5, otherdim: tuple[int, ...] = (-1,)):
        """
        Randomly flip along a dimension by flipping along all othre dimensions and complex conjugating.
        I.e. does ifft, flip along other dim, fft, complex conjugate.
        to simulate a flip along dim. otherdim must contain all reciprical dimensions except dim.

        Parameters
        ----------
        p, default 0.5
            Propability of flip along a dimension,  default 0.5 means 50% for all dimensions in dim)
        otherdim, default (-1,-2)
            Dimensions that are fourier transformed and will not be flipped in
        """
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        if isinstance(p, float):
            p = len(dim) * (p,)
        self.p = p
        self.otherdim = dim

    def forward(self, k):
        if torch.rand(1) < p:
            k = torch.fft.fft(torch.fft.fft(k, dim=self.otherdim), dim=self.otherdim, norm="forward").conj()
        return k


class RandomShuffleAlongDimensions(nn.Module):
    def __init__(self, p: float | tuple[float, ...] = 0.5, dim: int | tuple[int, ...] = (1,)):
        """
        Randomly shuffle along dimensions

        Parameters
        ----------
        p, default 0.5
            Propability of shuffle along a dimension,  default 0.5  means 50% for all dimensions in dim
        dim, default (-1,-2)
            Dimensions to shuffle in,
        """
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        if isinstance(p, float):
            p = len(dim) * (p,)
        self.p = p
        self.dim = dim

    def forward(self, x):
        for d, p in zip(self.dim, self.p):
            if torch.rand(1) < p:
                x = x[(slice(None),) * d + ((torch.randperm(x.shape[d])),)]
        return x


class RandomConjugte(nn.Module):
    def __init__(self, p: float):
        """
        Randomly complex conjugate the input

        Parameters
        ----------
        p
            Propability of complex conjugate
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            x = torch.conj(x)
        return x


class RandomKFlipUndersampled(nn.Module):
    def __init__(self, p: float, dim_fullysampled: int, dim_undersampled: int):
        """
        Randomly flip the input in k-space either along the fully sampled or the undersampled dimension.
        Flip along the fully sampled dimension is done by transforming to image space, flipping and transforming back.
        Flip along the undersampled dimension is done by flipping along the fully sampled dimension and complex conjugating.

        Parameters
        ----------
        p
            Propability of flip along one of the dimensions
        dim_fullysampled
            Dimension of the fully sampled k-space
        dim_undersampled
            Dimension of the undersampled k-space
        """
        super().__init__()
        self.p = p
        self.dim_fullysampled = dim_fullysampled
        self.dim_undersampled = dim_undersampled

    def forward(self, k):
        flip1, flip2 = torch.rand(2) < self.p
        if flip1 and flip2:
            # flip along both dimensions
            k = k.conj()
        elif flip1:
            # flip along under sampled dimension only
            k = torch.fft.fft(torch.fft.fft(k, dim=self.dim_fullysampled), dim=self.dim_fullysampled, norm="forward")
            k = k.conj()
        elif flip2:
            # flip along fully sampled dimension only
            k = torch.fft.fft(torch.fft.fft(k, dim=self.dim_fullysampled), dim=self.dim_fullysampled, norm="forward")

        return k


class RandomPhase(nn.Module):
    def __init__(self, p: float):
        """
        Randomly add a phase to the input

        Parameters
        ----------
        p
            Propability of adding a phase
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            x = x * torch.exp(1j * torch.rand(1) * 2 * torch.pi)
        return x
