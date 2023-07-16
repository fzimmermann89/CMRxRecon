import torch
import einops
from cmrxrecon.nets.unet import Unet
from cmrxrecon.nets.unet_andreas import Unet as UnetA
from . import CineModel


class MultiCoilDCLayer(torch.nn.Module):
    def __init__(self, Nc: int, embed_dim: int = 0):
        """
        Data-consistency layer for multiple receiver coils
        computes the argmin of min_x 1/2|| F_I x - y||_2**2 + lambda/2|| x - xnn||_2**2
        for each coil separately
        y.shape = [Nb, Nc, Nz, Nt, Nu, Nf]
        xnn.shape = [Nb, Nc, Nz, Nt, Nu, Nf]
        mask.shape = [Nb, 1, 1, Nu, 1]

        Parameters
        ----------
        Nc: Number of coils
        ebmed_dim:
            Embedding dimension for the lambda conditioning.
            If 0, static lambdas for each coil are used.
        """
        super().__init__()
        if embed_dim > 0:
            self.lambda_proj = torch.nn.Linear(embed_dim, Nc)
            with torch.no_grad():
                self.lambda_proj.weight.zero_()
                self.lambda_proj.bias[:] = 1e-1
            self.lambda_reg = None
        else:
            # static lambdas
            self.lambda_reg = torch.nn.Parameter(torch.ones(1, Nc, 1, 1, 1, 1)) * 1e-2
            self.lambda_proj = None

    def forward(self, k: torch.Tensor, xnn: torch.Tensor, mask: torch.Tensor, lambda_embed: torch.Tensor | None = None) -> torch.Tensor:
        if self.lambda_proj is not None:
            # use mlp to compute lambda
            if lambda_embed is None:
                lambda_embed = torch.zeros(k.shape[0], self.lambda_proj[0].in_features, device=k.device)
            lam = self.lambda_proj(lambda_embed)[:, :, None, None, None, None]  # [Nb, Nc, 1, 1, 1, 1]
        else:
            lam = self.lambda_reg
        mask = mask.unsqueeze(1)  # [Nb, Nc=1, Nz=1, Nz=1, Nu, Nf=1]
        fk = torch.where(mask, 1.0 / (1.0 + lam), 0.0)  # facor for k data, 0 for missing data
        fn = 1 - fk  # factor for xnn data
        knn = torch.fft.fft2(xnn, norm="ortho")
        xreg = fn * knn + fk * k
        xdc = torch.fft.ifft2(xreg, norm="ortho")
        return xdc


class Mapper(torch.nn.Module):
    def __init__(self, classes, eps=0.5):
        """
        Maps a value to a (soft) onehot vector and a scaled value

        Parameters
        ----------
        classes: list of values representing the individual classes
        eps: smoothing parameter
        """
        super().__init__()
        self.register_buffer("classes", torch.tensor(classes), persistent=False)
        self.scale = self.classes.max() - self.classes.min()
        self.shift = self.classes.min()
        self.eps = eps

    def forward(self, x):
        onehot = torch.tanh((1 / ((self.classes - x) ** 2 + self.eps)))
        scaled = torch.atleast_1d((x - self.shift) / self.scale)
        res = torch.cat([onehot, scaled], dim=-1)
        return res

    @property
    def out_dim(self):
        return len(self.classes) + 1


class Cascade(CineModel):
    def __init__(self, input_rss=False, lr=3e-3, weight_decay=1e-6, schedule=True, Nc: int = 10, T: int = 3, **kwargs):
        super().__init__()
        self.input_rss = input_rss
        embed_dim: int = 128
        self.norm1 = 0.0001  # rss.max()
        self.norm2 = self.norm1 * 10
        self.net = Unet(
            dim=2.5,
            channels_in=input_rss + 2 * Nc,
            channels_out=2 * Nc,
            layer=4,
            filters=32,
            padding_mode="circular",
            residual="inner",
            norm="group8",
            feature_growth=lambda d: (1, 2, 1.5, 1.34, 1, 1)[d],
            activation="leakyrelu",
            change_filters_last=False,
            emb_dim=embed_dim,
        )
        with torch.no_grad():
            self.net.last[0].bias.zero_()
            self.net.last[0].weight *= 0.1

        self.dc = torch.nn.ModuleList([MultiCoilDCLayer(Nc, embed_dim=embed_dim) for _ in range(T)])

        self.embed_augment_channels = 6
        self.embed_axis_channels = 2
        self.embed_acceleration_map = Mapper([2, 4, 6, 8, 10])
        embed_input_channels = self.embed_augment_channels + self.embed_axis_channels + self.embed_acceleration_map.out_dim
        self.embed_net = torch.nn.Sequential(
            torch.nn.Linear(embed_input_channels, embed_dim),
            torch.nn.SiLU(True),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.Dropout(0.2),
            torch.nn.SiLU(True),
        )
        self.embed_net_iter = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(True)) for _ in range(T)])

    def prepare_input(self, x, norm, include_rss=False):
        # input the coil reconstructions as channels alongsite the rss
        net_input = einops.rearrange(
            torch.view_as_real(x),
            "b c z t x y r -> (b z) (r c) t x y",
        )
        if include_rss:
            rss = x.abs().square().sum(1, keepdim=True).sqrt()

            net_input = torch.cat((net_input, einops.rearrange(rss, "b c z t x y -> (b z) c t x y")), 1)
        net_input = net_input * norm
        return net_input

    def prepare_output(self, x_net, x_res, norm, batch_size):
        # output the coil reconstructions as channels alongsite the rss
        x_net = einops.rearrange(x_net, "(b z) (r c) t x y -> b c z t x y r", b=batch_size, r=2).contiguous()
        x_net = x_net * norm
        x_net = torch.view_as_complex(x_net)
        x = x_net + x_res
        return x

    def forward(self, k: torch.Tensor, mask: torch.Tensor, **other) -> dict:
        # get all the conditioning information or defaults
        augmentinfo = other.get("augmentinfo", torch.zeros(k.shape[0], self.embed_augment_channels, device=k.device)).float()
        acceleration = other.get("acceleration", torch.ones(k.shape[0], device=k.device)).float()[:, None]
        accelerationinfo = self.embed_acceleration_map(acceleration)
        axis = other.get("axis", torch.zeros(k.shape[0], device=k.device)).float()[:, None]
        axisinfo = torch.cat((axis, 1 - axis), dim=-1)
        info = torch.cat((augmentinfo, axisinfo, accelerationinfo), dim=-1)

        z0 = self.embed_net(info)
        x0 = torch.fft.ifftn(k, dim=(-2, -1), norm="ortho")
        rss = x0.abs().square().sum(1).sqrt()

        x = [x0]
        for emb_net, dc in zip(self.embed_net_iter, self.dc):
            z = emb_net(z0)
            net_input = self.prepare_input(x[-1], self.norm1, include_rss=self.input_rss)
            x_net = self.net(net_input)
            x_net = self.prepare_output(x_net, x[-1], norm=self.norm2, batch_size=k.shape[0])
            x.append(dc(k, x_net, mask, z))

        pred = x[-1].abs().square().sum(1).sqrt()
        return dict(prediction=pred, rss=rss)
