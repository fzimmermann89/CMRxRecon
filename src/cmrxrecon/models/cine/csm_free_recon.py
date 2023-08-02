import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cmrxrecon.nets.unet_andreas import Unet
from cmrxrecon.nets.unet import Unet as Unet_felix

from cmrxrecon.models.utils.crop import crops_by_threshold
from . import CineModel
from cmrxrecon.models.utils.cg import conj_grad

from neptune.new.types import File as neptuneFile
from pytorch_lightning.loggers.neptune import NeptuneLogger


class MultiCoilDCLayer(nn.Module):

    """
    Simultaneous Data-consistency layer for multiple receiver coils

    computes the argmin of min_x 1/2|| F_I x - y||_2**2 + lmbda/2|| x - xnn||_2**2

    for each coil separately...

    y.shape = [Nb, Nc, Nz, Nt, Nu, Nf]
    xnn.shape = [Nb, Nc, Nz, Nt, Nu, Nf]
    mask.shape = [Nb, Nc, Nt, Nu, 1]

    TODO: extend it to work with spatio-temporal \Lambda parameter map
    """

    def __init__(self):
        super().__init__()

    def forward(self, y: torch.Tensor, xnn: torch.Tensor, mask: torch.Tensor, lambda_reg: torch.Tensor) -> torch.Tensor:
        kxnn = torch.fft.fftn(xnn, dim=(-2, -1), norm="ortho")
        xreg = mask * (lambda_reg / (1.0 + lambda_reg) * kxnn + 1.0 / (1.0 + lambda_reg) * y) + ~mask * kxnn
        # xreg = mask * (lambda_reg / (1.0 + lambda_reg) * kxnn + 1.0 / (1.0 + lambda_reg) * y) + (1 - mask) * kxnn

        xdc = torch.fft.ifftn(xreg, dim=(-2, -1), norm="ortho")
        return xdc


class MultiCoilImageCNN(nn.Module):

    """
    apply a simple CNN-block to each coil-separately
    """

    def __init__(self, cnn_block, mode="xyt"):
        super().__init__()

        self.cnn_block = cnn_block
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Nb, Nc, Nz, Nt, Nuf, Nf = x.shape

        x = torch.view_as_real(x)

        if self.mode == "xyt":
            pattern = "b c z t u f ch-> (b c z) ch u f t"
        elif self.mode == "xyz":
            pattern = "b c z t u f ch-> (b c t) ch u f z"

        x = rearrange(x, pattern)

        x = self.cnn_block(x)

        if self.mode == "xyt":
            pattern = "(b c z) ch u f t-> b c z t u f ch"
        elif self.mode == "xyz":
            pattern = "(b c t) ch u f z-> b c z t u f ch"

        x = rearrange(x, pattern, b=Nb, c=Nc, z=Nz, t=Nt, ch=2)

        x = torch.view_as_complex(x.contiguous())

        return x


class MultiCoilImageCNN_v2(nn.Module):

    """
    apply a simple CNN-block to a multi-coil image by stacking the channels
    """

    def __init__(self, cnn_block, mode="xyt"):
        super().__init__()

        self.cnn_block = cnn_block
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Nb, Nc, Nz, Nt, Nuf, Nf = x.shape

        x = torch.view_as_real(x)
        x = torch.concatenate([x[..., 0], x[..., 1]], dim=1)

        if self.mode == "xyt":
            pattern = "b c2 z t u f-> (b z) c2 u f t"
        elif self.mode == "xyz":
            pattern = "b c2 z t u f-> (b t) c2 u f z"

        x = rearrange(x, pattern)

        x = self.cnn_block(x)

        if self.mode == "xyt":
            pattern = "(b z) c2 u f t-> b c2 z t u f"
        elif self.mode == "xyz":
            pattern = "(b t) c2 u f z-> b c2 z t u f"

        x = rearrange(x, pattern, b=Nb, c2=2 * Nc, z=Nz, t=Nt)

        x = torch.stack([x[:, :Nc, ...], x[:, Nc:, ...]], dim=-1)

        x = torch.view_as_complex(x.contiguous())

        return x


class MLP(nn.Module):

    """
    apply a MLP to a vector containing meta-information about the currently
    considered sample
    """

    def __init__(self, in_features=5, out_features=1, hidden_dim=16, n_layers=4, activation=nn.LeakyReLU()):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        layers_list = [nn.Linear(in_features, hidden_dim), activation]

        for kl in range(n_layers):
            layers_list.extend([nn.Linear(hidden_dim, hidden_dim), activation])

        layers_list.extend([nn.Linear(hidden_dim, out_features)])

        self.mlp = nn.Sequential(*layers_list)

    def forward(self, meta_info_vect):
        return self.mlp(meta_info_vect)


class LambdaCNN(nn.Module):

    """
    estimate a lambda-map (> 0 for all locations)  from a complex-valued image
    """

    def __init__(self, cnn_block=None):
        super().__init__()
        if cnn_block is None:
            self.cnn_block = Unet(3, channels_in=2, channels_out=1, layer=1, filters=16, n_convs_per_stage=4)
        else:
            self.cnn_block = cnn_block

    def forward(self, x):
        Nb, Nz, Nt, Nuf, Nf = x.shape

        x = torch.view_as_real(x)

        x = rearrange(x, "b z t u f ch-> (b z) ch u f t")

        Lambda_map = self.cnn_block(x).squeeze(1)

        Lambda_map = rearrange(Lambda_map, "(b z) u f t-> b z t u f", b=Nb, z=Nz, t=Nt)

        return F.softplus(Lambda_map)


class CSMFreeReconNN(nn.Module):

    """ """

    def __init__(self, img_unet=None, T=1, normfactor=1e2, mode="xyt"):
        super().__init__()
        if img_unet is None:
            img_unet = Unet_felix(
                3, channels_in=20, channels_out=20, layer=2, filters=16, conv_per_enc_block=2, conv_per_dec_block=2
            )
        self.T = T

        self.mcdc = MultiCoilDCLayer()
        # self.img_cnn = MultiCoilImageCNN(img_unet, mode) #
        self.img_cnn = MultiCoilImageCNN_v2(img_unet, mode)

        # self.img_cnn_xyt = MultiCoilImageCNN_v2(img_unet_xyt, 'xyt')
        # self.img_cnn_xyz = MultiCoilImageCNN_v2(img_unet_xyz, 'xyz')

        # layer, n_convs_per_stage, filters = 3, 4, 64
        # pre_trained_weights_path = '/data/kofler01/projects/CMRxRecon/pre_trained_models/unet_E{}C{}K{}_T{}.pt'.format(layer, n_convs_per_stage, filters, T)
        # print('LOAD PRE-TRAINED MODEL')
        # self.img_cnn.load_state_dict(torch.load(pre_trained_weights_path))

        if self.T == 1:
            self.in_features_mlp = 3
        else:
            self.in_features_mlp = 4

        self.lambda_mlp = MLP(in_features=self.in_features_mlp, out_features=1, hidden_dim=5, n_layers=4)

        self.normfactor = normfactor

    def forward(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # zerof-filled recon
        # print('mask = {}'.format(mask.dtype))
        y = self.normfactor * y

        Nb, Nc, Nz, Nz, Nu, Nf = y.shape
        x = torch.fft.ifftn(y, dim=(-2, -1), norm="ortho")

        # get global meta-information
        R = torch.sum(mask) / Nu

        # print(R)

        # meta_vect = [Nb, R, lax, sax, ku]
        axis_id = 1 if (Nu, Nf) in [(204, 448), (168, 448), (132, 448)] else 2

        # meta_vect = torch.zeros(Nb,self.in_features_mlp).to(x.device)
        # meta_vect[:,0] = R
        # meta_vect[:,axis_id] = 1
        xrss = x.abs().square().sum(1).pow(0.5)

        for ku in range(self.T):
            print(ku)
            if axis_id == 1:
                # print('sax')
                meta_vect_data = torch.tensor([R, 1, 0]).to(x.device)
            elif axis_id == 2:
                # print('lax')
                meta_vect_data = torch.tensor([R, 0, 1]).to(x.device)

            if self.T > 1:
                # also use information about the iteration
                iteration_info = ku / (self.T - 1)
                meta_vect_data = torch.cat([meta_vect_data, torch.tensor([iteration_info]).to(x.device)], dim=-1)

            meta_vect = torch.stack(Nb * [meta_vect_data], dim=0)

            # lambda_reg = self.normfactor * F.softplus(self.lambda_mlp(meta_vect) )
            lambda_reg = self.normfactor * F.sigmoid(self.lambda_mlp(meta_vect))

            print(lambda_reg)

            with torch.no_grad():
                # xcrop, cuts = self.crop_op(x)
                proj = torch.sum(x.sum(1).abs().mean(1).mean(2), dim=-2)
                threshold = 0.00 * proj.max().item()
                cuts = crops_by_threshold(x.cpu().numpy(), (None, None, None, None, None, threshold))

            # xnn = self.img_cnn(x[cuts] * self.normfactor) * (1 / self.normfactor) + x[cuts]
            xnn = self.img_cnn(x[cuts]) + x[cuts]

            # xnn = self.img_cnn_xyz(self.img_cnn_xyt( x[cuts] * self.normfactor) ) * (1 / self.normfactor) + x[cuts]
            # xnn = self.img_cnn(x[cuts] * self.normfactor) * (1 / self.normfactor) + x[cuts]

            # xnn = self.img_cnn(x * self.normfactor) * (1 / self.normfactor) + x
            # pad xnn with zeros
            pad_sequence = (cuts[-1].start, Nf - cuts[-1].stop)
            xnn = torch.nn.functional.pad(xnn, pad_sequence)

            x = self.mcdc(y, xnn, mask, lambda_reg)

        p_k = torch.fft.fftn(x, dim=(-2, -1), norm="ortho")

        p_x = torch.fft.ifftn(p_k, dim=(-2, -1), norm="ortho").abs().square().sum(1).pow(0.5)

        csm = None
        return p_x, p_k, csm, xrss


class CSMFreeRecon(CineModel):
    def __init__(self, T=1, lr=1e-3, weight_decay=0.0, schedule=False, mode="xyt", phase="training", normfactor=1e4):
        super().__init__()
        # TODO: choose parameters

        self.net = CSMFreeReconNN(T=T, mode=mode, normfactor=normfactor)

        self.lr = lr
        self.weight_decay = weight_decay

        self.mode = mode

        self.phase = phase
        self.normfactor = normfactor

    def forward(self, k: torch.Tensor, mask: torch.Tensor, **other) -> dict:
        """
        JointModelBasedCSMImageRecon

        Parameters
        ----------
        k
            shape: (batch, coils, z, t, undersampled, fullysampled)
        mask
            shape: (batch, z, t, undersampled, fullysampled)
        csm
            shape: (batch, coils, z, undersampled, fullysampled)

        Returns
        -------
            x, ..., rss
        """

        idx = None
        if self.phase == "training":
            if self.mode == "xyz":
                # randomly choose an index for the time point and use all z-slices
                Nt = 12
                idx = np.random.randint(0, Nt)
                # print(idx)
                # print(k.shape)
                k = k[:, :, :, [idx], ...]
                # print(k.shape)
                # print(mask.shape)
                # mask = mask[:, :, [idx], ...]
                # print(mask.shape)

        p_x, p_k, p_csm, xrss = self.net(k, mask)
        return dict(prediction=p_x, p_k=p_k, p_csm=p_csm, rss=xrss), idx

    def training_step(self, batch, batch_idx):
        gt = self.normfactor * batch.pop("gt")
        # k_full = batch.pop("kf")
        ret, idx = self(**batch)

        # print('training')
        # print('pred.shape = {}'.format(ret["prediction"].shape))
        # print('gt.shape = {}'.format(gt.shape))

        if self.mode == "xyz":
            gt = gt[:, :, [idx], ...]

        # print('gt_new.shape = {}'.format(gt.shape))

        prediction, rss = ret["prediction"], ret["rss"]
        # rss = ret["rss"]

        loss = torch.nn.functional.mse_loss(prediction, gt)
        # print("KEYS OF ret: {}".format(ret.keys()))

        # MSE on the k-space data
        # k_prediction = ret["p_k"]
        # loss = torch.nn.functional.mse_loss(torch.view_as_real(k_prediction), torch.view_as_real(k_full))

        rss_loss = torch.nn.functional.mse_loss(rss, gt)
        self.log("train_advantage", (rss_loss - loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("rss_loss", rss_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # loss.backward(retain_graph=True)
        # print(self.net.lambda_reg.grad, self.net.lambda_reg)
        return loss

    def validation_step(self, batch, batch_idx):
        gt = self.normfactor * batch.pop("gt")
        ret, idx = self(**batch)

        # print('validation')
        # print('pred.shape = {}'.format(ret["prediction"].shape))
        # print('gt.shape = {}'.format(gt.shape))

        if self.mode == "xyz":
            gt = gt[:, :, [idx], ...]

        # print('gt_new.shape = {}'.format(gt.shape))

        prediction, rss = ret["prediction"], ret["rss"]
        loss = torch.nn.functional.mse_loss(prediction, gt)
        rss_loss = torch.nn.functional.mse_loss(rss, gt)
        self.log("val_advantage", (rss_loss - loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if batch_idx == 0:
            for logger in self.loggers:
                if isinstance(logger, NeptuneLogger):
                    # only for neptune logger, log the first image
                    img = prediction[0, 0, 0, :, :].detach().cpu().numpy()
                    img = img - img.min()
                    img = img / img.max()
                    logger.experiment["val/image"].log(neptuneFile.as_image(img))

    def test_step(self, batch, batch_idx):
        ret, idx = self(**batch)
        return ret["prediction"] / self.normfactor

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
        # # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        # # 	optimizer, max_lr=3e-3, total_steps=self.trainer.estimated_stepping_batches, pct_start=0.05, anneal_strategy="cos", cycle_momentum=True, div_factor=30, final_div_factor=1e3, verbose=False
        # # )
        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class CSMFreeReconDeluxeNN(nn.Module):

    """ """

    def __init__(
        self,
        # img_unet=Unet(3, channels_in=2, channels_out=2, layer=3, filters=32, n_convs_per_stage=2), #for  230
        # img_unet=Unet(3, channels_in=20, channels_out=20, layer=3, filters=96, n_convs_per_stage=4), #for  184
        img_unet=Unet(3, channels_in=20, channels_out=20, layer=3, filters=64, n_convs_per_stage=4),  # for  184
        lambda_unet=Unet(3, channels_in=2, channels_out=1, layer=1, filters=16, n_convs_per_stage=4),
        T=1,
        normfactor=1e2,
        mode="xyt",
    ):
        super().__init__()

        self.T = T

        self.mcdc = MultiCoilDCLayer()
        # self.img_cnn = MultiCoilImageCNN(img_unet, mode) #
        self.img_cnn = MultiCoilImageCNN_v2(img_unet, mode)
        self.lambda_cnn = LambdaCNN(lambda_unet)
        self.in_features_mlp = 3
        self.lambda_mlp = MLP(in_features=self.in_features_mlp, out_features=1, hidden_dim=5, n_layers=2)

        layer, n_convs_per_stage, filters = 3, 4, 64
        pre_trained_weights_path = "/data/kofler01/projects/CMRxRecon/pre_trained_models/unet_E{}C{}K{}.pt".format(
            layer, n_convs_per_stage, filters
        )
        self.img_cnn.load_state_dict(torch.load(pre_trained_weights_path))

        # meta

        self.normfactor = normfactor

    def forward(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # print('mask.shape')

        # initial image
        Nb, Nc, Nz, Nz, Nu, Nf = y.shape
        x = torch.fft.ifftn(y, dim=(-2, -1), norm="ortho")

        # get global meta-information
        R = torch.sum(mask) / Nu

        # print(R)

        # meta_vect = [Nb, R, lax, sax, ku]
        axis_id = 1 if (Nu, Nf) in [(204, 448), (168, 448), (132, 448)] else 2
        if axis_id == 1:
            # print('sax')
            meta_vect_data = torch.tensor([R, 1, 0]).to(x.device)
        elif axis_id == 2:
            # print('lax')
            meta_vect_data = torch.tensor([R, 0, 1]).to(x.device)

        meta_vect = torch.stack(Nb * [meta_vect_data], dim=0)
        # meta_vect = torch.zeros(Nb,self.in_features_mlp).to(x.device)
        # meta_vect[:,0] = R
        # meta_vect[:,axis_id] = 1
        t = F.sigmoid(self.lambda_mlp(meta_vect))

        # sum of coil-weighted images
        x0 = x.sum(1)

        # get lambda_map
        lambda_map = (1 / self.normfactor) * self.lambda_cnn(self.normfactor * x0)
        # lambda_map =  lambda_map.unsqueeze(1)

        xrss = x.abs().square().sum(1).pow(0.5)

        for ku in range(self.T):
            with torch.no_grad():
                # xcrop, cuts = self.crop_op(x)
                proj = torch.sum(x.sum(1).abs().mean(1).mean(2), dim=-2)
                print("threshold")
                threshold = 0.0 * proj.max().item()
                cuts = crops_by_threshold(x.cpu().numpy(), (None, None, None, None, None, threshold))

            # xnn = self.img_cnn(x[cuts] * self.normfactor) * (1 / self.normfactor) + x[cuts]
            # solve a system
            xnn = self.img_cnn(x[cuts] * self.normfactor) * (1 / self.normfactor) + x[cuts]

            # xnn = self.img_cnn(x * self.normfactor) * (1 / self.normfactor) + x
            # pad xnn with zeros
            pad_sequence = (cuts[-1].start, Nf - cuts[-1].stop)
            xnn = torch.nn.functional.pad(xnn, pad_sequence)

            # meta_vect[:,3] = ku / self.T
            t = self.lambda_mlp(meta_vect)

            # x = self.mcdc(y, xnn, mask, t * lambda_map)

            # solve a system of linear equations
            H = (
                lambda x: torch.fft.ifftn(mask * torch.fft.fftn(x, dim=(-2, -1), norm="ortho"), dim=(-2, -1), norm="ortho")
                + t * lambda_map * x
            )
            b = torch.fft.ifftn(y, dim=(-2, -1), norm="ortho") + t * lambda_map * xnn
            x = conj_grad(H, b, xnn, niter=4)

        p_k = torch.fft.fftn(x, dim=(-2, -1), norm="ortho")

        p_x = torch.fft.ifftn(p_k, dim=(-2, -1), norm="ortho").abs().square().sum(1).pow(0.5)

        csm = None
        return p_x, p_k, csm, xrss


class CSMFreeReconDeluxe(CineModel):
    def __init__(self, T=1, lr=1e-3, weight_decay=0.0, schedule=False, mode="xyt", phase="training"):
        super().__init__()
        # TODO: choose parameters

        self.net = CSMFreeReconDeluxeNN(T=T, mode=mode)

        self.lr = lr
        self.weight_decay = weight_decay

        self.mode = mode

        self.phase = phase

    def forward(self, k: torch.Tensor, mask: torch.Tensor, **other) -> dict:
        """
        JointModelBasedCSMImageRecon

        Parameters
        ----------
        k
            shape: (batch, coils, z, t, undersampled, fullysampled)
        mask
            shape: (batch, z, t, undersampled, fullysampled)
        csm
            shape: (batch, coils, z, undersampled, fullysampled)

        Returns
        -------
            x, ..., rss
        """

        idx = None
        if self.phase == "training":
            if self.mode == "xyz":
                # randomly choose an index for the time point and use all z-slices
                Nt = 12
                idx = np.random.randint(0, Nt)
                # print(idx)
                # print(k.shape)
                k = k[:, :, :, [idx], ...]
                # print(k.shape)
                # print(mask.shape)
                # mask = mask[:, :, [idx], ...]
                # print(mask.shape)

        p_x, p_k, p_csm, xrss = self.net(k, mask)
        return dict(prediction=p_x, p_k=p_k, p_csm=p_csm, rss=xrss), idx

    def training_step(self, batch, batch_idx):
        gt = batch.pop("gt")
        # k_full = batch.pop("kf")
        ret, idx = self(**batch)

        # print('training')
        # print('pred.shape = {}'.format(ret["prediction"].shape))
        # print('gt.shape = {}'.format(gt.shape))

        if self.mode == "xyz":
            gt = gt[:, :, [idx], ...]

        # print('gt_new.shape = {}'.format(gt.shape))

        prediction, rss = ret["prediction"], ret["rss"]
        # rss = ret["rss"]

        loss = torch.nn.functional.mse_loss(prediction, gt)
        # print("KEYS OF ret: {}".format(ret.keys()))

        # MSE on the k-space data
        # k_prediction = ret["p_k"]
        # loss = torch.nn.functional.mse_loss(torch.view_as_real(k_prediction), torch.view_as_real(k_full))

        rss_loss = torch.nn.functional.mse_loss(rss, gt)
        self.log("train_advantage", (rss_loss - loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("rss_loss", rss_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # loss.backward(retain_graph=True)
        # print(self.net.lambda_reg.grad, self.net.lambda_reg)
        return loss

    def validation_step(self, batch, batch_idx):
        gt = batch.pop("gt")
        ret, idx = self(**batch)

        # print('validation')
        # print('pred.shape = {}'.format(ret["prediction"].shape))
        # print('gt.shape = {}'.format(gt.shape))

        if self.mode == "xyz":
            gt = gt[:, :, [idx], ...]

        # print('gt_new.shape = {}'.format(gt.shape))

        prediction, rss = ret["prediction"], ret["rss"]
        loss = torch.nn.functional.mse_loss(prediction, gt)
        rss_loss = torch.nn.functional.mse_loss(rss, gt)
        self.log("val_advantage", (rss_loss - loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if batch_idx == 0:
            for logger in self.loggers:
                if isinstance(logger, NeptuneLogger):
                    # only for neptune logger, log the first image
                    img = prediction[0, 0, 0, :, :].detach().cpu().numpy()
                    img = img - img.min()
                    img = img / img.max()
                    logger.experiment["val/image"].log(neptuneFile.as_image(img))

    def test_step(self, batch, batch_idx):
        ret, idx = self(**batch)
        return ret["prediction"]

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.)

        # self.net.img_cnn.load_state_dict('/data/kofler01/projects/CMRxRecon/pre_trained_models/unet_E3C4K64.pt')
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
        # # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        # # 	optimizer, max_lr=3e-3, total_steps=self.trainer.estimated_stepping_batches, pct_start=0.05, anneal_strategy="cos", cycle_momentum=True, div_factor=30, final_div_factor=1e3, verbose=False
        # # )
        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
