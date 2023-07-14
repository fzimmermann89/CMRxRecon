import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cmrxrecon.nets.unet_andreas import Unet

from . import CineModel

from neptune.new.types import File as neptuneFile
from pytorch_lightning.loggers.neptune import NeptuneLogger

from cmrxrecon.models.utils import crops_by_threshold


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
        super(MultiCoilDCLayer, self).__init__()

    def forward(self, y: torch.Tensor, xnn: torch.Tensor, mask: torch.Tensor, lambda_reg: torch.Tensor) -> torch.Tensor:
        kxnn = torch.fft.fftn(xnn, dim=(-2, -1), norm="ortho")
        xreg = mask * (lambda_reg / (1.0 + lambda_reg) * kxnn + 1.0 / (1.0 + lambda_reg) * y) + ~mask * kxnn
        xdc = torch.fft.ifftn(xreg, dim=(-2, -1), norm="ortho")
        return xdc


class MultiCoilImageCNN(nn.Module):

    """
    apply a simple CNN-block to each coil-separately
    """

    def __init__(self, cnn_block, mode = 'xyt'):
        super().__init__()

        self.cnn_block = cnn_block
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Nb, Nc, Nz, Nt, Nuf, Nf = x.shape

        x = torch.view_as_real(x)

        if self.mode == 'xyt':
            pattern = "b c z t u f ch-> (b c z) ch u f t"
        elif self.mode == 'xyz':
            pattern = "b c z t u f ch-> (b c t) ch u f z"
            
        x = rearrange(x, pattern)

        x = self.cnn_block(x)
        
        if self.mode == 'xyt':
            pattern = "(b c z) ch u f t-> b c z t u f ch"
        elif self.mode == 'xyz':
            pattern = "(b c t) ch u f z-> b c z t u f ch"
            
        x = rearrange(x, pattern, b=Nb, c=Nc, z=Nz, t=Nt, ch=2)

        x = torch.view_as_complex(x.contiguous())

        return x
    
class MultiCoilImageCNN_v2(nn.Module):

    """
    apply a simple CNN-block to a multi-coil image by stacking the channels
    """

    def __init__(self, cnn_block, mode = 'xyt'):
        super().__init__()

        self.cnn_block = cnn_block
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Nb, Nc, Nz, Nt, Nuf, Nf = x.shape

        x = torch.view_as_real(x)
        x = torch.concatenate([x[...,0], x[...,1]],dim=1)

        if self.mode == 'xyt':
            pattern = "b c2 z t u f-> (b z) c2 u f t"
        elif self.mode == 'xyz':
            pattern = "b c2 z t u f-> (b t) c2 u f z"
            
        x = rearrange(x, pattern)

        x = self.cnn_block(x)
        
        if self.mode == 'xyt':
            pattern = "(b z) c2 u f t-> b c2 z t u f"
        elif self.mode == 'xyz':
            pattern = "(b t) c2 u f z-> b c2 z t u f"
            
        x = rearrange(x, pattern, b=Nb, c2=2*Nc, z=Nz, t=Nt)
        
        x = torch.stack([x[:,:Nc,...], x[:,Nc:,...]],dim=-1)

        x = torch.view_as_complex(x.contiguous())

        return x
    

class CSMFreeReconNN(nn.Module):

    """ """

    def __init__(self, 
                 #img_unet=Unet(3, channels_in=2, channels_out=2, layer=3, filters=32, n_convs_per_stage=2),
                 img_unet=Unet(3, channels_in=20, channels_out=20, layer=1, filters=32, n_convs_per_stage=4), 
                 T=1, 
                 lambda_init=1e1, 
                 normfactor=1e2,
                 mode = 'xyt'):
        super().__init__()

        self.T = T

        self.mdcd = MultiCoilDCLayer()
        #self.img_cnn = MultiCoilImageCNN(img_unet, mode) #
        self.img_cnn = MultiCoilImageCNN_v2(img_unet, mode)
        
        self.lambda_reg = torch.nn.Parameter(torch.log(torch.tensor([lambda_init])))

        self.normfactor = normfactor
        

    def forward(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # zerof-filled recon
        #print('yshape = {}'.format(y.shape))
        
        Nb, Nc, Nz, Nz, Nu, Nf = y.shape
        x = torch.fft.ifftn(y, dim=(-2, -1), norm="ortho")
        
        lambda_reg = torch.exp(self.lambda_reg)

        xrss = x.abs().square().sum(1).pow(0.5)

        for ku in range(self.T):
            
            with torch.no_grad():
                #xcrop, cuts = self.crop_op(x)
                proj = torch.sum(x.sum(1).abs().mean(1).mean(2),dim=-2)
                threshold = 0.03 * proj.max().item()
                cuts = crops_by_threshold(x.cpu().numpy(), (None,None,None,None,None,threshold))
                
            xnn = self.img_cnn(x[cuts] * self.normfactor) * (1 / self.normfactor) + x[cuts]
            
            
            #xnn = self.img_cnn(x * self.normfactor) * (1 / self.normfactor) + x
            #pad xnn with zeros
            pad_sequence = (cuts[-1].start, Nf - cuts[-1].stop)
            xnn = torch.nn.functional.pad(xnn, pad_sequence)

            x = self.mdcd(y, xnn, mask, lambda_reg)

        p_k = torch.fft.fftn(x, dim=(-2, -1), norm="ortho")

        p_x = torch.fft.ifftn(p_k, dim=(-2, -1), norm="ortho").abs().square().sum(1).pow(0.5)

        csm = None
        return p_x, p_k, csm, xrss


class CSMFreeRecon(CineModel):
    def __init__(self, T=1, lr=1e-3, weight_decay=0.0, schedule=False, mode = 'xyt', phase = 'training'):
        super().__init__()
        # TODO: choose parameters

        self.net = CSMFreeReconNN(T=T, mode = mode)

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
        if self.phase == 'training':
            if self.mode == 'xyz':
                #randomly choose an index for the time point and use all z-slices
                Nt = 12
                idx = np.random.randint(0,Nt)
                #print(idx)
                #print(k.shape)
                k = k[:, :, :, [idx], ...]
                #print(k.shape)
                #print(mask.shape)
                #mask = mask[:, :, [idx], ...]
                #print(mask.shape)
            
        p_x, p_k, p_csm, xrss = self.net(k, mask)
        return dict(prediction=p_x, p_k=p_k, p_csm=p_csm, rss=xrss), idx

    def training_step(self, batch, batch_idx):
        
        gt = batch.pop("gt")
        # k_full = batch.pop("kf")
        ret, idx = self(**batch)
        
       # print('training')
        #print('pred.shape = {}'.format(ret["prediction"].shape))
        #print('gt.shape = {}'.format(gt.shape))
        
        if self.mode == 'xyz':
            gt = gt[:,:,[idx],...]
        
        #print('gt_new.shape = {}'.format(gt.shape))
        
        prediction, rss = ret["prediction"], ret["rss"]
        # rss = ret["rss"]
        
        loss = torch.nn.functional.mse_loss(prediction, gt)
        #print("KEYS OF ret: {}".format(ret.keys()))

        # MSE on the k-space data
        # k_prediction = ret["p_k"]
        # loss = torch.nn.functional.mse_loss(torch.view_as_real(k_prediction), torch.view_as_real(k_full))

        rss_loss = torch.nn.functional.mse_loss(rss, gt)
        self.log("train_advantage", (rss_loss - loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("rss_loss", rss_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        #loss.backward(retain_graph=True)
        #print(self.net.lambda_reg.grad, self.net.lambda_reg)
        return loss
    
    def validation_step(self, batch, batch_idx):
        gt = batch.pop("gt")
        ret, idx = self(**batch)
        
        #print('validation')
        #print('pred.shape = {}'.format(ret["prediction"].shape))
        #print('gt.shape = {}'.format(gt.shape))
        
        if self.mode == 'xyz':
            gt = gt[:,:,[idx],...]
        
        #print('gt_new.shape = {}'.format(gt.shape))
        
        
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

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
        # # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        # # 	optimizer, max_lr=3e-3, total_steps=self.trainer.estimated_stepping_batches, pct_start=0.05, anneal_strategy="cos", cycle_momentum=True, div_factor=30, final_div_factor=1e3, verbose=False
        # # )
        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
