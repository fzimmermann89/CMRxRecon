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




class ProxG(nn.Module):

    """
    A CNN-block which can be interpreted to be the prox
    of the function g
    i.e. the network is applied in the image domain
    (--> use a 3D network)
    """

    def __init__(self, cnn_block, mode = 'xyt'):
        super(ProxG, self).__init__()

        self.cnn_block = cnn_block
        self.mode = mode

    def forward(self, x: torch.Tensor, KHy: torch.Tensor) -> torch.Tensor:
        Nb, Nc, Nz, Nt, Nuf, Nf = x.shape
        
        inp = torch.cat([x, KHy], dim=-5)

        inp = torch.view_as_real(inp)
        inp = torch.concatenate([inp[...,0], inp[...,1]],dim=1)

        if self.mode == 'xyt':
            pattern = "b c4 z t u f-> (b z) c4 u f t"
        elif self.mode == 'xyz':
            pattern = "b c4 z t u f-> (b t) c4 u f z"
            
        inp = rearrange(inp, pattern)
        
        x = self.cnn_block(inp)
        
        if self.mode == 'xyt':
            pattern = "(b z) c2 u f t-> b c2 z t u f"
        elif self.mode == 'xyz':
            pattern = "(b t) c2 u f z-> b c2 z t u f"
            
        x = rearrange(x, pattern, b=Nb, c2=2*Nc, z=Nz, t=Nt)
        
        x = torch.stack([x[:,:Nc,...], x[:,Nc:,...]],dim=-1)

        x = torch.view_as_complex(x.contiguous())

        return x
    

    
class ProxFStar(nn.Module):

    """
    apply a simple 2D CNN-block to a multi-coil image by stacking time points as channels,
    i.e. 2.5 Unet
    """

    def __init__(self, cnn_block):
        super(ProxFStar, self).__init__()

        #a 2D Unet
        self.cnn_block = cnn_block
        
    def forward(self, y: torch.Tensor, Kx: torch.tensor) -> torch.Tensor:
        Nb, Nc, Nz, Nt, Nuf, Nf = y.shape
        
        inp = torch.concat([y, Kx],dim=-3)

        inp = torch.view_as_real(inp)
        inp = torch.concatenate([inp[...,0], inp[...,1]],dim=1)

        pattern = "b c2 z t2 u f-> (b z c2) t2 u f"
                    
        inp = rearrange(inp, pattern)

        
        y = self.cnn_block(inp)
        
        pattern = " (b z c2) t u f-> b c2 z t u f"
                    
        y = rearrange(y, pattern, b=Nb, c2=2*Nc, z=Nz, t=Nt)
        
        y = torch.stack([y[:,:Nc,...], y[:,Nc:,...]],dim=-1)

        y = torch.view_as_complex(y.contiguous())

        return y



class CSMFreeMultiCoilPDNetwork(nn.Module):

    """
    apply a simple CNN-block to a multi-coil image by stacking the channels
    """

    def __init__(self,  
                 prox_g_unet = Unet(3, channels_in=40, channels_out=20, layer=3, n_convs_per_stage=2, filters=32),
                 prox_fstar_unet = Unet(2, channels_in=24, channels_out=12, layer=3, n_convs_per_stage=2, filters=32),
                 relax_unet = Unet(3, channels_in=40, channels_out=20, layer=3, n_convs_per_stage=2, filters=16),
                 T=4):
        super(CSMFreeMultiCoilPDNetwork, self).__init__()

        self.prox_g_cnn = ProxG(prox_g_unet)
        self.prox_fstar_cnn = ProxFStar(prox_fstar_unet)
        self.relax_cnn = ProxG(relax_unet)
        
        self.T = T
                
    def forward(self, k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        
        K = lambda x: mask * torch.fft.fftn(x, dim=(-2, -1), norm="ortho")
        KH =  lambda k: torch.fft.ifftn(k * mask, dim=(-2, -1), norm="ortho")
        
        x0 = KH(k)
        xbar = x0.clone()
        y = torch.zeros_like(k)
        for kT in range(self.T):    
            y = self.prox_fstar_cnn(y, K(xbar)) + y
            x1 = self.prox_g_cnn(x0, KH(y)) + x0
            
            if kT!=self.T -1:
                xbar = self.relax_cnn(x1, x0) + x0 
       
        xrss = x0.abs().square().sum(1).pow(0.5)
        x = x1.abs().square().sum(1).pow(0.5)
        return x, None, None, xrss


class CSMFreeMultiCoilPD(CineModel):
    def __init__(self,
                 T=4, 
                 lr=1e-4, 
                 weight_decay=0.0, 
                 schedule=False,
                 normfactor = 1e4):
        super().__init__()
        
        self.net = CSMFreeMultiCoilPDNetwork(T=T)

        self.lr = lr
        self.weight_decay = weight_decay
        
        self.normfactor = normfactor

    def forward(self, k: torch.Tensor, mask: torch.Tensor, **other) -> dict:
                
        p_x, p_k, p_csm, xrss = self.net(self.normfactor * k, mask)
        return dict(prediction=p_x, p_k=p_k, p_csm=p_csm, rss=xrss)

    def training_step(self, batch, batch_idx):
        
        gt = self.normfactor * batch.pop("gt")
        # k_full = batch.pop("kf")
        ret = self(**batch)
        prediction, rss = ret["prediction"], ret["rss"]
        
        loss = torch.nn.functional.mse_loss(prediction, gt)
        
        rss_loss = torch.nn.functional.mse_loss(rss, gt)
        self.log("train_advantage", (rss_loss - loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("rss_loss", rss_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        gt = self.normfactor * batch.pop("gt")
        ret = self(**batch)
       
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
        ret = self(**batch)
        return ret["prediction"] / self.normfactor
    
    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
          