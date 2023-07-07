import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


from cmrxrecon.nets.unet_andreas import Unet
from . import CineModel

from einops import rearrange

class MultiCoilDCLayer(nn.Module):
    
    """
    Simultaneous Data-consistency layer for multiple receiver coils
    
    computes the argmin of min_x 1/2|| F_I x - y||_2**2 + lmbda/2|| x - xnn||_2**2
    
    for each coil separately...
    
    y.shape = [Nb, Nc, Nz, Nt, Nu, Nf]
    xnn.shape = [Nb, Nc, Nz, Nt, Nu, Nf]
    mask.shape = [Nb, Nc, Nt, Nu, 1]
    """
    
    def __init__(self):
        super(MultiCoilDCLayer, self).__init__()
        
    def forward(self, y: torch.Tensor, xnn: torch.Tensor, mask: torch.Tensor, lambda_reg: torch.Tensor) -> torch.Tensor:
        
        kxnn = torch.fft.fftn(xnn, dim=(-2,-1),norm='ortho')
        xreg = mask * ( lambda_reg / (1. + lambda_reg) * kxnn + 1. / (1. + lambda_reg)  * y) + ~mask * kxnn
        xdc = torch.fft.ifftn(xreg, dim=(-2,-1),norm='ortho')
        return xdc
 

class MultiCoilImageCNN(nn.Module):
    
    """
    apply a simple CNN-block to a multi-coil image
    """
    
    def __init__(self, cnn_block ):
        super(MultiCoilImageCNN, self).__init__()
        
        self.cnn_block = cnn_block
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        Nb, Nc, Nz, Nt, Nuf, Nf = x.shape
        
        x = torch.view_as_real(x)
        
        x = rearrange(x, "b c z t u f ch-> (b c z) ch u f t")
        
        x = self.cnn_block(x)
        
        x = rearrange(x, "(b c z) ch u f t-> b c z t u f ch", b=Nb, c=Nc, z=Nz, ch=2)
        
        x = torch.view_as_complex(x.contiguous())
                
        return x
    

class CSMFreeReconNN(nn.Module):

    """
    

    """

    def __init__(self,
        img_unet = Unet(3, channels_in=2, channels_out=2, layer=1, filters=32, n_convs_per_stage=4),
        T=1,
        lambda_init = 1e-5,
        normfactor = 1e2
    ):
        super(CSMFreeReconNN, self).__init__()
            
        self.T = T
        
        self.mdcd = MultiCoilDCLayer()
        self.img_cnn = MultiCoilImageCNN(img_unet)
        
        self.lambda_reg = torch.nn.Parameter(torch.log(torch.tensor([lambda_init])))
        
        self.normfactor = normfactor
        
    def forward(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        
        #zerof-filled recon
        x = torch.fft.ifftn(y,dim=(-2,-1),norm='ortho')
        
        lambda_reg = torch.exp(self.lambda_reg)
        
        xrss = torch.fft.ifftn(y,dim=(-2,-1),norm='ortho').abs().square().sum(1).pow(0.5)
        
        for ku in range(self.T):
            
            xnn = self.img_cnn(x * self.normfactor) * (1 / self.normfactor) + x
            
            x = self.mdcd(y, xnn, mask, lambda_reg)
        
        
        p_k = torch.fft.fftn(x, dim=(-2,-1),norm='ortho')
         
        p_x = torch.fft.ifftn(p_k, dim=(-2,-1), norm='ortho').abs().square().sum(1).pow(0.5)
        
        csm = None
        return p_x, p_k, csm, xrss


class CSMFreeRecon(CineModel):
    def __init__(self, T=1, lr=1e-3,  weight_decay=0., schedule=False):
        super().__init__()
        # TODO: choose parameters

        self.net = CSMFreeReconNN()
        
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, k: torch.Tensor, mask: torch.Tensor,  **other) -> dict:
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
        p_x, p_k, p_csm, xrss = self.net(k, mask)
        return dict(prediction=p_x, p_k=p_k, p_csm=p_csm, rss=xrss)

    def training_step(self, batch, batch_idx):
        
        gt = batch.pop("gt")
        #k_full = batch.pop("kf")
        ret = self(**batch)
        
        #self.cdl_unit_norm_proj()
        
        #MSE loss on the images   
        
        prediction, rss = ret["prediction"], ret["rss"]
        #rss = ret["rss"]
        
        loss = torch.nn.functional.mse_loss(prediction, gt)
        print('KEYS OF ret: {}'.format(ret.keys()))
        
        #MSE on the k-space data
        #k_prediction = ret["p_k"]
        #loss = torch.nn.functional.mse_loss(torch.view_as_real(k_prediction), torch.view_as_real(k_full))
        
        rss_loss = torch.nn.functional.mse_loss(rss, gt)
        self.log("train_advantage", (rss_loss - loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("rss_loss", rss_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss 
    
    def configure_optimizers(self):
        #optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        return optimizer
        # # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        # # 	optimizer, max_lr=3e-3, total_steps=self.trainer.estimated_stepping_batches, pct_start=0.05, anneal_strategy="cos", cycle_momentum=True, div_factor=30, final_div_factor=1e3, verbose=False
        # # )
        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]