import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from cmrxrecon.models.utils.prox_ops import SoftShrinkAct
from cmrxrecon.models.utils.cg import conj_grad
from cmrxrecon.models.utils.op_norm import power_iteration
from cmrxrecon.models.utils.encoding import Dyn2DCartEncObj

from einops import rearrange

from . import CineModel

class ConvDicoLearnFISTANN(nn.Module):
    
    """
   
    """
    
    def __init__(self, 
                Dyn2DEncObj,
                T=1,
                n_filters = 8,
                kernel_size = [7,7,7],
                lambda_reg = 5e-7
                ):

        super(ConvDicoLearnFISTANN, self).__init__()
        
        #MR encoding objects
        self.Dyn2DEncObj  = Dyn2DEncObj
        
        #soft_thresholding
        self.soft_thresholding = SoftShrinkAct()
        
        #operator norms     
        self.op_norm_AHA = torch.sqrt(torch.tensor(1.)) #op-norm is one for appropriate csms
        
        #(log)-reg. parameter
        #lambda_log = torch.log(torch.tensor(lambda_reg)) 
        #lambda_log = nn.Parameter(-0.5 * torch.ones(1), requires_grad=True)
        #self.lambda_reg_log = nn.Parameter(lambda_log,requires_grad=True)
        
        self.lambda_reg_log = nn.Parameter( torch.log(lambda_reg * torch.ones(1)), requires_grad=True)
        
        self.T = T 
        
        self.n_filters = n_filters
        filter_shape = (n_filters, 1, ) + tuple([kernel_size[k] for k in range(len(kernel_size))])
        filter_init = 0.001*torch.randn(filter_shape)
		
        self.d_filter = nn.Parameter(filter_init, requires_grad= True)
        self.npad = tuple(2*[int(np.floor(kernel_size[k]/2)) for k in range(len(kernel_size))])
        
        
    def apply_D(self, s: torch.Tensor) -> torch.Tensor:
        
        """
        application of the dictionary to sparse codes 
        """
        
        Nb, Nz, n_filters, Nt, u, f = s.shape
        
        s = torch.view_as_real(s)
        s = torch.concat([s[...,0], s[...,1]], dim=2)

        #s = rearrange(s, "b z fil t u f ch -> (b z) ( fil ch ) u f t")
        s = rearrange(s, "b z fil t u f -> (b z)  fil  u f t")
        
        
        #prepare filter
        d = torch.cat(2*[self.d_filter],dim=0).to(s.device)
        Ds = F.conv_transpose3d(F.pad(s, self.npad,  mode = 'circular'), d, groups= 2, padding=[self.npad[0], self.npad[2], self.npad[4]])  
        Ds = Ds[:, :, self.npad[0]:-self.npad[1], self.npad[2]:-self.npad[3], self.npad[4]:-self.npad[5]]
        
        Ds = rearrange(Ds, "(b z) ch u f t -> b z t u f ch", b=Nb, z=Nz, ch=2)
        #Ds = rearrange(Ds, "(b z) ch u f t -> b z t u f ch", b=Nb, z=Nz, ch=2)
        
        Ds = torch.view_as_complex(Ds.contiguous())
        
        return Ds
        
    def apply_DH(self, x: torch.tensor) -> torch.Tensor:
        
        """
        application of the tranposed dictionary to an image
        """
        Nb, Nz, Nt, u, f = x.shape
        
        x = torch.view_as_real(x)
        
        x = rearrange(x, "b z t u f ch -> (b z) ch u f t")
        
        d = torch.cat(2*[self.d_filter],dim=0).to(x.device)
        DTx = F.conv3d(F.pad(x, self.npad,  mode = 'circular'), d, groups= 2, padding=[self.npad[0], self.npad[2], self.npad[4]])  
        DTx = DTx[:, :, self.npad[0]:-self.npad[1], self.npad[2]:-self.npad[3], self.npad[4]:-self.npad[5]]
        
        DTx = torch.stack([DTx[:,:int(self.n_filters), ...], DTx[:,int(self.n_filters):, ...]], dim=-1)
        #print(DTx.shape)
        
        #DTx = rearrange(DTx, "(b z) (fil ch) u f t -> b z fil t u f ch", b=Nb, z=Nz, ch=2, fil=self.n_filters)
        DTx = torch.view_as_complex(DTx.contiguous())
        DTx = rearrange(DTx, "(b z) fil u f t -> b z fil t u f", b=Nb, z=Nz, fil=self.n_filters)
        
        return DTx
    
    def apply_DHD(self, s: torch.Tensor) -> torch.Tensor:
        return self.apply_DH(self.apply_D(s))
    
    def apply_B(self, s: torch.tensor, csm: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        
        """
        the composition B = A D, where A is the foward operator and B the dictionary
        """
        
        return self.Dyn2DEncObj.apply_A(self.apply_D(s), csm, mask)
    
    def apply_BH(self, k: torch.tensor, csm: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        
        """
        the adjoint of  BH = DT AH, where A is the foward operator and B the dictionary
        """
        
        return self.apply_DH( self.Dyn2DEncObj.apply_AH(k, csm, mask) )
    
    def apply_BHB(self, s: torch.tensor, csm: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        
        return self.apply_BH( self.apply_B(s, csm, mask), csm, mask)
    
    def solve_normal_eqs(self, AHy: torch.Tensor, csm: torch.Tensor, mask: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        
        #solve a few few steps of CG for solving normal equations
        #to get better initialization
        with torch.no_grad():
            AHA = lambda x: self.Dyn2DEncObj.apply_AHA(x, csm, mask)
            x = conj_grad(AHA, AHy, x0, niter=8)
        return x    
        
    def forward(self, y: torch.Tensor, csm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        
        #initial recon
        AHy = self.Dyn2DEncObj.apply_AH(y, csm, mask)
        
        xrss = self.Dyn2DEncObj.apply_RSS(y)
        
        #get shape
        Nb, Nz, Nt, Nu, Nf = AHy.shape
                
        s = torch.zeros(Nb, Nz, self.n_filters, Nt, Nu, Nf, dtype=AHy.dtype).to(AHy.device)
        z = torch.zeros(Nb, Nz, self.n_filters, Nt, Nu, Nf, dtype=AHy.dtype).to(AHy.device)
        beta = torch.ones(Nb,Nz, self.n_filters, Nt, Nu, Nf, dtype=AHy.dtype).to(AHy.device)
        
        BHy = self.apply_BH(y, csm, mask)
                
        #estimate operator  norm
        BHB = lambda s: self.apply_DHD(s)
        #s0 = torch.rand(s.shape, dtype = s.dtype)
        s0 = torch.rand(1, 1, self.n_filters, 16, 16, 16, dtype = s.dtype)
        op_norm_BHB = power_iteration(BHB, s0, niter=36).abs()
        
        #threshold = self.SoftPlus(self.lambda_reg) / self.SoftPlus(self.c)
        lambda_reg = torch.exp(self.lambda_reg_log)
        threshold = lambda_reg / op_norm_BHB 
        
        for kiter in range(self.T):
            
            BHBz = self.apply_BHB(z, csm, mask)
            
            
            s_new = self.soft_thresholding( z - 1./op_norm_BHB * ( BHBz  - BHy ), 
                                        threshold)
                    
            #update beta
            beta_new = (1. + torch.sqrt(1+ 4 * beta**2 ) ) / 2.
            
            #update sparse codes
            z = s_new +  (beta - 1.) / beta_new * (s_new - s )
                        
            s = s_new
            beta = beta_new
        
        x = self.apply_D(s)
        
        
        
        # apply (full) forward model with estimated csms to xcnn
        p_k = self.Dyn2DEncObj.apply_A(x, csm, mask=None)

        # estimated image using RSS
        p_x = self.Dyn2DEncObj.apply_RSS(p_k)
        
        p_csm = csm
        
        return p_x, p_k, p_csm, xrss
    
    
class ConvDicoLearnFISTA(CineModel):
    def __init__(self, lr=1e-5, schedule=False):
        super().__init__()
        # TODO: choose parameters

        self.cdl_fista = ConvDicoLearnFISTANN(Dyn2DCartEncObj(), 
                                              T=12,
                                              n_filters = 32,
                                              kernel_size = [7,7,7],
                                              lambda_reg = 5e-2)

    def cdl_unit_norm_proj(self):
    	
    	"""
    	
    	"""
    	with torch.no_grad():
            print('project filters')
            n_filters=self.cdl_fista.n_filters
            for kf in range(n_filters):
                self.cdl_fista.d_filter[kf,...].div_(torch.norm(self.cdl_fista.d_filter[kf,...].flatten(), p=2, keepdim=True))
                
    def forward(self, k: torch.Tensor, mask: torch.Tensor, csm: torch.Tensor = None, **other) -> dict:
        """
        ConvDicoLearnFISTA

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
        p_x, p_k, p_csm, xrss = self.cdl_fista(k, csm, mask)
        return dict(prediction=p_x, p_k=p_k, p_csm=p_csm, rss=xrss)
    
    def training_step(self, batch, batch_idx):
        gt = batch.pop("gt")
        ret = self(**batch)
        
        #self.cdl_unit_norm_proj()
                
        prediction, rss = ret["prediction"], ret["rss"]
        loss = torch.nn.functional.mse_loss(prediction, gt)
        rss_loss = torch.nn.functional.mse_loss(rss, gt)
        self.log("train_advantage", (rss_loss - loss) / rss_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("rss_loss", rss_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss 

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        
        optimizer.step(closure = optimizer_closure)
        
        self.cdl_unit_norm_proj()
	