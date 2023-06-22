import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from cmrxrecon.models.grad_ops import GradOperators
from cmrxrecon.models.prox_ops import ClipAct
from cmrxrecon.models.utils.cg import conj_grad

from einops import rearrange

class Laplace2DCSM(nn.Module):
    def __init__(self):
        super(Laplace2DCSM, self).__init__()
        self.laplace_kernel = torch.tensor([[1., 1. , 1.], [1., -8., 1.], [1., 1. , 1.]]).unsqueeze(0)
        
    def apply_L(self, csm):
        
        Nb, Nc, Nz, u, f = csm.shape
        csm = torch.view_as_real(csm)
        
        csm = rearrange(csm, 'b c z u f r -> (b z c) r u f')
        
        kernel = torch.stack(2*[self.laplace_kernel], dim=0).to(csm.device)
        csm = F.conv2d(csm, kernel, groups=2, padding=1)
        
        csm = rearrange(csm, '(b z c) r u f -> b c z u f r', b=Nb, c=Nc, z=Nz)
        csm = torch.view_as_complex(csm.contiguous())
        
        return csm
    
    
class PDHG4DynMRIwTVPlusCSMs(nn.Module):
    
    """
    alternate T times of solving problems

    solve 

    min_{C,x} 1/2 || A(C,x) - y ||_2 + \lambda_xy ||\nabla_xy x ||_1 +   \lambda_t ||\nabla_t x ||_1 + \lambda_c/2 || (I - Laplace) C ||_2

    by alternating minimization by solving 

    min_x 1/2 || A_C x - y ||_2 + \lambda_xy ||\nabla_xy x ||_1 +   \lambda_t ||\nabla_t x ||_1

    min_C 1/2 || A_x C - y ||_2 + \lambda_c || (I - Laplace) C ||_2
     
    """
    
    def __init__(self, 
                Dyn2DEncObj,
                T=4,
                T_pdhg = 192,
                T_csm = 8,
                lambda_reg_xy = 5e-7,
                lambda_reg_t = 1e-6,
                lambda_reg_c = 2e-7,
                ):

        super(PDHG4DynMRIwTVPlusCSMs, self).__init__()
        
        #MR encoding objects
        self.Dyn2DEncObj  = Dyn2DEncObj
        
        #gradient operators and clipping function
        dim=3
        self.GradOps = GradOperators(dim, mode = 'forward')
        self.ClipAct = ClipAct()
        
        #operator norms     
        self.op_norm_AHA = torch.sqrt(torch.tensor(1.)) #op-norm is one for appropriate csms
        self.op_norm_GHG = torch.sqrt(torch.tensor(dim * 4.)) #can be estimtaed by uncommenting below, 
        self.L =  np.sqrt(self.op_norm_AHA ** 2 + self.op_norm_GHG ** 2 ) 
        
        #(log) constants depending on the operators
        self.tau = nn.Parameter(torch.tensor(10.),requires_grad=True) #starting value approximately  1/L
        self.sigma = nn.Parameter(torch.tensor(10.),requires_grad=True) #starting value approximately  1/L
        
        #theta should be in \in [0,1]
        self.theta = nn.Parameter(torch.tensor(10.),requires_grad=True) #starting value approximately  1
        
        #(log)-reg. parameter
        lambda_log_xy_init = torch.log(torch.tensor(lambda_reg_xy)) 
        self.lambda_reg_log_xy = nn.Parameter(lambda_log_xy_init,requires_grad=True)
        
        lambda_log_t_init = torch.log(torch.tensor(lambda_reg_t)) 
        self.lambda_reg_log_t = nn.Parameter(lambda_log_t_init,requires_grad=True)
        
        lambda_log_c_init = torch.log(torch.tensor(lambda_reg_c)) 
        self.lambda_reg_log_c = nn.Parameter(lambda_log_c_init,requires_grad=True)
        
        self.T = T 
        self.T_pdhg = T_pdhg
        self.T_csm = T_csm
        
        self.LaplaceOps = Laplace2DCSM()
        
    def apply_G4D(self, x: torch.Tensor) -> torch.Tensor:
        """ apply G to a 4D tensor also contaiing the slices"""
        
        Nb, Nz, Nt, u, f = x.shape
        
        #reshape to move z to slices
        x = rearrange(x, 'b z t u f -> (b z) u f t')
        
        Gx = self.GradOps.apply_G(x)
        
        Gx = rearrange(Gx, '(b z) ch u f t -> b ch z t u f', b=Nb, ch=3)
        
        return Gx
        
    def apply_GH4D(self, z: torch.tensor) -> torch.Tensor:
        """ apply G^H to a 4D tensor also contaiing the slices"""
        
        Nb, ch, Nz, Nt, u, f = z.shape
        
        z = rearrange(z, 'b ch z t u f -> (b z) ch u f t')
        
        GHz = self.GradOps.apply_GH(z)
        
        GHz = rearrange(GHz, '(b z) u f t -> b z t u f ', b=Nb)
        
        return GHz
    
    def solve_normal_eqs(self, AHy: torch.Tensor, csm: torch.Tensor, mask: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        
        #solve a few few steps of CG for solving normal equations
        #to get better initialization
        with torch.no_grad():
            AHA = lambda x: self.Dyn2DEncObj.apply_AHA(x, csm, mask)
            x = conj_grad(AHA, AHy, x0, niter=8)
        return x    
        
    def get_lambda_reg(self):
        
        #get xy- and t-lambda 
        lambda_reg_log_xy = torch.stack(2*[self.lambda_reg_log_xy])
        lambda_reg_log_t = self.lambda_reg_log_t.unsqueeze(0)
        
        #conatentae xy -and t-lambda
        lambda_reg_log = torch.cat([lambda_reg_log_t, lambda_reg_log_xy]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda_reg = torch.exp(lambda_reg_log)
        
        return lambda_reg
        
        
    def forward(self, y: torch.Tensor, mask: torch.Tensor, csm: torch.Tensor) -> torch.Tensor:
        
        L = self.L
        #sigma, tau, theta
        sigma = (1 / L ) * torch.sigmoid(self.sigma)    #\in (0,1/L)
        tau = (1 / L ) * torch.sigmoid(self.tau)        #\in (0,1/L)
        theta = torch.sigmoid(self.theta)               #\in (0,1)
        
        #RSS recon
        #xrss = self.Dyn2DEncObj.apply_RSS(y)
        
        #zerof filled reconstruction, i.e. A^H y with the estimated csms
        AHy = self.Dyn2DEncObj.apply_AH(y, csm, mask)
        
        #approximately solve the normal equations to get a better
        #initialization for the image
        #AHA = lambda x: self.Dyn2DEncObj.apply_AHA(x, csm, mask)
        #xneq = conj_grad(AHA, AHy, AHy, niter=8)
        
        #create x0 = r * exp(i * phi) 
        #with r = xrss (magnitude image) and phi = angle(xneq), 
        #where xneq is the approximate solution of the normal equations A(C)^H A x = A(C)^Hy
        #x0 = xrss * torch.exp(1j * xneq.angle())
        x0 = AHy
        
        Nb, Nz, Nt, us, fs = x0.shape
        device = x0.device
        
        xbar = x0.clone()
        x0 = x0.clone()
        
        #dual variable
        p = torch.zeros(y.shape, dtype = y.dtype).to(device)
        q = torch.zeros(Nb, 3, Nz, Nt, us, fs, dtype = x0.dtype).to(device)
        
        lambda_reg = self.get_lambda_reg()
        
        lambda_csm = torch.exp(self.lambda_reg_log_c)
        
        #csm_reg = csm.clone()
        
        for ka in range(self.T):
            
            print('outer iteration {} of {}'.format(ka+1, self.T))
            #RUN PDHG
            for ku in range(self.T_pdhg):
                
                #print(ku)
                #update p 
                p =  (p + sigma * (self.Dyn2DEncObj.apply_A(xbar, csm, mask) - y) ) / (1.+sigma) 
                
                q = self.ClipAct(q + sigma * self.apply_G4D(xbar), lambda_reg)
                x1 = x0 - tau * self.Dyn2DEncObj.apply_AH(p, csm, mask) - tau * self.apply_GH4D(q)
                
                #update xbar
                xbar = x1 + theta * (x1 -x0)
                x0 = x1
        
            #update csms using a projected gradient descent with l1-norm reg
            #for ku in range(self.T_csm):
                
            #solve CG
            if self.T_csm !=0:
                    
                eps = 1e-8
                AHxAx = lambda csm: self.Dyn2DEncObj.apply_AHxAx(csm, x1 + eps, mask) +  lambda_csm * (csm - 2 * self.LaplaceOps.apply_L(csm) - self.LaplaceOps.apply_L(self.LaplaceOps.apply_L(csm)))
                            
                rhs = self.Dyn2DEncObj.apply_AHx(y, x1 + eps, mask) 
                
                csm = conj_grad(AHxAx, rhs, csm, niter=self.T_csm)
                
                
                #project onto set
                norm_factor = torch.pow( torch.sum(csm.conj() * csm, dim=1, keepdim=True), -0.5)
                csm = norm_factor * csm
            
            
        return x1, csm
	