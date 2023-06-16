import torch
import torch.nn as nn
import einops
from cmrxrecon.nets.unet import Unet
from . import CineModel


def inner_product(t1,t2):
	
	if torch.is_complex(t1):
		
		innerp = torch.sum(t1.flatten() * t2.flatten().conj())
	else:
		innerp = torch.sum(t1.flatten() * t2.flatten())
	return innerp


def conj_grad(H, b, x, niter=4, tol=1e-16):
		
	#x is the starting value, b the rhs;
	r = H(x)
	r = b-r
	
	#initialize p
	p = r.clone()
	
	#old squared norm of residual
	sqnorm_r_old = inner_product(r,r)
	
	#sqnorm_b = inner_product(b,b)
	
	for kiter in range(niter):
		
		#print(kiter)
		#if sqnorm_r_old.item() / sqnorm_b.item()  > tol:
		
		#calculate Hp;
		d = H(p)

		#calculate step size alpha;
		inner_p_d = inner_product(p, d)
		
		alpha = sqnorm_r_old / inner_p_d

		#perform step and calculate new residual;
		x = torch.add(x, p, alpha= alpha.item())
		r = torch.add(r, d, alpha= -alpha.item())
		
		#new residual norm
		sqnorm_r_new = inner_product(r,r)
		#print('k={}; ||r||_2 = {}'.format(kiter, sqnorm_r_new))
		
		#calculate beta and update the norm;
		beta = sqnorm_r_new / sqnorm_r_old
		sqnorm_r_old = sqnorm_r_new

		p = torch.add(r,p,alpha=beta.item())
		#kiter+=1
		
	return x

class CSMNetV1(nn.Module):
	
	"""
	CNN to estimate CSM from y 
	(i.e. from the zero-filled recon of the center lines)
			 
	"""
	
	def __init__(self, net_csm):
		
		super(CSMNetV1, self).__init__()
		
		self.net_csm = net_csm
		
	def forward(self, y, n_center_lines=24):

		#according to 
		#https://arxiv.org/pdf/2004.06688.pdf; figure 1; eq (12)
		
		Nb, Nc, Nz, Nt, Nx, Ny = y.shape
		device = y.device
		
		#get middle 
		mask_center = torch.ones(Nb, 1, Nz, Nt, Nx, Ny).to(device)
		mask_center[...,n_center_lines//2:-n_center_lines//2,: ] = 0
		
		#mask k-space data
		ym = mask_center * y
		
		#temporal mean of zero-filled recon 
		x0 = torch.fft.ifftn(ym.mean(3), dim=(-2,-1), norm = 'ortho')
		norm_factor = torch.pow( torch.sum(x0.conj() * x0, dim=1, keepdim=True), -0.5)
		x0 = x0*norm_factor
		#transform to real-values tensor with two channels
		x0 = torch.view_as_real(x0)
		x0 = x0.permute(0,2,1,5,3,4).contiguous()
		x0 = x0.view(Nb*Nz, Nc*2, Nx, Ny)
		
		#apply NN
		csm = self.net_csm(x0)
		
		#bring back to complex-valued representation
		csm = csm.view(Nb, Nz, Nc, 2, Nx, Ny)
		csm = csm.permute(0,2,1,4,5,3).contiguous()
		csm = torch.view_as_complex(csm)	
		
		#normalize output
		norm_factor = torch.pow( torch.sum(csm.conj() * csm, dim=1, keepdim=True), -0.5)
		
		return norm_factor * csm 
	
	
class CSMNetV2(nn.Module):
	
	"""
	CNN to "refine" CSM from the CSMs initially estimated with ESPIRIT 
			 
	"""
	
	def __init__(self, net_csm):
		
		super(CSMNetV2, self).__init__()
		
		self.net_csm = net_csm
		
	def forward(self, csm):

		#according to 
		#https://arxiv.org/pdf/2004.06688.pdf; figure 1; eq (12)
		
		Nb, Nc, Nz, Nx, Ny = csm.shape
		
		#shift to real view
		csm = torch.view_as_real(csm)
		
		#reshape to have 2D input tensors
		csm = einops.rearrange(csm, 'b c z y x ch -> (b z) (c ch) y x')
		
		#apply 2D CNN
		csm = self.net_csm(csm)
		
		#rearrange
		csm = einops.rearrange(csm, '(b z) (c ch) y x -> b c z y x ch',b=1,c=10)
		
		#shift to complex view
		csm = torch.view_as_complex(csm.contiguous())
		
		#normalize output
		norm_factor = torch.pow( torch.sum(csm.conj() * csm, dim=1, keepdim=True), -0.5)
		
		return norm_factor * csm 
	
	
class ImgUNetSequence(nn.Module):
	
	"""
	a sequence of two 3D UNets which are applied to map a 4D input (x,y,z,t)
	to a 4D input by processing first
		2D + time: (x,y,t) --> (x,y,t)
	and then
		3D: (x,y,z) --> (x,y,z)
			 
	"""
	
	def __init__(self, net_xyz, net_xyt):
		
		super(ImgUNetSequence, self).__init__()
		
		self.net_xyt = net_xyt
		self.net_xyz = net_xyz
		
	def forward(self, x):
		
		#get shapes
		Nb, Nz, Nt, Ny, Nx = x.shape
		
		#change to real view
		x = torch.view_as_real(x)
		
		#from 4D to 3D (2D + time)
		x = einops.rearrange(x, 'b z t y x ch -> (b z) ch y x t')
		
		#apply spatio-temporal NN
		x = self.net_xyt(x)
		
		# switch time and slices dimensions; i.e. to three spatial dimensions)
		x = einops.rearrange(x, '(b z) ch y x t -> (b t) ch y x z', z=Nz)
		
		#apply 3D spatial NN
		x = self.net_xyz(x)
		
		#change back to original shape
		x = einops.rearrange(x, '(b t) ch y x z -> b z t y x ch', t=Nt)
		
		#and back to complex
		x = torch.view_as_complex(x.contiguous())
		
		return x

class JointCSMImageReconNN(nn.Module):
	def __init__(self,
		Dyn2DEncObj,
		net_img,
		net_csm
	):
		super().__init__()
		
		self.Dyn2DEncObj = Dyn2DEncObj
		self.net_img = net_img #learned mapping between two imaages
		self.net_csm = net_csm #learned mapping between two csms
		
	def forward(self, k: torch.Tensor, mask: torch.Tensor, csm: torch.Tensor) -> torch.Tensor:
		
		#expected input shape
		#y.shape = (Nb, Nc, Nz, Nt, Ny, Nx)
		#csm.shape = (Nb, Nc, Nz, Ny, Nx)
		#mask.shape (Nb, Nz, Nt, Ny, Nx)
		
		#estimate csms
		csm = self.net_csm(csm) 	#for version where csm is used;
		#csm = self.net_csm(y) 		#for version where y is used;
		
		#RSS recon
		xrss = self.Dyn2DEncObj.apply_RSS(k)
		
		#zerof filled reconstruction, i.e. A^H y with the estimated csms
		AHy = self.Dyn2DEncObj.apply_AH(k, csm, mask)
		
		#approximately solve the normal equations to get a better
		#initialization for the image
		AHA = lambda x: self.Dyn2DEncObj.apply_AHA(x, csm, mask)
		xneq = conj_grad(AHA, AHy, AHy, niter=4)
		
		#create x0 = r * exp(i * phi) 
		#with r = xrss (magnitude image) and phi = angle(xneq), 
		#where xneq is the approximate solution of the normal equations A(C)^H A x = A(C)^Hy
		x = xrss * torch.exp(1j * xneq.angle())
		
		#apply CNN
		x = self.net_img(x)
		
		#apply (full) forward model with estimated csms to xcnn
		kest = self.Dyn2DEncObj.apply_A(x, csm, mask=None)
		
		#estimated image using RSS
		#xest = self.Dyn2DEncObj.apply_RSS(kest)
		xest=x.abs()
		
		return xest, kest, csm
		
		
"""
	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-5)
		scheduler = torch.optim.lr_scheduler.OneCycleLR(
			optimizer, max_lr=3e-3, total_steps=self.trainer.estimated_stepping_batches, pct_start=0.05, anneal_strategy="cos", cycle_momentum=True, div_factor=30, final_div_factor=1e3, verbose=False
		)
		return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
"""
