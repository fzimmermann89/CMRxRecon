import torch
import matplotlib.pyplot as plt
import numpy as np
import h5py
import mat73
import torch.nn.functional as F

from src.cmrxrecon.mr_models.dyn2dcart import Dyn2DCartEncObj
from src.cmrxrecon.models.cine.pdhg_plus_csms import PDHG4DynMRIwTVPlusCSMs
from src.cmrxrecon.models.cine.cg import conj_grad
from src.cmrxrecon.data.cine_ds import CineDataDS

data_type = 'loader'
Dyn2D =Dyn2DCartEncObj()

if data_type == 'loader':
        
    CS = CineDataDS('/data/cardiac/files/MultiCoil/Cine/ProcessedTrainingSet/', return_csm=True) 
    data = CS[0]
    
    csm = data['csm'].unsqueeze(0)
    y = data['k'].unsqueeze(0)
    mask = data['mask'].unsqueeze(0)
    xgt = data['gt'].unsqueeze(0)
    
elif data_type == 'challenge':
    
    R='04'
    p_id = 'P084' # 084 to be found in /sax/512_204_10
    slice_type = 'sax'
    pname = '/data/cardiac/files/MultiCoil/Cine/ProcessedTrainingSet/sax/512_204_10/'
    f = h5py.File(pname+'{}.h5'.format(p_id), 'r')
    
    #get data
    #1) CSM
    csm = np.array(f['csm'])
    csm = torch.tensor(csm)
    csm = torch.view_as_complex(csm).permute(1,0,2,3).unsqueeze(0)
            
    data_mat_acc = mat73.loadmat('/data/cardiac/files/MultiCoil/Cine/TrainingSet/AccFactor{}/{}/cine_{}.mat'.format(R, p_id, slice_type))
    data_mat_full = mat73.loadmat('/data/cardiac/files/MultiCoil/Cine/TrainingSet/FullSample/{}/cine_{}.mat'.format(p_id, slice_type))
    data_mask = mat73.loadmat('/data/cardiac/files/MultiCoil/Cine/TrainingSet/AccFactor{}/{}/cine_{}_mask.mat'.format(R, p_id, slice_type))

    #define tensor and bring to shape (mb, Nc, Nz, Nt, Nx, Ny))
    y_np = data_mat_full['kspace_full'] #has shape (Nx, Ny, Nc, Nz, Nt )
    mask_np = data_mask['mask{}'.format(R)] #has shape (Nx, Ny, Nc, Nz, Nt )
    
    #format data
    y = torch.tensor(y_np).unsqueeze(0).permute(0,3,4,5,2,1)
    
    #fftshift the data for the NN
    y =torch.fft.fftn(
    	torch.fft.ifftshift(
    		torch.fft.ifftn(
    			torch.fft.fftshift(y,dim=(-1,-2)),
    			dim=(-1,-2)), 
    		dim=(-1,-2)),
    	dim=(-1,-2)) 
    
    #get shapes
    Nb, Nc, Nz, Nt, Nx, Ny = y.shape

   #mask
    data_mask = mat73.loadmat('/data/cardiac/files/MultiCoil/Cine/TrainingSet/AccFactor{}/{}/cine_{}_mask.mat'.format(R, p_id, slice_type))
    mask_np = data_mask['mask{}'.format(R)] 
    
    # mask out everything but acs
    mask = torch.tensor(mask_np).to(torch.float).permute(1,0)
    mask = torch.stack(Nt*[mask],dim=0)
    mask = torch.stack(Nz*[mask],dim=0)
    mask = torch.stack(Nb*[mask],dim=0)
    mask = torch.fft.fftshift(mask,dim=(-2,1))
    
    xgt = Dyn2D.apply_RSS(y)
    
yu = mask.unsqueeze(1) * y

AHy = Dyn2D.apply_AH(yu, csm, mask)
xrss = Dyn2D.apply_RSS(yu)

AHA = lambda x: Dyn2D.apply_AHA(x, csm, mask)
xneq = conj_grad(AHA, AHy, AHy, niter=8)



T=4 #number of alternations
T_pdhg=64 #iters for pdhg
T_csm=2 #iters for the second sub-problem for the csms
pdhg = PDHG4DynMRIwTVPlusCSMs(Dyn2D, 
    T=T,
    T_pdhg=T_pdhg,
    T_csm=T_csm,
    lambda_reg_xy=4e-7, 
    lambda_reg_t=5e-6,
    lambda_reg_c=1e-4).cuda()

with torch.no_grad():
    xpdhg, csm_est = pdhg(yu.cuda(), mask.cuda(), csm.cuda()) 
    xpdhg = xpdhg.cpu()
    csm_est = csm_est.cpu()
    
kz = 0
kt = 5

#arrs_list = [AHy, xneq, xrss, xpdhg, xgt]
arrs_list = [AHy.abs(), xneq.abs(), xrss.abs(), xpdhg.abs(), xgt.abs()]
errs_list = [arr - xgt for arr in arrs_list]
title_list = ['AHy', 'xneq', 'xrss', 'xpdhg', 'GT']

nfigs = len(arrs_list)
figsize = 3
fig,ax = plt.subplots(2, nfigs, figsize = (nfigs*figsize, 2 * figsize ))
for k in range(nfigs):
    print(k)
    ax[0,k].imshow(arrs_list[k][0,kz,kt,...].abs(), cmap = plt.cm.Greys_r, clim = [0,5e-4])
    mse = F.mse_loss(xgt.abs(), arrs_list[k].abs())
    ax[0,k].set_title('{}: MSE={}'.format(title_list[k],round(mse.item(),12)),fontsize=10)
    ax[1,k].imshow( 3 * errs_list[k][0,kz,kt,...].abs(), cmap = plt.cm.viridis, clim = [0,5e-4])
    
    ax[0,k].set_xticks([])
    ax[0,k].set_yticks([])
    ax[1,k].set_xticks([])
    ax[1,k].set_yticks([])
    
RR = round(torch.sum(mask).item()/mask.shape[3],2)
ax[1,-1].set_title('acc R={}'.format(RR))
    
fig.subplots_adjust(wspace=0.01, hspace=-0.6)





