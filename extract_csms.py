import torch
import numpy as np
import sigpy.mri as sp_mri
import os

import h5py

for view in ['lax', 'sax']:
	
	pname = '/data/cardiac/files/MultiCoil/Cine/ProcessedTrainingSet/{}/'.format(view)
	
	shapes_folders = os.listdir(pname)
	
	for folder in shapes_folders:
		
		shape_folder = pname + '/{}/'.format(folder)
		files_list = os.listdir(shape_folder)
		
		for file in files_list:
			
			print(folder, file)
			f = h5py.File(shape_folder+file, 'a')
			
			#get data 
			#NB. shape (2, 168, 12, 10, 448, 2)
			# with (Nz, undresampled, Nt, Nc, fully-sampled, real/imag)
			y_np = np.array(f['k'])
			y = torch.view_as_complex(torch.tensor(y_np)).mean(2)
			
			#fftshift the data for the CSM estimation
			y =torch.fft.fftn(
				torch.fft.ifftshift(
					torch.fft.ifftn(
						torch.fft.fftshift(y,dim=(-1,-3)),
						dim=(-1,-3)), 
					dim=(-1,-3)),
				dim=(-1,-3)) 
			
			y = y.permute(0,2,1,3)
			
			Nz, Nc, Ny, Nx = y.shape
			
			C_list = []
			for kz in range(Nz):
				
				threshold = 0.00025
				max_iter = 250
				
				espirit = sp_mri.app.EspiritCalib(np.array(y[kz,...]), max_iter=max_iter, thresh = threshold)
				C = espirit.run()
				C_list.append(C)
			
			csm0 = np.stack(C_list,axis=0)
			
			csm0 = np.stack( [np.real(csm0), np.imag(csm0)],axis=-1)
			
			if 'csm' in list(f.keys()):
				data = f['csm']
				data = csm0
			else:
				f.create_dataset('csm', data= csm0, dtype=np.float32)
			
			f.close()
			
				