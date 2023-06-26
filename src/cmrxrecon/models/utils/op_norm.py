import torch
from .cg import inner_product
		
def power_iteration(A, q0, niter = 128):
		
		"""
		power iteration to estimate the operator norm of a quadratic opeartor A;
		
		"""
		
		#no need to track gradients
		with torch.no_grad():
			print('estimating op-norm')
				
			qk = q0
			for k in range(niter):
				
				# apply the operator
				zk = A(qk)
			
				# calculate the norm
				zk_norm = torch.sqrt(inner_product(zk,zk))
				
				# re normalize the vector
				qk = zk / zk_norm
				
			Aqk = A(qk)
			op_norm = torch.sqrt(inner_product(Aqk,Aqk).cpu())
			
			print('op-norm = {}'.format(op_norm))
		return op_norm