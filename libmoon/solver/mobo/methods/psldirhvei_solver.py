"""
PSL + DirHV-EI

[1] Xi Lin, Zhiyuan Yang, Xiaoyuan Zhang, Qingfu Zhang. Pareto Set Learning for
Expensive Multiobjective Optimization. Advances in Neural Information Processing
Systems (NeurIPS) , 2022
[2] Liang Zhao and Qingfu Zhang. Hypervolume-Guided Decomposition for Parallel 
Expensive Multiobjective Optimization. IEEE Transactions on Evolutionary 
Computation, 28(2): 432-444, 2024.
"""

import torch
import torch.nn as nn
from torch import Tensor 
from botorch.utils.sampling import sample_simplex
from botorch.utils.multi_objective.hypervolume import Hypervolume
import math
from .base_psl_model import ParetoSetModel
from libmoon.solver.mobo.utils import lhs
from botorch.utils.transforms import unnormalize, normalize
from tqdm import tqdm
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from libmoon.solver.mobo.surrogate_models import GaussianProcess
torch.set_default_dtype(torch.float64)
from libmoon.solver.mobo.methods.base_solver_pslmobo import PSLMOBO
from botorch.utils.probability.utils import (
    log_ndtr as log_Phi,
    log_phi, # Logarithm of standard normal pdf
    log_prob_normal_in,
    ndtr as Phi, # Standard normal CDF
    phi, # Standard normal PDF
)
class PSLDirHVEISolver(PSLMOBO):
    def __init__(self, problem, x_init, MAX_FE, BATCH_SIZE):
        super().__init__(problem, x_init, MAX_FE, BATCH_SIZE)
        self.solver_name = 'psldirhvei'


    def _get_xis(self, ref_vecs):
        # ref_vecs is generated via simplex-lattice design
        # temp = 1.1 * ref_vecs - self.z
        dir_vecs = ref_vecs / torch.norm(ref_vecs, dim=1, keepdim=True)
        # Eq. 11, compute the intersection points
        div_dir = 1.0 / dir_vecs
        A = self.train_y_nds - self.z  # L*M
        G = torch.ger(div_dir[:, 0], A[:, 0])  # N*L, f1
        for j in range(1, self.n_obj):
            G = torch.max(G, torch.ger(div_dir[:, j], A[:, j]))  # N*L, max(fi,fj)
        
        # minimum of mTch for each direction vector
        Lmin = torch.min(G, dim=1, keepdim=True).values.data  # N*1  one for each direction vector
        # N*M  Intersection points
        xis = self.z + torch.mul(Lmin, dir_vecs)
        return xis, dir_vecs
	
    def _train_psl(self):
        self.psmodel = ParetoSetModel(self.n_var, self.n_obj)
        # optimizer
        optimizer = torch.optim.Adam(self.psmodel.parameters(), lr=1e-3)
        # t_step Pareto Set Learning with Gaussian Process
        for t_step in range(self.n_steps):
            self.psmodel.train()
            
            # sample n_pref_update preferences L1=1
            pref_vec = sample_simplex(d=self.n_obj, n=self.n_pref_update-self.n_obj).to(torch.float64)
            pref_vec = torch.cat((pref_vec, torch.eye(self.n_obj)),dim=0)
            pref_vec = torch.clamp(pref_vec, min=1.e-6) 
            
            xis, dir_vecs = self._get_xis(pref_vec) 
            # get the current coressponding solutions
            x = self.psmodel(pref_vec)
            # TODO, train GPs using Gpytorch
            out = self.gps.evaluate(x.detach().cpu().numpy(), cal_std=True, cal_grad=True) 
            mean, std, mean_grad, std_grad = torch.from_numpy(out['F']), torch.from_numpy(out['S']), torch.from_numpy(out['dF']), torch.from_numpy(out['dS'])
            
            xi_minus_u = xis - mean  # N*M
            tau = xi_minus_u / std  # N*M
            alpha_i = xi_minus_u * Phi(tau) + std * phi(tau)  # N*M
   
            alpha_mean_grad = (-Phi(tau)*alpha_i).unsqueeze(2).repeat(1,1,self.n_var) * mean_grad
            alpha_std_grad = (phi(tau)*alpha_i).unsqueeze(2).repeat(1,1,self.n_var) * std_grad
            dirhvei_grad = -torch.sum( alpha_mean_grad + alpha_std_grad, dim=1)
             
            # gradient-based pareto set model update 
            optimizer.zero_grad()
            self.psmodel(pref_vec).backward(dirhvei_grad)
            optimizer.step()  
            
    def _batch_selection(self, batch_size):
        # sample n_candidate preferences default:1000
        self.psmodel.eval()  # Sets the module in evaluation mode.
        pref = sample_simplex(d=self.n_obj, n=self.n_candidate).to(torch.float64)
        pref = torch.clamp(pref, min=1.e-6) 
        
        # generate correponding solutions, get the predicted mean/std
        with torch.no_grad():
            candidate_x = self.psmodel(pref).to(torch.float64) 
            # TODO, train GPs using Gpytorch
            out = self.gps.evaluate(candidate_x.detach().cpu().numpy(), cal_std=True, cal_grad=False)
            candidate_mean, candidata_std = torch.from_numpy(out['F']), torch.from_numpy(out['S'])
        xis, dir_vecs = self._get_xis(pref)  
        EIDs = torch.zeros(self.n_candidate,self.n_candidate)
        for i in range(self.n_candidate):
            temp_mean = candidate_mean[i:i+1].repeat(self.n_candidate,1)
            temp_std = candidata_std[i:i+1].repeat(self.n_candidate,1)
            xi_minus_u = xis - temp_mean  # N*M
            tau = xi_minus_u / temp_std  # N*M
            alpha_i = xi_minus_u * Phi(tau) + temp_std * phi(tau)  # N*M
            EIDs[i,:] = torch.prod(alpha_i, dim=1)  
        Qb = []
        temp = EIDs.clone()
        beta = torch.zeros(self.n_candidate)  
        for i in range(batch_size):
            index = torch.argmax(torch.sum(temp, dim=1))
            Qb.append(index.item())
            beta = beta + temp[index, :]
            # Update temp: [EI_D(x|\lambda) - beta]_+
            temp = EIDs - beta[None, :].repeat(self.n_candidate, 1)
            temp[temp < 0] = 0
      
        # evaluate the selected n_sample solutions
        X_new = candidate_x[Qb]
        return X_new  
    
     
    
   
    
 