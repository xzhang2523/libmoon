# -*- coding: utf-8 -*-
"""
[1] Liang Zhao and Qingfu Zhang. Hypervolume-Guided Decomposition for Parallel 
Expensive Multiobjective Optimization. IEEE Transactions on Evolutionary 
Computation, 28(2): 432-444, 2024.
"""
import torch
from libmoon.solver.mobo.methods.base_solver_mobod import MOBOD
from botorch.utils.probability.utils import (
    log_ndtr as log_Phi,
    log_phi, # Logarithm of standard normal pdf
    log_prob_normal_in,
    ndtr as Phi, # Standard normal CDF
    phi, # Standard normal PDF
)

class DirHVEGOSolver(MOBOD):
    def __init__(self, problem, x_init, MAX_FE, BATCH_SIZE):
        super().__init__(problem, x_init, MAX_FE, BATCH_SIZE)
        self.solver_name = 'dirhvego'

    def _get_acquisition(self, u, sigma, ref_vec, pref_inc):
        '''
        Parameters:
            ref_vec: direction vector
            pref_inc :  preference-conditional incumbent

        Returns
            preference-conditional EI: DirHV-EI(X|pref_vec)

        '''
        xi_minus_u = pref_inc - u  # N*M
        tau = xi_minus_u / sigma  # N*M
        alpha_i = xi_minus_u * Phi(tau) + sigma * phi(tau)  # N*M
        return torch.prod(alpha_i, dim=1)

    def _step(self, batch_size):
        # Calculate the Intersection points and Direction vectors
        xis, dir_vecs = self._get_xis(self.ref_vecs)
        # Use MOEA/D to maximize DirHV-EI
        candidate_x, candidate_mean, candidata_std = self._moead_GR(dir_vecs, xis)
        
        # Find q solutions with the greedy algorithm
        # Compute EI_D for all the points in Q
        pop_size = self.ref_vecs.shape[0]
        EIDs = torch.zeros(pop_size,pop_size)
        for i in range(pop_size):
            temp_mean = candidate_mean[i:i+1].repeat(pop_size,1)
            temp_std = candidata_std[i:i+1].repeat(pop_size,1)
            EIDs[i, :] = self._get_acquisition(temp_mean, temp_std, dir_vecs, xis)
 
        Qb = []
        temp = EIDs.clone()
        beta = torch.zeros(pop_size)
        for i in range(batch_size):
            index = torch.argmax(torch.sum(temp, dim=1))
            Qb.append(index.item())
            beta = beta + temp[index, :]
            # Update temp: [EI_D(x|\lambda) - beta]_+
            temp = EIDs - beta[None, :].repeat(pop_size, 1)
            temp[temp < 0] = 0
            
        # evaluate the selected n_sample solutions
        X_new = candidate_x[Qb]
        return X_new 
    
    def _get_xis(self, ref_vecs):
        # ref_vecs is generated via simplex-lattice design
        temp = 1.1 * ref_vecs - self.z
        dir_vecs = temp / torch.norm(temp, dim=1, keepdim=True)
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
   



 