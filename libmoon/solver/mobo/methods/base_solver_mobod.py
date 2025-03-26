'''
    Main algorithm framework for  Decomposition-based Multi-objective Bayesian Optimization.

    [1] Liang Zhao and Qingfu Zhang. Hypervolume-Guided Decomposition for Parallel
    Expensive Multiobjective Optimization. IEEE Transactions on Evolutionary
    Computation, 28(2): 432-444, 2024.
'''

import numpy as np
import torch 
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from libmoon.solver.mobo.utils.lhs import lhs
from libmoon.solver.mobo.surrogate_models import GaussianProcess
import math
from tqdm import tqdm
from libmoon.metrics import compute_hv



class MOBOD(object):
    def __init__(self, problem, x_init, MAX_FE, BATCH_SIZE):
        self.n_var = problem.n_var
        self.n_obj = problem.n_obj
        self.x_init = x_init
        self.n_init = x_init.shape[0]
        self.MAX_FE = MAX_FE
        self.BATCH_SIZE = BATCH_SIZE
        self.max_iter = math.ceil((MAX_FE - self.n_init)/BATCH_SIZE)
        self.problem = problem
        self.bounds = torch.from_numpy(np.vstack((problem.lbound,problem.ubound)))
         
        self.ref_point = None
        self.x, self.y = None, None
        
    def _get_acquisition(self, u, sigma, ref_vec, pref_inc):
       # calculate the preference conditional acquisition functions
       pass
       
    def _step(self,batch_size):
        # optimize the preference conditional acquisition functions
        # subset selection
        pass     
     
    def solve(self):
        # get initial samples
        # x_init = torch.from_numpy(lhs(self.n_var,samples=self.n_init))
        x_init = self.bounds[0,...] + (self.bounds[1,...] - self.bounds[0,...]) *  self.x_init
        y_init = self.problem.evaluate(x_init) 
        self._record(x_init, y_init)
        # generate reference vectors
        params_H =  [199,19] # parameter for weight vectors, for M = 2,3,4,5,6 respectively.
        if self.n_obj == 2 or self.n_obj == 3:
            # simplex-lattice design
            self.ref_vecs = torch.from_numpy(get_reference_directions("uniform", self.n_obj, n_partitions=params_H[self.n_obj-2]))
        else:
            # TODO
            pass

        hv_dict = {}
        hv_dict[self.n_init] = compute_hv(self.y.detach().cpu().numpy(), self.problem.problem_name)
        for i in tqdm(range(self.max_iter)):
            # Scale the objective values 
            train_x = self.x.clone()
            min_vals, _ = torch.min(self.y, dim=0)
            max_vals, _ = torch.max(self.y, dim=0)
            train_y = torch.div(torch.sub(self.y, min_vals), torch.sub(max_vals, min_vals)) 
            self.z = -0.01*torch.ones((1,self.n_obj))
            self.train_y_nds = train_y[self.idx_nds[0]].clone()
            
            # Bulid GP model for each objective function
            # TODO, train GPs using Gpytorch
            self.gps =  GaussianProcess(self.n_var, self.n_obj) 
            self.gps.fit(train_x.detach().cpu().numpy(), train_y.detach().cpu().numpy())
            
            # greedy batch selection
            batch_size = min(self.MAX_FE - self.x.size(0),self.BATCH_SIZE)  
            new_x = self._step(batch_size)   
            
            # observe new values
            new_obj = self.problem.evaluate(new_x)
            self._record(new_x, new_obj)
            hv_val = compute_hv(self.y.detach().cpu().numpy(), self.problem.problem_name)
            print('Iteration: %d, HV: %.4f'%(i,hv_val))
            hv_dict[(i+1)*batch_size + self.n_init] = hv_val

        res = {}
        res['x'] = self.x.detach().numpy()
        res['y'] = self.y.detach().numpy()
        res['idx_nds'] = self.idx_nds
        res['hv'] = hv_dict
        return res
            
      

    def _record(self, new_x, new_obj):
        # after add new observations
        if self.x is None:
            self.x = new_x.clone()
            self.y = new_obj.clone()
        else:
            self.x = torch.cat((self.x, new_x),dim=0)
            self.y = torch.cat((self.y, new_obj),dim=0)
        
        # nondominated sorting
        nds = NonDominatedSorting()
        self.idx_nds = nds.do(self.y.detach().cpu().numpy())
      
  

    def _moead_GR(self, ref_vecs, pref_incs):
        # using MOEA/D-GR to solve subproblems
        maxIter = 50
        pop_size = self.ref_vecs.shape[0] # pop_size
        T = int(np.ceil(0.1 * pop_size).item())  # size of neighbourhood: 0.1*N
        B = torch.argsort(torch.cdist(ref_vecs, ref_vecs), dim=1)[:, :T]
 

        # the initial population for MOEA/D
        x_ini = torch.from_numpy(lhs(self.n_var,samples=pop_size))
        pop_x = self.bounds[0,...] + (self.bounds[1,...] - self.bounds[0,...]) *  x_ini
        # gp poterior
        # TODO, train GPs using Gpytorch
        out = self.gps.evaluate(pop_x.detach().cpu().numpy(), cal_std=True, cal_grad=False)
        pop_mean, pop_std = torch.from_numpy(out['F']), torch.from_numpy(out['S'])
        # calculate the values of preference conditional acquisition functions
        pop_acq = self._get_acquisition(pop_mean, pop_std, ref_vecs, pref_incs)

        # optimization
        for gen in range(maxIter - 1):
            for i in range(pop_size):
                if torch.rand(1) < 0.8:  # delta
                    P = B[i, np.random.permutation(B.shape[1])]
                else:
                    P = np.random.permutation(pop_size)
                # generate an offspring 1*d
                off_x = self._operator_DE(pop_x[i:i+1, :], pop_x[P[0:1], :], pop_x[P[1:2], :])
                # TODO, train GPs using Gpytorch
                out = self.gps.evaluate(off_x.detach().cpu().numpy(), cal_std=True, cal_grad=False)
                off_mean, off_std = torch.from_numpy(out['F']), torch.from_numpy(out['S'])
                 
                # Global Replacement  MOEA/D-GR
                # Find the most approprite subproblem and its neighbourhood
                acq_all = self._get_acquisition(off_mean.repeat(pop_size,1), off_std.repeat(pop_size,1),ref_vecs, pref_incs)
                best_index = np.argmax(acq_all)
                P = B[best_index, :]  # replacement neighborhood

                offindex = P[pop_acq[P] < acq_all[P]]
                if len(offindex) > 0:
                    pop_x[offindex, :] = off_x.repeat(len(offindex), 1)
                    pop_mean[offindex, :] = off_mean.repeat(len(offindex), 1)
                    pop_std[offindex, :] = off_std.repeat(len(offindex), 1)
                    pop_acq[offindex] = acq_all[offindex]

        return pop_x, pop_mean, pop_std

    def _operator_DE(self, Parent1, Parent2, Parent3):
        '''
            generate one offspring by P1 + 0.5*(P2-P3) and polynomial mutation.
        '''
        # Parameter
        CR = 1
        F = 0.5
        proM = 1
        disM = 20
        #
        N, D = Parent1.shape
        # Differental evolution
        Site = torch.rand(N, D) < CR
        Offspring = Parent1.clone()
        Offspring[Site] = Offspring[Site] + F * (Parent2[Site] - Parent3[Site])
        # Polynomial mutation
        Lower = self.bounds[0:1,...]
        Upper = self.bounds[1:2,...]

        U_L = Upper - Lower
        Site = torch.rand(N, D) < proM / D
        mu = torch.rand(N, D)
        temp = torch.logical_and(Site, mu <= 0.5)
        Offspring = torch.min(torch.max(Offspring, Lower), Upper)
     
        delta1 = (Offspring - Lower) / U_L
        delta2 = (Upper - Offspring) / U_L
        #  mu <= 0.5
        val = 2. * mu + (1 - 2. * mu) * ((1. - delta1).pow(disM + 1))
        Offspring[temp] = Offspring[temp] + ((val[temp]).pow(1.0 / (disM + 1)) - 1.) * U_L[temp]
        # mu > 0.5
        temp = torch.logical_and(Site, mu > 0.5)
        val = 2. * (1.0 - mu) + 2. * (mu - 0.5) * ((1. - delta2).pow(disM + 1))
        Offspring[temp] = Offspring[temp] + (1.0 - (val[temp]).pow(1.0 / (disM + 1))) * U_L[temp]
    
        return Offspring
 