import numpy as np
import torch 

 
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from libmoon.solver.mobo.utils.lhs import lhs
from libmoon.solver.mobo.surrogate_models import GaussianProcess
import math
from tqdm import tqdm
from botorch.utils.transforms import unnormalize, normalize
from libmoon.metrics import compute_hv
'''
    Main algorithm framework for  Decomposition-based Multi-objective Bayesian Optimization.
    
    [1] Liang Zhao and Qingfu Zhang. Hypervolume-Guided Decomposition for Parallel 
    Expensive Multiobjective Optimization. IEEE Transactions on Evolutionary 
    Computation, 28(2): 432-444, 2024.
'''
class PSLMOBO(object):
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
        self.x = None
        
        self.n_steps = 1000 # number of learning steps
        self.n_pref_update = 10 # number of sampled preferences per step
        self.n_candidate = 1000  # number of sampled candidates on the approxiamte PF
    
    def _train_psl(self):
        pass
    
    def _batch_selection(self, batch_size):
        pass
    
    def solve(self):
        # get initial samples
        x_init = self.bounds[0,...] + (self.bounds[1,...] - self.bounds[0,...]) *  self.x_init
        y_init = self.problem.evaluate(x_init) 
        self._record(x_init, y_init)

        hv_dict = {}
        hv_dict[self.n_init] = compute_hv(self.y.detach().cpu().numpy(), self.problem.problem_name)
        print('Iteration: %d, HV: %.4f' % (0, compute_hv(self.y.detach().cpu().numpy(), self.problem.problem_name)))

        for i in tqdm(range(self.max_iter)):
            # solution normalization x: [0,1]^d, y: [0,1]^m
            train_x = normalize(self.x, self.bounds) 
            min_vals, _ = torch.min(self.y, dim=0)
            max_vals, _ = torch.max(self.y, dim=0)
            train_y = torch.div(torch.sub(self.y, min_vals), torch.sub(max_vals, min_vals))   
            self.z =  -0.1*torch.ones((1,self.n_obj))  
            self.train_y_nds = train_y[self.idx_nds[0]].clone()
            
            # train GP surrogate model  
            # TODO, train GPs using Gpytorch
            self.gps =  GaussianProcess(self.n_var, self.n_obj) 
            self.gps.fit(train_x.detach().cpu().numpy(), train_y.detach().cpu().numpy()) 
            
            # intitialize the model and optimizer 
            self._train_psl()   
            
            # greedy batch selection
            batch_size = min(self.MAX_FE - self.x.size(0),self.BATCH_SIZE)
            X_new = self._batch_selection(batch_size)
            # observe new values
            new_x = unnormalize(X_new, bounds=self.bounds)
            new_obj = self.problem.evaluate(new_x)
            self._record(new_x, new_obj)
            # self.y: HV
            # print()
            hv_val = compute_hv(self.y.detach().cpu().numpy(), self.problem.problem_name)
            print('Iteration: %d, HV: %.4f' % ((i+1), hv_val))
            hv_dict[(i+1) * batch_size + self.n_init] = hv_val
            
        res = {}
        res['x'] = self.x.detach().numpy()
        res['y'] = self.y.detach().numpy()
        res['idx_nds'] = self.idx_nds
        # res['idx_nds'] = self.idx_nds
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