
import numpy as np
import torch
import argparse
from matplotlib import pyplot as plt
from problem.mop import mop

from numpy import array


class RE21(mop):
    def __init__(self, n_var=4, n_obj=2, lbound=np.zeros(4), ubound=np.ones(4)):
        self.problem_name = 'RE21'
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_cons = 0

        self.n_original_constraints = 0
        self.ideal = array([1237.8414230005742, 0.002761423749158419])
        self.nadir = array([2086.36956042, 0.00341421356237])

        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma
        self.ubound = np.full(self.n_var, 3 * tmp_val)
        self.lbound = np.zeros(self.n_var)
        self.lbound[0] = tmp_val
        self.lbound[1] = np.sqrt(2.0) * tmp_val
        self.lbound[2] = np.sqrt(2.0) * tmp_val
        self.lbound[3] = tmp_val


    def _evaluate_numpy(self, x):

        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]
        f = np.zeros( (len(x), self.n_obj) )

        F = 10.0
        sigma = 10.0
        E = 2.0 * 1e5
        L = 200.0

        f[:,0] = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
        f[:,1] = ((F * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))

        # f_arr = np.stack((f1,f2), axis=1)
        f_arr_norm = (f - self.ideal) / (self.nadir - self.ideal)
        return f_arr_norm


    def _evaluate_torch(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        F = 10.0
        sigma = 10.0
        E = 2.0 * 1e5
        L = 200.0

        f1 = L * ( (2 * x1) + np.sqrt(2.0) * x2 + torch.sqrt(x3) + x4 )
        f2 = ((F * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))
        f_arr = torch.stack((f1, f2), dim=1)
        f_arr_norm = (f_arr - self.ideal) / (self.nadir - self.ideal)
        return f_arr_norm

class RE22(mop):
    def __init__(self, n_var=3, n_obj=2, lbound=np.zeros(30),
                 ubound=np.ones(30)):


        self.n_var=n_var
        self.n_obj=n_obj
        self.problem_name = 'RE22'
        self.n_cons = 0
        self.n_original_constraints = 2

        self.ideal = np.array([5.88, 0.0])
        self.nadir = np.array([361.262944647, 180.01547])


        self.ubound = np.zeros(self.n_var)
        self.lbound = np.zeros(self.n_var)

        self.lbound[0] = 0.2
        self.lbound[1] = 0.0
        self.lbound[2] = 0.0
        self.ubound[0] = 15
        self.ubound[1] = 20
        self.ubound[2] = 40

        self.n_var = n_var
        self.n_obj = n_obj

        self.feasible_vals = np.array(
            [0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32, 1.40, 1.55, 1.58, 1.60,
             1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80, 3.0, 3.08, 3, 10, 3.16, 3.41,
             3.52, 3.60, 3.72, 3.95, 3.96, 4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53,
             5.72, 6.0, 6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0, 11.06, 11.85,
             12.0, 13.0, 14.0, 15.0])


    def _evaluate_numpy(self, x):
        n_sub = len(x)

        f = np.zeros( (n_sub, self.n_obj) )
        g = np.zeros( (n_sub, self.n_original_constraints)  )
        # Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
        idx_arr = [np.abs(np.asarray(self.feasible_vals) - x0).argmin() for x0 in x[:,0]]
        x1 = array([self.feasible_vals[idx] for idx in idx_arr])
        x2 = x[:,1]
        x3 = x[:,2]

        # First original objective function
        f[:,0] = (29.4 * x1) + (0.6 * x2 * x3)
        # Original constraint functions
        g[:,0] = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
        g[:,1] = 4.0 - (x3 / x2)
        g = np.where(g < 0, -g, 0)
        f[:,1] = g[:,0] + g[:,1]
        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        return f_norm

    def _evaluate_torch(self, x):
        pass


class RE23(mop):
    def __init__(self, n_var=4, n_obj=2, lbound=np.zeros(2),
                 ubound=np.ones(2)):
        self.problem_name = 'RE23'
        self.n_obj = n_obj
        self.n_var = n_var
        self.n_cons = 0
        self.n_original_constraints = 3

        self.ideal = array([15.9018007813, 0.0])
        self.nadir = array([481.608088535, 44.2819047619])

        self.ubound = np.zeros(self.n_var)
        self.lbound = np.zeros(self.n_var)
        self.lbound[0] = 1
        self.lbound[1] = 1
        self.lbound[2] = 10
        self.lbound[3] = 10
        self.ubound[0] = 100
        self.ubound[1] = 100
        self.ubound[2] = 200
        self.ubound[3] = 240

    def _evaluate_numpy(self, x):
        f = np.zeros(self.n_obj )
        g = np.zeros(self.n_original_constraints)

        x1 = 0.0625 * int(np.round(x[:,0]))
        x2 = 0.0625 * int(np.round(x[:,1]))
        x3 = x[:,2]
        x4 = x[:,3]

        # First original objective function
        f[:,0] = (0.6224 * x1 * x3 * x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)

        # Original constraint functions
        g[:,0] = x1 - (0.0193 * x3)
        g[:,1] = x2 - (0.00954 * x3)
        g[:,2] = (np.pi * x3 * x3 * x4) + ((4.0 / 3.0) * (np.pi * x3 * x3 * x3)) - 1296000
        g = np.where(g < 0, -g, 0)
        f[:,1] = g[:,0] + g[:,1] + g[:,2]

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)

        return f_norm





class RE24(mop):
    def __init__(self, n_var=2, n_obj=2, lbound=np.zeros(2),
                 ubound=np.ones(2)):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )

        self.problem_name = 'RE24'
        self.n_obj = 2
        self.n_var = 2

        self.n_cons = 0
        self.n_original_constraints = 4

        self.ubound = np.zeros(self.n_var)
        self.lbound = np.zeros(self.n_var)

        self.lbound[0] = 0.5
        self.lbound[1] = 0.5

        self.ubound[0] = 4
        self.ubound[1] = 50

        self.ideal = np.array([60.5, 0.0])
        self.nadir = np.array([481.608088535, 44.2819047619])




    def _evaluate_numpy(self, x):
        n_sub = len(x)
        # f = np.zeros(self.n_objectives)
        g = np.zeros( (n_sub, self.n_original_constraints) )

        x1 = x[:,0]
        x2 = x[:,1]

        # First original objective function
        f1 = x1 + (120 * x2)

        E = 700000
        sigma_b_max = 700
        tau_max = 450
        delta_max = 1.5
        sigma_k = (E * x1 * x1) / 100
        sigma_b = 4500 / (x1 * x2)
        tau = 1800 / x2
        delta = (56.2 * 10000) / (E * x1 * x2 * x2)

        g[:,0] = 1 - (sigma_b / sigma_b_max)
        g[:,1] = 1 - (tau / tau_max)
        g[:,2] = 1 - (delta / delta_max)
        g[:,3] = 1 - (sigma_b / sigma_k)
        g = np.where(g < 0, -g, 0)
        f2 = g[:,0] + g[:,1] + g[:,2] + g[:,3]

        f_arr = np.stack((f1, f2), axis=1)
        f_norm = (f_arr - self.ideal) / (self.nadir - self.ideal)

        return f_norm


    def _evaluate_torch(self, x):
        pass



