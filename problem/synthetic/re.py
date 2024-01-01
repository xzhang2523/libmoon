
import numpy as np
import torch
import argparse
from matplotlib import pyplot as plt
from problem.mop import mop



class RE21(mop):
    def __init__(self, n_var=4, n_obj=2, lower_bound=np.zeros(4), upper_bound=np.ones(4)):
        self.problem_name = 'RE21'
        self.n_constraints = 0
        self.n_original_constraints = 0

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lower_bound=lower_bound,
                         upper_bound=upper_bound, )

        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma
        self.ub = np.full(self.n_var, 3 * tmp_val)
        self.lb = np.zeros(self.n_var)
        self.lb[0] = tmp_val
        self.lb[1] = np.sqrt(2.0) * tmp_val
        self.lb[2] = np.sqrt(2.0) * tmp_val
        self.lb[3] = tmp_val

    def evaluate(self, x):
        # f = np.zeros(self.n_obj)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        F = 10.0
        sigma = 10.0
        E = 2.0 * 1e5
        L = 200.0

        f1 = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
        f2 = ((F * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))

        return np.stack( (f1,f2), axis=1)