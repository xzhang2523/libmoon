from libmoon.solver.gradient.methods.base_solver import GradBaseSolver, AggCore
from torch.autograd import Variable
from torch.optim import SGD
from torch import Tensor
from libmoon.util.constant import get_agg_func, solution_eps, get_hv_ref
import torch
from tqdm import tqdm
from pymoo.indicators.hv import HV
from matplotlib import pyplot as plt
import numpy as np
from libmoon.model.simple import PFLModel
import os
from torch import nn
from libmoon.util.prefs import pref2angle
criterion = nn.MSELoss()

class UMODSolver(GradBaseSolver):
    '''
        UniformSolver is a bi-level agg solver.
    '''
    def __init__(self, problem, prefs, step_size, n_epoch, tol, folder_name):
        self.problem = problem
        self.prefs = prefs
        self.solver_name = 'UMOD'
        self.core_solver = AggCore(prefs, agg_name='mTche')
        self.pref_adjust_epoch = 2000
        self.pfl_train_epoch = 2000
        self.folder_name = folder_name
        super().__init__(step_size, n_epoch, tol, self.core_solver)

    def solve(self, x_init):
        return super().solve(self.problem, x_init, self.prefs)