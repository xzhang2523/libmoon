from torch.autograd import Variable
from torch.optim import SGD
from torch import Tensor
from libmoon.util_global.constant import get_agg_func, solution_eps, get_hv_ref
import torch
from tqdm import tqdm
from pymoo.indicators.hv import HV
import numpy as np

class GradBaseSolver:
    def __init__(self, step_size, n_iter, tol):
        self.step_size = step_size
        self.n_iter = n_iter
        self.tol = tol
    def solve(self, problem, x, prefs, args):
        '''
            :param problem:
            :param x:
            :param agg:
            :return:
                is a dict with keys: x, y
        '''
        # The abstract class cannot be implemented directly.
        raise NotImplementedError

class GradAggSolver(GradBaseSolver):
    def __init__(self, problem, step_size, n_iter, tol, agg):
        self.agg = agg
        self.problem = problem

        super().__init__(step_size, n_iter, tol)


    def solve(self, x, prefs):
        x = Variable(x, requires_grad=True)
        ind = HV(ref_point = get_hv_ref(self.problem.problem_name))

        hv_arr = []
        y_arr = []
        x_arr = []
        prefs = Tensor(prefs)
        optimizer = SGD([x], lr=self.step_size)
        agg_func = get_agg_func(self.agg)
        res = {}
        for i in tqdm(range(self.n_iter)):
            y = self.problem.evaluate(x)
            hv_arr.append(ind.do(y.detach().numpy()))
            agg_val = agg_func(y, prefs)
            optimizer.zero_grad()
            torch.sum(agg_val).backward()
            optimizer.step()
            y_arr.append(y.detach().numpy())
            if 'lbound' in dir(self.problem):
                x.data = torch.clamp(x.data, torch.Tensor(self.problem.lbound) + solution_eps, torch.Tensor(self.problem.ubound)-solution_eps)


        res['x'] = x.detach().numpy()
        res['y'] = y.detach().numpy()
        res['hv_history'] = np.array(hv_arr)
        res['y_history'] = np.array(y_arr)
        res['x_history'] = np.array(y_arr)
        return res