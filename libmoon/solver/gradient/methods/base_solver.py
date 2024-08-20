from torch.autograd import Variable
from torch.optim import SGD
from torch import Tensor
from libmoon.util_global.constant import get_agg_func, solution_eps, get_hv_ref
import torch
from tqdm import tqdm
from pymoo.indicators.hv import HV
import numpy as np

from libmoon.util_global.grad_util import get_moo_Jacobian_batch



class GradBaseSolver:
    def __init__(self, step_size, epoch, tol, core_solver):
        self.step_size = step_size
        self.epoch = epoch
        self.tol = tol
        self.core_solver = core_solver
        self.is_agg = (self.core_solver.core_name == 'AggCore')

    def solve(self, problem, x , prefs):
        '''
            :param problem:
            :param x:
            :param agg:
            :return:
                is a dict with keys: x, y.
        '''
        self.n_prob, self.n_obj = prefs.shape[0], prefs.shape[1]
        xs_var = Variable(x, requires_grad=True)
        optimizer = SGD([xs_var], lr=self.step_size)
        ind = HV(ref_point=get_hv_ref(problem.problem_name))
        hv_arr, y_arr = [], []
        for iter_idx in tqdm(range(self.epoch)):
            fs_var = problem.evaluate(xs_var)
            y_np = fs_var.detach().numpy()
            y_arr.append(y_np)
            hv_arr.append(ind.do(y_np))
            Jacobian_arr = get_moo_Jacobian_batch(xs_var, fs_var, self.n_obj)
            y_detach = fs_var.detach()
            optimizer.zero_grad()
            if self.is_agg:
                agg_name = self.core_solver.solver_name.split('_')[-1]
                agg_func = get_agg_func(agg_name)
                agg_val = agg_func(fs_var, torch.Tensor(prefs).to(fs_var.device))
                torch.sum(agg_val).backward()
            else:
                if self.core_solver.core_name in ['EPOCore', 'MGDAUBCore', 'RandomCore']:
                    weights = torch.stack([self.core_solver.get_alpha(Jacobian_arr[idx], y_detach[idx], idx) for idx in range( self.n_prob) ])
                # elif self.core_solver.core_name == 'AggCore':
                elif self.core_solver.core_name in ['PMTLCore', 'MOOSVGDCore', 'HVGradCore']:
                    # assert False, 'Unknown core_name'
                    if self.core_solver.core_name == 'HVGradCore':
                        weights = self.core_solver.get_alpha_array(y_detach)
                    else:
                        weights = self.core_solver.get_alpha_array(Jacobian_arr, y_detach)
                else:
                    assert False, 'Unknown core_name'

                torch.sum(weights * fs_var).backward()


            optimizer.step()
            if 'lbound' in dir(problem):
                x.data = torch.clamp(x.data, torch.Tensor(problem.lbound) + solution_eps,
                                     torch.Tensor(problem.ubound) - solution_eps)
        res = {}
        res['x'] = x.detach().numpy()
        res['y'] = y_np
        res['hv_arr'] = hv_arr
        res['y_arr'] = y_arr
        return res



class GradAggSolver(GradBaseSolver):
    def __init__(self, problem, step_size, epoch, tol, agg):
        self.agg = agg
        self.problem = problem
        super().__init__(step_size, epoch, tol)

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
        for i in tqdm(range(self.epoch)):
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