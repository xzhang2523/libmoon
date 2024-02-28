"""
The class HvMaximization is based on the algorithm described by
Wang, Hao, et al.
"Hypervolume indicator gradient ascent multimnist-objective optimization."
International conference on evolutionary multimnist-criterion optimization. Springer, Cham, 2017.
"""

import numpy as np
import torch

from .functions_evaluation import fastNonDominatedSort
from .functions_hv_grad_3d import grad_multi_sweep_with_duplicate_handling
from .base_solver import GradBaseSolver
from torch.autograd import Variable


from tqdm import tqdm
from pymoo.indicators.hv import HV
from ...util_global.constant import solution_eps, get_hv_ref_dict

"""
The class HvMaximization is based on the algorithm described by
Wang, Hao, et al.
"Hypervolume indicator gradient ascent multi-objective optimization."
International conference on evolutionary multi-criterion optimization. Springer, Cham, 2017.
"""

import numpy as np
import torch

from .functions_evaluation import fastNonDominatedSort
from .functions_hv_grad_3d import grad_multi_sweep_with_duplicate_handling


class HvMaximization(object):
    """
    Mo optimizer for calculating dynamic weights using higamo style hv maximization
    based on Hao Wang et al.'s HIGA-MO
    uses non-dominated sorting to create multiple fronts, and maximize hypervolume of each
    """

    def __init__(self, n_mo_sol, n_mo_obj, ref_point, obj_space_normalize=True):
        super(HvMaximization, self).__init__()
        self.name = 'hv_maximization'
        self.ref_point = np.array(ref_point)
        self.n_mo_sol = n_mo_sol
        self.n_mo_obj = n_mo_obj
        self.obj_space_normalize = obj_space_normalize

    def compute_weights(self, mo_obj_val):
        n_mo_obj = self.n_mo_obj
        n_mo_sol = self.n_mo_sol

        # non-dom sorting to create multiple fronts
        hv_subfront_indices = fastNonDominatedSort(mo_obj_val)
        dyn_ref_point = 1.1 * np.max(mo_obj_val, axis=1)
        for i_obj in range(0, n_mo_obj):
            dyn_ref_point[i_obj] = np.maximum(self.ref_point[i_obj], dyn_ref_point[i_obj])
        number_of_fronts = np.max(hv_subfront_indices) + 1  # +1 because of 0 indexing

        obj_space_multifront_hv_gradient = np.zeros((n_mo_obj, n_mo_sol))
        for i_fronts in range(0, number_of_fronts):
            # compute HV gradients for current front
            temp_grad_array = grad_multi_sweep_with_duplicate_handling(mo_obj_val[:, (hv_subfront_indices == i_fronts)],
                                                                       dyn_ref_point)
            obj_space_multifront_hv_gradient[:, (hv_subfront_indices == i_fronts)] = temp_grad_array

        # normalize the hv_gradient in obj space (||dHV/dY|| == 1)
        normalized_obj_space_multifront_hv_gradient = np.zeros((n_mo_obj, n_mo_sol))
        for i_mo_sol in range(0, n_mo_sol):
            w = np.sqrt(np.sum(obj_space_multifront_hv_gradient[:, i_mo_sol] ** 2.0))
            if np.isclose(w, 0):
                w = 1
            if self.obj_space_normalize:
                normalized_obj_space_multifront_hv_gradient[:, i_mo_sol] = obj_space_multifront_hv_gradient[:,
                                                                           i_mo_sol] / w
            else:
                normalized_obj_space_multifront_hv_gradient[:, i_mo_sol] = obj_space_multifront_hv_gradient[:, i_mo_sol]

        dynamic_weights = torch.tensor(normalized_obj_space_multifront_hv_gradient, dtype=torch.float)
        return (dynamic_weights)



class GradHVSolver(GradBaseSolver):
    def __init__(self, step_size, max_iter, tol):

        super().__init__(step_size, max_iter, tol)

    def solve(self, problem, x, prefs, args):
        if args.n_obj != 2:
            assert False, 'hvgrad only supports 2 obj problem'

        hv_maximizer = HvMaximization(args.n_prob, args.n_obj, get_hv_ref_dict(args.problem_name))


        x = Variable(x, requires_grad=True)
        optimizer = torch.optim.SGD([x,], lr=self.step_size)
        hv_ind = HV(ref_point=get_hv_ref_dict(args.problem_name))
        hv_arr = [0] * self.max_iter
        y_arr=[]
        for iter_idx in tqdm(range(self.max_iter)):
            y = problem.evaluate(x)
            y_np = y.detach().numpy()
            y_arr.append(y_np)
            hv_arr[iter_idx] = hv_ind.do(y_np)
            weight = hv_maximizer.compute_weights(y_np.T)
            weight = torch.tensor(weight.T, dtype=torch.float)
            optimizer.zero_grad()
            torch.sum(weight*y).backward()
            optimizer.step()
            if 'lbound' in dir(problem):
                x.data = torch.clamp(x.data, torch.Tensor(problem.lbound) + solution_eps, torch.Tensor(problem.ubound)-solution_eps )


        res = {}
        res['x'] = x.detach().numpy()
        res['y'] = y.detach().numpy()
        res['hv_arr'] = hv_arr
        res['y_arr'] = y_arr

        return res