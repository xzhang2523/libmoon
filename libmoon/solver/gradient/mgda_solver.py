import torch.autograd

from .mgda_core import solve_mgda

from .base_solver import GradBaseSolver
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from torch import Tensor
import numpy as np
from ...util_global.constant import solution_eps, get_hv_ref_dict
from pymoo.indicators.hv import HV


'''
    MGDA solver, published in: 
    1. Multiple-gradient descent algorithm (MGDA) for multiobjective optimizationAlgorithme de descente à gradients multiples pour lʼoptimisation multiobjectif
    2. Sener, Ozan, and Vladlen Koltun. "Multi-task learning as multimnist-objective optimization." Advances in neural information processing systems 31 (2018).
'''


class MGDASolver(GradBaseSolver):
    def __init__(self, step_size, max_iter, tol):
        super().__init__(step_size, max_iter, tol)


    def solve(self, problem, x, prefs, args):
        x = Variable(x, requires_grad=True)
        optimizer = SGD([x], lr=self.step_size)

        ind = HV(ref_point=get_hv_ref_dict(args.problem_name))
        hv_arr = []
        y_arr = []

        for i in tqdm(range(self.max_iter)):
            grad_arr = [0] * args.n_prob
            y = problem.evaluate(x)
            y_np = y.detach().numpy()
            y_arr.append(y_np)
            hv_arr.append(ind.do(y_np))

            for prob_idx in range( args.n_prob ):
                grad_arr[prob_idx] = [0] * args.n_obj
                for obj_idx in range(args.n_obj):
                    y[prob_idx][obj_idx].backward(retain_graph=True)
                    grad_arr[prob_idx][obj_idx] = x.grad[prob_idx].clone()
                    x.grad.zero_()
                grad_arr[prob_idx] = torch.stack(grad_arr[prob_idx])

            grad_arr = torch.stack(grad_arr)
            gw_arr = [solve_mgda(G, return_coeff=True) for G in grad_arr]
            optimizer.zero_grad()
            weights = Tensor( np.array([gw[1] for gw in gw_arr]) )
            # weights = Tensor( np.array([1.0, 0.0]) )
            torch.sum(weights * y).backward()
            optimizer.step()

            if 'lbound' in dir(problem):
                x.data = torch.clamp(x.data, torch.Tensor(problem.lbound) + solution_eps, torch.Tensor(problem.ubound) - solution_eps )

        res = {}
        res['x'] = x.detach().numpy()
        res['y'] = y.detach().numpy()
        res['hv_arr'] = hv_arr
        res['y_arr'] = y_arr

        return res


if __name__ == '__main__':
    print()