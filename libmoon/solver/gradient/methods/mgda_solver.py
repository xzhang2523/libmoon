import torch.autograd

from libmoon.solver.gradient.methods.core.mgda_core import solve_mgda
from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from torch import Tensor
import numpy as np
from libmoon.util_global.constant import solution_eps, get_hv_ref
from pymoo.indicators.hv import HV

from libmoon.problem.synthetic import ZDT1, ZDT2
from libmoon.solver.gradient.methods.core.core_solver import MGDACore
from matplotlib import pyplot as plt


'''
    MGDA solver, published in: 
    1. Multiple-gradient descent algorithm (MGDA) for multiobjective optimizationAlgorithme de descente à gradients multiples pour lʼoptimisation multiobjectif
    2. Sener, Ozan, and Vladlen Koltun. "Multi-task learning as multimnist-objective optimization." Advances in neural information processing systems 31 (2018).
'''
# class MGDAUBSolver(GradBaseSolver):
#     def __init__(self, step_size, max_iter, tol):
#         super().__init__(step_size, max_iter, tol)
#
#     def solve(self, problem, x, prefs, args):
#         x = Variable(x, requires_grad=True)
#         optimizer = SGD([x], lr=self.step_size)
#         ind = HV(ref_point=get_hv_ref(args.problem_name))
#         hv_arr = []
#         y_arr = []
#         for i in tqdm(range(self.max_iter)):
#             grad_arr = [0] * args.n_prob
#             y = problem.evaluate(x)
#             y_np = y.detach().numpy()
#             y_arr.append(y_np)
#             hv_arr.append(ind.do(y_np))
#             for prob_idx in range( args.n_prob ):
#                 grad_arr[prob_idx] = [0] * args.n_obj
#                 for obj_idx in range(args.n_obj):
#                     y[prob_idx][obj_idx].backward(retain_graph=True)
#                     grad_arr[prob_idx][obj_idx] = x.grad[prob_idx].clone()
#                     x.grad.zero_()
#                 grad_arr[prob_idx] = torch.stack(grad_arr[prob_idx])
#             grad_arr = torch.stack(grad_arr)
#             gw_arr = [solve_mgda(G, return_coeff=True) for G in grad_arr]
#             optimizer.zero_grad()
#             weights = Tensor( np.array([gw[1] for gw in gw_arr]) )
#             torch.sum(weights * y).backward()
#             optimizer.step()
#             if 'lbound' in dir(problem):
#                 x.data = torch.clamp(x.data, torch.Tensor(problem.lbound) + solution_eps, torch.Tensor(problem.ubound) - solution_eps )
#         res = {}
#         res['x'] = x.detach().numpy()
#         res['y'] = y.detach().numpy()
#         res['hv_arr'] = hv_arr
#         res['y_arr'] = y_arr
#
#         return res
class MGDAUBSolver(GradBaseSolver):
    def __init__(self, step_size, max_iter, tol):
        super().__init__(step_size, max_iter, tol)
        self.weight_solver_cls = MGDACore()
    def solve(self, problem, x, prefs):
        return super().solve(problem, x, prefs, self.weight_solver_cls)



if __name__ == '__main__':
    n_prob = 10
    n_var=10
    problem = ZDT1(n_var=n_var)
    solver = MGDAUBSolver(0.1, 1000, 1e-6)
    x = torch.rand(n_prob, n_var)
    prefs = torch.rand(n_prob, 2)
    res = solver.solve(problem, x, prefs)
    y_arr = res['y']


    plt.scatter(y_arr[:, 0], y_arr[:, 1])
    plt.show()

