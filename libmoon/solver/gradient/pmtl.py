import torch

from .base_solver import GradBaseSolver
from matplotlib import pyplot as plt
from .min_norm_solvers_numpy import MinNormSolver
import numpy as np



from torch.autograd import Variable
from tqdm import tqdm
from ...util_global.constant import solution_eps

from .mgda_core import solve_mgda




def get_d_moomtl(grads):
    """
        calculate the gradient direction for MOO-MTL
    """
    nobj, dim = grads.shape
    sol, nd = MinNormSolver.find_min_norm_element(grads)
    return sol


def get_d_paretomtl(grads, value, weights, i):
    # calculate the gradient direction for Pareto MTL
    nobj, dim = grads.shape

    # check active constraints
    normalized_current_weight = weights[i] / np.linalg.norm(weights[i])
    normalized_rest_weights = np.delete(weights, (i), axis=0) / np.linalg.norm(np.delete(weights, (i), axis=0), axis=1,
                                                                     keepdims=True)
    # shape: (9, 2)
    w = normalized_rest_weights - normalized_current_weight
    # solve QP
    gx = np.dot(w, value / np.linalg.norm(value))
    idx = gx > 0
    vec = np.concatenate((grads, np.dot(w[idx], grads)), axis=0)
    # use MinNormSolver to solve QP
    # vec.shape:

    # sol, nd = MinNormSolver.find_min_norm_element( vec )
    _, sol = solve_mgda( torch.Tensor(vec), return_coeff=True)

    # reformulate ParetoMTL as linear scalarization method, return the weights
    weight0 = sol[0] + np.sum(np.array([sol[j] * w[idx][j - 2, 0] for j in np.arange(2, 2 + np.sum(idx))]))
    weight1 = sol[1] + np.sum(np.array([sol[j] * w[idx][j - 2, 1] for j in np.arange(2, 2 + np.sum(idx))]))
    weight = np.stack([weight0, weight1])

    return weight




def get_d_paretomtl_init(grads, value, weights, i):
    # calculate the gradient direction for Pareto MTL initialization
    nobj, dim = grads.shape

    # check active constraints
    normalized_current_weight = weights[i] / np.linalg.norm(weights[i])
    normalized_rest_weights = np.delete(weights, (i), axis=0) / np.linalg.norm(np.delete(weights, (i), axis=0), axis=1,
                                                                               keepdims=True)
    w = normalized_rest_weights - normalized_current_weight
    gx = np.dot(w, value / np.linalg.norm(value))
    idx = gx > 0


    if np.sum(idx) <= 0:
        return np.zeros(nobj)
    if np.sum(idx) == 1:
        sol = np.ones(1)
    else:
        vecs = np.dot(w[idx], grads)
        _, sol = solve_mgda( torch.Tensor(vecs), return_coeff=True)
        # print()

    # calculate the weights
    weight0 = np.sum(np.array([sol[j] * w[idx][j, 0] for j in np.arange(0, np.sum(idx))]))
    weight1 = np.sum(np.array([sol[j] * w[idx][j, 1] for j in np.arange(0, np.sum(idx))]))
    weight = np.stack([weight0, weight1])

    return weight


def circle_points(r, n):
    # generate evenly distributed preference vector
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles




class PMTLSolver(GradBaseSolver):
    def __init__(self, step_size, max_iter, tol):
        super().__init__(step_size, max_iter, tol)

    def solve(self, problem, x, prefs, args):
        if args.n_obj != 2:
            assert False, 'hvgrad only supports 2 obj problem'

        x = Variable(x, requires_grad=True)
        warmup_iter = self.max_iter // 5
        optimizer = torch.optim.SGD([x], lr=self.step_size)

        y_arr = []
        for iter_idx in tqdm( range(self.max_iter) ):
            y = problem.evaluate(x)
            y_np = y.detach().numpy()
            y_arr.append(y_np)

            grad_arr = [0] * args.n_prob
            for prob_idx in range(args.n_prob):
                grad_arr[prob_idx] = [0] * args.n_obj
                for obj_idx in range(args.n_obj):
                    y[prob_idx][obj_idx].backward(retain_graph=True)
                    grad_arr[prob_idx][obj_idx] = x.grad[prob_idx].clone()
                    x.grad.zero_()
                grad_arr[prob_idx] = torch.stack(grad_arr[prob_idx])

            grad_arr = torch.stack(grad_arr)
            grad_arr_np = grad_arr.detach().numpy()
            if iter_idx < warmup_iter:
                weights = [ get_d_paretomtl_init(grad_arr_np[i], y_np[i], prefs, i) for i in range(args.n_prob) ]
            else:
                weights = [ get_d_paretomtl(grad_arr_np[i], y_np[i], prefs, i) for i in range(args.n_prob) ]

            optimizer.zero_grad()
            torch.sum(torch.tensor(weights) * y).backward()
            optimizer.step()

            if 'lbound' in dir(problem):
                x.data = torch.clamp(x.data, torch.Tensor(problem.lbound) + solution_eps, torch.Tensor(problem.ubound) - solution_eps)

        res={}
        res['x'] = x.detach().numpy()
        res['y'] = y.detach().numpy()
        res['hv_arr'] = [0]
        res['y_arr'] = y_arr
        return res