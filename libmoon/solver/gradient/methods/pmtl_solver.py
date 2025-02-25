import torch
from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
from libmoon.solver.gradient.methods.core.min_norm_solvers_numpy import MinNormSolver
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from libmoon.util.constant import solution_eps
from libmoon.solver.gradient.methods.core.mgda_core import solve_mgda

class PMTLCore():
    def __init__(self, n_obj, n_var, n_epoch, prefs):
        '''
        Input:
            problem: Problem
            n_epoch: int
            warmup_epoch: int
            prefs: (n_prob, n_obj)
        '''
        self.core_name = 'PMTLCore'
        self.n_obj, self.n_var = n_obj, n_var
        self.n_epoch = n_epoch
        self.warmup_epoch = n_epoch // 5
        self.prefs_np = prefs.numpy() if type(prefs) == torch.Tensor else prefs
        self.stage1_finish = False

    def get_alpha_array(self, Jacobian_array, losses, epoch_idx):
        '''
            Input:Jacobian_array: (n_prob, n_obj, n_var)
                    losses: (n_prob, n_obj)
                    epoch_idx: int
            Return: (n_prob, n_obj)
        '''
        if type(losses) == torch.Tensor:
            losses_np = losses.detach().numpy()
        else:
            losses_np = losses
        n_prob = losses_np.shape[0]
        Jacobian_array_np = Jacobian_array.detach().numpy()
        # if epoch_idx < self.warmup_epoch:
        #     print('Stage 1')
        self.has_print = False
        if not self.stage1_finish:
            res = [get_d_paretomtl_init(Jacobian_array_np[i], losses_np[i], self.prefs_np, i)
                       for i in range(n_prob)]
            weights = [res[i][0] for i in range(n_prob)]
            finish = all([res[i][1] for i in range(n_prob)])
            self.stage1_finish = finish
        else:
            # just print once for the first time
            if not self.has_print:
                print('Begin second stage')
                self.has_print = True
            weights = [get_d_paretomtl(Jacobian_array_np[i], losses_np[i], self.prefs_np, i)
                       for i in range(n_prob)]
        # else:
        #     print('Stage 2')
        #     weights = [get_d_paretomtl(Jacobian_array_np[i], losses_np[i], self.prefs_np, i)
        #                for i in range(n_prob)]

        weights = torch.Tensor(np.array(weights)).to(Jacobian_array.device)
        return weights


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
    # sol, nd = MinNormSolver.find_min_norm_element( vec )
    sol = solve_mgda( torch.Tensor(vec))
    # reformulate ParetoMTL as linear scalarization method, return the weights
    weight0 = sol[0] + np.sum(np.array([sol[j] * w[idx][j - 2, 0] for j in np.arange(2, 2 + np.sum(idx))]))
    weight1 = sol[1] + np.sum(np.array([sol[j] * w[idx][j - 2, 1] for j in np.arange(2, 2 + np.sum(idx))]))
    weight = np.stack([weight0, weight1])
    return weight


def get_d_paretomtl_init(grads, value, weights, i):
    nobj, dim = grads.shape     # (2, 10)
    normalized_current_weight = weights[i] / np.linalg.norm(weights[i])
    normalized_rest_weights = np.delete(weights, (i), axis=0) / np.linalg.norm(np.delete(weights, (i), axis=0), axis=1,
                                                                               keepdims=True)
    w_diff = normalized_rest_weights - normalized_current_weight
    gx = np.dot(w_diff, value/np.linalg.norm(value))
    # gx: what is the meaning? Eq. (9)
    idx = gx > 0
    finish = True
    if np.sum(idx) <= 0:
        return np.zeros(nobj), finish

    if np.sum(idx) == 1:
        sol = np.ones(1)
    else:
        vecs = np.dot(w_diff[idx], grads)
        sol = solve_mgda( torch.Tensor(vecs) )
    # calculate the weights
    w_index = w_diff[idx]
    weight0 = sol @ w_index[:, 0]
    weight1 = sol @ w_index[:, 1]
    weight = np.stack([weight0, weight1])

    finish = False
    return weight, finish


class PMTLSolver(GradBaseSolver):
    def __init__(self, problem, prefs, step_size, n_epoch, tol, folder_name):
        self.solver_name = 'PMTL'
        self.problem = problem
        self.prefs = prefs
        self.folder_name = folder_name
        self.core_solver = PMTLCore(n_obj=problem.n_obj,
                                    n_var=problem.n_var,
                                    n_epoch=n_epoch,
                                    prefs=prefs)

        self.warmup_epoch = n_epoch // 5
        super().__init__(step_size, n_epoch, tol, self.core_solver)


    def solve(self, x_init):
        res = super().solve(self.problem, x_init, self.prefs)
        return res