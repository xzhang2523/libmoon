from libmoon.solver.gradient.methods.core.mgda_core import solve_mgda
import torch

import numpy as np
import cvxpy as cp
import cvxopt
from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
from torch.autograd import Variable
from tqdm import tqdm
import torch
from torch.optim import SGD
from numpy import array
from pymoo.indicators.hv import HV
import warnings
warnings.filterwarnings("ignore")
from libmoon.util.constant import solution_eps, get_hv_ref
from libmoon.util.gradient import get_moo_Jacobian
from libmoon.problem.synthetic.zdt import ZDT1

from libmoon.solver.gradient.methods.gradhv import HVMaxSolver
# D:\pycharm_project\libmoon\libmoon\solver\gradient\methods\gradhv.py

from libmoon.solver.gradient.methods.pmtl import get_d_paretomtl_init, get_d_paretomtl
# D:\pycharm_project\libmoon\libmoon\solver\gradient\methods\pmtl.py

from libmoon.solver.gradient.methods.moosvgd import get_svgd_alpha_array
# D:\pycharm_project\libmoon\libmoon\solver\gradient\methods\moosvgd.py

from libmoon.solver.gradient.methods.pmgda_solver import solve_pmgda, constraint, get_Jhf
# D:\pycharm_project\libmoon\libmoon\solver\gradient\methods\pmgda_solver.py

class EPO_LP(object):
    # Paper: https://proceedings.mlr.press/v119/mahapatra20a.html
    # Paper: https://arxiv.org/abs/2010.06313
    def __init__(self, m, n, r, eps=1e-4):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.n = n
        self.r = r
        self.eps = eps
        self.last_move = None
        self.a = cp.Parameter(m)        # Adjustments
        self.C = cp.Parameter((m, m))   # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)       # d_bal^TG
        self.rhs = cp.Parameter(m)      # RHS of constraints for balancing
        self.alpha = cp.Variable(m)     # Variable to optimize
        obj_bal = cp.Maximize(self.alpha @ self.Ca)   # objective for balance
        constraints_bal = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Simplex
                           self.C @ self.alpha >= self.rhs]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance
        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent
        constraints_res = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Restrict
                           self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
                           self.C @ self.alpha >= 0]
        constraints_rel = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Relaxed
                           self.C @ self.alpha >= 0]
        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP dominance
        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0     # Stores the latest non-uniformity


    def get_alpha(self, l, G, r_in=None, C=False, relax=False):
        r = r_in.numpy() if type(r_in) == torch.Tensor else r_in
        r = self.r if r is None else r
        assert len(l) == len(G) == len(r) == self.m, "length != m"

        rl, self.mu_rl, self.a.value = adjustments(l, r)
        self.C.value = G if C else G @ G.T
        self.Ca.value = self.C.value @ self.a.value

        if self.mu_rl > self.eps:
            J = self.Ca.value > 0
            if len(np.where(J)[0]) > 0:
                J_star_idx = np.where(rl == np.max(rl))[0]
                self.rhs.value = self.Ca.value.copy()
                self.rhs.value[J] = -np.inf     # Not efficient; but works.
                self.rhs.value[J_star_idx] = 0
            else:
                self.rhs.value = np.zeros_like(self.Ca.value)
            self.gamma = self.prob_bal.solve(solver=cp.GLPK, verbose=False)
            # self.gamma = self.prob_bal.solve(verbose=False)
            self.last_move = "bal"
        else:
            if relax:
                self.gamma = self.prob_rel.solve(solver=cp.GLPK, verbose=False)
            else:
                self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            # self.gamma = self.prob_dom.solve(verbose=False)
            self.last_move = "dom"
        return self.alpha.value


def mu(rl, normed=False):
    # Modified by Xiaoyuan to handle negative issue.
    # if len(np.where(rl < 0)[0]):
    #     raise ValueError(f"rl<0 \n rl={rl}")
    #     return None
    rl = np.clip(rl, 0, np.inf)
    m = len(rl)
    l_hat = rl if normed else rl / rl.sum()
    eps = np.finfo(rl.dtype).eps
    l_hat = l_hat[l_hat > eps]
    return np.sum(l_hat * np.log(l_hat * m))


def adjustments(l, r=1):
    m = len(l)
    rl = r * l
    l_hat = rl / rl.sum()
    mu_rl = mu(l_hat, normed=True)
    eps = 1e-3   # clipping by eps is to avoid log(0), zxy Dec. 5.
    a = r * ( np.log( np.clip(l_hat * m, eps, np.inf) ) - mu_rl)
    return rl, mu_rl, a


def solve_epo(Jacobian, losses, pref, epo_lp):
    '''
        input: Jacobian: (m,n).
        losses : (m,).
        pref: (m,) inv.
        return : gw: (n,). alpha: (m,)
    '''
    if type(pref) == torch.Tensor:
        pref = pref.cpu().numpy()
    pref = np.array(pref)

    G = Jacobian.detach().clone()
    if type(losses) == torch.Tensor:
        losses_np = losses.detach().clone().cpu().numpy().squeeze()
    else:
        losses_np = losses
    m = G.shape[0]
    n = G.shape[1]
    GG = G @ G.T
    GG = GG.cpu().numpy()

    alpha = epo_lp.get_alpha(losses_np, G=GG, C=True)
    if alpha is None:   # A patch for the issue in cvxpy
        alpha = pref / pref.sum()
    return alpha


class EPOCore():
    def __init__(self, n_var, prefs):
        '''
            Input:
            n_var: int, number of variables.
            prefs: (n_prob, n_obj).
        '''
        self.core_name = 'EPOCore'
        self.prefs = prefs
        self.n_prob, self.n_obj = prefs.shape[0], prefs.shape[1]
        self.n_var = n_var
        prefs_np = prefs.cpu().numpy() if type(prefs) == torch.Tensor else prefs
        self.epo_lp_arr = [EPO_LP(m=self.n_obj, n = self.n_var, r=1/pref) for pref in prefs_np]

    def get_alpha(self, Jacobian, losses, idx):
        alpha = solve_epo(Jacobian, losses, self.prefs[idx], self.epo_lp_arr[idx])
        return torch.Tensor(alpha)


import json
class PMGDACore():
    def __init__(self, n_var, prefs):
        '''
            Input:
            n_var: int, number of variables.
            prefs: (n_prob, n_obj).
        '''
        self.core_name = 'PMGDACore'
        self.prefs = prefs
        self.n_prob, self.n_obj = prefs.shape[0], prefs.shape[1]
        self.n_var = n_var
        prefs_np = prefs.cpu().numpy() if type(prefs) == torch.Tensor else prefs
        self.config_name = 'D:\\pycharm_project\\libmoon\\libmoon\\config\\pmgda.json'
        json_file = open(self.config_name, 'r')
        self.config = json.load(json_file)
        self.h_eps = self.config['h_eps']
        self.sigma = self.config['sigma']


    def get_alpha(self, Jacobian, losses, idx):
        '''
            Input:
            Jacobian: (n_obj, n_var), torch.Tensor
            losses: (n_obj,), torch.Tensor
            idx: int
        '''
        # (1) get the constraint value
        losses_var = Variable(losses, requires_grad=True)
        h_var = constraint(losses_var, pref=self.prefs[idx])
        h_val = h_var.detach().cpu().clone().numpy()
        h_var.backward()
        Jacobian_h_losses = losses_var.grad.detach().clone()
        # shape: (n_obj)
        alpha = solve_pmgda(Jacobian, Jacobian_h_losses, h_val, self.h_eps, self.sigma)
        return torch.Tensor(alpha).to(Jacobian.device)

'''
    MGDASolver. 
'''
class MGDAUBCore():
    def __init__(self, n_var, prefs):
        self.core_name = 'MGDAUBCore'

    def get_alpha(self, Jacobian, losses, idx):
        alpha = solve_mgda(Jacobian)
        return alpha


class RandomCore():
    def __init__(self, n_var, prefs):
        self.core_name = 'RandomCore'

    def get_alpha(self, Jacobian, losses, idx):
        n_obj = len(losses)
        return torch.rand(n_obj)


class AggCore():
    def __init__(self, n_var, prefs, solver_name):
        self.core_name = 'AggCore'
        self.solver_name = solver_name
        self.agg_name = solver_name.split('_')[-1]

    def get_alpha(self, Jacobian, losses, idx):
        assert False, 'RandomCore does not have get_alpha method.'
        return None



class MOOSVGDCore():
    def __init__(self, n_var, prefs):
        self.core_name = 'MOOSVGDCore'

    def get_alpha_array(self, Jacobian_arr, losses_arr):
        '''
            Input:
            Jacobian_arr: (n_prob, m, n)
            losses_arr: (n_prob, m)
            Return: (n_prob, m)
        '''
        alpha_array = get_svgd_alpha_array(Jacobian_arr, losses_arr, None)
        return alpha_array


class HVGradCore():
    def __init__(self, n_obj, n_var, problem_name):
        self.core_name = 'HVGradCore'
        # problem = get_problem(problem_name=problem_name, n_var=n_var)
        self.n_obj, self.n_var = n_obj, n_var
        self.problem_name = problem_name

    def get_alpha_array(self, losses):
        '''
            Input : losses: (n_prob, n_obj)
            Return: (n_prob, n_obj)
        '''
        losses_np = losses.detach().numpy()
        n_prob = losses_np.shape[0]

        hv_maximizer = HVMaxSolver(n_prob, self.n_obj, get_hv_ref(self.problem_name))
        weight = hv_maximizer.compute_weights(losses_np.T).T
        return weight


class PMTLCore():
    def __init__(self, n_obj, n_var, total_epoch, warmup_epoch, prefs):
        '''
        Input:
            problem: Problem
            total_epoch: int
            warmup_epoch: int
            prefs: (n_prob, n_obj)
        '''
        self.core_name = 'PMTLCore'
        self.n_obj, self.n_var = n_obj, n_var
        self.total_epoch = total_epoch
        self.warmup_epoch = warmup_epoch
        self.prefs_np = prefs.numpy() if type(prefs) == torch.Tensor else prefs


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
        if epoch_idx < self.warmup_epoch:
            weights = [get_d_paretomtl_init(Jacobian_array_np[i], losses_np[i], self.prefs_np, i) for i in range(n_prob)]
        else:
            weights = [get_d_paretomtl(Jacobian_array_np[i], losses_np[i], self.prefs_np, i) for i in range(n_prob)]
        weights = torch.Tensor(np.array(weights)).to(Jacobian_array.device)
        return weights




if __name__ == '__main__':
    #
    n_prob = 5
    n_obj = 2
    n_var = 10
    problem = ZDT1(n_var)
    pref_1d = np.linspace(1e-3, 1-1e-3, n_prob)
    prefs = np.c_[pref_1d, 1 - pref_1d]

    epo_core = EPOCore(problem.n_var, prefs)
    Jacobian = torch.rand(n_obj, n_prob)
    losses = torch.rand(n_obj)
    alpha_arr = epo_core.get_alpha(Jacobian, losses)
    print()