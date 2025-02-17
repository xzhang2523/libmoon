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

from libmoon.solver.gradient.methods.gradhv_solver import HVMaxSolver
from libmoon.solver.gradient.methods.pmtl_solver import get_d_paretomtl_init, get_d_paretomtl
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


'''
    MGDASolver. 
'''
class MGDAUBCore():
    def __init__(self):
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