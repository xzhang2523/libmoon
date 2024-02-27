import numpy as np
import cvxpy as cp
import cvxopt

from .base_solver import GradBaseSolver
from torch.autograd import Variable
from tqdm import tqdm
import torch
from torch.optim import SGD
from numpy import array
from pymoo.indicators.hv import HV
import warnings
warnings.filterwarnings("ignore")

from ...util_global.constant import solution_eps



class EPO_LP(object):
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


    def get_alpha(self, l, G, r=None, C=False, relax=False):

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
    if len(np.where(rl < 0)[0]):
        raise ValueError(f"rl<0 \n rl={rl}")
        return None
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



def solve_epo(grad_arr, losses, pref, epo_lp):

    '''
        input: grad_arr: (m,n).
        losses : (m,).
        pref: (m,) inv.

        return : gw: (n,).
    '''
    if type(pref) == torch.Tensor:
        pref = pref.numpy()

    pref = np.array(pref)
    G = grad_arr.detach().clone().numpy()

    if type(losses) == torch.Tensor:
        losses_np = losses.detach().clone().numpy().squeeze()
    else:
        losses_np = losses

    m = G.shape[0]
    n = G.shape[1]
    GG = G @ G.T

    # epo_lp = EPO_LP(m=m, n=n, r=np.array(pref))

    alpha = epo_lp.get_alpha(losses_np, G=GG, C=True)
    if alpha is None:   # A patch for the issue in cvxpy
        alpha = pref / pref.sum()
    gw = alpha @ G

    # return torch.Tensor(gw).unsqueeze(0)
    return torch.Tensor(gw), alpha







class EPOSolver(GradBaseSolver):
    def __init__(self, step_size, max_iter, tol):
        super().__init__(step_size, max_iter, tol)


    def solve(self, problem, x, prefs, args):
        x = Variable(x, requires_grad=True)

        epo_arr = [  EPO_LP(m=args.n_obj, n=args.n_var, r=np.array( 1/pref )) for pref in prefs ]
        optimizer = SGD([x], lr=self.step_size)

        ref_point = array([2.0, 2.0])
        ind = HV(ref_point=ref_point)
        hv_arr = []
        y_arr = []


        for i in tqdm( range(self.max_iter) ):

            # optimizer.zero_grad()
            y = problem.evaluate(x)
            y_arr.append(y.detach().numpy() )

            alpha_arr = [0] * args.n_prob
            for prob_idx in range( args.n_prob ):
                grad_arr = [0] * args.n_obj
                for obj_idx in range(args.n_obj):
                    y[prob_idx][obj_idx].backward(retain_graph=True)
                    grad_arr[obj_idx] = x.grad[prob_idx].clone()
                    x.grad.zero_()

                grad_arr = torch.stack(grad_arr)
                _, alpha = solve_epo(grad_arr, losses=y[prob_idx], pref=prefs[prob_idx], epo_lp=epo_arr[prob_idx])
                alpha_arr[prob_idx] = alpha

            optimizer.zero_grad()
            alpha_arr = torch.Tensor( np.array(alpha_arr) )
            torch.sum(alpha_arr * y).backward()
            optimizer.step()

            if 'lbound' in dir(problem):
                x.data = torch.clamp(x.data, torch.Tensor(problem.lbound) + solution_eps, torch.Tensor(problem.ubound)-solution_eps )


        res = {}
        res['x'] = x.detach().numpy()
        res['y'] = y.detach().numpy()
        res['hv_arr'] = [0]
        res['y_arr'] = y_arr

        return res