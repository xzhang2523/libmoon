from libmoon.solver.gradient.methods.base_solver import GradBaseSolver

from libmoon.util.constant import root_name
import os
from torch.optim import SGD
from tqdm import tqdm
from pymoo.indicators.hv import HV
import math
import torch
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch import Tensor
from libmoon.solver.gradient.methods.core.mgda_core import solve_mgda
from libmoon.util.constant import solution_eps, get_hv_ref

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.config_name = os.path.join(root_name, 'config', 'pmgda.json')

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

def get_nn_pmgda_componets(loss_vec, pref):
    '''
        return: h_val, grad_h, J_hf
    '''
    # Here, use a single small bp graph
    loss_vec_var = Variable(loss_vec, requires_grad=True)
    h = constraint(loss_vec_var, pref=pref )
    h.backward()
    J_hf = loss_vec_var.grad
    h_val = h.detach().clone().item()
    # grad_h = J_hf @ Jacobian?
    return h_val, J_hf

def ts_to_np(grad_arr):
    g_np_arr = [0] * len(grad_arr)
    for idx, g_ts in enumerate(grad_arr):
        g_np_arr[idx] = g_ts.detach().clone().cpu().numpy()[0]
    return np.array(g_np_arr)

def pbi(f, lamb):
    lamb_ts = torch.Tensor(lamb)
    lamb0 = lamb_ts / torch.norm(lamb_ts)
    lamb0 = lamb0.double()
    d1 = f.squeeze().double() @ lamb0
    d2 = torch.norm(f.squeeze().double() - d1 * lamb0)
    return d1, d2

def constraint(loss_arr, pref=Tensor([0, 1])):
    '''
        # Just consider two types of constraints.
        # -- Type (1), the `exact' constraint.
        # -- Type (2), the `ROI' constraint.
    '''
    if type(pref) == np.ndarray:
        pref = Tensor(pref)
    constraint_mtd = 'pbi'
    if constraint_mtd == 'cel':
        eps = 1e-3
        loss_arr_0 = torch.clip(loss_arr / torch.sum(loss_arr), eps)
        res = torch.sum(loss_arr_0 * torch.log(loss_arr_0 / pref)) + torch.sum(
            pref * torch.log(pref / loss_arr_0))
        d2 = res.unsqueeze(0)
    elif constraint_mtd == 'cos':
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        pref_ts = pref.to(loss_arr.device)
        d2 = (1 - cos(loss_arr.unsqueeze(0), pref_ts.unsqueeze(0)))
    else:
        _, d2 = pbi(loss_arr, pref)
        d2 = d2.unsqueeze(0)
    return d2

def cosmos(losses, pref, coeff=5.0):
    if type(pref) == np.ndarray:
        pref = Tensor(pref)
    d1 = losses @ pref
    d2 = losses @ pref / torch.norm(losses) / torch.norm(pref)
    return d1 - coeff * d2

def get_cosmos_Jhf(loss, pref):
    loss = Variable(Tensor(loss), requires_grad=True)
    cosmos_loss = cosmos(loss, pref)
    cosmos_loss.backward()
    Jhf = loss.grad.detach().clone().cpu().numpy()
    return Jhf

def solve_pmgda(Jacobian, Jacobian_h_losses, h_val, h_tol, sigma):
    '''
        Input:
        Jacobian: (n_obj, n_var) : Tensor
        grad_h: (1, n_var)
        h_val: (1,) : float
        Jhf: (m,)
        Output:
        alpha: (m,)
    '''
    grad_h = Jacobian_h_losses @ Jacobian
    Jacobian_ts = Jacobian.detach().clone().to(device)
    grad_h_np = grad_h.detach().clone().cpu().numpy()
    G_ts = torch.cat((Jacobian, grad_h.unsqueeze(0)), dim=0).detach()
    G_norm = torch.norm(G_ts, dim=1, keepdim=True)
    G_n = G_ts / (G_norm + 1e-4)
    GGn = (G_ts @ G_n.T).clone().cpu().numpy()
    (m, n) = Jacobian_ts.shape
    condition = h_val < h_tol
    if condition:
        mu_prime = solve_mgda(Jacobian_ts)
    else:
        # Do the correction step. Eq. (20) in the main paper.
        # The total optimization number is m+2. A is the constraint matrix, and b is the constraint vector.
        A1 = -GGn
        A_tmp = - np.ones((m + 1, 1))
        A_tmp[-1][0] = 0
        A1 = np.c_[A1, A_tmp]
        b1 = np.zeros(m + 1)
        b1[-1] = - sigma * np.linalg.norm(grad_h_np)
        # A2 plus A3 are the simplex constraint, A2 is for non-zero constraints.
        A2 = np.c_[-np.eye(m + 1), np.zeros((m + 1, 1))]
        b2 = -np.zeros(m + 1)
        # A3, A4 are for the sum-equal-one constraint
        A3 = np.ones((1, m + 2))
        A3[0][-1] = 0.0
        b3 = np.ones(1)

        A4 = -np.ones((1, m + 2))
        A4[0][-1] = 0.0
        b4 = -np.ones(1)
        A_all = np.concatenate((A1, A2, A3, A4), 0)
        b_all = np.r_[b1, b2, b3, b4]
        A_matrix = matrix(A_all)  # The constraint matrix.
        b_matrix = matrix(b_all)  # The constraint vector.
        c = np.zeros(m + 2)  # The objective function.
        c[-1] = 1
        c_matrix = matrix(c)
        sol = solvers.lp(c_matrix, A_matrix, b_matrix)
        res = np.array(sol['x']).squeeze()
        # print( len(res) )
        mu, coeff = res[:-1], res[-1]
        # gw = G_n.T @ torch.Tensor(mu).to(G_n.device)
        # coeff, Eq. (18) in the main paper.
        mu_prime = get_pmgda_DWA_coeff(mu, Jacobian_h_losses, G_norm, m)
    return mu_prime

def get_pmgda_DWA_coeff(mu, Jhf, G_norm, m):
    '''
        This function is to compute the coefficient of the dynamic weight adjustment.
        Please ref the Eq. (18) for the formulation in the main paper.
    '''
    mu_prime = np.zeros( m )
    for i in range( m ):
        mu_prime[i] = mu[i] / G_norm[i] + mu[m] / G_norm[m] * Jhf[i]
    return mu_prime

def solve_mgda_null(G_tilde):
    # G_tilde.shape: (m+1, n)
    GG = G_tilde @ G_tilde.T
    # GG.shape : (m+1, m+1)
    Q = matrix(GG.astype(np.double))
    m, n = G_tilde.shape
    p = matrix(np.zeros(m))
    G = -np.eye(m)
    G[-1][-1] = 0.0
    G = matrix(G)

    h = matrix(np.zeros(m))
    A = np.ones(m)
    A[-1] = 0
    A = matrix(A, (1, m))
    b = matrix(1.0)
    sol = solvers.qp(Q, p, G, h, A, b)
    res = np.array(sol['x']).squeeze()
    gw = res @ G_tilde
    return gw

def get_Jhf(f_arr, pref, return_h=False):
    f = Variable(f_arr, requires_grad=True)
    h = constraint(f, pref=pref)
    h.backward()
    Jhf = f.grad.detach().clone().cpu().numpy()
    if return_h:
        return Jhf, float(h.detach().clone().cpu().numpy())
    else:
        return Jhf


class PMGDASolver(GradBaseSolver):
    # The PGMDA paper: http://arxiv.org/abs/2402.09492.
    def __init__(self, problem, prefs, step_size=1e-3, n_epoch=500, tol=1e-3,
                 sigma=0.1, h_tol=1e-3, folder_name=None):
        self.folder_name=folder_name
        self.problem = problem
        self.sigma = sigma
        self.h_tol = h_tol
        self.n_epoch = n_epoch
        self.pmgda_core = PMGDACore(n_var=problem.n_var, prefs=prefs)
        self.prefs = prefs
        self.solver_name = 'PMGDA'
        super().__init__(step_size, n_epoch, tol, self.pmgda_core)

    def solve(self, x_init):
        return super().solve(self.problem, x_init, self.prefs)