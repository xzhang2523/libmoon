import numpy as np
from libmoon.solver.gradient.methods.mgda_core import solve_mgda
from libmoon.solver.gradient.methods.epo_solver import EPO_LP
import torch
from libmoon.solver.gradient.methods.gradhv import HvMaximization
from libmoon.util_global.constant import get_hv_ref_dict
from libmoon.solver.gradient.methods.pmgda_core import solve_pmgda
from libmoon.solver.gradient.methods.pmtl import get_d_paretomtl_init, get_d_paretomtl
import math

from torch.autograd import Variable


class CoreHVGrad:
    '''
        Conventions used in this paper. Use _vec to denote a vec, and use _mat to denote a matrix.
    '''
    def __init__(self, args):
        self.args = args
        self.hv_solver = HvMaximization(args.n_sub, args.n_obj, get_hv_ref_dict(args.dataset) )

    def get_alpha(self, loss_mat):
        '''
            Input: loss_mat.shape (K, n_obj).
            Output: alpha_mat.shape (K, n_obj).
        '''
        if type(loss_mat) == torch.Tensor:
            loss_mat = loss_mat.detach().cpu().numpy().copy()
        alpha_mat = self.hv_solver.compute_weights(loss_mat.T).T
        return alpha_mat

class CorePMGDA:
    def __init__(self, args):
        self.args = args

    def get_alpha(self, Jacobian, grad_h, h_val, args, return_coeff=True, Jhf=None):
        _, alpha = solve_pmgda(Jacobian, grad_h, h_val, args, return_coeff=return_coeff, Jhf=Jhf)
        return alpha
        #     Input:
        #     Jacobian: (m,n)
        #     grad_h: (1,n)
        #     h_val: (1,)
        #     args: args
        #     return_coeff: bool
        #     Jhf: (m,) .
        #
        #     Output:
        #     if use_coeff:
        #         gw: (n,)
        #         coeff: (m,)
        #     else:
        #         gw: (n,)





# from libmoon.solver.gradient.methods.moosvgd import solve_moo_svgd


# def kernel_functional_rbf(losses):
#     '''
#         input losses: (n_prob, n_obj)
#         output kernel_matrix: (n_prob, n_prob)
#         comments: This function is used to compute the kernel matrix for SVGD.
#     '''
#     n = losses.shape[0]
#     # losses shape : (10,) * (3,)
#     pairwise_distance = torch.norm(losses[:, None] - losses, dim=2).pow(2)
#     h = median(pairwise_distance) / math.log(n)
#     A = 5e-6  # Noted, this bandwith parameter is important.
#     kernel_matrix = torch.exp(-pairwise_distance / A*h)  # 5e-6 for zdt1,2,3, zxy, Dec 5, 2023
#     return kernel_matrix



def median(tensor):
    """
        torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.


def kernel_functional_rbf(losses):
    '''
        input losses: (n_prob, n_obj)
        output kernel_matrix: (n_prob, n_prob)
        comments: This function is used to compute the kernel matrix for SVGD.
    '''
    n = losses.shape[0]
    pairwise_distance = torch.norm(losses[:, None] - losses, dim=2).pow(2)
    h = median(pairwise_distance) / math.log(n)
    A = 1  # Noted, this bandwith parameter is very important.
    kernel_matrix = torch.exp(-pairwise_distance / A*h)  # 5e-6 for zdt1,2,3, zxy, Dec 5, 2023
    return kernel_matrix



class CoreMOOSVGD:
    def __init__(self, args):
        # pass
        self.args = args
        self.n_sub = args.n_sub


    def get_alpha(self, Jacobian_arr, loss_mat):

        # Jacobian_arr.shape: (n_sub, n_obj, n_var)
        # loss_mat.shape: (n_sub, n_obj)

        Jacobian_arr = torch.stack(Jacobian_arr)
        n_sub = len(Jacobian_arr)
        mgda_alpha_mat = [0] * n_sub
        for idx, Jacobian in enumerate(Jacobian_arr):
            _, alpha = solve_mgda(Jacobian, return_coeff=True)
            mgda_alpha_mat[idx] = alpha

        mgda_alpha_mat = np.array(mgda_alpha_mat)   # shape: (n_sub, n_obj)
        mgda_alpha_mat_ts = torch.Tensor(mgda_alpha_mat)

        loss_mat_var = Variable(loss_mat, requires_grad=True)
        kernel = kernel_functional_rbf(loss_mat_var)    # shape: (n_sub, n_sub)
        # term_A = kernel.detach().mm(g_mgda)  # shape: (n_sub, n_var)

        term_A = kernel.detach() @ mgda_alpha_mat_ts # shape: (n_sub, n_obj)
        term_B = - 0.5 * torch.autograd.grad(kernel.sum(), loss_mat_var, allow_unused=True)[0]  # (n_prob, n_obj)

        alpha_mat = (term_A - term_B) / self.n_sub
        return alpha_mat


class CoreMGDA:
    def __init__(self):
        pass

    def get_alpha(self, G):
        # G.shape: (m,n). G is the shorthand for Jacobian matrix.
        _, alpha = solve_mgda(G, return_coeff=True)
        return alpha



class CoreGrad:
    def __init__(self):
        pass

class CoreEPO(CoreGrad):
    def __init__(self, pref):
        self.pref = pref
        self.epo_lp = EPO_LP(m=len(pref), n=1, r=1/np.array(pref))

    def get_alpha(self, G, losses):
        '''
            Input: G: (m,n), losses: (m,)
        '''
        if type(G) == torch.Tensor:
            G = G.detach().cpu().numpy().copy()
        if type(losses) == torch.Tensor:
            losses = losses.detach().cpu().numpy().copy()
        GG = G @ G.T
        alpha = self.epo_lp.get_alpha(losses, G=GG, C=True)
        return alpha


class CoreAgg(CoreGrad):
    def __init__(self, pref, agg_mtd='ls'):
        self.agg_mtd = agg_mtd
        self.pref = pref

    def get_alpha(self, G, losses):
        if self.agg_mtd == 'ls':
            alpha = self.pref
        elif self.agg_mtd == 'mtche':
            idx = np.argmax(losses)
            alpha = np.zeros_like(self.pref )
            alpha[idx] = 1.0
        else:
            assert False
        return alpha


class CorePMTL(CoreGrad):
    def __init__(self, args, pref_mat):
        self.pref_mat = pref_mat.cpu().numpy().copy()
        self.args = args

    def get_alpha(self, Jacobian_arr, loss_mat, is_warmup=True):
        '''
            Input: Jacobian_arr: (K, n_obj, n), loss_mat: (K, n_obj)
            Output: alpha_mat: (K, n_obj)
        '''
        if type(Jacobian_arr) == torch.Tensor:
            Jacobian_arr = Jacobian_arr.detach().cpu().numpy().copy()

        if type(loss_mat) == torch.Tensor:
            loss_mat = loss_mat.detach().cpu().numpy().copy()

        if is_warmup:
            alpha_mat = [ get_d_paretomtl_init(Jacobian_arr[i], loss_mat[i], self.pref_mat, i) for i in range(self.args.n_sub) ]
        else:
            alpha_mat = [ get_d_paretomtl(Jacobian_arr[i], loss_mat[i], self.pref_mat, i) for i in range(self.args.n_sub) ]

        return np.array(alpha_mat)



if __name__ == '__main__':
    # agg = CoreAgg( pref=np.array([1, 0]) )
    # x = torch.rand(10, 1)
    # res = median(x)
    # loss_mat = torch.rand(10, 3)
    # kernel = kernel_functional_rbf(loss_mat)
    # print()
    pass


