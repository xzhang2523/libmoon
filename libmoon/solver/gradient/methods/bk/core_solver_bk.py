import matplotlib.pyplot as plt
import numpy as np
from libmoon.solver.gradient.methods.core.mgda_core import solve_mgda
from libmoon.solver.gradient.methods.epo_solver import EPO_LP
import torch
from libmoon.solver.gradient.methods.gradhv import HVMaxSolver
from libmoon.util.constant import get_hv_ref
from libmoon.solver.gradient.methods.pmgda_core import solve_pmgda
from libmoon.solver.gradient.methods.pmtl import get_d_paretomtl_init, get_d_paretomtl
import math
from torch import nn
from libmoon.solver.gradient.methods.uniform_solver import train_pfl_model
from torch.autograd import Variable
from torch.optim import SGD
from libmoon.util.mtl import get_angle_range
from libmoon.util.xy_util import pref2angle, angle2pref

from tqdm import tqdm
import os

class CoreHVGrad:
    '''
        Conventions used in this paper. Use _vec to denote a vec, and use _mat to denote a matrix.
    '''
    def __init__(self, n_prob, n_obj, dataset_name):
        self.hv_solver = HVMaxSolver(n_prob, n_obj, get_hv_ref(dataset_name) )

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
    def __init__(self):
        pass

    def get_alpha(self, Jacobian, grad_h, h_val, h_tol, sigma, return_coeff=True, Jhf=None):
        _, alpha = solve_pmgda(Jacobian, grad_h, h_val, h_tol, sigma, return_coeff=return_coeff, Jhf=Jhf)
        return alpha

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
    def __init__(self, n_prob):
        # pass
        self.n_prob = n_prob
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def get_alpha(self, Jacobian_arr, loss_mat):
        Jacobian_arr = torch.stack(Jacobian_arr)
        n_sub = len(Jacobian_arr)
        mgda_alpha_mat = [0] * n_sub
        for idx, Jacobian in enumerate(Jacobian_arr):
            _, alpha = solve_mgda(Jacobian, return_coeff=True)
            mgda_alpha_mat[idx] = alpha

        mgda_alpha_mat = np.array(mgda_alpha_mat)   # shape: (n_sub, n_obj)
        mgda_alpha_mat_ts = torch.Tensor(mgda_alpha_mat).to(self.device)  # shape: (n_sub, n_obj)

        loss_mat_var = Variable(loss_mat, requires_grad=True)
        kernel = kernel_functional_rbf(loss_mat_var).to(self.device)    # shape: (n_sub, n_sub)
        # term_A = kernel.detach().mm(g_mgda)  # shape: (n_sub, n_var)

        term_A = kernel.detach() @ mgda_alpha_mat_ts # shape: (n_sub, n_obj)
        term_B = - 0.5 * torch.autograd.grad(kernel.sum(), loss_mat_var, allow_unused=True)[0]  # (n_prob, n_obj)

        alpha_mat = (term_A - term_B) / self.n_prob
        return alpha_mat


class CoreMGDA:
    def __init__(self):
        pass
    def get_alpha(self, G):
        # G.shape: (m,n). G is the shorthand for Jacobian matrix.
        _, alpha = solve_mgda(G, return_coeff=True)
        return alpha


class CoreUniform:
    def __init__(self, device, folder_name, dataset_name, uniform_pref_update):
        self.device = device
        self.loss_mat_ts_arr = []
        self.pref_mat_ts_arr = []
        self.folder_name = folder_name
        self.pfl_model = PFLModel(n_obj=2).to(self.device)
        self.dataset_name = dataset_name
        self.uniform_pref_update = uniform_pref_update



    def visualize(self, pref_history, loss_history, update_idx):
        pref = np.linspace(0, 1, 100)
        pref = np.stack([pref, 1-pref], axis=1)
        pref = torch.Tensor(pref).to(self.device)

        # pref_angle = pref2angle(pref)

        predict = self.pfl_model(pref)

        fig = plt.figure()
        predict_np = predict.detach().cpu().numpy()
        pref_np = pref.detach().cpu().numpy()


        fig = plt.figure()
        for pref_elem, predict_elem in zip(pref_np, predict_np):
            plt.plot([pref_elem[0], predict_elem[0]], [pref_elem[1], predict_elem[1]], color='black', linestyle='--')

        plt.scatter(pref_np[:, 0], pref_np[:, 1], color='red')
        plt.scatter(predict_np[:, 0], predict_np[:, 1], color='blue')

        pref_history_np = torch.cat(pref_history).detach().cpu().numpy()
        loss_history_np = torch.cat(loss_history).detach().cpu().numpy()

        plt.scatter(pref_history_np[:, 0], pref_history_np[:, 1], color='green', s=100)
        plt.scatter(loss_history_np[:, 0], loss_history_np[:, 1], color='yellow', s=100)

        plt.xlabel('$L_1$')
        plt.ylabel('$L_2$')
        fig_name = os.path.join(self.folder_name, 'pref_loss_{}.pdf'.format(update_idx) )
        plt.savefig(fig_name)
        print('Save fig to {}'.format(fig_name))




    def update_pref_mat(self, pref_mat, loss_mat, pref_history, loss_history, update_idx):
        '''
            Input: pref_mat: (m,n), loss_mat: (m,n)
            Output: pref_mat: (m,n)
        '''
        print('update_idx: ', update_idx)

        print('len pref history', len(pref_history))
        print('len loss_history', len(loss_history))

        pref_history_ts = torch.cat(pref_history).to(self.device)
        loss_history_ts = torch.cat(loss_history).to(self.device)
        criterion = nn.MSELoss()
        pfl_optimizer = torch.optim.Adam(self.pfl_model.parameters(), lr=0.01)
        pfl_model = train_pfl_model(self.folder_name, update_idx, self.pfl_model, pfl_optimizer, criterion,
                                    pref_history_ts, loss_history_ts )

        angle_lower, angle_upper = get_angle_range(self.dataset_name)
        pref_mat_angle = pref2angle(pref_mat)
        prefs_angle_var = Variable(pref_mat_angle, requires_grad=True)
        prefs_optimizer = SGD([prefs_angle_var], lr=1e-4)

        mms_arr = []

        print('Preference updating...')
        for _ in tqdm(range(self.uniform_pref_update)):
            y_pred = pfl_model(prefs_angle_var)
            mms_val = compute_MMS(y_pred)
            prefs_optimizer.zero_grad()
            mms_val.backward()
            prefs_optimizer.step()
            prefs_angle_var.data = torch.clamp(prefs_angle_var.data, angle_lower, angle_upper)
            mms_arr.append( mms_val.item() )

        fig = plt.figure()
        plt.plot(mms_arr)
        fig_name = os.path.join(self.folder_name, 'mms_{}.pdf'.format(update_idx) )
        plt.savefig(fig_name)

        pref_mat = angle2pref(prefs_angle_var.data)
        return pref_mat


class CoreGrad:
    def __init__(self):
        pass

class CoreEPO(CoreGrad):
    def __init__(self, pref):
        self.pref = pref
        self.epo_lp = EPO_LP(m=len(pref), n=1, r=1/np.array( self.pref.cpu() ) )

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
    def __init__(self, pref_mat):
        self.pref_mat = pref_mat.cpu().numpy().copy()
        self.n_prob = pref_mat.shape[0]

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
            alpha_mat = [ get_d_paretomtl_init(Jacobian_arr[i], loss_mat[i], self.pref_mat, i) for i in range(self.n_prob) ]
        else:
            alpha_mat = [ get_d_paretomtl(Jacobian_arr[i], loss_mat[i], self.pref_mat, i) for i in range(self.n_prob) ]

        return np.array(alpha_mat)





if __name__ == '__main__':
    print()


