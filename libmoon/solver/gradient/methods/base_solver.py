import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.optim import SGD, Adam
from libmoon.util.constant import get_agg_func, solution_eps, get_hv_ref
import torch
from pymoo.indicators.hv import HV
import numpy as np
from libmoon.util.gradient import get_moo_Jacobian_batch
from libmoon.model.simple import PFLModel
from libmoon.util.prefs import pref2angle, angle2pref, get_uniform_pref
import os
from libmoon.metrics.metrics import compute_lmin
from torch import Tensor
criterion = torch.nn.MSELoss()
from tqdm import tqdm

def umod_train_pfl_model(folder_name, update_idx, pfl_model, pfl_optimizer,
                    criterion, prefs, y, pfl_epoch=2000):
    prefs = torch.Tensor(prefs)
    y = torch.Tensor(y)
    prefs_angle = pref2angle(prefs)
    loss_arr = []
    for _ in range(pfl_epoch):
        y_hat = pfl_model(prefs_angle)
        loss = criterion(y_hat, y)
        loss_arr.append(loss.item())
        pfl_optimizer.zero_grad()
        loss.backward()
        pfl_optimizer.step()
    fig = plt.figure()
    plt.plot(loss_arr)
    plt.xlabel('Iteration')
    plt.ylabel('PFL Loss')
    fig_name = os.path.join(folder_name, 'loss_{}.pdf'.format(update_idx) )
    plt.savefig(fig_name)
    print('Save to {}'.format(fig_name))
    return pfl_model

def umod_adjust_pref(prefs, pfl_model, n_adjust_epoch, main_epoch_idx, folder_name):
    prefs_angle = pref2angle(prefs)
    prefs_angle_var = Variable(prefs_angle, requires_grad=True)
    optimizer = Adam([prefs_angle_var], lr=1e-3)
    lmin_arr = []
    for _ in range(n_adjust_epoch):
        y_pred = pfl_model(prefs_angle_var)
        lmin_val = compute_lmin(y_pred)
        optimizer.zero_grad()
        (-lmin_val).backward()   # To max the pairwise distance.
        optimizer.step()
        prefs_angle_var.data = torch.clamp(prefs_angle_var.data,
                                           0, np.pi/2)
        lmin_arr.append(lmin_val.item())
    fig = plt.figure()
    plt.plot(lmin_arr)
    plt.xlabel('Iteration')
    plt.ylabel('Lmin')
    fig_name = os.path.join(folder_name, 'lmin_{}.pdf'.format(main_epoch_idx) )
    plt.savefig(fig_name)
    print('Save to {}'.format(fig_name))
    return angle2pref(prefs_angle_var.data)

class AggCore():
    def __init__(self, prefs, agg_name):
        self.core_name = 'AggCore'
        self.agg_name = agg_name
        self.prefs = prefs

    def get_alpha(self, Jacobian, losses):
        if self.agg_name == 'LS':
            return torch.Tensor(self.prefs)
        else:
            assert False, 'Has not implemented.'

class GradBaseSolver:
    def __init__(self, step_size, epoch, tol, core_solver, verbose=False):
        self.step_size = step_size
        self.epoch = epoch
        self.tol = tol
        self.core_solver = core_solver
        self.is_agg = (self.core_solver.core_name == 'AggCore')
        try:
            self.agg_name = self.core_solver.agg_name
        except:
            pass

    def solve(self, problem, x, prefs):
        '''
            :param problem:
            :param x:
            :return:
                is a dict with keys: x, y.
        '''
        if self.solver_name == 'UMOD':
            self.pfl_model = PFLModel(n_obj=problem.n_obj)
            self.pfl_optimizer = torch.optim.Adam(self.pfl_model.parameters(), lr=1e-3)

        self.n_prob, self.n_obj = prefs.shape[0], prefs.shape[1]
        xs_var = Variable(x, requires_grad=True)
        optimizer = Adam([xs_var], lr=self.step_size)
        ind = HV(ref_point=get_hv_ref(problem.problem_name))
        hv_arr, y_arr = [], []

        # For UMOD solver, we need to store (pref, y) pairs.
        pref_y_pairs = []
        if self.verbose:
            iteration_container = tqdm(range(self.epoch))
        else:
            iteration_container = range(self.epoch)

        for epoch_idx in iteration_container:
            fs_var = problem.evaluate(xs_var)
            y_np = fs_var.detach().numpy()
            y_arr.append(y_np)
            hv_arr.append(ind.do(y_np))
            Jacobian_array = get_moo_Jacobian_batch(xs_var, fs_var, self.n_obj)
            y_detach = fs_var.detach()
            optimizer.zero_grad()

            if self.solver_name in ['GradAgg', 'UMOD']:
                agg_func = get_agg_func(self.agg_name)
                agg_val = agg_func(fs_var, torch.Tensor(prefs).to(fs_var.device))
                torch.sum(agg_val).backward()
            else:
                if self.core_solver.core_name in ['EPOCore', 'MGDAUBCore', 'PMGDACore', 'RandomCore']:
                    alpha_array = torch.stack(
                        [self.core_solver.get_alpha(Jacobian_array[idx], y_detach[idx], idx) for idx in
                         range(self.n_prob)])
                elif self.core_solver.core_name in ['PMTLCore', 'MOOSVGDCore', 'GradHVCore']:
                    if self.core_solver.core_name == 'GradHVCore':
                        alpha_array = self.core_solver.get_alpha_array(y_detach)
                    elif self.core_solver.core_name == 'PMTLCore':
                        alpha_array = self.core_solver.get_alpha_array(Jacobian_array, y_np, epoch_idx)
                    elif self.core_solver.core_name == 'MOOSVGDCore':
                        alpha_array = self.core_solver.get_alpha_array(Jacobian_array, y_detach)
                    else:
                        assert False, 'Unknown core_name:{}'.format(self.core_solver.core_name)
                else:
                    assert False, 'Unknown core_name'
                torch.sum(alpha_array * fs_var).backward()

            optimizer.step()



            if 'lbound' in dir(problem):
                x.data = torch.clamp(x.data, torch.Tensor(problem.lbound) + solution_eps,
                                     torch.Tensor(problem.ubound) - solution_eps)

            if problem.problem_name in ['MOKL']:
                x.data = torch.clamp(x.data, min=0)
                x.data = x.data / torch.sum(x.data, dim=1, keepdim=True)
            # print('x.data', x.data)
            # assert False

            if self.solver_name == 'UMOD':
                if epoch_idx % self.pfl_train_epoch == 0 and epoch_idx != 0:
                    pref_y_pairs.append((prefs, y_np))
                    print('Pair len: {}'.format(len(pref_y_pairs)) )
                    prefs_np = prefs.detach().numpy()
                    plt.scatter(prefs_np[:,0], prefs_np[:,1])
                    plt.scatter(y_np[:,0], y_np[:,1])
                    # train the relationship between prefs_np and y_np
                    prefs_all = torch.cat([Tensor(pair[0]) for pair in pref_y_pairs], axis=0)
                    y_all = torch.cat([Tensor(pair[1]) for pair in pref_y_pairs], axis=0)

                    umod_train_pfl_model(
                        folder_name=self.folder_name,
                        update_idx=epoch_idx,
                        pfl_model=self.pfl_model,
                        pfl_optimizer=self.pfl_optimizer,
                        criterion=criterion,
                        prefs=prefs_all,
                        y=y_all
                    )     # Use all historical data to train the model.

                    prefs_new = umod_adjust_pref(prefs, pfl_model=self.pfl_model, n_adjust_epoch=self.pref_adjust_epoch,
                                                 main_epoch_idx=epoch_idx, folder_name=self.folder_name)

                    pref_test = get_uniform_pref(n_prob=100, dtype='Tensor')
                    y_test = self.pfl_model(pref2angle(pref_test))
                    y_test_np = y_test.detach().numpy()
                    for (pp, yy) in zip(prefs_np, y_np):
                        plt.plot([pp[0], yy[0]], [pp[1], yy[1]], color='grey', linestyle='dashed')
                    plt_umod = False
                    if plt_umod:
                        plt.scatter(y_test_np[:, 0], y_test_np[:, 1])
                        plt.scatter(prefs_new[:,0], prefs_new[:,1], label='New prefs')
                        plt.legend()
                        plt.show()
                    prefs = prefs_new
        res = {}
        res['x'] = x.detach().numpy()
        res['y'] = y_np
        res['hv_history'] = hv_arr
        res['y_history'] = y_arr
        return res


class GradAggSolver(GradBaseSolver):
    def __init__(self, problem, prefs, step_size=1e-3, n_epoch=500, tol=1e-3,
                 agg_name='LS', folder_name=None):

        self.folder_name = folder_name
        self.step_size=step_size
        self.n_epoch = n_epoch
        self.tol=tol
        self.problem = problem
        self.prefs = prefs
        self.solver_name = 'GradAgg'
        self.core_solver = AggCore(prefs, agg_name)
        super().__init__(step_size, n_epoch, tol, core_solver=self.core_solver)

    def solve(self, x_init):
        return super().solve(self.problem, x_init, self.prefs)