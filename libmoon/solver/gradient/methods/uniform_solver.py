from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
from torch.autograd import Variable
from torch.optim import SGD
from torch import Tensor
from libmoon.util.constant import get_agg_func, solution_eps, get_hv_ref
import torch
from tqdm import tqdm
from pymoo.indicators.hv import HV
from matplotlib import pyplot as plt
import numpy as np
from libmoon.model.simple import PFLModel
from torch import nn
criterion = nn.MSELoss()
import os
from libmoon.util.xy_util import pref2angle


def train_pfl_model(folder_name, update_idx, pfl_model, pfl_optimizer, criterion, prefs, y):
    prefs_angle = pref2angle(prefs)
    # prefs_angle = prefs_angle.unsqueeze(1)
    loss_arr = []
    for _ in range(1000):
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

    return pfl_model




class UniformSolver(GradBaseSolver):
    '''
        UniformSolver is a bilevel agg solver.
    '''
    def __init__(self, step_size, max_iter, tol):
        super().__init__(step_size, max_iter, tol)


    def solve(self, problem, x, prefs, args):
        x = Variable(x, requires_grad=True)
        # ref_point = array([2.0, 2.0])
        ind = HV(ref_point = get_hv_ref(args.problem_name))
        hv_arr = []
        y_arr = []
        prefs = Tensor(prefs)
        optimizer = SGD([x], lr=self.step_size)
        agg_func = get_agg_func(args.agg)
        res = {}


        pfl_model = PFLModel(n_obj=problem.n_obj)
        pfl_optimizer = torch.optim.Adam(pfl_model.parameters(), lr=1e-3)

        for _ in range(5):
            for i in tqdm(range(self.max_iter)):
                y = problem.evaluate(x)
                hv_arr.append(ind.do(y.detach().numpy()))
                agg_val = agg_func(y, prefs)
                optimizer.zero_grad()
                torch.sum(agg_val).backward()
                optimizer.step()
                y_arr.append( y.detach().numpy() )
                if 'lbound' in dir(problem):
                    x.data = torch.clamp(x.data, torch.Tensor(problem.lbound) + solution_eps, torch.Tensor(problem.ubound)-solution_eps)

            # Training the PFL model on the solutions.
            pfl_model = train_pfl_model(pfl_model, pfl_optimizer, criterion, y.detach(), prefs)

            # Update the prefs using the PFL model.
            prefs_var = Variable(prefs, requires_grad=True)
            prefs_optimizer = SGD([prefs_var], lr=1e-4)


            mms_arr = []
            for _ in range(1000):
                y_pred = pfl_model(prefs_var)
                mms_val = compute_MMS(y_pred)

                prefs_optimizer.zero_grad()
                mms_val.backward()
                prefs_optimizer.step()
                prefs_var.data = torch.clamp(prefs_var.data, 0, 1)
                prefs_var.data = prefs_var.data / torch.sum(prefs_var.data, axis=1, keepdim=True)
                mms_arr.append(mms_val.item())


            prefs = prefs_var.data
            use_mms_plt = True
            if use_mms_plt:
                plt.plot(mms_arr)
                plt.xlabel('Iteration')
                plt.ylabel('MMS')
                plt.title('MMS curve')
                plt.show()
                assert False
            use_plt = False
            if use_plt:
                prefs_np = prefs.detach().numpy()
                prefs_np_l2 = prefs_np / np.linalg.norm(prefs_np, axis=1, keepdims=True)
                for pref in prefs_np_l2:
                    plt.plot([0, pref[0]], [0, pref[1]], label='Preference', color='grey', linestyle='dashed')
                y_np = y.detach().numpy()
                plt.scatter(y_np[:,0], y_np[:,1], label='Solutions')
                plt.show()
                assert False




        res['x'] = x.detach().numpy()
        res['y'] = y.detach().numpy()
        res['hv_arr'] = hv_arr
        res['y_arr'] = y_arr
        return res