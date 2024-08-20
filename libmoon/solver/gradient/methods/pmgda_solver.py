from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm

import torch
from numpy import array
from pymoo.indicators.hv import HV
import numpy as np
from libmoon.util_global.constant import solution_eps
from libmoon.solver.gradient.methods.pmgda_core import solve_pmgda, constraint, get_Jhf

from libmoon.util_global.constant import solution_eps, get_hv_ref



class PMGDASolver(GradBaseSolver):
    # The PGMDA paper: http://arxiv.org/abs/2402.09492.
    def __init__(self, problem, step_size, n_iter, tol, sigma, h_tol):
        self.problem = problem
        self.sigma = sigma
        self.h_tol = h_tol
        super().__init__(step_size, n_iter, tol)


    def solve(self, x, prefs):
        # The implement of PMGDA is close to EPO
        n_obj, n_var, n_prob = self.problem.n_obj, self.problem.n_var, len(prefs)

        x = Variable(x, requires_grad=True)
        optimizer = SGD([x], lr=self.step_size)

        ref_point = get_hv_ref(self.problem.problem_name)
        ind = HV(ref_point=ref_point)

        hv_arr = []
        y_arr = []
        x_arr = []


        for i in tqdm(range(self.n_iter)):
            y = self.problem.evaluate(x)
            y_np = y.detach().numpy()

            y_arr.append( y_np )
            x_arr.append( x.detach().numpy() )

            hv_arr.append(
                ind.do( y_np )
            )

            alpha_arr = [0] * n_prob
            for prob_idx in range( n_prob ):
                Jacobian = torch.autograd.functional.jacobian(lambda ph: self.problem.evaluate(ph).squeeze(),
                                                              x[prob_idx].unsqueeze(0))
                Jacobian = torch.squeeze(Jacobian)
                pref = prefs[prob_idx]
                # (Step 2). Get the gradient of the constraint.
                h = constraint( y[prob_idx].unsqueeze(0), pref=pref)
                h.backward(retain_graph=True)
                grad_h = x.grad[prob_idx].detach().clone()
                x.grad.zero_()
                h_val = float(h.detach().clone().numpy())
                Jhf = get_Jhf(y[prob_idx], pref)

                # replace it to mgda loss
                _, alpha = solve_pmgda(Jacobian, grad_h, h_val, self.h_tol, self.sigma, return_coeff=True, Jhf=Jhf)  # combine the gradient information
                alpha_arr[prob_idx] = alpha

            optimizer.zero_grad()
            alpha_arr = torch.Tensor(np.array(alpha_arr))
            torch.sum(alpha_arr * y).backward()
            optimizer.step()

            if 'lbound' in dir(self.problem):
                x.data = torch.clamp(x.data, torch.Tensor(self.problem.lbound) + solution_eps,
                                     torch.Tensor(self.problem.ubound) - solution_eps)

        res = {}
        res['x_opt'] = x.detach().numpy()
        res['y_opt'] = y.detach().numpy()

        res['hv_history'] = np.array(hv_arr)
        res['y_history'] = np.array(y_arr)
        res['x_history'] = np.array(x_arr)

        return res