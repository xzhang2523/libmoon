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






class PMGDASolver(GradBaseSolver):
    # The PGMDA paper: http://arxiv.org/abs/2402.09492.

    def __init__(self, step_size, max_iter, tol):
        super().__init__(step_size, max_iter, tol)


    def solve(self, problem, x, prefs, args):
        # The implement of PMGDA is close to EPO

        x = Variable(x, requires_grad=True)

        optimizer = SGD([x], lr=self.step_size)

        ref_point = array([2.0, 2.0])
        ind = HV(ref_point=ref_point)
        hv_arr = []
        y_arr = []

        for i in tqdm(range(self.max_iter)):
            y = problem.evaluate(x)
            y_arr.append(y.detach().numpy())

            alpha_arr = [0] * args.n_prob
            for prob_idx in range(args.n_prob):
                Jacobian = torch.autograd.functional.jacobian(lambda ph: problem.evaluate(ph).squeeze(),
                                                              x[prob_idx].unsqueeze(0))
                Jacobian = torch.squeeze(Jacobian)
                pref = prefs[prob_idx]
                # (Step 2). Get the gradient of the constraint.
                h = constraint( y[prob_idx].unsqueeze(0), pref=pref, args=args )
                h.backward(retain_graph=True)
                grad_h = x.grad[prob_idx].detach().clone()
                x.grad.zero_()
                h_val = float(h.detach().clone().numpy())
                Jhf = get_Jhf(y[prob_idx], pref, args)

                # replace it to mgda loss
                _, alpha = solve_pmgda(Jacobian, grad_h, h_val, args, return_coeff=True, Jhf=Jhf)  # combine the gradient information
                alpha_arr[prob_idx] = alpha

            optimizer.zero_grad()
            alpha_arr = torch.Tensor(np.array(alpha_arr))
            torch.sum(alpha_arr * y).backward()
            optimizer.step()

            if 'lbound' in dir(problem):
                x.data = torch.clamp(x.data, torch.Tensor(problem.lbound) + solution_eps,
                                     torch.Tensor(problem.ubound) - solution_eps)

        res = {}
        res['x'] = x.detach().numpy()
        res['y'] = y.detach().numpy()
        res['hv_arr'] = [0]
        res['y_arr'] = y_arr
        return res





