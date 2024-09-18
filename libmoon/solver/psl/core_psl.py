import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from libmoon.model.simple import SimplePSLModel
from torch.autograd import Variable
from libmoon.util import get_problem
from libmoon.util.constant import get_problem, FONT_SIZE, get_agg_func
from libmoon.util.gradient import get_moo_Jacobian_batch

# D:\pycharm_project\libmoon\libmoon\solver\gradient\methods\core\core_solver.py
from libmoon.solver.gradient.methods.core.core_solver import EPOCore, PMGDACore
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BasePSLSolver:
    def __init__(self, problem, batch_size=256,
                 device='cuda', lr=1e-3, epoch=1000, solver_name='agg_tche', use_es=False):
        self.problem = problem
        self.batch_size = batch_size
        self.device = device
        self.epoch = epoch
        self.model = SimplePSLModel(problem).to(self.device)
        self.use_es = use_es
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.solver_name = solver_name
        self.is_agg = True if self.solver_name.startswith('agg') else False
        self.agg = solver_name.split('_')[1] if self.is_agg else None
        self.n_obj, self.n_var = problem.n_obj, problem.n_var

    def solve(self):
        loss_history = []
        for _ in tqdm(range(self.epoch)):
            prefs = torch.Tensor(np.random.dirichlet(np.ones(self.n_obj),
                                                     self.batch_size)).to(self.device)
            # shape: (batch_size, n_obj)
            xs = self.model(prefs)
            xs_var = Variable(xs.detach(), requires_grad=True)
            # shape: (batch_size, n_var)
            fs = self.problem.evaluate(xs)
            fs_detach = fs.detach()
            fs_var = self.problem.evaluate(xs_var)
            # shape: (batch_size, n_obj)

            # if self.use_es:
            #     # Part 1. Estimating term A.
            #     agg_func = agg_dict[args.agg]
            #     fs_var = Variable(fs, requires_grad=True)
            #     g = agg_func(fs_var, prefs)
            #     loss_g = torch.mean(g)
            #     loss_history.append(loss_g.cpu().detach().numpy())
            #     g.sum().backward()
            #     termA = (fs_var.grad).unsqueeze(1)
            #     # Part 2. Estimating term B.
            #     termB = ES_gradient_estimation_batch(problem, xs.cpu().detach().numpy())
            #     termB = torch.Tensor(termB).to(args.device)
            #     xs = model(prefs)
            #     res = termA @ termB
            #     loss = torch.mean(res @ xs.unsqueeze(2))
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            # else:
            if self.is_agg:
                agg_func = get_agg_func(self.agg)
                g = agg_func(fs, prefs)
                loss = torch.mean(g)
            elif self.solver_name in ['epo', 'pmgda']:
                # Use core_epo_cls
                Jacobian_arr = get_moo_Jacobian_batch(xs_var, fs_var, self.n_obj)
                # shape: (batch_size, n_obj, n_var)
                if self.solver_name == 'epo':
                    core_solver = EPOCore(n_var=self.n_var, prefs=prefs)
                else:
                    core_solver = PMGDACore(n_var=self.n_var, prefs=prefs)

                alpha_arr = torch.stack([core_solver.get_alpha(Jacobian_arr[idx], fs_detach[idx], idx)
                                         for idx in range(self.batch_size)]).to(self.device)
                # shape: (batch_size, n_obj)
                loss = torch.mean(alpha_arr * fs)

            else:
                assert False, '{} Not implemented yet'.format(self.solver_name)

            loss_history.append(loss.cpu().detach().numpy())
            self.optimizer.zero_grad()

            loss.backward()
            # gradient clip here
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

        return self.model, loss_history
