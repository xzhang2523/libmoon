import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from libmoon.solver.psl.model import SimplePSLModel

from torch.autograd import Variable
from libmoon.util_global import get_problem
from libmoon.util_global.constant import get_problem, FONT_SIZE
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class AggPSLSolver:
    def __init__(self, problem, batch_size=256, device='cuda', lr=1e-3, epoch=1000, agg='tche', use_es=False):
        self.problem = problem
        self.batch_size = batch_size
        self.device = device
        self.epoch = epoch
        self.model = SimplePSLModel(problem).to(self.device)

        self.use_es = use_es
        self.agg = agg
        self.optimizer = torch.optim.Adam( self.model.parameters(), lr=lr )


    def solve(self):
        n_obj = self.problem.n_obj
        loss_history = []
        for _ in tqdm( range(self.epoch) ):
            prefs = torch.Tensor(np.random.dirichlet(np.ones(n_obj),
                                                     self.batch_size)).to(self.device)
            xs = self.model(prefs)
            fs = self.problem.evaluate(xs)
            if self.use_es:
                # Part 1. Estimating term A.
                agg_func = agg_dict[args.agg]
                fs_var = Variable(fs, requires_grad=True)
                g = agg_func(fs_var, prefs)
                loss_g = torch.mean(g)
                loss_history.append(loss_g.cpu().detach().numpy())
                g.sum().backward()
                termA = (fs_var.grad).unsqueeze(1)
                # Part 2. Estimating term B.
                termB = ES_gradient_estimation_batch(problem, xs.cpu().detach().numpy())
                termB = torch.Tensor(termB).to(args.device)
                xs = model(prefs)
                res = termA @ termB
                loss = torch.mean(res @ xs.unsqueeze(2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                agg_func = agg_dict[self.agg]
                g = agg_func(fs, prefs)
                loss = torch.mean(g)
                loss_history.append(loss.cpu().detach().numpy())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.model


if __name__ == '__main__':
    from libmoon.util_global import uniform_pref
    from torch import Tensor

    agg='ls'
    problem = get_problem(problem_name='ZDT1', n_var=10)
    solver = AggPSLSolver(problem, batch_size=128, device='cuda', lr=1e-4, epoch=1000, agg=agg, use_es=False)
    model = solver.solve()

    prefs = uniform_pref(n_prob=100, n_obj=problem.n_obj, clip_eps=1e-2)
    eval_y_np = problem.evaluate( model( Tensor(prefs).cuda() ) ).cpu().detach().numpy()

    prefs_scale = prefs * 0.4
    for pref, y in zip(prefs_scale, eval_y_np):
        plt.scatter(pref[0], pref[1], color='blue', s=40)
        plt.scatter(y[0], y[1], color='orange', s=40)
        plt.plot([pref[0], y[0]], [pref[1], y[1]], color='tomato', linewidth=0.5, linestyle='--')


    plt.plot(prefs_scale[:, 0], prefs_scale[:, 1], color='blue', linewidth=1, label='Preference')
    plt.plot(eval_y_np[:, 0], eval_y_np[:, 1], color='orange', linewidth=1, label='Objectives')
    plt.xlabel('$f_1$', fontsize=FONT_SIZE)
    plt.ylabel('$f_2$', fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)


    folder_name = 'D:\\pycharm_project\\libmoon\\output\\psl'
    fig_name = os.path.join(folder_name, '{}.pdf'.format(agg))
    plt.savefig(fig_name, bbox_inches='tight')
    print('Figure saved to {}'.format(fig_name))

    plt.show()