import numpy as np
import torch
import torch.nn.functional as F
from libmoon.problem.synthetic.vlmop import VLMOP1, VLMOP2
from tqdm import tqdm
from libmoon.util.prefs import get_random_prefs, get_uniform_pref
from libmoon.util.constant import get_agg_func
from matplotlib import pyplot as plt


from libmoon.util.constant import save_pickle, plot_loss, plot_fig_2d
import os

class SimplePSLLoRAModel(torch.nn.Module):
    def __init__(self, n_obj, n_var, step_size=1e-3):
        super(SimplePSLLoRAModel, self).__init__()
        # self.n_tasks = n_tasks
        self.lr = step_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.Theta = torch.nn.Parameter(torch.rand(n_obj + 1, n_var))
        self.optimizer = torch.optim.Adam([self.Theta], lr=self.lr)

    def forward(self, prefs):
        gen_theta = torch.cat((torch.ones(len(prefs), 1),
                               torch.tensor(prefs).clone().detach() ), dim=1)
        # gen_theta.shape: (n_problem, n_obj+1)
        variable = torch.matmul(gen_theta, self.Theta)
        return variable

    def optimize(self, problem, epoch):
        loss_arr = []
        loss_history = []
        for epoch_idx in tqdm(range(epoch)):
            # self.optimizer.zero_grad()
            prefs = get_random_prefs(128, self.n_obj)
            variable = self.forward(prefs)
            objective = problem.evaluate(variable)
            agg_func = get_agg_func('mtche')
            agg_val = agg_func(objective, prefs)
            self.optimizer.zero_grad()
            loss = torch.mean(agg_val)
            loss.backward()
            self.optimizer.step()
            loss_arr.append(loss.clone().detach().item())
        return loss_arr


    def evaluate(self, prefs):
        variable = psl_model.forward(prefs)
        objective = problem.evaluate(variable)
        objective_np = objective.clone().detach().numpy()
        variable_np = variable.clone().detach().numpy()
        return objective_np, variable_np


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--step-size', type=float, default=1e-3)
    parser.add_argument('--n-obj', type=int, default=2)
    parser.add_argument('--n-var', type=int, default=2)
    args = parser.parse_args()
    problem = VLMOP2(n_var=args.n_var, n_obj=args.n_obj)
    psl_model = SimplePSLLoRAModel( n_var=args.n_var, n_obj=args.n_obj, step_size=args.step_size)
    loss_arr = psl_model.optimize(problem, args.epoch)

    # plt.figure()
    uniform_prefs = get_uniform_pref(n_prob=10, n_obj=args.n_obj)
    objective_np, variable_np = psl_model.evaluate(uniform_prefs)
    plt.scatter(objective_np[:, 0], objective_np[:, 1])

    folder_name = os.path.join('D:\\pycharm_project\\libmoon\\Output\\psl', problem.problem_name)
    os.makedirs(folder_name, exist_ok=True)

    plot_loss(folder_name=folder_name, loss_arr=loss_arr)
    save_pickle(folder_name=folder_name, res=res)
    plot_fig_2d(folder_name=folder_name, loss=objective_np, prefs=uniform_prefs)


