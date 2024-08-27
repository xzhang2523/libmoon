# Chen et al. Efficient Pareto Manifold Learning with Low-Rank Structure. ICML. 2024.
# Zhong et al. Panacea: Pareto Alignment via Preference Adaptation for LLMs. ArXiv. 2024.
import sys
sys.path.append('D:\\pycharm_project\\libmoon')

import torch
from libmoon.problem.synthetic.vlmop import VLMOP1, VLMOP2
from tqdm import tqdm
from libmoon.util.prefs import get_random_prefs, get_uniform_pref
from libmoon.util.constant import get_agg_func
from matplotlib import pyplot as plt
from libmoon.util.constant import save_pickle, plot_loss, plot_fig_2d
import os
from libmoon.model.fair_model import FullyConnected
from libmoon.util.mtl import model_from_dataset, mtl_dim_dict, mtl_setting_dict
from libmoon.util.mtl import get_dataset, get_mtl_prefs
from libmoon.problem.mtl.objectives import from_name
import numpy as np

from torch.nn import functional as F


class MTLPSLLoRAModel(torch.nn.Module):
    def __init__(self, problem_name, step_size=1e-3, batch_size=128, solver_name='agg_mtche'):
        super(MTLPSLLoRAModel, self).__init__()
        self.step_size = step_size
        self.problem_name = problem_name
        self.dataset = get_dataset(self.problem_name)
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.settings = mtl_setting_dict[self.problem_name]
        self.batch_size = batch_size
        self.base_model = model_from_dataset(self.problem_name)
        self.params = list(self.base_model.parameters())
        self.params_shape = [p.shape for p in self.params]
        # [torch.Size([128, 88]), torch.Size([128]), torch.Size([128, 128]), torch.Size([128]), torch.Size([1, 128]),
        #  torch.Size([1])]
        self.free_rank = 30
        # Kaiming Initialization (He Initialization)

        # Scale factor
        scale = 1e-2
        self.A1 = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.params_shape[0][0], self.free_rank)) * scale)
        self.B1 = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.free_rank, self.params_shape[0][1])) * scale)
        self.A2 = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.params_shape[2][0], self.free_rank)) * scale)
        self.B2 = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.free_rank, self.params_shape[2][1])) * scale)

        all_parameters = [self.A1, self.B1, self.A2, self.B2] + list(self.base_model.parameters())
        self.optimizer = torch.optim.SGD(all_parameters, lr=self.step_size)
        self.obj_arr = from_name( self.settings['objectives'], self.dataset.task_names() )
        self.n_obj = 2
        self.agg_name = solver_name.split('_')[-1]

    def set_pref(self, prefs):
        W0 = prefs[0] * self.A1 @ self.B1
        W1 = prefs[0] * self.A2 @ self.B2
        params = list( self.base_model.parameters() )
        params[0].data.add_(W0)
        params[2].data.add_(W1)

    #     RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
    def optimize(self, epoch):
        loss_epoch = []
        for epoch_idx in tqdm(range(epoch)):
            loss_batch = []
            for batch_idx, batch in enumerate(self.train_loader):
                prefs = get_random_prefs(1, self.n_obj).squeeze()
                self.set_pref(prefs)
                logits = self.base_model(batch['data'])
                loss_vec = torch.stack([obj(logits['logits'], **batch) for obj in self.obj_arr])
                loss = get_agg_func(self.agg_name)(torch.atleast_2d(loss_vec), torch.atleast_2d(prefs))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_batch.append(loss.clone().detach().item())
            loss_epoch.append(np.mean(loss_batch))
        return loss_epoch


    def evaluate(self, prefs_batch):
        '''
            Input: prefs_batch: (n_prob, n_obj)
            Output: loss_pref: (n_prob, n_obj)
        '''
        with torch.no_grad():
            loss_pref = []
            for prefs in prefs_batch:
                self.set_pref(prefs)
                loss_batch = []
                for batch_idx, batch in enumerate(self.train_loader):
                    logits = self.base_model(batch['data'])
                    loss_vec = torch.stack( [obj(logits['logits'], **batch) for obj in self.obj_arr] )
                    loss_batch.append( loss_vec.clone().detach().numpy() )
                loss_pref.append( np.mean(loss_batch, axis=0) )
            return np.array(loss_pref)




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
        variable = self.forward(prefs)
        objective = self.problem.evaluate(variable)
        objective_np = objective.clone().detach().numpy()
        variable_np = variable.clone().detach().numpy()
        return objective_np, variable_np


def evaluate_synthetic(problem, model):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--step-size', type=float, default=1e-3)
    parser.add_argument('--n-obj', type=int, default=2)
    parser.add_argument('--n-var', type=int, default=2)
    args = parser.parse_args()
    problem = VLMOP1(n_var=args.n_var, n_obj=args.n_obj)
    psl_model = SimplePSLLoRAModel( n_var=args.n_var, n_obj=args.n_obj, step_size=args.step_size)
    loss_arr = psl_model.optimize(problem, args.epoch)

    uniform_prefs = get_uniform_pref(n_prob=10, n_obj=args.n_obj)
    objective_np, variable_np = psl_model.evaluate(uniform_prefs)
    plt.scatter(objective_np[:, 0], objective_np[:, 1])

    folder_name = os.path.join('D:\\pycharm_project\\libmoon\\Output\\psl_lora', problem.problem_name)
    os.makedirs(folder_name, exist_ok=True)
    res = {}
    res['y'] = objective_np
    res['loss'] = loss_arr
    res['prefs'] = uniform_prefs
    plot_loss(folder_name=folder_name, loss_arr=loss_arr)
    save_pickle(folder_name=folder_name, res=res)
    plot_fig_2d(folder_name=folder_name, loss=objective_np, prefs=uniform_prefs)


if __name__ == '__main__':
    '''
        Evaluate the performance on fairness classification problem.
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--eval-num', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--step-size', type=float, default=1e-3)
    parser.add_argument('--n-obj', type=int, default=2)
    parser.add_argument('--problem-name', type=str, default='adult')
    parser.add_argument('--solver-name', type=str, default='agg_mtche')

    args = parser.parse_args()
    psl_model = MTLPSLLoRAModel(problem_name=args.problem_name, step_size=args.step_size,
                                batch_size=args.batch_size, solver_name=args.solver_name)

    print('Training...')
    history = psl_model.optimize(epoch=args.epoch)
    uniform_prefs = get_uniform_pref(n_prob=args.eval_num, n_obj=psl_model.n_obj)
    print('Evaluating...')
    eval_res = psl_model.evaluate(uniform_prefs)
    folder_name = os.path.join('D:\\pycharm_project\\libmoon\\Output\\psl_lora', args.problem_name)
    os.makedirs(folder_name, exist_ok=True)

    res = {}
    res['y'] = eval_res
    res['loss'] = history
    res['prefs'] = uniform_prefs

    plot_loss(folder_name=folder_name, loss_arr=res['loss'])
    save_pickle(folder_name=folder_name, res=res)
    plot_fig_2d(folder_name=folder_name, loss=res['y'], prefs=uniform_prefs, axis_equal=False)