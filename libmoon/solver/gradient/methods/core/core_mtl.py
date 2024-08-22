import matplotlib.pyplot as plt
from torch.utils import data
from libmoon.problem.mtl.objectives import from_name
from libmoon.util.mtl import model_from_dataset, mtl_dim_dict, mtl_setting_dict

import argparse
import torch
import numpy as np
from tqdm import tqdm
from libmoon.util.constant import get_agg_func, normalize_vec, root_name
from libmoon.util.gradient import calc_gradients_mtl, flatten_grads
import os
from libmoon.util.mtl import get_dataset


class GradBaseMTLSolver:
    def __init__(self, problem_name, batch_size, step_size, epoch, core_solver, prefs):
        self.step_size = step_size
        self.problem_name = problem_name
        self.epoch = epoch
        self.n_prob = len(prefs)
        self.batch_size = batch_size
        self.dataset = get_dataset(self.problem_name)
        self.settings = mtl_setting_dict[self.problem_name]
        self.prefs = prefs
        self.core_solver = core_solver

        train_dataset = get_dataset(self.problem_name, type='train')
        test_dataset = get_dataset(self.problem_name, type='test')
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=0)
        self.obj_arr = from_name( self.settings['objectives'], self.dataset.task_names() )
        self.model_arr = [model_from_dataset(self.problem_name) for _ in range( self.n_prob )]
        self.optimizer_arr = [ torch.optim.Adam(model.parameters(), lr=self.step_size)
                               for model in self.model_arr ]
        self.update_counter = 0
        self.solver_name = core_solver.core_name
        self.is_agg = self.solver_name.startswith('Agg')
        self.agg_name = core_solver.agg_name if self.is_agg else None


    def solve(self):
        prefs = self.prefs
        n_prob = len(prefs)

        loss_history = []
        for epoch_idx in tqdm( range(self.epoch) ):
            loss_mat_batch = []
            for batch_idx, batch in enumerate(self.train_loader):
                # Step 1, get Jacobian_array and fs.
                loss_mat = [0] * n_prob
                Jacobian_array = [0] * n_prob
                for pref_idx, pref in enumerate(self.prefs):
                    logits = self.model_arr[pref_idx](batch)
                    # batch.update(logits)
                    loss_vec = torch.stack( [obj(logits['logits'], **batch) for obj in self.obj_arr] )
                    loss_mat[pref_idx] = loss_vec

                    if not self.is_agg:
                        Jacobian_ = calc_gradients_mtl(batch, self.model_arr[pref_idx], self.obj_arr)
                        Jacobian = torch.stack([flatten_grads(elem) for elem in Jacobian_])
                        Jacobian_array[pref_idx] = Jacobian

                if not self.is_agg:
                    Jacobian_array = torch.stack(Jacobian_array)
                    # shape: (n_prob, n_obj, n_param)

                loss_mat = torch.stack(loss_mat)
                loss_mat_detach = loss_mat.detach()
                loss_mat_np = loss_mat.detach().numpy()
                # shape: (n_prob, n_obj)
                loss_mat_batch.append(loss_mat_np)

                for idx in range(n_prob):
                    self.optimizer_arr[idx].zero_grad()

                # Step 2, get alpha_array
                if self.is_agg:
                    agg_func = get_agg_func(self.agg_name)
                    agg_val = agg_func(loss_mat, torch.Tensor(prefs).to(loss_mat.device))
                    # shape: (n_prob)
                    torch.sum(agg_val).backward()
                else:
                    if self.core_solver.core_name in ['EPOCore', 'MGDAUBCore', 'PMGDACore', 'RandomCore']:
                        alpha_array = torch.stack(
                            [self.core_solver.get_alpha(Jacobian_array[idx], loss_mat_detach[idx], idx) for idx in
                             range(self.n_prob)])
                    elif self.core_solver.core_name in ['PMTLCore', 'MOOSVGDCore', 'HVGradCore']:
                        # assert False, 'Unknown core_name'
                        if self.core_solver.core_name == 'HVGradCore':
                            alpha_array = self.core_solver.get_alpha_array(loss_mat_detach)
                        elif self.core_solver.core_name == 'PMTLCore':
                            alpha_array = self.core_solver.get_alpha_array(Jacobian_array, loss_mat_np, epoch_idx)
                        elif self.core_solver.core_name == 'MOOSVGDCore':
                            alpha_array = self.core_solver.get_alpha_array(Jacobian_array, loss_mat_detach)
                        else:
                            assert False, 'Unknown core_name'
                    else:
                        assert False, 'Unknown core_name'
                    torch.sum(alpha_array * loss_mat).backward()

                for idx in range(n_prob):
                    self.optimizer_arr[idx].step()


            loss_mat_batch_mean = np.mean(np.array(loss_mat_batch), axis=0)
            loss_history.append(loss_mat_batch_mean)


        res = {'loss_history': loss_history,
               'loss' : loss_history[-1]}
        return res



if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is available")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--solver', type=str, default='uniform')

    parser.add_argument('--agg', type=str, default='tche')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-sub', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n-obj', type=int, default=2)

    parser.add_argument('--update-counter', type=int, default=0)
    parser.add_argument('--uniform-update-counter', type=int, default=0)
    parser.add_argument('--uniform-update-iter', type=int, default=2000)
    # For pmgda
    parser.add_argument('--h-eps', type=float, default=5e-3)
    parser.add_argument('--sigma', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    if self.solver == 'agg':
        args.task_name = 'agg_{}'.format(args.agg)
    else:
        args.task_name = self.solver

    output_folder_name = os.path.join(root_name, 'output', 'mtl', args.task_name, self.problem_name, '{}'.format(args.seed))
    os.makedirs(output_folder_name, exist_ok=True)
    args.output_folder_name = output_folder_name

    epo_solver = MTL_Solver(problem_name='adult', n_prob=5)
    loss_pref_final = epo_solver.solve()

    plt.scatter(loss_pref_final[:,0], loss_pref_final[:,1])
    plt.show()
