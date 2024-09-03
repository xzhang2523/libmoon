import matplotlib.pyplot as plt
from torch.utils import data
from libmoon.problem.mtl.objectives import from_name
from libmoon.problem.mtl.model_utils import model_from_dataset, dim_dict
from libmoon.problem.mtl.settings import adult_setting, credit_setting, compass_setting, mnist_setting, fashion_setting, fmnist_setting
import argparse
import torch
import numpy as np
from tqdm import tqdm
from libmoon.util_global.constant import get_agg_func, normalize_vec
from libmoon.util_global.grad_util import calc_gradients, flatten_grads

import os
from libmoon.util_global.constant import root_name
from libmoon.util.mtl import get_dataset
from libmoon.solver.gradient.methods.bk.core_solver_bk import CorePMGDA
from libmoon.solver.gradient.methods.pmgda_core import get_nn_pmgda_componets
from libmoon.solver.gradient.methods.bk.core_solver_bk import CoreEPO



class MTL_Solver:
    def __init__(self, n_prob, batch_size, lr, epoch, solver, dataset_name,
                 architecture, obj_normalization, agg, seed, h_tol=5e-3, sigma=0.8):
        print('Batch size: {}'.format(batch_size))
        print('Learning rate: {}'.format(lr))
        print('Sigma: {}'.format(sigma))
        self.solver = solver
        self.agg = agg
        self.seed = seed
        self.lr = lr
        self.epoch = epoch
        self.dataset_name = dataset_name
        self.obj_normalization = obj_normalization
        self.architecture = architecture
        self.n_prob = n_prob
        self.batch_size = batch_size
        self.dataset = get_dataset( self.dataset_name )
        self.h_tol = h_tol
        self.sigma = sigma

        self.setting_dict = {
            'adult': adult_setting,
            'credit': credit_setting,
            'compass': compass_setting,
            'mnist' : mnist_setting,
            'fashion' : fashion_setting,
            'fmnist' : fmnist_setting,
        }

        self.settings = self.setting_dict[ self.dataset_name ]
        self.trainloader = data.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=0)
        self.obj_arr = from_name( self.settings['objectives'], self.dataset.task_names() )
        self.model_arr = [model_from_dataset(self.dataset_name, self.architecture, dim=dim_dict[self.dataset_name])
                          for _ in range( n_prob )]
        self.optimizer_arr = [ torch.optim.Adam(model.parameters(), lr=self.lr)
                               for model in self.model_arr ]
        self.update_counter = 0


    def solve(self, pref_mat):
        if type(pref_mat) == np.ndarray:
            pref_mat = torch.Tensor(pref_mat)
        epoch_loss_pref = []
        for _ in tqdm( range(self.epoch) ):
            loss_batch = []
            for b, batch in enumerate(self.trainloader ):
                loss_mat = [0] * len(pref_mat)
                for pref_idx, pref in enumerate(pref_mat):

                    logits = self.model_arr[pref_idx](batch)
                    batch.update(logits)
                    loss_vec = [0] * 2
                    for idx, obj in enumerate(self.obj_arr):
                        loss_vec[idx] = obj(**batch)

                    loss_vec = torch.stack(loss_vec)
                    if self.obj_normalization:
                        loss_vec = normalize_vec( loss_vec, problem=self.dataset_name )

                    if self.solver.start_with('agg'):
                        agg_func = get_agg_func(self.agg, self.co)
                        scalar_loss = torch.squeeze(agg_func(loss_vec.unsqueeze(0), pref.unsqueeze(0)))
                    elif self.solver == 'agg':
                        gradients, obj_values = calc_gradients(batch, self.model_arr[pref_idx], self.obj_arr)
                        Jacobian = torch.stack([flatten_grads(gradients[idx]) for idx in range(2)])
                        if self.solver == 'epo':
                            core_epo = CoreEPO(pref)
                            alpha = torch.Tensor(core_epo.get_alpha(Jacobian, loss_vec))
                        elif self.solver == 'mgda':
                            from libmoon.solver.gradient.methods.bk.core_solver_bk import CoreMGDA
                            core_mgda = CoreMGDA()

                            alpha = torch.Tensor(core_mgda.get_alpha(Jacobian))
                        elif self.solver == 'pmgda':
                            h_val, Jhf = get_nn_pmgda_componets(loss_vec, pref)
                            grad_h = torch.Tensor(Jhf) @ Jacobian
                            core_pmgda = CorePMGDA()
                            alpha = core_pmgda.get_alpha(Jacobian, grad_h, h_val, self.h_tol, self.sigma,
                                                         return_coeff=True, Jhf=Jhf)
                            alpha = torch.Tensor(alpha)
                        scalar_loss = torch.sum(alpha * loss_vec)
                    elif self.solver == 'uniform':
                        agg_func = get_agg_func('mtche')
                        scalar_loss = torch.squeeze(agg_func(loss_vec.unsqueeze(0), pref.unsqueeze(0)))
                    else:
                        scalar_loss = torch.sum(alpha * loss_vec)

                    self.optimizer_arr[pref_idx].zero_grad()
                    scalar_loss.backward()
                    self.optimizer_arr[pref_idx].step()
                    self.update_counter += 1
                    loss_mat[pref_idx] = loss_vec.detach().cpu().numpy()
                loss_mat = np.array(loss_mat)
                loss_batch.append(loss_mat)



            loss_batch = np.array(loss_batch)
            epoch_loss_pref.append(np.mean(loss_batch, axis=0))
        epoch_loss_pref = np.array(epoch_loss_pref)
        loss_pref_final = epoch_loss_pref[-1, :, :]

        return loss_pref_final, epoch_loss_pref



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

    output_folder_name = os.path.join(root_name, 'output', 'mtl', args.task_name, self.dataset_name, '{}'.format(args.seed))
    os.makedirs(output_folder_name, exist_ok=True)
    args.output_folder_name = output_folder_name

    epo_solver = MTL_Solver(dataset_name='adult', n_prob=5)
    loss_pref_final = epo_solver.solve()

    plt.scatter(loss_pref_final[:,0], loss_pref_final[:,1])
    plt.show()
