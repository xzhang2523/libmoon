from torch.utils import data
from libmoon.problem.mtl.objectives import from_name
from libmoon.problem.mtl.model_utils import model_from_dataset, dim_dict
from libmoon.problem.mtl.settings import adult_setting, credit_setting, compass_setting, mnist_setting, fashion_setting, fmnist_setting
from libmoon.solver.gradient.methods.bk.core_solver_bk import CoreUniform
from libmoon.solver.gradient.methods.bk.core_solver_bk import CoreMGDA


import argparse
import torch
import numpy as np
from tqdm import tqdm
from libmoon.util_global.constant import get_agg_func, normalize_vec
from libmoon.util_global.grad_util import calc_gradients, flatten_grads

from libmoon.util.mtl import get_dataset

from libmoon.solver.gradient.methods.bk.core_solver_bk import CorePMGDA
from libmoon.solver.gradient.methods.pmgda_core import get_nn_pmgda_componets

from libmoon.solver.gradient.methods.bk.core_solver_bk import CoreEPO




class MTL_Set_Solver:
    def __init__(self, folder_name, n_prob, batch_size, lr, epoch, solver, dataset_name,
                 architecture, obj_normalization, agg, seed, device, uniform_update_iter=4000, uniform_pref_update=2000, h_tol=5e-3, sigma=0.8):

        print('Batch size: {}'.format(batch_size))
        print('Learning rate: {}'.format(lr))
        print('Sigma: {}'.format(sigma))
        self.folder_name = folder_name
        self.device = device
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
        self.model_arr = [model_from_dataset(self.dataset_name, self.architecture, dim=dim_dict[self.dataset_name]).to(self.device)
                          for _ in range( n_prob )]
        self.optimizer_arr = [torch.optim.Adam(model.parameters(), lr=self.lr)
                               for model in self.model_arr]
        self.update_counter = 0
        self.uniform_update_iter = uniform_update_iter
        self.uniform_pref_update = uniform_pref_update
        self.uniform_pref_history = []
        self.uniform_loss_history = []
        self.pmtl_warmup_iter_counter = 0
        self.pmtl_warmup_iter = 2000


    def solve(self, pref_mat):
        if type(pref_mat) == np.ndarray:
            pref_mat = torch.Tensor(pref_mat).to(self.device)
        epoch_loss_pref = []

        for _ in tqdm( range(self.epoch) ):
            loss_batch = []
            for b, batch in enumerate(self.trainloader):
                batch = {key: val.to(self.device) for key, val in batch.items()}
                loss_mat = []  # shape: (K, n_obj)
                for k in range(self.n_prob):
                    logits = self.model_arr[k](batch)
                    batch.update(logits)
                    loss_vec = [0] * 2
                    for idx, obj in enumerate( self.obj_arr ):
                        loss_vec[idx] = obj(**batch)
                    loss_vec = torch.stack(loss_vec)
                    loss_mat.append(loss_vec)
                loss_mat = torch.stack(loss_mat)
                assert self.solver in ['hvgrad', 'moosvgd', 'pmtl']
                if self.solver in ['moosvgd', 'pmtl']:
                    Jacobian_arr = []
                    for pref_idx in range(self.n_prob):
                        gradients, obj_values = calc_gradients(batch, self.model_arr[pref_idx], self.obj_arr)
                        Jacobian = torch.stack([flatten_grads(gradients[idx]) for idx in range(2)])
                        Jacobian_arr.append(Jacobian)
                if self.solver == 'hvgrad':
                    from libmoon.solver.gradient.methods.bk.core_solver_bk import CoreHVGrad
                    solver = CoreHVGrad(n_prob=self.n_prob, n_obj=2, dataset_name=self.dataset_name)
                    alpha_mat = solver.get_alpha(loss_mat)
                elif self.solver == 'pmtl':
                    from libmoon.solver.gradient.methods.bk.core_solver_bk import CorePMTL
                    solver = CorePMTL(pref_mat)
                    is_warmup = self.pmtl_warmup_iter_counter < self.pmtl_warmup_iter
                    alpha_mat = solver.get_alpha(Jacobian_arr=Jacobian_arr, loss_mat=loss_mat, is_warmup=is_warmup)
                elif self.solver == 'moosvgd':
                    from libmoon.solver.gradient.methods.bk.core_solver_bk import CoreMOOSVGD
                    solver = CoreMOOSVGD(n_prob=self.n_prob)
                    alpha_mat = solver.get_alpha(Jacobian_arr, loss_mat)
                for k in range(self.n_prob):
                    self.optimizer_arr[k].zero_grad()
                if type(alpha_mat) == np.ndarray:
                    alpha_mat = torch.Tensor(alpha_mat)

                alpha_mat = alpha_mat.to(self.device)
                loss = torch.sum(alpha_mat * loss_mat)
                loss.backward()
                for k in range(self.n_prob):
                    self.optimizer_arr[k].step()
                    self.update_counter += 1
                loss_batch.append(loss_mat.detach().cpu().numpy())
            loss_batch = np.array(loss_batch)
            epoch_loss_pref.append(np.mean(loss_batch, axis=0))
        epoch_loss_pref = np.array(epoch_loss_pref)
        loss_final = epoch_loss_pref[-1, :, :]

        return loss_final, epoch_loss_pref, pref_mat


class MTL_Pref_Solver:
    def __init__(self, folder_name, n_prob, batch_size, lr, epoch, solver, dataset_name,
                 architecture, obj_normalization, agg, seed, device, cosmos_hp=1.0, uniform_update_iter=4000, uniform_pref_update=2000, h_tol=5e-3, sigma=0.8):

        print('Batch size: {}'.format(batch_size))
        print('Learning rate: {}'.format(lr))
        print('Sigma: {}'.format(sigma))
        self.folder_name = folder_name
        self.device = device
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
        self.model_arr = [model_from_dataset(self.dataset_name, self.architecture, dim=dim_dict[self.dataset_name]).to(self.device)
                          for _ in range( n_prob )]

        from libmoon.util_global.constant import get_param_num
        print('Model: {}'.format( get_param_num(self.model_arr[0]) ))

        self.optimizer_arr = [torch.optim.Adam(model.parameters(), lr=self.lr)
                               for model in self.model_arr]

        self.update_counter = 0
        self.uniform_update_iter = uniform_update_iter
        self.uniform_pref_update = uniform_pref_update

        self.uniform_pref_history = []
        self.uniform_loss_history = []
        self.cosmos_hp = cosmos_hp


    def solve(self, pref_mat):
        if type(pref_mat) == np.ndarray:
            pref_mat = torch.Tensor(pref_mat).to(self.device)
        epoch_loss_pref = []

        for _ in tqdm( range(self.epoch) ):
            loss_batch = []
            for b, batch in enumerate(self.trainloader ):
                batch = {key: val.to(self.device) for key, val in batch.items()}
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
                    if self.solver in ['epo', 'mgda', 'pmgda']:
                        gradients, obj_values = calc_gradients(batch, self.model_arr[pref_idx], self.obj_arr)
                        Jacobian = torch.stack( [flatten_grads(gradients[idx]) for idx in range(2)] )
                        if self.solver == 'epo':
                            core_epo = CoreEPO(pref)
                            alpha = torch.Tensor(core_epo.get_alpha(Jacobian, loss_vec))
                        elif self.solver == 'mgda':
                            core_mgda = CoreMGDA()
                            alpha = torch.Tensor(core_mgda.get_alpha(Jacobian))
                        elif self.solver == 'pmgda':
                            h_val, Jhf = get_nn_pmgda_componets(loss_vec, pref)
                            grad_h = Jhf @ Jacobian
                            core_pmgda = CorePMGDA()
                            alpha = core_pmgda.get_alpha(Jacobian, grad_h, h_val, self.h_tol, self.sigma, return_coeff=True, Jhf=Jhf)
                            alpha = torch.Tensor(alpha)

                        alpha = alpha.to(self.device)
                        scalar_loss = torch.sum(alpha * loss_vec)

                    elif self.solver in ['agg', 'uniform']:
                        agg_func = get_agg_func(self.agg, cosmos_hp=self.cosmos_hp)
                        scalar_loss = torch.squeeze(agg_func(loss_vec.unsqueeze(0), pref.unsqueeze(0)))

                    else:
                        assert False, 'Invalid solver'

                    self.optimizer_arr[pref_idx].zero_grad()
                    scalar_loss.backward()
                    self.optimizer_arr[pref_idx].step()
                    self.update_counter += 1
                    loss_mat[pref_idx] = loss_vec.detach().cpu().numpy()

                loss_mat = np.array(loss_mat)
                loss_batch.append(loss_mat)
                if self.solver == 'uniform':
                    uniform_solver = CoreUniform(self.device, folder_name=self.folder_name, dataset_name=self.dataset_name, uniform_pref_update=self.uniform_pref_update)
                    update_idx = self.update_counter // self.uniform_update_iter
                    if self.update_counter % self.uniform_update_iter == 0:
                        self.uniform_pref_history.append( pref_mat )
                        self.uniform_loss_history.append( torch.Tensor(loss_mat).to(self.device) )
                        pref_mat = uniform_solver.update_pref_mat(pref_mat, loss_mat, self.uniform_pref_history, self.uniform_loss_history, update_idx)
            loss_batch = np.array(loss_batch)
            epoch_loss_pref.append(np.mean(loss_batch, axis=0))
        epoch_loss_pref = np.array(epoch_loss_pref)
        loss_final = epoch_loss_pref[-1, :, :]
        return loss_final, epoch_loss_pref, pref_mat


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
