import matplotlib.pyplot as plt
from torch.utils import data
from libmoon.problem.mtl.objectives import from_name

from libmoon.util.mtl import model_from_dataset, mtl_dim_dict, mtl_setting_dict
from libmoon.util.mtl import get_dataset


import torch
import numpy as np
from tqdm import tqdm
from libmoon.util.constant import get_agg_func, root_name
from libmoon.util.gradient import calc_gradients_mtl, flatten_grads
from libmoon.model.hypernet import HyperNet, LeNetTarget
from libmoon.util.prefs import get_random_prefs, get_uniform_pref
from libmoon.util.network import numel


class GradBasePSLMTLSolver:
    def __init__(self, problem_name, batch_size, step_size, epoch, device, solver_name):
        self.step_size = step_size
        self.problem_name = problem_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device
        self.solver_name = solver_name
        train_dataset = get_dataset(self.problem_name, type='train')
        test_dataset = get_dataset(self.problem_name, type='test')
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,
                                                       num_workers=0)
        # For hypernetwork model, we have the hypernet and target network.
        self.hnet = HyperNet(kernel_size=(9, 5)).to(self.device)
        self.net = LeNetTarget(kernel_size=(9, 5)).to(self.device)

        num_param_hnet, num_param_net = numel(self.hnet), numel(self.net)
        print('Number of parameters in hnet: {:.2f}M, net: {:.2f}K'.format(num_param_hnet/1e6, num_param_net/1e3))

        self.optimizer = torch.optim.Adam(self.hnet.parameters(), lr=self.step_size)
        self.dataset = get_dataset(self.problem_name)
        self.settings = mtl_setting_dict[self.problem_name]
        self.obj_arr = from_name( self.settings['objectives'], self.dataset.task_names() )
        self.is_agg = self.solver_name.startswith('agg')
        self.agg_name = self.solver_name.split('_')[-1] if self.is_agg else None

    def solve(self):
        loss_epoch = []
        for epoch_idx in tqdm(range(self.epoch)):
            loss_batch = []
            for batch_idx, batch in enumerate(self.train_loader):
                ray = torch.from_numpy(
                    np.random.dirichlet((1, 1), 1).astype(np.float32).flatten()
                ).to(self.device)  # ray.shape (1,2), everytime, only sample one preference.
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                # batch['data'].shape: (batch_size, 1, 36, 36)
                self.hnet.train()
                self.optimizer.zero_grad()
                weights = self.hnet(ray)      # len(weights) = 10
                num_target = numel(weights)    # numel(weights) = 31910
                logits_l, logits_r = self.net(batch['data'], weights)
                logits_array = [logits_l, logits_r]

                loss_vec = torch.stack([obj(logits, **batch) for (logits,obj) in zip(logits_array, self.obj_arr)])
                if self.is_agg:
                    loss_vec = torch.atleast_2d(loss_vec)
                    ray = torch.atleast_2d(ray)
                    loss = torch.sum( get_agg_func(self.agg_name)(loss_vec, ray) )
                elif self.solver_name in ['epo', 'pmgda']:
                    # Here, we also need the Jacobian matrix.
                    grads = []
                    for i, loss in enumerate(loss_vec):
                        g = torch.autograd.grad(loss, weights, retain_graph=True)
                        flat_grad = self._flattening(g)
                        grads.append(flat_grad.data)
                    print()

                else:
                    assert False, 'Unknown solver_name'
                loss_batch.append( loss.cpu().detach().numpy() )
                loss.backward()
                self.optimizer.step()
            loss_epoch.append( np.mean(np.array(loss_batch)) )
        res = {'train_loss': loss_epoch}
        return res

    def eval(self, n_eval):
        uniform_prefs = torch.Tensor(get_uniform_pref(n_eval)).to(self.device)
        loss_pref = []
        for pref_idx, pref in tqdm(enumerate(uniform_prefs)):
            loss_batch = []
            for batch_idx, batch in enumerate(self.train_loader):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                weights = self.hnet(pref)
                logits_l, logits_r = self.net(batch['data'], weights)
                logits_array = [logits_l, logits_r]
                loss_vec = torch.stack([obj(logits, **batch) for logits, obj in zip(logits_array, self.obj_arr)])
                loss_batch.append(loss_vec.cpu().detach().numpy())
            loss_pref.append(np.mean(np.array(loss_batch), axis=0))

        res = {}
        res['eval_loss'] = np.array(loss_pref)
        res['prefs'] = uniform_prefs.cpu().detach().numpy()
        return res



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
                    # model input: data
                    logits = self.model_arr[pref_idx](batch['data'])
                    loss_vec = torch.stack( [obj(logits['logits'], **batch) for obj in self.obj_arr] )
                    loss_mat[pref_idx] = loss_vec
                    if not self.is_agg:
                        Jacobian_ = calc_gradients_mtl(batch['data'], batch, self.model_arr[pref_idx], self.obj_arr)
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
                    elif self.core_solver.core_name in ['PMTLCore', 'MOOSVGDCore', 'GradHVCore']:
                        if self.core_solver.core_name == 'GradHVCore':
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
    print()
