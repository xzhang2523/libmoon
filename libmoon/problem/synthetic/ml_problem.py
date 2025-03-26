import matplotlib.pyplot as plt

from libmoon.problem.synthetic.mop import BaseMOP
import torch
import numpy as np
from torch import Tensor
from torch import nn

import argparse
'''
    (1) Regression, adpats from: Hu et al. Revisiting Scalarization in Multi-Task Learning:
            A Theoretical Perspective. NeurIPS 2023.
'''


class VirationalInference():
    def __init__(self):
        pass

class MOOGaussian(BaseMOP):
    def __init__(self, n_var=1, n_obj=2):
        super().__init__(n_var=n_var, n_obj=n_obj, lbound=None, ubound=None)
        self.dist1 = torch.distributions.Normal(0, 1)
        self.dist2 = torch.distributions.Normal(1, 1)
        self.problem_name = 'moogaussian'

    def _evaluate_torch(self, x):
        n_prob = len(x)
        result_arr = []
        for idx in range(n_prob):
            dist = torch.distributions.Normal(x[idx], 1)
            kl1 = torch.distributions.kl.kl_divergence(self.dist1, dist)
            kl2 = torch.distributions.kl.kl_divergence(self.dist2, dist)
            result = torch.stack((kl1, kl2), dim=1)
            result_arr.append(result.squeeze())
        res = torch.stack(result_arr)
        return res

    def _get_pf(self, n_points: int = 100):
        mu_arr = np.linspace(0, 1, n_points)
        div_arr = []
        for mu in mu_arr:
            dist = torch.distributions.Normal(mu, 1)
            kl_div1 = torch.distributions.kl.kl_divergence(self.dist1, dist)
            kl_div2 = torch.distributions.kl.kl_divergence(self.dist2, dist)
            div_arr.append([float(kl_div1.numpy()), float(kl_div2.numpy())])
        return np.array(div_arr)



class MODM(BaseMOP):
    def __init__(self, n_var=5, n_obj=2, mu1=None, Sigma1=None,
                         mu2=None, Sigma2=None):
        '''
            Description:
                MODM problem. KL divergence matching between two multi-variate Gaussian distributions.
        '''
        super().__init__(n_var=n_var, n_obj=n_obj)

        if mu1 is None:
            mu1 = torch.zeros(n_var)
            Sigma1 = torch.eye(n_var)
        if mu2 is None:
            mu2 = torch.ones(n_var)
            Sigma2 = torch.eye(n_var)
        self.mu1 = mu1
        self.Sigma1 = Sigma1
        self.mu2 = mu2
        self.Sigma2 = Sigma2
        self.dist1 = torch.distributions.MultivariateNormal(self.mu1, self.Sigma1)
        self.dist2 = torch.distributions.MultivariateNormal(self.mu2, self.Sigma2)
        self.problem_name = 'modm'

    def _evaluate_torch(self, x):
        '''
            Input: x: torch.Tensor, shape: (n_prob, n_var)
            Output: torch.Tensor, shape: (n_prob, n_obj)
        '''
        n_prob, n_var = x.shape
        Sigma = torch.eye(self.n_var)

        kl_arr = []
        for prob_idx in range(n_prob):
            x_i = x[prob_idx]
            dist = torch.distributions.MultivariateNormal(x_i, Sigma)
            kl1 = torch.distributions.kl.kl_divergence(dist, self.dist1)
            kl2 = torch.distributions.kl.kl_divergence(dist, self.dist2)
            kl_arr.append(torch.stack([kl1, kl2]))
        return torch.stack(kl_arr)


    def _get_pf(self, n_points: int = 100):
        mu_10 = self.mu1[0].numpy()
        mu_20 = self.mu2[0].numpy()
        mu_idx = np.linspace(mu_10, mu_20, n_points)
        mu = torch.repeat(mu_idx, self.n_var, 1).T
        div_arr = []
        Sigma = torch.eye(self.n_var)
        for mu_i in mu:
            dist = torch.distributions.MultivariateNormal(mu_i, Sigma)
            kl_div1 = torch.distributions.kl.kl_divergence(dist, self.dist1)
            kl_div2 = torch.distributions.kl.kl_divergence(dist, self.dist2)
            div_arr.append([float(kl_div1.numpy()), float(kl_div2.numpy())])
        return np.array(div_arr)



class MOOVAE:
    def __init__(self):
        pass

    def _evaluate_torch(self, x):
        pass


class LinearRegreesion(BaseMOP):
    def __init__(self, n_var=5, n_obj=2):
        lbound = np.zeros(n_var)
        ubound = np.ones(n_var)
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound)
        # k : n_obj. k regression tasks.
        self.problem_name = 'regression'
        self.p = 5
        self.q = n_var // self.p
        self.n_var = n_var
        self.n = 40
        # self.k = n_obj
        np.random.seed(0)
        self.X = Tensor( np.random.rand(self.n, self.p) )
        self.Y = Tensor( np.random.rand(self.n, n_obj) )

    def _evaluate_torch(self, W: torch.Tensor):
        # W.shape: (n_prob, p)
        W_ = W.reshape(-1, self.p, self.q)
        n_prob = W.shape[0]
        loss_prob = []
        for prob_idx in range(n_prob):
            W_idx = W_[prob_idx,:,:]
            Z = (self.X @ W_idx)
            Proj = Z @ torch.linalg.pinv(Z.T @ Z) @ Z.T
            loss_arr = []
            for k in range(self.n_obj):
                loss_arr.append(- self.Y[:, k].T @ Proj @ self.Y[:, k] + self.Y[:, k].T @ self.Y[:, k])
            loss = torch.stack(loss_arr)
            loss_prob.append(loss)
        loss_prob = torch.stack(loss_prob)
        return loss_prob
    def _evaluate_numpy(self, x: np.ndarray):
        pass


class NNRegression(nn.Module):
    '''
    '''
    def __init__(self, n_var=5, n_obj=2):
        super(NNRegression, self).__init__()
        lbound = np.zeros(n_var)
        ubound = np.ones(n_var)
        # k : n_obj. k (m) regression tasks.
        # n is the sample numble.
        # p is the input dimension.
        # q is the task specific layer.
        self.problem_name = 'regression_nn'
        self.p = 5
        self.q = 1
        self.n_var = n_var
        self.n = 40
        self.X = Tensor( np.random.rand(self.n, self.p) )
        self.backbone = nn.Sequential(
            nn.Linear(self.p, 60),
            nn.ReLU(),
            nn.Linear(60, 60),
            nn.ReLU(),
            nn.Linear(60, self.q)
        )
        self.task1 = nn.Sequential(
            nn.Linear(self.q, 60),
            nn.ReLU(),
            nn.Linear(60, 1)
        )
        self.task2 = nn.Sequential(
            nn.Linear(self.q, 60),
            nn.ReLU(),
            nn.Linear(60, 1)
        )

    def generate_data(self):
        mid = self.backbone(self.X)
        self.Y1 = self.task1(mid)
        self.Y2 = self.task2(mid)
        self.Y = torch.cat((self.Y1, self.Y2), dim=1)
        return self.X, self.Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example script')
    parser.add_argument('--n-var', type=int, default=5)
    parser.add_argument('--n-prob', type=int, default=10)
    problem = MOOVAE()
    obj = problem.evaluate(torch.rand(10, 5))