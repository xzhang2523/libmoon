from libmoon.problem.synthetic.mop import BaseMOP
import torch
import numpy as np
from torch import Tensor
from torch import nn

'''
    Adpats from: Hu et al. Revisiting Scalarization in Multi-Task Learning:
            A Theoretical Perspective. NeurIPS 2023. 
'''

class VirationalInference():
    def __init__(self):
        pass

class DivergenceMacthing(BaseMOP):
    def __init__(self, n_var=1, n_obj=2):
        super().__init__(n_var=n_var, n_obj=n_obj, lbound=None, ubound=None)
        self.dist1 = torch.distributions.Normal(0, 1)
        self.dist2 = torch.distributions.Normal(1, 1)
        self.problem_name = 'divergence'

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
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.mu1 = torch.zeros(n_var)
        self.Sigma1 = torch.eye(n_var)

        self.mu2 = torch.ones(n_var)
        self.Sigma2 = torch.eye(n_var)

        self.dist1 = torch.distributions.multivariate_normal.MultivariateNormal(mu1, Sigma1)
        self.dist2 = torch.distributions.multivariate_normal.MultivariateNormal(mu2, Sigma2)
        self.problem_name = 'modm'


    def _evaluate_torch(self, x):
        Sigma = torch.eye(self.n_var)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(x, Sigma)

        kl1 = torch.distributions.kl.kl_divergence(self.dist1, dist)

        # result_arr = []
        # for idx in range(n_prob):
        #     dist = torch.distributions.Normal(x[idx], 1)
        #     kl1 = torch.distributions.kl.kl_divergence(self.dist1, dist)
        #     kl2 = torch.distributions.kl.kl_divergence(self.dist2, dist)
        #     result = torch.stack((kl1, kl2), dim=1)
        #     result_arr.append(result.squeeze())
        # res = torch.stack(result_arr)
        # return res

    def _get_pf(self, n_points: int = 100):
        mu_arr = np.linspace(0, 1, n_points)
        div_arr = []
        for mu in mu_arr:
            dist = torch.distributions.Normal(mu, 1)
            kl_div1 = torch.distributions.kl.kl_divergence(self.dist1, dist)
            kl_div2 = torch.distributions.kl.kl_divergence(self.dist2, dist)
            div_arr.append([float(kl_div1.numpy()), float(kl_div2.numpy())])
        return np.array(div_arr)



class LinearRegreesion(BaseMOP):
    def __init__(self, n_var=5, n_obj=2):
        lbound = np.zeros(n_var)
        ubound = np.ones(n_var)
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound)
        # k : n_obj. k regression tasks.
        # n is the sample numble
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
            # Z.shape: (n, q)
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
        # super().__init__(n_var=n_var,
        #                  n_obj=n_obj,
        #                  lbound=lbound,
        #                  ubound=ubound)

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
    # problem = DivergenceMacthing()
    # pf = problem._get_pf()
    # plt.plot(pf[:, 0], pf[:, 1], '-')
    # plt.show()

    n_var = 5
    problem = MODM(n_var=n_var)
    x = torch.rand(2, n_var)
    print(problem.evaluate(x))





