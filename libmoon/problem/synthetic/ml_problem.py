from libmoon.problem.synthetic.mop import BaseMOP
import torch
import numpy as np
from torch import Tensor

'''
    Adpats from: Hu et al. Revisiting Scalarization in Multi-Task Learning: A Theoretical Perspective. NeurIPS 2023. 
'''
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
        # q, K ?
        # X.shape: n*p. n samples, p features.
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
        # y_hat = W @ self.X.T
        # shape: (n_prob, self.n)
        # loss_prob = []
        # for prob_idx in range( n_prob ):
        #     loss = torch.sum(torch.pow(self.Y - y_hat[prob_idx,:].unsqueeze(1), 2), axis=0)
        #     loss_prob.append(loss)
        # loss_prob = torch.stack(loss_prob, axis=0)
        # return loss_prob

    def _evaluate_numpy(self, x: np.ndarray):
        pass


if __name__ == '__main__':
    lr_problem = LinearRegreesion()
    p = 5
    W = torch.squeeze(torch.rand(p))
    loss = lr_problem.evaluate( W )
    print(loss)
    # print(lr_problem.X)