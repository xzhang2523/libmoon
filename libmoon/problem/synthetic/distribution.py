import numpy as np
import torch
from libmoon.problem.synthetic.mop import BaseMOP
from torch import Tensor


class MOKL(BaseMOP):
    def __init__(self, mu_arr=None, Sigma_arr=None):
        self.n_obj = m = len(mu_arr)
        self.mu_arr = mu_arr
        self.Sigma_arr = Sigma_arr
        self.n_var = len(self.mu_arr[0])
        self.problem_name = "MOKL"


    def _evaluate_torch(self, prefs_arr: torch.Tensor):
        # prefs are the coefficients.
        f_arr_all = []
        for prefs in prefs_arr:
            Sigma_inverse_arr_arr = [p * torch.inverse(Sigma_) for p, Sigma_ in zip(prefs, self.Sigma_arr)]
            Sigma = torch.inverse(torch.sum( torch.stack(Sigma_inverse_arr_arr), axis=0))

            mu_arr_arr = [p * torch.inverse(Sigma_) @ mu
                          for p, mu, Sigma_ in zip(prefs, self.mu_arr, self.Sigma_arr)]
            mu = Sigma @ torch.sum( torch.stack(mu_arr_arr), axis=0)
            f_arr = []

            for obj_idx in range(self.n_obj):
                mu_i = self.mu_arr[obj_idx]
                Sigma_i = self.Sigma_arr[obj_idx]
                term1 = torch.log(torch.det(Sigma_i)) - torch.log(torch.det(Sigma))
                term2 = (mu - mu_i) @ torch.inverse(Sigma_i) @ (mu - mu_i)
                term3 = torch.trace(torch.inverse(Sigma_i) @ Sigma)
                fi = 0.5 * (term1 + term2 + term3 - self.n_var)
                f_arr.append(fi)
            f_arr = torch.stack(f_arr)
            f_arr_all.append(f_arr)
        return torch.stack(f_arr_all)


if __name__ == '__main__':

    mu_arr = [Tensor([1,2]), Tensor([2,3])]
    Sigma_arr = [Tensor(np.array([[1, 0.5], [0.5,1]])), Tensor(np.array([[1,0], [0,1]]))]

    problem = MOKL(mu_arr=mu_arr, Sigma_arr=Sigma_arr)
    prefs = torch.Tensor([0.5, 0.5])
    res = problem._evaluate_torch(prefs)
    # print()




