from libmoon.problem.synthetic.mop import BaseMOP
import numpy as np
import torch


class F1(BaseMOP):
    def __init__(self,
                 n_var: int,
                 n_obj: int=None,
                 lbound: np.ndarray=None,
                 ubound: np.ndarray=None,
                 n_cons: int = 0,
                 ) -> None:

        self.n_dim = n_var
        self.n_obj = 2
        self.lbound = torch.zeros(n_var).float()
        self.ubound = torch.ones(n_var).float()

    def _evaluate_torch(self, x):
        n = x.shape[1]

        sum1 = sum2 =  0.0
        count1 = count2 =  0.0

        for i in range(2,n+1):
            yi = x[:,i-1] - torch.pow(2 * x[:,0] - 1, 2)
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 )))

        objs = torch.stack([f1,f2]).T

        return objs


    def _evaluate_numpy(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0.0
        count1 = count2 = 0.0

        for i in range(2, n + 1):
            yi = x[:, i - 1] - np.power(2 * x[:, 0] - 1, 2)
            yi = yi * yi

            if i % 2 == 0:
                sum2 += yi
                count2 += 1.0
            else:
                sum1 += yi
                count1 += 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - np.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = np.stack([f1, f2]).T

        return objs


if __name__ == '__main__':
    x = np.random.random((100, 30))
    problem = F1(n_var=6)

    y = problem.evaluate(x)
    print(y)
    print(y.shape)
    print()




