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

class F7(BaseMOP):
    def __init__(self):
        self.n_dim = 2
        self.n_obj = 2
        self.lbound = torch.tensor([-1, -1])
        self.ubound = torch.tensor([1, 1])

    def forward(self, theta):
        def h1(theta):
            return torch.log(torch.max(torch.abs(0.5 * (-theta[:,0]) - 7) - torch.tanh(-theta[:,1]), torch.tensor(0.000005))) + 6

        def h2(theta):
            return torch.log(torch.max(torch.abs(0.5 * (-theta[:,0]) + 3) - torch.tanh(-theta[:,1]) + 2, torch.tensor(0.000005))) + 6

        def g1(theta):
            return (((-theta[:,0] + 7) ** 2 + 0.1 * ((-theta[:,1]) ** 2 - 8) ** 2) / 10) - 20

        def g2(theta):
            return (((-theta[:,0] - 7) ** 2 + 0.1 * ((-theta[:,1]) ** 2 - 8) ** 2) / 10) - 20

        def c1(theta):
            return torch.max(torch.tanh(0.5 * theta[:,0]), torch.tensor(0.0))

        def c2(theta):
            return torch.max(torch.tanh(-0.5 * theta[:,1]), torch.tensor(0.0))

        def f1(theta):
            return c1(theta) * h1(theta) + c2(theta) * g1(theta)

        def f2(theta):
            return c1(theta) * h2(theta) + c2(theta) * g2(theta)

        return torch.stack([f1(theta), f2(theta)]).T



if __name__ == '__main__':


    x = np.random.random((100, 30))
    problem = F1(n_var=6)

    y = problem.evaluate(x)
    print(y)
    print(y.shape)
    print()




