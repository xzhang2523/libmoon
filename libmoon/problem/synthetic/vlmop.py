import matplotlib.pyplot as plt
import torch
import numpy as np

from libmoon.problem.synthetic.mop import mop



class VLMOP1(mop):
    def __init__( self, n_var=10, n_obj=2, lbound=-np.ones(10), ubound=np.ones(10) ):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'VLMOP1'

    def _evaluate_torch(self, x):
        f1 = 1 - torch.exp(-1 * torch.sum((x - 1 / np.sqrt(self.n_var))**2, dim=1))
        f2 = 1 - torch.exp(-1 * torch.sum((x + 1 / np.sqrt(self.n_var))**2, dim=1))
        return torch.stack((f1, f2), dim=1)

    def _evaluate_numpy(self, x):
        f1 = 1 - np.exp(-1 * np.sum((x - 1 / np.sqrt(self.n_var)) ** 2, axis=1 ) )
        f2 = 1 - np.exp(-1 * np.sum((x + 1 / np.sqrt(self.n_var)) ** 2, axis=1 ) )
        return np.stack((f1, f2), axis=1)


    def get_pf(self):
        x = torch.linspace(-1 / np.sqrt(self.n_var), 1 / np.sqrt(self.n_var), 100)
        x = torch.tile(x.unsqueeze(1), (1, self.n_var))
        with torch.no_grad():
            return self._evaluate_torch(x).numpy()




class VLMOP2(mop):
    def __init__(self, n_var=10, n_obj=2, lbound=-np.ones(10), ubound=np.ones(10)):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'VLMOP2'

    def _evaluate_torch(self, x):
        f1 = torch.norm(x - 1 / np.sqrt(self.n_var), dim=1)**2 / 4
        f2 = torch.norm(x + 1 / np.sqrt(self.n_var), dim=1)**2 / 4
        return torch.stack((f1, f2), dim=1)

    def _evaluate_numpy(self, x):
        f1 = np.linalg.norm(x - 1 / np.sqrt(self.n_var), axis=1)**2 / 4
        f2 = np.linalg.norm(x + 1 / np.sqrt(self.n_var), axis=1)**2 / 4
        return np.stack((f1, f2), axis=1)

    def get_pf(self):
        x = torch.linspace(-1 / np.sqrt(self.n_var), 1 / np.sqrt(self.n_var), 100)
        x = torch.tile(x.unsqueeze(1), (1, self.n_var))
        with torch.no_grad():
            return self._evaluate_torch(x).numpy()


if __name__ == '__main__':
    problem = VLMOP2()
    pf = problem.get_pf()
    plt.plot(pf[:, 0], pf[:, 1], '-')
    plt.show()
    print()