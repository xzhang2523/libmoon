import matplotlib.pyplot as plt
import torch
import numpy as np


class VLMOP1:
    def __init__(self):
        self.n_var = 10
        self.n_obj = 2


    def evaluate(self, x):
        f1 = 1 - torch.exp(-1 * torch.sum((x - 1 / np.sqrt(self.n_var))**2, dim=1))
        f2 = 1 - torch.exp(-1 * torch.sum((x + 1 / np.sqrt(self.n_var))**2, dim=1))
        return torch.stack((f1, f2), dim=1)

    def get_pf(self):
        x = torch.linspace(-1 / np.sqrt(self.n_var), 1 / np.sqrt(self.n_var), 100)
        x = torch.tile(x.unsqueeze(1), (1, self.n_var))
        with torch.no_grad():
            return self.evaluate(x).numpy()




class VLMOP2:
    def __init__(self):
        self.n_var = 10
        self.n_obj = 2

    def evaluate(self, x):
        f1 = torch.norm(x - 1 / np.sqrt(self.n_var), dim=1)**2
        f2 = torch.norm(x + 1 / np.sqrt(self.n_var), dim=1)**2

        return torch.stack((f1, f2), dim=1)

    def get_pf(self):
        x = torch.linspace(-1 / np.sqrt(self.n_var), 1 / np.sqrt(self.n_var), 100)
        x = torch.tile(x.unsqueeze(1), (1, self.n_var))
        with torch.no_grad():
            return self.evaluate(x).numpy()




if __name__ == '__main__':
    problem = VLMOP2()
    pf = problem.get_pf()
    plt.plot(pf[:, 0], pf[:, 1], '-')
    plt.show()
    print()