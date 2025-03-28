import torch
from torch import Tensor

def mokl(mu1, mu2, Std1, Std2, pref0):
    # step 1, convert std1 into diag.
    Std1_mtx = torch.diag_embed(Std1)
    Std2_mtx = torch.diag_embed(Std2)
    Sigma_output = []
    for mtx1, mtx2 in zip(Std1_mtx, Std2_mtx):
        mtx = torch.inverse(pref0 * torch.inverse(mtx1) + (1-pref0) * torch.inverse(mtx2))
        Sigma_output.append(mtx)
    Sigma_output = torch.stack(Sigma_output)
    mu_output = []
    for mu1_i, mu2_i, Sigma_i, std1_i, std2_i in zip(mu1, mu2, Sigma_output, Std1_mtx, Std2_mtx):
        mu = Sigma_i @ (pref0 * torch.inverse(std1_i) @ mu1_i + (1-pref0) * torch.inverse(std2_i) @ mu2_i )
        mu_output.append(mu)
    mu_output = torch.stack(mu_output)
    return mu_output, Sigma_output


if __name__ == '__main__':
    print()
