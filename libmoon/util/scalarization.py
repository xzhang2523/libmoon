import numpy as np
import torch
from torch import Tensor
'''
    numpy evaluation may have some problems, please check. 
    Here, an interesting thing is that, PBI and COSMOS are not MOO agg functions.
    I.e., the optimas may not be Pareto optimal.
'''
def soft_tche(f_arr, w, mu=1e-1, z=0, normalization=False):
    inner = w * (f_arr - z) / mu
    if type(f_arr) == Tensor:
        val = mu * torch.logsumexp(inner, axis=1)
        return val
    else:
        val = mu * np.log( np.sum(np.exp(inner), axis=1) )
        return val

def soft_mtche(f_arr, w, mu=0.1, z=0, normalization=False):
    return soft_tche(f_arr, 1/w, mu, z, normalization)

def aasf(f_arr, w, z=0, rho=0.1):
    return soft_mtche(f_arr, w, z=z) + rho * ls(f_arr, w, z=z)

def pnorm(f_arr, w, z=0, rho=0.1, p=2):
    # return soft_mtche(f_arr, w, z=z) + rho * ls(f_arr, w, z=z)
    inner = w * (f_arr - z)
    if type(f_arr) == Tensor:
        val = torch.norm(inner, p=p, dim=1)
        # return torch.pow(val, 1/p)
    else:
        val = np.linalg.norm(inner, ord=p, axis=1)
        # return np.power(val, 1/p)
    return val

def tche(f_arr, w, z=0):
    '''
        Tchebycheff scalarization function
        Input:
            f_arr: (n_prob, n_obj)
            w: (n_prob, n_obj).
        Return:
            (n_prob, )
    '''
    if type(f_arr) == Tensor:
        idx = torch.argmax(w * (f_arr - z), axis=1)
        return f_arr[torch.arange(f_arr.shape[0]), idx]
        # return [0]

    elif type(f_arr) == np.ndarray:
        idx = np.argmax(w * (f_arr - z), axis=1)
        return f_arr[np.arange(f_arr.shape[0]), idx]
        # return np.max(w * (f_arr - z), axis=1)
    else:
        raise Exception('type not supported')

def mtche(f_arr, w, z=0):
    return tche(f_arr, 1/w, z)

def ls(f_arr, w, z=0):
    if type(f_arr) == Tensor:
        return torch.sum(w * (f_arr - z), axis=1)
    elif type(f_arr) == np.ndarray:
        return np.sum(w * (f_arr - z), axis=1)
    else:
        raise Exception('type not supported')


def invagg(f_arr, w, z=1):
    elem_arr = (z - f_arr)
    elem_arr = torch.pow(elem_arr, w)
    if type(f_arr) == Tensor:
        return 1 / torch.prod(elem_arr, axis=1)
    else:
        return 1 / np.prod(elem_arr, axis=1)


def pbi(f_arr, w, coeff=1, z=0):
    if type(f_arr) == Tensor:
        w0 = w / torch.norm(w, dim=1).unsqueeze(1)
        d1 = torch.sum(f_arr * w0, axis=1)
        d2 = torch.norm(f_arr - d1.unsqueeze(1) * w0, dim=1)
        return d1 + coeff * d2
    else:
        w0 = w / np.linalg.norm(w)
        d1 = np.sum(f_arr * w0, axis=1)
        d2 = np.linalg.norm(f_arr - np.outer(d1, w0), axis=1)
        return d1 + coeff * d2

def cosmos(f_arr, w, coeff=10, z=0):
    if type(f_arr) == Tensor:
        w0 = w / torch.norm(w, dim=1).unsqueeze(1)

        d1 = torch.sum(f_arr * w0, axis=1)
        d2 = d1 / torch.norm(f_arr, dim=1)
        return d1 - coeff * d2
    else:
        w0 = w / np.linalg.norm(w)
        d1 = np.sum(f_arr * w0, axis=1)
        d2 = d1 / np.linalg.norm(f_arr, axis=1)
        return d1 - coeff * d2



if __name__ == '__main__':
    f_arr = torch.rand(100, 2)
    w = torch.Tensor([1, 1])
    z = torch.rand(100, 2)

    # print(Tche(f_arr, w, z))
    print(pbi(f_arr, w))