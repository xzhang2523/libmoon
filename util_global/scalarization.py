import numpy as np
import torch
from torch import Tensor

'''
    numpy evaluation may have some problems, please check. 
    Here, an interesting thing is that, PBI and COSMOS are not MOO agg functions.
    I.e., the optimas may not be Pareto optimal.
'''

def tche(f_arr, w, z=0):
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


def pbi(f_arr, w, coeff=5, z=0):

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
        d1 = f_arr @ w0
        d2 = f_arr @ w0 / np.linalg.norm(f_arr, axis=1)
        return d1 - coeff * d2



if __name__ == '__main__':
    f_arr = torch.rand(100, 2)
    w = torch.Tensor([1, 1])
    z = torch.rand(100, 2)

    # print(Tche(f_arr, w, z))
    print(pbi(f_arr, w))