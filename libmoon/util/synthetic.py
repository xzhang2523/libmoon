import torch
import numpy as np

def synthetic_init(problem, prefs):

    n_prob = len(prefs)
    if 'lbound' in dir(problem):
        if problem.problem_name == 'VLMOP1':
            x0 = torch.rand(n_prob, problem.n_var) * 2 / np.sqrt(problem.n_var) - 1 / np.sqrt(problem.n_var)
        else:
            x0 = torch.rand(n_prob, problem.n_var)
    else:
        x0 = torch.rand(n_prob, problem.n_var ) * 20 - 10

    return x0