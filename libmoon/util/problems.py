from ..problem.synthetic import VLMOP1, VLMOP2, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from ..problem.synthetic import MAF1
from ..problem.synthetic.distribution import MOKL
from ..problem.synthetic.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4
from ..problem.synthetic.re_problem import RE21, RE22, RE23, RE24, RE25, RE31, RE37, RE41, RE42
from ..problem.synthetic.ml_problem import LinearRegreesion, MOOGaussian

from torch import Tensor
import numpy as np

def get_problem(problem_name, n_var=10):
    mu_arr = [Tensor([1, 2]), Tensor([2, 3])]
    Sigma_arr = [Tensor(np.array([[1, 0.5], [0.5, 1]])), Tensor(np.array([[1, 0], [0, 1]]))]

    problem_dict = {
        'ZDT1': ZDT1(n_var=n_var),
        'ZDT2': ZDT2(n_var=n_var),
        'ZDT3': ZDT3(n_var=n_var),
        'ZDT4': ZDT4(n_var=n_var),
        'ZDT6': ZDT6(n_var=n_var),
        'DTLZ1': DTLZ1(n_var=n_var),
        'DTLZ2': DTLZ2(n_var=n_var),
        'DTLZ3': DTLZ3(n_var=n_var),
        'DTLZ4': DTLZ4(n_var=n_var),
        'VLMOP1': VLMOP1(n_var=n_var),
        'VLMOP2': VLMOP2(n_var=n_var),
        'MAF1': MAF1(n_var=n_var),
        'RE21': RE21(),
        'RE22': RE22(),
        'RE23': RE23(),
        'RE24': RE24(),
        'RE25': RE25(),
        'RE31': RE31(),
        'RE37': RE37(),
        'RE41': RE41(),
        'RE42': RE42(),
        'regression': LinearRegreesion(),
        'moogaussian': MOOGaussian(),
        'MOKL' : MOKL(mu_arr, Sigma_arr),
    }
    problem_cls = problem_dict[problem_name]
    return problem_cls