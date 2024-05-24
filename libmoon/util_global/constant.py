import numpy as np

from .scalarization import ls, mtche, tche, pbi, cosmos, invagg, soft_tche, soft_mtche
from libmoon.util_global.problems import get_problem


import os
from numpy import array
import torch

FONT_SIZE = 20
solution_eps = 1e-5

nadir_point_dict = {
    'adult': array([0.6, 0.12]),
    'compass': array([0.52, 0.35]),
    'credit': array([0.52, 0.015]),
}

ideal_point_dict = {
    'adult': array([0.3, 0.01]),
    'compass': array([0.00, 0.00]),
    'credit': array([0.37, 0.00]),
}

def normalize_vec(x, problem ):
    ideal = ideal_point_dict[problem]
    nadir = nadir_point_dict[problem]
    if type(x) == torch.Tensor:
        return (x - torch.Tensor(ideal)) / ( torch.Tensor(nadir) - torch.Tensor(ideal))
    else:
        return (x - ideal) / (nadir - ideal)

agg_dict = {
    'ls' : ls,
    'mtche' : mtche,
    'tche' : tche,
    'pbi' : pbi,
    'cosmos' : cosmos,
    'invagg' : invagg,
    'softtche' : soft_tche,
    'softmtche': soft_mtche,
}

all_indicators = ['hv', 'igd', 'spacing', 'sparsity', 'uniform', 'soft uniform', 'maxgd']
oracle_indicators = ['hv', 'spacing', 'sparsity', 'uniform', 'soft uniform']


max_indicators = {'hv', 'uniform', 'soft uniform'}

beautiful_ind_dict = {
    'hv': 'HV',
    'igd': 'IGD',
    'spacing': 'Spacing',
    'sparsity': 'Sparsity',
    'uniform': 'Uniform',
    'soft uniform': 'SUniform',
    'maxgd': 'MaxGD',
}

scale_dict = {
    'hv': 1,
    'igd': 100,
    'maxgd': 10,
    'spacing': 100,
    'sparsity': 100,
    'uniform': 10,
    'soft uniform': 10,
}



def get_hv_ref(problem_name):
    # return nadir_point_dict[problem_name] + 1e-2
    hv_ref_dict = {
        'VLMOP1': array([1.0, 1.0]),
        'adult': array([2.0, 2.0]),
        'VLMOP2': array([4.0, 4.0]),
        'MAF1': array([2.0, 2.0, 2.0]),
        'mnist': array([3.0, 3.0]),
        'fmnist': array([3.0, 3.0]),
    }

    if problem_name in hv_ref_dict:
        return hv_ref_dict[problem_name]
    else:
        problem = get_problem(problem_name)
        n_obj = problem.n_obj
        return np.ones(n_obj) * 2.0




def get_hv_ref_dict(problem_name):
    if problem_name.startswith('ZDT'):
        ref = array([1.0, 1.0])
    else:
        ref = nadir_point_dict[problem_name]
    return ref + 1e-2

root_name = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def is_pref_based(mtd):
    if mtd in ['epo', 'mgda', 'agg', 'pmgda']:
        return True
    else:
        return False


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('cuda is available')
    else:
        device = torch.device("cpu")
        print('cuda is not available')
    return device

def get_param_num(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


color_arr = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'grey', 'black', 'yellow']

beautiful_dict = {
    # For MOEA use.
    # ['MOEADURAW', 'MCEAD', 'LMPFE', 'nsga3', 'sms', 'moead', 'adjust']
    'MOEADURAW' : 'MOEA/D-URAW',
    'MCEAD' : 'MCEA/D',
    'MOEADUR' : 'MOEA/D-UR',
    'LMPFE' : 'LMPFE',
    'DEAGNG': 'DEA-GNG',
    'IMMOEA': 'IM-MOEA/D',
    'nsga3' : 'NSGA-III',
    'sms' : 'SMS-EMOA',
    'moead' : 'MOEA/D',
    'adjust' : 'UMOEA/D',
    'Subset' : 'Subset',
    'awa': 'AWA',

    # For Gradient use.
    'epo': 'EPO',
    'mgda': 'MGDA',
    'pmgda': 'PMGDA',
    'agg-ls': 'AGG-LS',
    'uniform': 'UMOD',
    'agg_ls': 'Agg-LS',
    'agg_tche': 'Agg-Tche',
    'agg_pbi': 'Agg-PBI',
    'hvgrad': 'HVGrad',
    'pmtl': 'PMTL',
}


color_arr = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'grey', 'black', 'yellow']
marker_arr = ['o', 'x', '+', 'v', 's', 'p', 'D', 'h', '8', '1']

