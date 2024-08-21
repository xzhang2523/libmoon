import numpy as np

from .scalarization import ls, mtche, tche, pbi, cosmos, invagg, soft_tche, soft_mtche
from libmoon.util_global.problems import get_problem

import os
from numpy import array
import torch

FONT_SIZE = 20
FONT_SIZE_2D = 20
FONT_SIZE_3D = 20

solution_eps = 1e-5

nadir_point_dict = {
    'adult': array([0.6, 0.12]),
    'compass': array([0.52, 0.34]),
    'credit': array([0.52, 0.016]),
    'mnist': array([0.36, 0.36]),
    'fashion': array([0.6, 0.6]),
    'fmnist': array([0.6, 0.6]),
}

ideal_point_dict = {
    'adult': array([0.3, 0.01]),
    'compass': array([0.04, 0.04]),
    'credit': array([0.32, 0.002]),
    'mnist': array([0.2, 0.2]),
    'fashion': array([0.4, 0.4]),
    'fmnist': array([0.2, 0.4]),
}

def normalize_vec(x, problem ):
    ideal = ideal_point_dict[problem]
    nadir = nadir_point_dict[problem]
    if type(x) == torch.Tensor:
        return (x - torch.Tensor(ideal)) / ( torch.Tensor(nadir) - torch.Tensor(ideal))
    else:
        return (x - ideal) / (nadir - ideal)

def get_agg_func(agg, cosmos_hp=8.0):
    if agg == 'ls':
        return ls
    elif agg == 'mtche':
        return mtche
    elif agg == 'tche':
        return tche
    elif agg == 'pbi':
        return pbi
    elif agg == 'cosmos':
        cosmos_func = lambda f_arr, w, z=0: cosmos(f_arr, w, cosmos_hp, z)
        return cosmos_func
    elif agg == 'invagg':
        return invagg
    elif agg == 'softtche':
        return soft_tche
    elif agg == 'softmtche':
        return soft_mtche
    else:
        raise ValueError('Invalid agg function')


all_indicators = ['hv', 'igd', 'spacing', 'sparsity', 'uniform', 'soft uniform', 'maxgd']
oracle_indicators = ['hv', 'spacing', 'sparsity', 'uniform', 'soft uniform']

max_indicators = {'hv', 'uniform', 'soft uniform'}

beautiful_ind_dict = {
    'spacing': 'Spacing',
    'sparsity': 'Sparsity',
    'inner_product': 'IP',
    'cross_angle': 'Cross Angle',
    'pbi' : 'PBI',
    'hv': 'HV',
    'lmin': 'Lmin',
    'soft_lmin': 'Soft Lmin',
    'maxgd': 'MaxGD',
    'igd': 'IGD',
}

min_key_array = ['spacing', 'sparsity', 'pbi', 'cross_angle', 'inner_product'  ]

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
        return np.ones(n_obj) * 1.2

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

# color_arr = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'grey', 'black', 'yellow'] * 100

beautiful_dict = {
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
    'mgdaub': 'MGDA-UB',
    'pmgda': 'PMGDA',
    'agg-ls': 'AGG-LS',
    'uniform': 'UMOD',
    'agg_ls': 'Agg-LS',
    'agg_tche': 'Agg-Tche',
    'agg_softtche': 'Agg-SoftTche',
    'agg_mtche': 'Agg-mTche',
    'agg_cosmos': 'Agg-COSMOS',
    'agg_pbi': 'Agg-PBI',
    'hvgrad': 'HVGrad',
    'pmtl': 'PMTL',
    'random' : 'Random',
    'moosvgd' : 'MOO-SVGD'
}

all_mtd_arr = ['epo', 'pmgda', 'agg_ls', 'agg_tche', 'agg_mtche', 'agg_cosmos', 'agg_pbi', 'hvgrad', 'uniform', 'pmtl']

import seaborn as sns
color_arr = sns.color_palette() + ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'grey', 'black', 'yellow']
marker_arr = ['o', 'x', '+', 'v', 's', 'p', 'D', 'h', '8', '1']


plt_2d_pickle_size = 16
plt_2d_marker_size = 10
plt_2d_label_size = 20


