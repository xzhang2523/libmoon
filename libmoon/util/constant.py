import numpy as np
from .scalarization import (ls, mtche, tche, pbi, cosmos, invagg, soft_tche,
                            soft_mtche, aasf, pnorm)
from libmoon.util.problems import get_problem
import os
from numpy import array
import torch


FONT_SIZE = 20
FONT_SIZE_2D = 20
FONT_SIZE_3D = 20
solution_eps = 1e-5
from matplotlib import pyplot as plt

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
    if agg == 'LS':
        return ls
    elif agg == 'AASF':
        return aasf
    elif agg == 'PNorm':
        return pnorm
    elif agg == 'mTche':
        return mtche
    elif agg == 'Tche':
        return tche
    elif agg == 'PBI':
        return pbi
    elif agg == 'COSMOS':
        cosmos_func = lambda f_arr, w, z=0: cosmos(f_arr, w, cosmos_hp, z)
        return cosmos_func
    elif agg == 'invagg':
        return invagg
    elif agg == 'STche':
        return soft_tche
    elif agg == 'SmTche':
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
    'span': 'Span',
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
        'regression': array([8.0, 8.0]),
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
    'epo': 'EPO',
    'mgdaub': 'MGDA-UB',
    'pmgda': 'PMGDA',
    'agg-ls': 'AGG-LS',
    'uniform': 'UMOD',
    'agg_ls': 'Agg-LS',
    'agg_pnorm': 'Agg-pNorm',
    'agg_aasf': 'Agg-AASF',
    'agg_tche': 'Agg-Tche',
    'agg_softtche': 'Agg-SoftTche',
    'agg_softmtche': 'Agg-SoftmTche',
    'agg_mtche': 'Agg-mTche',
    'agg_cosmos': 'Agg-COSMOS',
    'agg_pbi': 'Agg-PBI',
    'hvgrad': 'HVGrad',
    'pmtl': 'PMTL',
    'random' : 'Random',
    'moosvgd' : 'MOO-SVGD',
    'dirhvego' : 'DirHV-EGO',
    'psldirhvei': 'PSL-DirHV-EI',
    'pslmobo': 'PSL-MOBO',

}

paper_dict = {
    'epo' : 'Mahapatra et al 2020',
    'mgdaub' : 'Sener et al 2018',
    'pmgda' : 'Zhang et al 2024',
    'random' : 'Lin et al 2021',
    'moosvgd' : 'Liu et al 2021',
    'pmtl' : 'Lin et al 2019',
    'hvgrad' : 'Deist et al 2021',
    'agg_ls' : 'Miettinen et al 1999',
    'agg_tche' : 'Zhang et al 2007',
    'agg_pbi' : 'Zhang et al 2007',
    'agg_cosmos' : 'Ruchte et al 2007',
    'agg_softtche' : 'Lin et al 2024',
    'agg_softmtche' : 'Lin et al 2024',
    'agg_mtche' : 'Ma et al 2017',
}

all_mtd_arr = ['epo', 'pmgda', 'agg_ls', 'agg_tche', 'agg_mtche',
               'agg_cosmos', 'agg_pbi', 'hvgrad', 'uniform', 'pmtl']

import seaborn as sns
color_arr = sns.color_palette() + ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'grey', 'black', 'yellow']
marker_arr = ['o', 'x', '+', 'v', 's', 'p', 'D', 'h', '8', '1']


plt_2d_tickle_size = 12
plt_2d_marker_size = 10
plt_2d_label_size = 20


def plot_fig_2d(folder_name, loss, prefs, use_plt='False', axis_equal=True, line=True):
    plt.figure()
    rho = np.max([np.linalg.norm(elem) for elem in loss])
    prefs_l2 = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)

    if axis_equal:
        plt.axis('equal')

    plt.xlabel('$L_1$', fontsize=plt_2d_label_size)
    plt.ylabel('$L_2$', fontsize=plt_2d_label_size)
    plt.xticks(fontsize=plt_2d_tickle_size)
    plt.yticks(fontsize=plt_2d_tickle_size)
    for pref in prefs_l2:
        plt.plot([0, rho * pref[0]], [0, rho * pref[1]], color='grey',
                 linestyle='--', linewidth=2)
    plt.scatter(loss[:, 0], loss[:, 1])
    file_name = os.path.join(folder_name, 'res.pdf')
    plt.savefig(file_name, dpi=1200, bbox_inches='tight')
    print('Save to {}'.format(file_name))
    if use_plt == 'True':
        plt.show()


def save_pickle(folder_name, res):
    import pickle
    pickle_name = os.path.join(folder_name, 'res.pickle')
    with open(pickle_name, 'wb') as f:
        pickle.dump(res, f)
    print('Save pickle to {}'.format(pickle_name))

def plot_loss(folder_name, loss_arr):
    plt.figure()
    plt.plot(loss_arr)
    plt.xlabel('Epoch', fontsize=plt_2d_label_size)
    plt.ylabel('Loss', fontsize=plt_2d_label_size)
    plt.xticks(fontsize=plt_2d_tickle_size)
    plt.yticks(fontsize=plt_2d_tickle_size)
    plt.grid()
    file_name = os.path.join(folder_name, 'loss.pdf')
    plt.savefig(file_name, format='pdf', dpi=1200, bbox_inches='tight')
    print('Loss save to {}'.format(file_name))
