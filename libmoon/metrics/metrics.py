import torch
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting



def adj_matrix(solutions):
    n = len(solutions)
    if type(solutions) == torch.Tensor:
        mat = torch.cdist(solutions, solutions, p=2)
        off_diag = mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    else:
        from scipy.spatial.distance import cdist
        mat = cdist(solutions, solutions, metric='euclidean')
        off_diag = np.reshape(mat.flatten()[:-1], (n - 1, n + 1))[:, 1:].flatten()
    return off_diag

def compute_MMS(solutions):
    mat = adj_matrix(solutions)
    if type(mat) == torch.Tensor:
        sp = - torch.min( mat )
    else:
        sp = - np.min(mat)
    return sp

def compute_soft_MMS(solutions):
    off_diag = adj_matrix(solutions)
    # Here gamma = 1
    K = 20
    if type(off_diag) == torch.Tensor:
        sp = - torch.log( torch.sum( torch.exp( - K * off_diag ) ) ) / K
    else:
        sp = - np.log( np.sum( np.exp( -K * off_diag ) ) ) / K
    return sp



def compute_sparsity_mit( objs ):
    '''
        objs: objective arrays
    '''
    non_dom = NonDominatedSorting().do(objs, only_non_dominated_front=True)
    objs = objs[non_dom]
    sparsity_sum = 0
    for objective in range(objs.shape[-1]):
        objs_sort = np.sort(objs[:, objective])
        sp = 0
        for i in range(len(objs_sort) - 1):
            sp += np.power(objs_sort[i] - objs_sort[i + 1], 2)
        sparsity_sum += sp
    if len(objs) > 1:
        sparsity = sparsity_sum / (len(objs) - 1)
    else:
        sparsity = 0
    return sparsity


def compute_spacing( sols ):
    n = len( sols )
    from scipy.spatial.distance import cdist
    mat = cdist(sols, sols, metric='euclidean')

    np.fill_diagonal(mat, np.inf)

    sp_arr = - np.min(mat, axis=1)
    return np.std(sp_arr)


def compute_hv( sols, ref_point=np.array([1.0,1.0])):
    n,m = sols.shape
    from pymoo.indicators.hv import HV
    ind = HV(ref_point=ref_point)
    hv_val = ind(sols)
    return hv_val


def compute_pbi(sols, prefs, coeff=5.0):
    pref_l2 = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
    pbi_arr = []
    for sol, pref in zip(sols, pref_l2):
        d1 = np.dot(sol, pref)
        d2 = np.linalg.norm(sol -  pref / np.linalg.norm(pref) * d1)
        pbi_val = d1 + coeff * d2
        pbi_arr.append(pbi_val)

    return np.mean(pbi_arr)




def compute_inner_product(sols, prefs):
    sum_value = sols * prefs
    ip_values_vec = np.sum(sum_value, axis=1)
    return np.mean(ip_values_vec)


def compute_cross_angle(sols, prefs):
    arccos_value_arr = []
    for sol, pref in zip(sols, prefs):
        arccos_value = np.dot(sol, pref) / ( np.linalg.norm(sol) * np.linalg.norm(pref) )
        arccos_value = np.clip(arccos_value, -1, 1)
        angle = np.arccos(arccos_value) * 180 / np.pi
        arccos_value_arr.append( angle )
    return np.mean(arccos_value_arr)






def compute_indicators(objs, prefs):
    mms = -compute_MMS(objs)
    soft_mms = compute_soft_MMS(objs)
    spacing = compute_spacing(objs)
    sparsity = compute_sparsity_mit(objs)
    hv = compute_hv(objs)

    inner_product = compute_inner_product(objs, prefs)
    cross_angle = compute_cross_angle(objs, prefs)
    pbi = compute_pbi(objs, prefs)

    return {
        'uniform': mms,
        'soft uniform': soft_mms,
        'spacing': spacing,
        'sparsity': sparsity,
        'hv': hv,
        'inner_product': inner_product,
        'cross_angle': cross_angle,
        'pbi': pbi
    }