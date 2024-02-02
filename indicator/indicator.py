import torch
import numpy as np

def angle2pref(angle, n_obj):
    if n_obj ==3:
        angle1 = angle[:,0]
        angle2 = angle[:,1]

        if type(angle) == torch.Tensor:
            pref1 = torch.cos(angle1) ** 2
            pref2 = torch.sin(angle1) ** 2 * torch.cos(angle2) ** 2
            pref3 = torch.sin(angle1) ** 2 * torch.sin(angle2) ** 2
            pref = torch.stack((pref1, pref2, pref3), dim=1)
        else:
            pref1 = np.cos(angle1) ** 2
            pref2 = np.sin(angle1) ** 2 * np.cos(angle2) ** 2
            pref3 = np.sin(angle1) ** 2 * np.sin(angle2) ** 2
            pref = np.c_[pref1, pref2, pref3]
    elif n_obj == 2:
        if type(angle) == torch.Tensor:
            pref1 = torch.cos(angle[:,0]) ** 2
            pref2 = torch.sin(angle[:,0]) ** 2
            pref = torch.stack((pref1, pref2), dim=1)
        else:
            pref1 = np.cos(angle[:,0]) ** 2
            pref2 = np.sin(angle[:,0]) ** 2
            pref = np.c_[pref1, pref2]
    else:
        assert False, 'n_obj should be 2 or 3'
    return pref



def pref2angle(pref, n_obj):
    if n_obj == 3:
        pref1 = pref[:,0]
        pref2 = pref[:,1]

        if type(pref) == torch.Tensor:
            angle1 = torch.acos(torch.sqrt(pref1))
            angle2 = torch.acos(torch.sqrt(pref2 / torch.sin(angle1) ** 2))
            angle = torch.stack((angle1, angle2), dim=1)
        else:

            angle1 = np.arccos( np.sqrt(pref1) )
            use_debug = False
            if use_debug:
                print('angle1', angle1)

            angle2 = np.zeros_like(angle1)
            for idx, ang1 in enumerate(angle1):
                if ang1 != 0:
                    cos_th = np.sqrt(pref2[idx] / (np.sin(ang1) ** 2) )
                    cos_th = np.clip(cos_th, 0, 1)
                    angle2[idx] = np.arccos( cos_th  )
            angle = np.c_[angle1, angle2]

    else:
        if type(pref) == torch.Tensor:
            angle1 = torch.acos(torch.sqrt(pref[:,0]))
            angle = angle1.unsqueeze(1)
        else:
            angle1 = np.arccos(np.sqrt(pref[:,0]))
            angle = np.c_[angle1]

    return angle


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






from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
def compute_sparsity_mit( obj ):
    '''
    obj: objective arrays
    '''
    non_dom = NonDominatedSorting().do(obj, only_non_dominated_front=True)
    objs = obj[non_dom]
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


def compute_hv( sols ):
    n,m = sols.shape
    from pymoo.indicators.hv import HV

    ref_point = 1.2 * np.ones(m)
    ind = HV(ref_point=ref_point)
    # print("HV", ind(A))
    hv_val = ind(sols)
    return hv_val

def compute_indicators(objs):
    mms = compute_MMS(objs)
    soft_mms = compute_soft_MMS(objs)
    spacing = compute_spacing(objs)
    sparsity = compute_sparsity_mit(objs)
    hv = compute_hv(objs)
    return {
        'mms': mms,
        'soft_mms': soft_mms,
        'spacing': spacing,
        'sparsity': sparsity,
        'hv': hv
    }