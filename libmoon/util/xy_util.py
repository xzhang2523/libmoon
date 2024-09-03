from libmoon.util.constant import max_indicators
import numpy as np
import torch

def set_indicators_rank(Indicators, indicator_dict_dict_saved, mtd_arr):
    for indicator in Indicators:
        mtd_val = [ indicator_dict_dict_saved[mtd][indicator] for mtd in mtd_arr ]
        if indicator in max_indicators:
            arg_sort = np.argsort( -np.copy(mtd_val) )
        else:
            arg_sort = np.argsort( np.copy(mtd_val) )

        for idx, mtd in enumerate(mtd_arr):
            indicator_dict_dict_saved[mtd]['{}_rank'.format(indicator)] = arg_sort.tolist().index(idx)

def get_indicator(problem_name, mtd_name, num_seed, use_save=False):

    indicator_dict_seed = []
    for seed_idx in range(1, num_seed+1):
        # from xy_util import save_indicators
        # obj = get_obj_by_mtd(problem_name=problem_name, mtd_name=mtd_name, seed_idx=seed_idx)
        indicator_dict = 0
        assert False
        indicator_dict_seed.append(indicator_dict)

    mean_indicator_dict = {}
    std_indicator_dict = {}
    for key in indicator_dict_seed[0].keys():
        mean_indicator_dict[key] = np.mean([indicator_dict[key] for indicator_dict in indicator_dict_seed])
        std_indicator_dict[key] = np.std([indicator_dict[key] for indicator_dict in indicator_dict_seed])
    return mean_indicator_dict, std_indicator_dict



def pref2angle(pref):
    if type(pref) == torch.Tensor:
        angle = torch.arctan2(pref[:,1], pref[:,0])
        angle = angle.unsqueeze(1)
    else:
        angle = np.arctan2(pref[:,0], pref[:,1])
    return angle

def angle2pref(angle):
    if type(angle) == torch.Tensor:
        return torch.squeeze(torch.stack([torch.cos(angle), torch.sin(angle)], dim=1))
    else:
        return np.stack([np.cos(angle), np.sin(angle)], axis=1)


def random_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)