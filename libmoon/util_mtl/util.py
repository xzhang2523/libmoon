import numpy as np
from libmoon.problem.mtl.loaders import Adult, Credit, Compas, MultiMNISTData
from libmoon.util_global.constant import nadir_point_dict, ideal_point_dict


def get_dataset(dataset_name):
    if dataset_name == 'adult':
        dataset_ = Adult(split="train")
    elif dataset_name == 'credit':
        dataset_ = Credit(split="train")
    elif dataset_name == 'compass':
        dataset_ = Compas(split="train")
    elif dataset_name == 'mnist':
        dataset_ = MultiMNISTData(dataset='mnist', split='train')
    elif dataset_name == 'fashion':
        dataset_ = MultiMNISTData(dataset='fashion', split='train')
    elif dataset_name == 'fmnist':
        dataset_ = MultiMNISTData(dataset='fmnist', split='train')
    else:
        raise ValueError('Invalid dataset name')
    return dataset_

def get_angle_range(dataset, return_degrees=False):
    p1 = [nadir_point_dict[dataset][0], ideal_point_dict[dataset][1]]
    p2 = [ideal_point_dict[dataset][0], nadir_point_dict[dataset][1]]
    th1 = np.arctan2(p1[1], p1[0])
    th2 = np.arctan2(p2[1], p2[0])
    if return_degrees:
        th1 = np.rad2deg(th1)
        th2 = np.rad2deg(th2)
    return th1, th2


import torch
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

def get_mtl_prefs(dataset, n_prob, obj_normalization=True):
    if obj_normalization:
        theta_arr = np.linspace(0, np.pi/2, n_prob)
    else:
        p1 = [nadir_point_dict[dataset][0], ideal_point_dict[dataset][1]]
        p2 = [ideal_point_dict[dataset][0], nadir_point_dict[dataset][1]]
        th1 = np.arctan2(p1[1], p1[0])
        th2 = np.arctan2(p2[1], p2[0])
        theta_arr = np.linspace(th1, th2, n_prob)
    prefs = np.c_[np.cos(theta_arr), np.sin(theta_arr)]
    prefs = prefs / np.sum(prefs, axis=1)[:, None]
    return prefs




if __name__ == '__main__':
    # dataset = 'adult'
    # ang1, ang2 = get_angle_range(dataset, return_degrees=True)
    # print()

    prefs = np.array([[1, 0], [0, 1]])
    prefs = torch.Tensor([[1, 0], [0, 1]])
    angle = pref2angle( prefs )
    prefs_out = angle2pref(angle)

    print()



