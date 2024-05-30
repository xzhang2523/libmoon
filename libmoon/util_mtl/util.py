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
    prefs = prefs / np.linalg.norm(prefs, axis=1)[:, None]

    return prefs





