from libmoon.util.constant import max_indicators
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import cv2

# Read an image
# image = cv2.imread("image.jpg")



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


class FolderDataset(Dataset):
    def __init__(self, folder_name):
        self.folder_name = folder_name
        file_names = [f for f in os.listdir(self.folder_name) if os.path.isfile(os.path.join(self.folder_name, f))]
        image_array = []
        for idx, file_name in enumerate(file_names):
            file_path = os.path.join(self.folder_name, file_name)
            image = cv2.imread(file_path)
            image_array.append(image)
        self.image_array = image_array

    def __getitem__(self, idx):
        return self.image_array[idx]

    def __len__(self):
        return len(self.image_array)


def random_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)