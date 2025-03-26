import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from libmoon.util.constant import beautiful_dict
from libmoon.util.constant import root_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--seed-num', type=int, default=1)
    parser.add_argument('--problem-name', type=str, default='ZDT1')
    args = parser.parse_args()

    mtd_arr = ['dirhvego', 'psldirhvei', 'pslmobo']

    tmp_name = 'D:\\pycharm_project\\libmoon\\Output\\mobo\\{}\\{}\\seed_{}\\res.pickle'
    for mtd in mtd_arr:
        # res = pickle.load(open(file_name, 'rb'))
        val_arr = []
        for seed in range(args.seed_num):
            file_name = tmp_name.format(args.problem_name, mtd, seed)
            with open(file_name, 'rb') as f:
                res = pickle.load(f)
                idx = np.sort(np.array(list(res['hv'].keys())))
                val = np.sort(np.array(list(res['hv'].values())) )
                extended_idx = np.repeat(idx, 2)[1:]
                extended_val = np.repeat(val, 2)[:-1]
                val_arr.append(extended_val)

        val_mean = np.mean(array(val_arr), axis=0)
        val_std = np.std(array(val_arr), axis=0)
        plt.step(extended_idx, val_mean, where='pre', label=beautiful_dict[mtd], linewidth=2)
        plt.fill_between(extended_idx, val_mean - val_std, val_mean + val_std, alpha=0.2)

    plt.xlabel('FE', fontsize=18)
    plt.ylabel('HV', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)
    folder_name = 'D:\\pycharm_project\\libmoon\\script\\rebuttal\\mobo'
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(os.path.join(folder_name, '{}.pdf'.format(args.problem_name) ), bbox_inches='tight')
    plt.show()