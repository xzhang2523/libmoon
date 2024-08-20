
import pickle
from libmoon.metrics import compute_indicators






if __name__ == '__main__':
    problem_name = 'VLMOP1'
    seed_num=2

    pickle_name_template = 'D:\\pycharm_project\\libmoon\\Output\\discrete\\{}\\{}\\seed_{}\\res.pickle'
    mtd_arr = ['epo', 'mgdaub', 'random']
    for mtd in mtd_arr:
        indicator_arr = [0,] * seed_num
        for seed_idx in range(seed_num):
            pickle_name = pickle_name_template.format(problem_name, mtd, seed_idx)
            with open(pickle_name, 'rb') as f:
                res = pickle.load(f)
                objs = res['y']
                prefs = res['prefs']
                indicator_dict = compute_indicators(objs, prefs)
