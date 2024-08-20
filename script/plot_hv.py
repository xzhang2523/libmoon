import pickle

import matplotlib.pyplot as plt

if __name__ == '__main__':
    mtd_arr = ['epo', 'mgdaub', 'random']
    tmp = 'D:\\pycharm_project\\libmoon\\Output\\{}\\ZDT1\\res.pkl'

    for mtd in mtd_arr:
        with open(tmp.format(mtd), 'rb') as f:
            res = pickle.load(f)
        # print(res.keys())
        plt.plot(res['hv_arr'], label=mtd)

    plt.legend('upper right', fontsize=19)
    plt.show()