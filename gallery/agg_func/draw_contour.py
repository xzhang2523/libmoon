import numpy as np
from numpy import array
from libmoon.util.constant import get_agg_func, beautiful_dict
from matplotlib import pyplot as plt
import os


if __name__ == '__main__':
    # agg_name_arr = ['agg_ls', 'agg_mtche', 'agg_tche', 'agg_pbi', 'agg_softtche', 'agg_softmtche']
    agg_name_arr = ['agg_softtche',]
    for agg_name in agg_name_arr:
        function_name = agg_name.split('_')[1]
        function = get_agg_func(function_name)
        x = np.linspace(-5, 5, 1000)
        y = np.linspace(-5, 5, 1000)
        prefs = array([0.5, 0.5])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = function(np.atleast_2d(array([X[i, j], Y[i, j]])), np.atleast_2d(prefs))

        # Plot the contour
        plt.figure()
        contour = plt.contour(X, Y, Z, 20)  # 20 levels
        plt.clabel(contour, inline=True, fontsize=8)
        plt.title('Contour plot of {}'.format(beautiful_dict[agg_name]))
        plt.xlabel('$f_1$', fontsize=20)
        plt.ylabel('$f_2$', fontsize=20)

        fig_name = os.path.join('D:\\pycharm_project\\libmoon\\gallery', '{}.pdf'.format(function_name))
        plt.savefig(fig_name, dpi=1200, bbox_inches='tight')
        print('Save fig to {}'.format(fig_name))
    # plt.show()
