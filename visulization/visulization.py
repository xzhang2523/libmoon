import numpy as np
from matplotlib import pyplot as plt
from util_global.constant import FONT_SIZE

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def vis_res(res, problem, prefs, args):


    if args.n_obj == 2:
        fig = plt.figure(figsize=(8, 20))
        plt.subplot(3,1,1)
        plt.title(args.solver, fontsize=FONT_SIZE)
        plt.scatter(res['y'][:, 0], res['y'][:, 1], label='solution')
        pf = problem.get_pf()
        plt.plot(pf[:, 0], pf[:, 1], 'r--', label='PF')
        plt.xlabel('$f_1$', fontsize=FONT_SIZE)
        plt.ylabel('$f_2$', fontsize=FONT_SIZE)
        pf_norm = np.max( np.linalg.norm(pf, axis=1) )
        prefs_norm = prefs / np.linalg.norm(prefs, axis=1)[:, np.newaxis] * pf_norm
        if args.plt_pref_flag == 'Y':
            for idx, pref in enumerate(prefs_norm) :
                if idx == 0:
                    plt.plot([0, pref[0]], [0, pref[1]], 'k--', linewidth=1, label='Prefs')
                else:
                    plt.plot([0, pref[0]], [0, pref[1]], 'k--', linewidth=1)
        plt.legend(fontsize=FONT_SIZE)

        plt.subplot(3,1,2)
        plt.plot(res['hv_arr'])
        plt.xlabel('iteration', fontsize=FONT_SIZE)
        plt.ylabel('hypervolume', fontsize=FONT_SIZE)

        plt.subplot(3,1,3)
        for xx in res['x']:
            plt.plot(xx, 'o-', color='k')
        plt.xlabel('variable index', fontsize=FONT_SIZE)
        plt.ylabel('variable value', fontsize=FONT_SIZE)
    elif args.n_obj==3:
        # Generate random data

        # Create figure and axis objects
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the data points
        # ax.scatter(x, y, z)

        ax.scatter(res['y'][:, 0], res['y'][:, 1], res['y'][:, 2], label='solution',)
        if args.problem_name == 'maf1':
            # ax draw rectangle.
            x1 = np.array([0, 0, 2])
            y1 = np.array([0, 2, 0])
            z1 = np.array([2, 0, 0])  # z1 should have 3 coordinates, right?
            ax.scatter(x1, y1, z1)

            # 1. create vertices from points
            verts = [list(zip(x1, y1, z1))]
            # 2. create 3d polygons and specify parameters
            srf = Poly3DCollection(verts, alpha=.25, facecolor='#800000')
            # 3. add polygon to the figure (current axes)
            plt.gca().add_collection3d(srf)


            # ax.scatter(0, 0, 0, label='PF', c='r')





        # Set labels and title
        ax.set_xlabel('$f_1$', fontsize=FONT_SIZE)
        ax.set_ylabel('$f_2$', fontsize=FONT_SIZE)
        ax.set_zlabel('$f_3$', fontsize=FONT_SIZE)

        # ax2 = fig.add_subplot(312)
        # ax2.plot(res['hv_arr'])


    else:
        raise Exception('n_obj not supported')



