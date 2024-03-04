import numpy as np
from matplotlib import pyplot as plt
from libmoon.util_global.constant import FONT_SIZE, root_name
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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

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

        ax.set_xlabel('$f_1$', fontsize=FONT_SIZE)
        ax.set_ylabel('$f_2$', fontsize=FONT_SIZE)
        ax.set_zlabel('$f_3$', fontsize=FONT_SIZE)
        ax.view_init(elev=30, azim=125)

        # ax2 = fig.add_subplot(312)
        # ax2.plot(res['hv_arr'])
    else:
        raise Exception('n_obj not supported')


import matplotlib.cm as cm
# import matplotlib.animation as animation

from matplotlib.animation import FuncAnimation
import matplotlib.animation as manimation
from matplotlib.animation import PillowWriter

from libmoon.util_global.constant import FONT_SIZE

def vedio_res(res, problem, prefs, args):
    print('vedio making')
    from matplotlib.animation import FuncAnimation
    from matplotlib.animation import FFMpegWriter

    # Create some data
    # x = np.linspace(0, 2 * np.pi, 100)
    # y = np.sin(x)
    subsample = 20
    y_arr = res['y_arr'][::subsample]
    n_frame = len( y_arr )

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)

    ax.set_xlabel('$f_1$', fontsize=FONT_SIZE)
    ax.set_ylabel('$f_2$', fontsize=FONT_SIZE)

    # Create a function to update the plot for each frame of the animation
    if args.solver == 'agg':
        file_name = '{}_{}_{}'.format(args.problem_name, args.solver, args.agg)
    else:
        file_name = '{}_{}'.format(args.problem_name, args.solver)
    def update(frame):
        ax.clear()
        ax.scatter(y_arr[frame][:,0], y_arr[frame][:,1])

        pf = problem.get_pf()

        ax.plot(pf[:,0], pf[:,1], 'k', linewidth=1)

        ax.set_xlim(0, 1.2)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel('$f_1$', fontsize=FONT_SIZE)
        ax.set_ylabel('$f_2$', fontsize=FONT_SIZE)

        ax.set_title(file_name, fontsize=FONT_SIZE)


    # Create the animation
    ani = FuncAnimation(fig, update, frames=n_frame, interval=100)

    # Define the writer for the animation using FFMpegWriter
    writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)



    import os
    folder_name = os.path.join(root_name, 'solver', 'gradient', 'output')
    mp4_file_name = os.path.join(folder_name, file_name + '.mp4')



    # Save the animation as a video file
    ani.save(mp4_file_name, writer=writer)
    print('Vedio saved: {}'.format(mp4_file_name))

    if args.use_plt=='Y':
        plt.show()







