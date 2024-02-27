import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_simplex(ax, p1, p2, p3, color='k'):
    # x1 = np.array([0, 0, 2])
    # y1 = np.array([0, 2, 0])
    # z1 = np.array([2, 0, 0])  # z1 should have 3 coordinates, right?
    # ax.scatter(x1, y1, z1)
    # 1. create vertices from points
    verts = [list(zip(p1, p2, p3))]
    # 2. create 3d polygons and specify parameters
    srf = Poly3DCollection(verts, alpha=.25, facecolor='#800000')
    # 3. add polygon to the figure (current axes)
    plt.gca().add_collection3d(srf)


def plot_unit_sphere(ax):
    u, v = np.mgrid[0:np.pi / 2:30j, 0:np.pi / 2:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color='b', alpha=0.2)