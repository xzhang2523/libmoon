import numpy as np
from sklearn.neighbors import KernelDensity


if __name__ == '__main__':
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)

    print()
