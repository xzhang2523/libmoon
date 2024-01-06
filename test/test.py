import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
import argparse


t1 = 1
t2 = 4
t3 = 4
m=3
p=0.8


def f(theta):
    th1 = theta[:,0]
    th2 = theta[:,1]
    f1 = np.exp(-th1*t1-th2*t1)
    f2 = np.exp(-th1*t2-th2*t2)
    f3 = np.exp(-th1*t3-th2*t3)
    return np.stack((f1, f2), axis=1)

    # return np.stack((f1, f2, f3), axis=1)


# G: a local radius
# def Gi(x):

def perturb(X):
    num_sol = len(X)
    kde = KernelDensity( kernel='gaussian', bandwidth=0.2 ).fit(X)
    p=0.5
    # res = p * kde.sample(num_sol) + (1-p) * np.random.rand(num_sol, 2)
    # print()
    prob = np.exp(kde.score_samples(X))
    prob = prob / np.sum(prob)
    X_new = [0,] * len(X)
    for i in range(len(X)):
        if np.random.rand() < prob[i] * 3:
            x_new = p * kde.sample(1) + (1-p) * np.random.rand(1,2)
            X_new[i] = x_new.squeeze()
        else:
            X_new[i] = X[i]

    for x in X_new:
        print(x)

    # print(X_new)
    X_new = np.stack(X_new)
    # print(X_new)

    return X_new


def knn(y, y_arr, k):
    distances = np.linalg.norm(y_arr - y, axis=1)
    # distances = np.delete(distances, 0)
    distances_sorted = np.sort(distances)
    return distances_sorted[k]


def resample(X):
    y_arr = f(X)
    r_arr = np.zeros( len(y_arr) )
    for i in range(len(y_arr)):
        yi = y_arr[i]
        r_arr[i] = knn(yi, y_arr, k=3)
    G = np.power(r_arr, m)
    G_norm = G / np.sum(G)
    draw = np.random.choice(range(len(y_arr)), len(y_arr), p=G_norm)
    x_new = X[draw]
    return x_new, r_arr



if __name__ == '__main__':
    # th1 = th2 = np.linspace(0, 100, 400)
    # th1, th2 = np.meshgrid(th1, th2)
    # theta = np.stack((th1.flatten(), th2.flatten()), axis=1)
    #
    # res = f(theta)
    # ax = plt.subplot(111, projection='3d')
    # ax.scatter(res[:,0], res[:,1], res[:,2])
    # plt.show()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-sol', type=int, default=10)
    args = parser.parse_args()


    plt.figure( figsize=(4,8) )
    X = np.random.rand(args.num_sol, 2) * 1


    for iter in range(5):
        y_origin = f(X)
        X_old = np.copy(X)
        X, r_arr = resample(X)
        X = perturb(X)
        y_new = f(X)

    plt.subplot(2,1,1)
    plt.scatter(X_old[:,0], X_old[:,1], c='r', label='original', marker='x')
    plt.scatter(X[:,0], X[:,1], c='b', label='resampled')

    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')

    plt.legend()


    plt.subplot(2,1,2)
    plt.scatter(y_origin[:,0], y_origin[:,1], c='r', label='original', marker='x')
    plt.scatter(y_new[:,0], y_new[:,1], c='b', label='resampled')
    plt.legend()
    plt.xlabel('$f_1$')
    plt.ylabel('$f_2$')
    plt.show()