
import numpy as np

def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)



def uniform_pref(number, n_obj=2, clip_eps=0, mode='uniform'):

    if n_obj == 2:
        # Just generate linear uniform preferences
        pref_1 = np.linspace(clip_eps, 1-clip_eps, number)
        pref_2 = 1 - pref_1
        prefs = np.stack((pref_1, pref_2), axis=1)

    else:
        prefs = das_dennis(number, n_obj)
        prefs = np.clip(prefs, clip_eps, 1-clip_eps)
        prefs = prefs / prefs.sum(axis=1, keepdims=True)

    return prefs

