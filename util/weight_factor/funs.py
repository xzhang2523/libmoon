import numpy as np
from .das_dennis import das_dennis


def uniform_pref(n_partition, n_obj, clip_eps=0, mtd='uniform'):

    if n_obj == 2:
        pref_1 = np.linspace(clip_eps, 1-clip_eps, n_partition)
        pref_2 = 1 - pref_1
        prefs = np.stack((pref_1, pref_2), axis=1)
    else:
        prefs = das_dennis(n_partition, n_obj)

        prefs = np.clip(prefs, clip_eps, 1-clip_eps)
        prefs = prefs / prefs.sum(axis=1, keepdims=True)

    return prefs