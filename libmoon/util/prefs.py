import numpy as np
import torch

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

def get_uniform_pref(n_prob, n_obj=2, clip_eps=0, mode='uniform', dtype='Tensor'):
    if n_obj == 2:
        # Just generate linear uniform preferences
        pref_1 = np.linspace(clip_eps, 1-clip_eps, n_prob)
        pref_2 = 1 - pref_1
        prefs = np.stack((pref_1, pref_2), axis=1)
    else:
        from pymoo.util.ref_dirs import get_reference_directions
        prefs = get_reference_directions("energy", n_obj, n_prob, seed=1)
        prefs = np.clip(prefs, clip_eps, 1-clip_eps)
        prefs = prefs / prefs.sum(axis=1)[:, None]
    if dtype == 'Tensor':
        return torch.Tensor(prefs)
    else:
        return prefs

def get_x_init(n_prob, n_var, lbound=None, ubound=None):
    if type(lbound)==type(None):
        x_init = torch.rand(n_prob, n_var)
    else:
        x_init = torch.rand(n_prob, n_var) * (ubound - lbound) + lbound
    return x_init

def get_random_prefs(batch_size, n_obj, type='Tensor'):
    import torch
    if type == 'Tensor':
        return torch.distributions.dirichlet.Dirichlet(torch.ones(n_obj)).sample((batch_size,)).squeeze()
    else:
        return np.random.dirichlet(np.ones(n_obj), batch_size)

def pref2angle(pref):
    if type(pref) == torch.Tensor:
        angle = torch.arctan2(pref[:,1], pref[:,0])
        angle = angle.unsqueeze(1)
    else:
        angle = np.arctan2(pref[:,0], pref[:,1])
    return angle

def angle2pref(angle):
    if type(angle) == torch.Tensor:
        return torch.squeeze(torch.stack([torch.cos(angle), torch.sin(angle)], dim=1))
    else:
        return np.stack([np.cos(angle), np.sin(angle)], axis=1)