import numpy as np 




def population_initialization(n_pop, 
                              problem, 
                              method: str='random'): 
    
    lb = problem.get_lower_bound
    ub = problem.get_upper_bound 
    dim = problem.get_number_variable
    if method == 'random': 
        return lb + (ub - lb) * np.random.rand(n_pop, dim)


def repair_clamp(Xp, xl, xu):

    I = np.where(Xp < xl)
    Xp[I] = xl[I]

    I = np.where(Xp > xu)
    Xp[I] = xu[I]

    return Xp


def dominance_min(u, v): 

    assert u.ndim * v.ndim == 1 
    assert len(u) == len(v) 

    for i in range(len(u)): 
        if v[i] < u[i]: return False
    
    if (u==v).all(): return False

    return True



def neighborhood_selection(n_pop, neighbors, n_selects=2, prob=.9): 

    assert neighbors.ndim == 1 

    if np.random.random() < prob:
        P = np.random.choice(neighbors, n_selects, replace=False)
    else:
        P = np.random.permutation(n_pop)[:n_selects]

    return P


