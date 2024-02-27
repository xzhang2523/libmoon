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



def OperatorDE(Parent1, Parent2, Parent3, mop):
    '''
        generate one offspring by P1 + 0.5*(P2-P3) and polynomial mutation.
    '''
    # Parameter
    CR = 1
    F = 0.5
    proM = 1
    disM = 20
    #
    N, D = Parent1.shape
    # Differental evolution
    Site = np.random.rand(N, D) < CR
    Offspring = Parent1.copy()
    Offspring[Site] = Offspring[Site] + F * (Parent2[Site] - Parent3[Site])
    # Polynomial mutation
    Lower = np.atleast_2d( mop.get_lower_bound )  # numpy  Upper=np.array(Upper)[None,:]
    Upper = np.atleast_2d( mop.get_upper_bound)  # Lower = np.atleast_2d(Lower)
    U_L = Upper - Lower
    Site = np.random.rand(N, D) < proM / D
    mu = np.random.rand(N, D)
    temp = np.logical_and(Site, mu <= 0.5)
    Offspring = np.minimum(np.maximum(Offspring, Lower), Upper)
    delta1 = (Offspring - Lower) / U_L
    delta2 = (Upper - Offspring) / U_L
    #  mu <= 0.5
    val = 2. * mu + (1 - 2. * mu) * (np.power(1. - delta1, disM + 1))
    Offspring[temp] = Offspring[temp] + (np.power(val[temp], 1.0 / (disM + 1)) - 1.) * U_L[temp]
    # mu > 0.5
    temp = np.logical_and(Site, mu > 0.5)
    val = 2. * (1.0 - mu) + 2. * (mu - 0.5) * (np.power(1. - delta2, disM + 1))
    Offspring[temp] = Offspring[temp] + (1.0 - np.power(val[temp], 1.0 / (disM + 1))) * U_L[temp]

    return Offspring






if __name__ == '__main__':
    from libmoon.problem.synthetic.zdt import ZDT1

    problem = ZDT1()

    # p1 = np.array( [[1,2,3],] )
    # p2 = np.array( [[2,3,4],] )

    p1 = np.random.random( (1, 30) )
    p2 = np.random.random( (1, 30) )
    p3 = np.random.random( (1, 30) )


    print(p1)
    print(p2)

    res = _OperatorDE(p1, p2, p1, problem)
    print( res )

