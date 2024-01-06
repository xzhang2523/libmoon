import numpy as np 

from solver.moea.utils.utils_ea import repair_clamp





def cross_sbx(X, xl, xu, eta=15, prob_var=0.5, prob_bin=0.5, eps=1.0e-14, n_offsprings=1):
    n_parents, n_var = X.shape

    # the probability of a crossover for each of the variables
    cross = np.random.random((n_var)) < prob_var

    # when solutions are too close -> do not apply sbx crossover
    too_close = np.abs(X[0] - X[1]) <= eps

    # disable if two individuals are already too close
    cross[too_close] = False
    # disable crossover when lower and upper bound are identical
    cross[xl == xu] = False

    # assign y1 the smaller and y2 the larger value
    y1 = np.min(X, axis=0)[cross]
    y2 = np.max(X, axis=0)[cross]

    # mask all the values that should be crossovered
    _xl = xl[cross]
    _xu = xu[cross]
    eta = np.full((n_var, ), eta)[cross]
    prob_bin = np.full((n_var, ), prob_bin)[cross]


    # random values for each individual
    rand = np.random.random(len(eta))
    def calc_betaq(beta):
        alpha = 2.0 - np.power(beta, -(eta + 1.0))

        mask, mask_not = (rand <= (1.0 / alpha)), (rand > (1.0 / alpha))

        betaq = np.zeros(mask.shape)
        betaq[mask] = np.power((rand * alpha), (1.0 / (eta + 1.0)))[mask]
        betaq[mask_not] = np.power((1.0 / (2.0 - rand * alpha)), (1.0 / (eta + 1.0)))[mask_not]

        return betaq

    # difference between all variables
    delta = (y2 - y1)

    beta = 1.0 + (2.0 * (y1 - _xl) / delta)
    betaq = calc_betaq(beta)
    c1 = 0.5 * ((y1 + y2) - betaq * delta)

    beta = 1.0 + (2.0 * (_xu - y2) / delta)
    betaq = calc_betaq(beta)
    c2 = 0.5 * ((y1 + y2) + betaq * delta)

    # with the given probability either assign the value from the first or second parent
    b = np.random.random(len(prob_bin)) < prob_bin
    tmp = np.copy(c1[b])
    c1[b] = c2[b]
    c2[b] = tmp

    # first copy the unmodified parents
    Q = np.copy(X)

    # copy the positions where the crossover was done
    Q[0, cross] = c1
    Q[1, cross] = c2

    Q[0] = repair_clamp(Q[0], xl, xu)
    Q[1] = repair_clamp(Q[1], xl, xu)

    if n_offsprings == 1:
        rand = np.random.random() < 0.5
        Q[0, rand] = Q[1, rand]
        Q = Q[[0]]

    return Q


def mut_pm(X, xl, xu, eta=15, prob=None, at_least_once=False):
    n, n_var = X.shape
    if prob is None: prob = min(0.5, 1 / n_var) 

    eta = np.full((n,), eta)
    prob = np.full((n,), prob)

    Xp = np.full(X.shape, np.inf)

    mut = mut_binomial(n, n_var, prob, at_least_once=at_least_once)
    mut[:, xl == xu] = False

    Xp[:, :] = X

    _xl = np.repeat(xl[None, :], X.shape[0], axis=0)[mut]
    _xu = np.repeat(xu[None, :], X.shape[0], axis=0)[mut]

    X = X[mut]
    eta = np.tile(eta[:, None], (1, n_var))[mut]

    delta1 = (X - _xl) / (_xu - _xl)
    delta2 = (_xu - X) / (_xu - _xl)

    mut_pow = 1.0 / (eta + 1.0)

    rand = np.random.random(X.shape)
    mask = rand <= 0.5
    mask_not = np.logical_not(mask)

    deltaq = np.zeros(X.shape)

    xy = 1.0 - delta1
    val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (eta + 1.0)))
    d = np.power(val, mut_pow) - 1.0
    deltaq[mask] = d[mask]

    xy = 1.0 - delta2
    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (eta + 1.0)))
    d = 1.0 - (np.power(val, mut_pow))
    deltaq[mask_not] = d[mask_not]

    # mutated values
    _Y = X + deltaq * (_xu - _xl)

    # back in bounds if necessary (floating point issues)
    _Y[_Y < _xl] = _xl[_Y < _xl]
    _Y[_Y > _xu] = _xu[_Y > _xu]

    # set the values for output
    Xp[mut] = _Y

    return Xp


def row_at_least_once_true(M):
    _, d = M.shape
    for k in np.where(~np.any(M, axis=1))[0]:
        M[k, np.random.randint(d)] = True
    return M


def mut_binomial(n, m, prob, at_least_once=True):
    prob = np.ones(n) * prob
    M = np.random.random((n, m)) < prob[:, None]

    if at_least_once:
        M = row_at_least_once_true(M)

    return M





