"""
    The function fastNonDominatedSort is based on the sorting algorithm described by
    Deb, Kalyanmoy, et al.
    "A fast and elitist multiobjective genetic algorithm: NSGA-II."
    IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
"""


import numpy as np
import pdb
from .functions_hv_python3 import HyperVolume


def determine_non_dom_mo_sol(mo_obj_val):
    # get set of non-dominated solutions, returns indices of non-dominated and booleans of dominated mo_sol
    n_mo_sol = mo_obj_val.shape[1]
    domination_rank = fastNonDominatedSort(mo_obj_val)
    non_dom_indices = np.where(domination_rank == 0)
    non_dom_indices = non_dom_indices[0] # np.where returns a tuple, so we need to get the array inside the tuple
    # non_dom_mo_sol = mo_sol[:,non_dom_indices]
    # non_dom_mo_obj_val = mo_obj_val[:,non_dom_indices]
    mo_sol_is_non_dominated = np.zeros(n_mo_sol,dtype = bool)
    mo_sol_is_non_dominated[non_dom_indices] = True
    mo_sol_is_dominated = np.bitwise_not(mo_sol_is_non_dominated)
    return(non_dom_indices,mo_sol_is_dominated)

def fastNonDominatedSort(objVal):
    # Based on Deb et al. (2002) NSGA-II
    N_OBJECTIVES = objVal.shape[0]
    N_SOLUTIONS = objVal.shape[1]

    rankIndArray = - 999 * np.ones(N_SOLUTIONS, dtype = int) # -999 indicates unassigned rank
    solIndices = np.arange(0,N_SOLUTIONS) # array of 0 1 2 ... N_SOLUTIONS
    ## compute the entire domination matrix
    # dominationMatrix: (i,j) is True if solution i dominates solution j
    dominationMatrix = np.zeros((N_SOLUTIONS,N_SOLUTIONS), dtype = bool)
    for p in solIndices:
        objValA = objVal[:,p][:,None] # add [:,None] to preserve dimensions
        # objValArray =  np.delete(objVal, obj = p axis = 1) # dont delete solution p because it messes up indices
        dominates = checkDomination(objValA,objVal)
        dominationMatrix[p,:] = dominates

    # count the number of times a solution is dominated
    dominationCounter = np.sum(dominationMatrix, axis = 0)

    ## find rank 0 solutions to initialize loop
    isRankZero = (dominationCounter == 0) # column and row binary indices of solutions that are rank 0

    rankZeroRowInd = solIndices[isRankZero]
    # mark rank 0's solutions by -99 so that they are not considered as members of next rank
    dominationCounter[rankZeroRowInd] = -99
    # initialize rank counter at 0
    rankCounter = 0
    # assign solutions in rank 0 rankIndArray = 0
    rankIndArray[isRankZero] = rankCounter

    isInCurRank = isRankZero
    # while the current rank is not empty
    while not (np.sum(isInCurRank) == 0):
        curRankRowInd = solIndices[isInCurRank] # column and row numbers of solutions that are in current rank
        # for each solution in current rank
        for p in curRankRowInd:
            # decrease domination counter of each solution dominated by solution p which is in the current rank
            dominationCounter[dominationMatrix[p,:]] -= 1 #dominationMatrix[p,:] contains indices of the solutions dominated by p
        # all solutions that now have dominationCounter == 0, are in the next rank
        isInNextRank = (dominationCounter == 0)
        rankIndArray[isInNextRank] = rankCounter + 1
        # mark next rank's solutions by -99 so that they are not considered as members of future ranks
        dominationCounter[isInNextRank] = -99
        # increase front counter
        rankCounter += 1
        # check which solutions are in current rank (next rank became current rank)
        isInCurRank = (rankIndArray == rankCounter)
        if not np.all(isInNextRank == isInCurRank): # DEBUGGING, if it works fine, replace above assignment
            pdb.set_trace()
    return(rankIndArray)

def checkDomination(objValA,objValArray):
    dominates = ( np.any(objValA < objValArray, axis = 0) & np.all(objValA <= objValArray , axis = 0) )
    return(dominates)

def compute_hv_in_higher_dimensions(mo_obj_val,ref_point):
    n_mo_obj = mo_obj_val.shape[0]
    n_mo_sol = mo_obj_val.shape[1]
    assert len(ref_point) == n_mo_obj
    # initialize hv computation instance
    hv_computation_instance = HyperVolume(tuple(ref_point))
    # turn numpy array to list of tuples
    list_of_mo_obj_val = list()
    for i_mo_sol in range(n_mo_sol):
        list_of_mo_obj_val.append(tuple(mo_obj_val[:,i_mo_sol]))

    hv = float(hv_computation_instance.compute(list_of_mo_obj_val))
    return(hv)