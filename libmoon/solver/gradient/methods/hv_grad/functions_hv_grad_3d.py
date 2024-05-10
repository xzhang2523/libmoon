# This is file follows Emmerich's work

'''
    The function grad_multi_sweep and its subfunctions in this file are based on the algorithm described in:
    Emmerich, Michael, and André Deutz.
    "Time complexity and zeros of the hypervolume metrics gradient field."
    EVOLVE-a bridge between probability, set oriented numerics,
    and evolutionary computation III. Springer, Heidelberg, 2014. 169-193.
'''
import numpy as np
import copy
from .functions_hv_python3 import HyperVolume
from .functions_evaluation import determine_non_dom_mo_sol

def determine_mo_sol_in_exterior(mo_obj_val,ref_point):
    # select only mo-solutions that are in the exterior
    ref_point_temp = ref_point[:,None] # add axis so that comparison works
    exterior_booleans = np.any(mo_obj_val > ref_point_temp, axis = 0)
    exterior_indices = np.where(exterior_booleans == True)
    return(exterior_indices,exterior_booleans)

def determine_mo_sol_in_interior(mo_obj_val,ref_point):
    # select only mo-solutions that are in the interior
    ref_point_temp = ref_point[:,None] # add axis so that comparison works
    interior_booleans = np.all(mo_obj_val < ref_point_temp, axis = 0)
    interior_indices = np.where(interior_booleans == True)
    return(interior_indices,interior_booleans)

def determine_mo_sol_on_ref_boundary(mo_obj_val,ref_point):
    # select only mo-solutions that are on the reference boundary
    ref_point_temp = ref_point[:,None] # add axis so that comparison works
    boundary_booleans = np.logical_and( np.all(mo_obj_val <= ref_point_temp, axis = 0) , np.any(mo_obj_val == ref_point_temp, axis = 0) )
    boundary_indices = np.where(boundary_booleans == True)
    return(boundary_indices,boundary_booleans)

def compute_domination_properties(mo_obj_val):
    '''
    compute properties needed for HV gradient computation
    '''
    n_mo_sol = mo_obj_val.shape[1]
    # mo_sol i stricly dominates j (all entries in i  < j ) if strong_domination_matrix[i,j] = True
    strong_domination_matrix = np.zeros((n_mo_sol,n_mo_sol), dtype = np.bool_)
    for i in range(0,n_mo_sol):
        cur_col = mo_obj_val[:,i][:,None]
        strong_domination_matrix[i,:] = np.all(cur_col < mo_obj_val,axis = 0)
    # mo_sol i weakly dominates j (all entries in i  <= j and at least one entry i < j ) if weak_domination_matrix[i,j] = True
    weak_domination_matrix = np.zeros((n_mo_sol,n_mo_sol), dtype = np.bool_)
    for i in range(0,n_mo_sol):
        cur_col = mo_obj_val[:,i][:,None]
        weak_domination_matrix[i,:] = np.logical_and( np.all(cur_col <= mo_obj_val,axis = 0) , np.any(cur_col < mo_obj_val,axis = 0) )
    # a mo_sol i is strongly dominated if any other solutions j strongly dominates it (any True in the column strong_domination_matrix[:,i] )
    is_strongly_dominated = np.any(strong_domination_matrix, axis = 0)
    # a mo_sol i is weakly dominated if any other solutions j weakly dominates it (any True in the column weak_domination_matrix[:,i] )
    is_weakly_dominated = np.any(weak_domination_matrix, axis = 0)
    # weakly but not strongly dominated
    is_weakly_but_not_strongly_dominated = np.logical_and( np.logical_not(is_strongly_dominated) , np.any(weak_domination_matrix, axis = 0) )

    # no other solution weakly dominates it
    is_not_weakly_dominated = np.logical_not(is_weakly_dominated)

    # mo_sol i shares coordinate with j if at least 1 entry i = j
    coordinate_sharing_matrix = np.zeros((n_mo_sol,n_mo_sol), dtype = np.bool_)
    for i in range(0,n_mo_sol):
        cur_col = mo_obj_val[:,i][:,None]
        coordinate_sharing_matrix[i,:] = np.any(cur_col == mo_obj_val,axis = 0)

    ## is not weakly dominated but share some coordinate with another not weakly dominated mo_sol
    # set diagonal entries of coordinate_sharing_matrix to zero to make check work (we do not care of shared coordinates with itself)
    subset_of_coordinate_sharing_matrix_with_diag_zero = copy.copy(coordinate_sharing_matrix)
    np.fill_diagonal(subset_of_coordinate_sharing_matrix_with_diag_zero,False) # inplace operation
    # select subset of columns so that the comparison is for all mo-solutions but only with respect to all non-weakly dominated solutions
    subset_of_coordinate_sharing_matrix_with_diag_zero = subset_of_coordinate_sharing_matrix_with_diag_zero[:,is_not_weakly_dominated]
    # check per row if any coordinate is shared with a non-weakly dominated solution
    has_shared_coordinates = np.any(subset_of_coordinate_sharing_matrix_with_diag_zero,axis = 1)
    is_not_weakly_dominated_and_shares_coordinates_with_other_not_weakly_dominated = np.logical_and( is_not_weakly_dominated , has_shared_coordinates  )

    ## is not weakly dominated and share no coordinate with another not weakly dominated mo_sol
    is_not_weakly_dominated_and_shares_no_coordinates_with_other_not_weakly_dominated = np.logical_and( is_not_weakly_dominated , np.logical_not(has_shared_coordinates)  )


    return(is_strongly_dominated,is_weakly_dominated,is_weakly_but_not_strongly_dominated,is_not_weakly_dominated,is_not_weakly_dominated_and_shares_coordinates_with_other_not_weakly_dominated,is_not_weakly_dominated_and_shares_no_coordinates_with_other_not_weakly_dominated)


def compute_subsets(mo_obj_val,ref_point):
    is_strongly_dominated,is_weakly_dominated,is_weakly_but_not_strongly_dominated, \
    is_not_weakly_dominated,is_not_weakly_dominated_and_shares_coordinates_with_other_not_weakly_dominated, \
    is_not_weakly_dominated_and_shares_no_coordinates_with_other_not_weakly_dominated = compute_domination_properties(mo_obj_val)
    ## Z: indices of mo-solutions for which all partial derivatives are zero
    # E: in exterior of reference space
    E,_ = determine_mo_sol_in_exterior(mo_obj_val,ref_point)
    # S: that are strictly dominated
    S = np.where(is_strongly_dominated == True)
    # Z = E cup S
    Z = np.union1d(E,S)

    ## U: indices of mo-solutions for which some partial derivatives are undefined
    # D: that are non-dominated but have duplicate coordinates
    D = np.where(is_not_weakly_dominated_and_shares_coordinates_with_other_not_weakly_dominated == True)
    # W: that are weakly dominated, i.e. not strictly dominated but there is a weakly better (Pareto dominating) solution
    W = np.where(is_weakly_but_not_strongly_dominated == True)
    # B: that are on the boundary of the reference space
    B,_ = determine_mo_sol_on_ref_boundary(mo_obj_val,ref_point)
    # # U = D cup (W \ E) cup (B \ S)
    # U = np.union1d( np.union1d(D, np.setdiff1d(W,E)) , np.setdiff1d(B,S) )
    #DEVIATION FROM EMMERICH & DEUTZ PAPER:
    # U = (D \ E) cup (W \ E) cup (B \ S)
    U = np.union1d( np.union1d( np.setdiff1d(D,E) , np.setdiff1d(W,E) ) , np.setdiff1d(B,S) )

    ## P: indices of mo-solutions for which all partial derivatives are positive (negative??? which one makes sense in our case)
    # N: that are not Pareto dominated AND have no duplicate coordinate with any other non-dominated solution
    N = np.where(is_not_weakly_dominated_and_shares_no_coordinates_with_other_not_weakly_dominated == True)
    # I: that are in the interior of the reference space
    I,_ = determine_mo_sol_in_interior(mo_obj_val,ref_point)
    # P = N intersect I
    P = np.intersect1d(N,I)

    return(P,U,Z)

def grad_multi_sweep_with_duplicate_handling(mo_obj_val,ref_point):
    # find unique mo_obj_val (it also sorts columns which is unnecessary but gets fixed in when using mapping_indices)
    unique_mo_obj_val, mapping_indices = np.unique(mo_obj_val, axis = 1, return_inverse = True)
    # compute hv_grad for unique mo-solutions
    unique_hv_grad = grad_multi_sweep(unique_mo_obj_val,ref_point)
    # assign the same gradients to duplicate mo_obj_val (and undo the unnecessary sorting)
    hv_grad = unique_hv_grad[:,mapping_indices]
    return(hv_grad)

def grad_multi_sweep(mo_obj_val,ref_point):
    '''
    Based on:
    Emmerich, Michael, and André Deutz.
    "Time complexity and zeros of the hypervolume metrics gradient field."
    EVOLVE-a bridge between probability, set oriented numerics,
    and evolutionary computation III. Springer, Heidelberg, 2014. 169-193.
    '''
    n_obj = mo_obj_val.shape[0]
    n_mo_sol = mo_obj_val.shape[1]
    assert n_obj == len(ref_point)
    hv_grad = np.zeros_like(mo_obj_val)



    P, U, Z = compute_subsets(mo_obj_val,ref_point)
    #####
    if not (len(U) == 0):
        # raise ValueError("Partial derivatives might be only one-sided in indice" + str(U))
        print("Partial derivatives might be only one-sided in indices" + str(U))
        print(mo_obj_val[:,U])
        hv_grad[:,U] = 0

    for k in range(0,n_obj):
        temp_ref_point = copy.copy(ref_point)
        temp_ref_point = np.delete(temp_ref_point,k,axis = 0)
        temp_ref_point = tuple(temp_ref_point)
        hv_instance = HyperVolume(temp_ref_point)
        sorted_P = copy.copy(mo_obj_val[:,P])
        # descending order sorting
        sort_order= np.argsort(-sorted_P[k,:])
        sorted_P = sorted_P[:,sort_order]
        # remove k-th row
        sorted_P = np.delete(sorted_P,k,0)
        # initialize queue by turning array of columns into list of columns
        Q = sorted_P.T.tolist() # it should be possible to delete this row, Q is overwritten in the next line
        Q = list()
        for i in range(sorted_P.shape[1]):
            Q.append(tuple(sorted_P[:,i]))
        queue_index = len(P) # this initialization is actually index of last queue entry +1. The +1 is convenient because the while loop will always update the index by -1, so it all matches up in the end
        T = list()
        while (len(Q) > 0):
            # take last element in list
            q = Q.pop()
            # compute hypervolume contribution of q when added to T

            if len(T) == 0:
                T_with_q = list()
                T_with_q.append(q)
                hv_contribution = hv_instance.compute(T_with_q)
            else:
                T_with_q = copy.copy(T)
                T_with_q.append(q)
                hv_contribution = hv_instance.compute(T_with_q) - hv_instance.compute(T)
            # queue_index is the index of q
            queue_index = queue_index - 1 # -1 because the counter is always lagging 1 and it was initialized with +1
            # mo_sol_index is the index of q in mo_obj_val
            mo_sol_index = P[sort_order[queue_index]]
            hv_grad[k,mo_sol_index] = hv_contribution

            ## add q to T and remove points dominated by q
            # initialize T by q in first iteration
            if len(T) == 0:
                T.append(q)
                # T = q
            else:
               ## remove columns in T that are dominated by q
               # loop through T
                i = 0
                while (i < len(T)):
                    # remove entry if dominated by q
                    if check_weak_domination_in_tuple(q,T[i]):
                        del T[i]
                    else:
                        # if entry not deleted, move to next entry
                        i = i + 1

                # add q to T
                T.append(q)
    return(hv_grad)

def check_weak_domination_in_tuple(tuple_A,tuple_B):
    assert len(tuple_A) == len(tuple_B)
    # initialize as True
    A_weakly_dominates_B = True
    for i in range(len(tuple_B)):
        if tuple_A[i] > tuple_B[i]:
            A_weakly_dominates_B = False
            break
    return(A_weakly_dominates_B)