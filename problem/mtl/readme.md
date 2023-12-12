MTL steps.

    Step 0:
        Define the loss function.

Type (1), If preference-based method (agg, pmgda, epo, ....)
    
    Step 1:
        get losses, get grad
    
    Step 2:
        get alpha and then backward propogation. 

Type (2), If set-based method, (Grad-hv, moo-svgd)

    Step 1:
        get losses_arr, get grad_arr

    Step 2:
        get alpha_arr / gw_arr and then backward propogation.

    It is noted that, the original code moo-svgd does not offer a MTL implement. 
