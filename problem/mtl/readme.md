MTL steps.

(1) If preference-based method (agg, pmgda, epo, ....)
    
    Step 1:
        get losses, get grad
    
    Step 2:
        get alpha and then backward propogation. 


(2) If set-based method, (Grad-hv, moo-svgd)

    Step 1:
        get losses_arr, get grad_arr

    Step 2:
        get alpha_arr / gw_arr and then backward propogation.