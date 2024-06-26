"""
Runing the psl+lcb method on ZDT1
 
"""
import torch
import time
  
  
from  methods.pslmobo_solver import PSLMOBOSolver  

import matplotlib.pyplot as plt
 
torch.set_default_dtype(torch.float64)

from libmoon.problem.synthetic.zdt import ZDT1
 
MAX_FE = 200
BATCH_SIZE = 5  # batch size   
# minimization
problem =  ZDT1(n_var=8,n_obj=2)
n_init = 11*problem.n_var-1 
ts = time.time()    
solver_lcb = PSLMOBOSolver(problem, n_init, MAX_FE, BATCH_SIZE) 
res = solver_lcb.solve() 
elapsed = time.time() - ts
res['elapsed'] = elapsed 



use_fig = True
if use_fig:
    fig = plt.figure()
    plt.scatter(res['y'][res['idx_nds'][0],0], res['y'][res['idx_nds'][0],1], label='Solutions')
    plt.plot(problem.get_pf()[:,0], problem.get_pf()[:,1], label='PF')
 
    plt.legend(fontsize=16)
    plt.xlabel('$f_1$', fontsize=18)
    plt.ylabel('$f_2$', fontsize=18)
 
    plt.show()
 
   
    
 
 
 
       
    
 
 

