git remote add mootorch git@github.com:xzhang2523/mootorch.git


# plugins
problem, 
solver (GradSolver, EvoSolver, PSL solver)


# Coding conventions
1. use x_arr to represent a vector, 
    for x_arr and y_arr. shape:
    
    x: (n_prob, n_var)
    y: (n_prob, n_obj)
    pref: (n_prob, n_obj)

2. For problems, use upper letter. 

3. For methods, use small letter. 



# Tricky things along development. 
(1) For MGDA for synthetic problems, x clip with a small margin 1e-5 will help the numerical gradient
problems. 

(2) For EPO the recommended learning rate is 1e-2. 
