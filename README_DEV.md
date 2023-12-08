git remote add mootorch git@github.com:xzhang2523/mootorch.git

# plugins
problem
solver


# conventions
1. use x_arr to represent a vector, 
    for x_arr and y_arr. shape:
    
    x: (n_prob, n_var)
    y: (n_prob, n_obj)
    pref: (n_prob, n_obj)





# Tricky thing along development. 
(1) For MGDA for synthetic problems, x clip with a small margin 1e-5 will help the numerical gradient
problems. 

(2) For EPO the recommended learning rate is 1e-2. 

(3) 


