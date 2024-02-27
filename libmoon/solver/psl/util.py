

import numpy as np

def get_random_prefs(batch_size, n_obj):
    return np.random.dirichlet(np.ones(n_obj), batch_size)



# if __name__ == '__main__':
#     prefs = get_random_prefs(10, 2)
    # print()
