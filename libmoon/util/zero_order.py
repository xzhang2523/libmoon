import numpy as np
from libmoon.util.constant import get_problem, FONT_SIZE, get_agg_func


def ES_gradient_estimation_batch(problem, x0_batch, sigma=0.1, P=20):
    g_arr = [ES_gradient_estimation(problem, x0[np.newaxis, :], sigma, P) for x0 in x0_batch]
    res = np.stack(g_arr, axis=0)
    return res

def ES_gradient_estimation(problem, theta, sigma, P):
    n = theta.shape[-1]
    sum_list = np.zeros( (problem.n_obj, n))
    for i in range(P):
        noise = np.random.normal(size=theta.shape)
        new_x1 = (theta + sigma * noise)
        new_x2 = (theta - sigma * noise)
        if hasattr(problem, 'lbound'):
            new_x1 = np.clip(new_x1, problem.lbound, problem.ubound)
            new_x2 = np.clip(new_x2, problem.lbound, problem.ubound)
        result1 = problem.evaluate(new_x1)
        result2 = problem.evaluate(new_x2)
        sum_list += ((result1 - result2).T) @ noise
    return sum_list / (2 * sigma * P)


if __name__ == '__main__':
    # Example usage
    # Initialize theta as a 2D vector
    theta = np.array([0.5, 0.5])
    sigma = 0.1
    P = 100


