import numpy as np

def objective_function(x):
    # Define your objective function here
    return np.array([0.5 * x**2, 0.5 * (x - 0.5)**2])

def gES(theta, sigma, P):
    sum_list = np.zeros(2)
    for i in range(P):
        noise = np.random.normal()
        result = objective_function(theta + sigma * noise)

        sum_list += (result - objective_function(theta - sigma * noise) ) / noise

    return sum_list / (2 * sigma * P)



if __name__ == '__main__':
    # Example usage
    theta = 0.5
    sigma = 0.1
    P = 100

    result = gES(theta, sigma, P)
    print("Result:", result)