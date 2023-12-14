import torch



# Define the function for which to compute the vector-Jacobian product
def func(x):
    return torch.tensor([
        x[0] + 2 * x[1],
        x[1]**2 - x[2],
        3 * x[0] * x[2]
    ])




if __name__ == '__main__':
    # Define the input variables
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    # Compute the function evaluation
    y = func(x)
    jac = torch.autograd.functional.jacobian(y, x)
    # Compute the vector-Jacobian product
    # v = torch.tensor([1.0, 2.0, 3.0])
    # vjp = torch.autograd.grad(y, x, v, retain_graph=True)
    # print(vjp)