'''
    This file contains the utility functions for the gradient descent solver.
'''
import torch

def get_grads_from_model(loss, model):
    G = [0,] * len(loss)
    for idx, l in enumerate(loss):
        model.zero_grad()
        l.backward(retain_graph=True)
        G[idx] = get_flatten_grad(model)
    return torch.stack(G)

def get_flatten_grad(model):
    grad = []
    for param in model.parameters():
        if param.grad is not None:
            grad.append(param.grad.view(-1))
        else:
            grad.append(torch.zeros_like(param.view(-1)))
    grad = torch.cat(grad)
    return grad


def numel_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
