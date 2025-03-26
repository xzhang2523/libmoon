import torch



def get_moo_Jacobian_batch(x_batch, y_batch, n_obj):
    '''
        Input : x_batch: (batch_size, n_var)
                y_batch: (batch_size, n_obj)
                n_obj: int
        Return: grad_batch: (batch_size, n_obj, n_var)
    '''
    grad_batch = []
    batch_size = len(x_batch)
    for batch_idx in range(batch_size):
        # grad_arr.append(get_moo_Jacobian(xs, ys, n_obj))
        grad_arr_idx = [0,] * n_obj
        for obj_idx in range(n_obj):
            y_batch[batch_idx][obj_idx].backward(retain_graph = True)
            grad_arr_idx[obj_idx] = x_batch.grad[batch_idx].clone()
            x_batch.grad.zero_()
        grad_batch.append( torch.stack(grad_arr_idx) )
    return torch.stack(grad_batch)

def get_moo_Jacobian(x, y, n_obj):
    grad_arr = [0] * n_obj
    for obj_idx in range(n_obj):
        y[obj_idx].backward(retain_graph=True)
        grad_arr[obj_idx] = x.grad.clone()
        x.grad.zero_()
    grad_arr = torch.stack(grad_arr)
    return grad_arr

def flatten_grads(grads_dict):
    return torch.cat( [v.view(-1) for _, v in grads_dict.items()] )

def calc_gradients_mtl(data, batch, model, objectives):
    # store gradients and objective values
    gradients = []
    obj_values = []
    for i, objective in enumerate(objectives):
        model.zero_grad()
        logits = model(data)
        output = objective(logits['logits'], **batch)
        output.backward()
        obj_values.append(output.item())
        gradients.append({})
        private_params = model.private_params() if hasattr(model, 'private_params') else []
        for name, param in model.named_parameters():
            not_private = all([p not in name for p in private_params])
            if not_private and param.requires_grad and param.grad is not None:
                gradients[i][name] = param.grad.data.detach().clone()
    return gradients

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