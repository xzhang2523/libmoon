
def numel(model):
    if type(model) == dict:
        return sum(p.numel() for p in model.values() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)