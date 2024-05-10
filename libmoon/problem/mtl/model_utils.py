from libmoon.problem.mtl.models import MultiLeNet
from libmoon.problem.mtl.models import FullyConnected

def model_from_dataset(dataset, **kwargs):
    if dataset == 'adult':
        return FullyConnected(**kwargs)
    elif dataset == 'credit':
        return FullyConnected(**kwargs)
    elif dataset == 'compass':
        return FullyConnected(**kwargs)
    elif dataset == 'multi_mnist' or dataset == 'multi_fashion_mnist' or dataset == 'multi_fashion':
        return MultiLeNet(**kwargs)
    else:
        raise ValueError("Unknown model name {}".format(dataset))


dim_dict = {
    'adult' : (88,),
    'credit' : (90,),
    'compass' : (20,),
}
