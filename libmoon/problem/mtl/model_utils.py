from libmoon.problem.mtl.models import MultiLeNet
from libmoon.problem.mtl.models import FullyConnected


def model_from_dataset(dataset_name, architecture, **kwargs):
    if dataset_name == 'adult':
        return FullyConnected(architecture, **kwargs)
    elif dataset_name == 'credit':
        return FullyConnected(architecture, **kwargs)
    elif dataset_name == 'compass':
        return FullyConnected(architecture, **kwargs)
    elif dataset_name in ['mnist','fashion','fmnist'] :
        return MultiLeNet(**kwargs)
    else:
        raise ValueError("Unknown model name {}".format(dataset_name))



dim_dict = {
    'adult' : (88,),
    'credit' : (90,),
    'compass' : (20,),
    'mnist' : (1,36,36),
    'fashion' : (1,36,36),
    'fmnist' : (1,36,36),
}
