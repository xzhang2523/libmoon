import torch.nn as nn

'''
    MTL problems need . 
'''



class MultiLeNet(nn.Module):

    def __init__(self, dim, **kwargs):

        '''
            :param dim: a 3d-array. [chanel, height, width]
            :param kwargs:
        '''
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(dim[0], 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(720, 50),
            nn.ReLU(),
        )
        self.private_left = nn.Linear(50, 10)
        self.private_right = nn.Linear(50, 10)


    def forward(self, batch):
        x = batch['data']
        x = self.shared(x)
        return dict(logits_l=self.private_left(x), logits_r=self.private_right(x))

    def private_params(self):
        return ['private_left.weight', 'private_left.bias', 'private_right.weight', 'private_right.bias']





class FullyConnected(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(dim[0], 60),
            nn.ReLU(),
            nn.Linear(60, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
        )

    def forward(self, batch):
        x = batch['data']
        return dict(logits=self.f(x))





if __name__ == '__main__':
    from libmoon.util_global.constant import root_name
    import os
    import pickle




    pickle_name = os.path.join(root_name, 'problem', 'mtl', 'data', 'multimnist', 'mnist.pickle')
    with open(pickle_name, 'rb') as f:
        data = pickle.load(f)

    model = MultiLeNet([3, 32, 32])
    print('hello world')