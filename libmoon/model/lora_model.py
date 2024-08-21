import torch
from torch import nn
from libmoon.util.network import numel
import torch.nn.functional as F

from libmoon.problem.mtl.objectives import from_name
loss_func_arr = from_name( names=['CrossEntropyLoss', 'CrossEntropyLoss'], task_names=['l', 'r'] )
from matplotlib import pyplot as plt



class LeNetTargetLoRA(nn.Module):
    '''
        LeNet target network
    '''
    def __init__(self,
                 kernel_size,
                 n_kernels=10,
                 out_dim=10,
                 target_hidden_dim=50,
                 n_conv_layers=2,
                 n_tasks=2
                 ):

        super().__init__()
        assert len(kernel_size) == n_conv_layers, (
            'kernel size should be the same as the number of conv layers'
            'conv layers holding kernel size for earch conv layer'
        )
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.out_dim = out_dim
        self.n_conv_layers = n_conv_layers
        self.n_tasks = n_tasks
        self.target_hidden_dim = target_hidden_dim


    def forward(self, x, weights=None, A=None, ray=None):
        # weights['conv0.weights'].shape : (bs, 810)
        x = F.conv2d(
            x,
            weight=weights['conv0.weights'].reshape(
                self.n_kernels, 1, self.kernel_size[0], self.kernel_size[0]
            ),
            bias=weights['conv0.bias'],
            stride=1,
        )

        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        for i in range(1, self.n_conv_layers):
            if i == 1:
                base_weight = weights[f"conv{i}.weights"]
                base_weight = base_weight + (ray[0] * A).squeeze()

            if i==1:
                x = F.conv2d(
                    x,
                    base_weight.reshape(
                        int(2 ** i * self.n_kernels), int(2 ** (i - 1) * self.n_kernels), self.kernel_size[i],
                        self.kernel_size[i]
                    ),
                    bias=weights[f"conv{i}.bias"],
                    stride=1,
                )
            else:
                x = F.conv2d(
                    x,
                    weight=weights[f"conv{i}.weights"].reshape(
                        int(2 ** i * self.n_kernels), int(2 ** (i - 1) * self.n_kernels), self.kernel_size[i],
                        self.kernel_size[i]
                    ),
                    bias=weights[f"conv{i}.bias"],
                    stride=1,
                )
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = F.linear(
            x,
            weight=weights['hidden0.weights'].reshape(
                self.target_hidden_dim, x.shape[-1]
            ),
            bias=weights['hidden0.bias'],
        )

        logits = []
        for j in range(self.n_tasks):
            logits.append(
                F.linear(
                    x,
                    weight=weights[f"task{j}.weights"].reshape(
                        self.out_dim, self.target_hidden_dim
                    ),
                    bias=weights[f"task{j}.bias"],
                )
            )
        return logits



if __name__ == '__main__':
    import argparse
    from libmoon.problem.mtl.loaders import MultiMNISTData
    from libmoon.util_global.constant import get_device
    from tqdm import tqdm
    import numpy as np
    from mtl import HyperNet

    parse = argparse.ArgumentParser()
    parse.add_argument('--n-epoch', type=int, default=5)
    parse.add_argument('--batch-size', type=int, default=128)
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--n-obj', type=int, default=2)
    parse.add_argument('--model', type=str, default='lenet')
    parse.add_argument('--ray-hidden', type=int, default=100)


    from torch.autograd import Variable

    dataset = MultiMNISTData('mnist', 'train')
    args = parse.parse_args()
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=0)
    dataset_test = MultiMNISTData('mnist', 'test')
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True,
                                              num_workers=0)

    args.device = get_device()
    if args.model == 'lenet':
        hnet = HyperNet([9, 5]).to(args.device)
        net = LeNetTargetLoRA([9, 5]).to(args.device)
    else:
        assert False, 'model not supported'

    loss_history = []
    A = Variable(torch.Tensor(np.random.randn(5000, 1))).to(args.device)
    A.requires_grad = True


    ray_input = torch.Tensor([0.5, 0.5]).to(args.device)
    weights = hnet(ray_input)

    for k,v in weights.items():
        weights[k] = Variable(v, requires_grad=True).to(args.device)

    # print()
    optimized_var = list(weights.values())

    A_optimizer = torch.optim.Adam([A]+optimized_var, lr=args.lr)




    print('Training...')
    for idx in tqdm(range(args.n_epoch)):
        for i, batch in enumerate(loader):
            ray = torch.from_numpy(
                np.random.dirichlet((1, 1), 1).astype(np.float32).flatten()
            ).to(args.device)  # ray.shape (1,2)

            batch_ = {}
            for k, v in batch.items():
                batch_[k] = v.to(args.device)


            # dict_keys(['conv0.weights', 'conv0.bias', 'conv1.weights', 'conv1.bias', 'hidden0.weights', 'hidden0.bias',
            #            'task0.weights', 'task0.bias', 'task1.weights', 'task1.bias'])

            logits_l, logits_r = net(batch_['data'], weights, A, ray)

            batch_['logits_l'] = logits_l
            batch_['logits_r'] = logits_r
            loss_arr = torch.stack([loss(**batch_) for loss in loss_func_arr])
            A_optimizer.zero_grad()

            loss_arr = torch.atleast_2d(loss_arr)
            ray = torch.atleast_2d(ray)
            # print()
            loss = torch.sum(loss_arr * ray )

            (loss).backward()
            loss_item = loss.cpu().detach().numpy()
            loss_history.append(loss_item)
            A_optimizer.step()


    ray_1d = torch.linspace(0.1, 0.9, 10).to(args.device)
    ray_2d = torch.stack([ray_1d, 1 - ray_1d], dim=1).to(args.device)


    print('Evaluating...')
    loss_ray = []

    for idx, ray in tqdm(enumerate(ray_2d)):
        loss_batch_arr = []
        for i, batch in enumerate(loader):


            batch_ = {}
            for k, v in batch.items():
                batch_[k] = v.to(args.device)
            # ray = torch.from_numpy(
            #     np.random.dirichlet((1, 1), 1).astype(np.float32).flatten()
            # ).to(args.device)  # ray.shape (1,2)
            ray_input = torch.Tensor([0.5, 0.5]).to(args.device)
            weights = hnet(ray_input)
            # dict_keys(['conv0.weights', 'conv0.bias', 'conv1.weights', 'conv1.bias', 'hidden0.weights', 'hidden0.bias',
            #            'task0.weights', 'task0.bias', 'task1.weights', 'task1.bias'])
            logits_l, logits_r = net(batch_['data'], weights, A, ray)
            batch_['logits_l'] = logits_l
            batch_['logits_r'] = logits_r
            loss_arr = torch.stack([loss(**batch_) for loss in loss_func_arr])
            loss_batch_arr.append( loss_arr.detach().cpu().numpy() )

        loss_batch_arr = np.array(loss_batch_arr)
        loss_ray.append( np.mean(loss_batch_arr, axis=0) )

    loss_ray = np.array(loss_ray)

    plt.scatter(loss_ray[:,0], loss_ray[:,1])
    plt.plot(loss_ray[:, 0], loss_ray[:, 1])
    plt.show()

    print('done')

