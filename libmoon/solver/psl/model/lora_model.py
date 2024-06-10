import torch
from torch import nn
from libmoon.util_global.constant import get_param_num
import torch.nn.functional as F

class LeNetTarget(nn.Module):
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


    def forward(self, x, weights=None):
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

    print()




