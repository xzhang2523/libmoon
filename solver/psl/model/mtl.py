import torch
from torch import nn
from util_global.constant import get_param_num


class HyperNet(nn.Module):
    def __init__(self,
        kernel_size,
        ray_hidden_dim=100,
        out_dim=10,
        target_hidden_dim=50,
        n_kernels=10,
        n_conv_layers=2,
        n_hidden=1,
        n_tasks=2):

        super().__init__()
        self.n_conv_layers = n_conv_layers
        self.n_hidden = n_hidden
        self.n_tasks = n_tasks

        assert len(kernel_size) == n_conv_layers, (
            'kernel size should be the same as the number of conv layers'
            'conv layers holding kernel size for earch conv layer'
        )

        self.ray_mlp = nn.Sequential(
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim)
        )

        self.conv_0_weights = nn.Linear(
            ray_hidden_dim, n_kernels * kernel_size[0] * kernel_size[0]
        )
        self.conv_0_bias = nn.Linear(ray_hidden_dim, n_kernels)

        for i in range(1, n_conv_layers):
            # previous number of kernels
            p = 2**(i-1)*n_kernels
            # current number of kernels
            c = 2**i*n_kernels

            setattr(
                self, f"conv_{i}_weights", nn.Linear(ray_hidden_dim, c*p*kernel_size[i]*kernel_size[i]),
            )
            setattr(self, f"conv_{i}_bias", nn.Linear(ray_hidden_dim, c))

        latent = 25
        self.hidden_0_weights = nn.Linear(
            ray_hidden_dim, target_hidden_dim*2**i*n_kernels*latent
        )
        self.hidden_0_bias=nn.Linear(ray_hidden_dim, target_hidden_dim)

        for j in range(n_tasks):
            setattr(
                self,
                f"task_{j}_weights",
                nn.Linear(ray_hidden_dim, target_hidden_dim*out_dim)
            )
            setattr(self, f"task_{j}_bias", nn.Linear(ray_hidden_dim, out_dim))


    def shared_parameters(self):
        return list([p for n,p in self.named_parameters() if 'task' not in n])

    def forward(self, ray):
        features = self.ray_mlp(ray)
        # features.shape: (batch_size, ray_hidden_dim)
        out_dict={}  # task 1, task 2. Task specfic parameters.

        layer_types = ['conv', 'hidden', 'task']
        for i in layer_types:
            if i == 'conv':
                n_layers = self.n_conv_layers
            elif i == 'hidden':
                n_layers = self.n_hidden
            elif i == 'task':
                n_layers = self.n_tasks

            for j in range(n_layers):
                out_dict[f"{i}{j}.weights"] = getattr(self, f"{i}_{j}_weights")(
                    features
                )
                out_dict[f"{i}{j}.bias"] = getattr(self, f"{i}_{j}_bias")(
                    features
                ).flatten()
        return out_dict




if __name__ == '__main__':
    prefs = torch.rand(10, 2)
    hyper_model = HyperNet(kernel_size=[5,5])
    model_num = get_param_num(hyper_model)   #model num: 3,186,850. It is too large, unacceptable.
    print('model_num', model_num)
    out_dict = hyper_model(prefs)
    # print()