from typing import Optional
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import math
from torch import Tensor
import random


class LoRAModule(nn.Module):
    members: nn.ModuleList

    def __init__(self, n, reinit):
        super().__init__()
        self.n = n
        self.reinit = reinit

    def _clone(self, m: nn.Module) -> nn.Module:
        return copy.deepcopy(m)

    def get_weight(self):
        raise NotImplementedError

    def _create_new_(self, module: nn.Module):
        module = self._clone(module)
        if self.reinit:
            module.reset_parameters()
        return module

    def reset_parameters(self):
        for m in self.members:
            m.reset_parameters()


class LoRAConv(LoRAModule):
    def __init__(self, module: nn.Conv2d, n: int, reinit: bool, r: int, lora_alpha: float, gamma_init: float):
        super().__init__(n, reinit)
        self.main_module = module
        # copy the attributes of the conv layer
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.kernel_size = module.kernel_size[0]
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.stride = module.stride
        self.has_bias = module.bias is not None
        self.r = min(r, self.in_channels, self.out_channels)
        self.lora_alpha = lora_alpha

        # print(r, self.kernel_size)
        self.lora_a_members = nn.ParameterList([nn.Parameter(
            module.weight.new_zeros((self.r * self.kernel_size, self.in_channels * self.kernel_size)))
            for _ in range(n)])
        self.lora_b_members = nn.ParameterList([nn.Parameter(
            module.weight.new_zeros((self.out_channels // self.groups * self.kernel_size, self.r * self.kernel_size)))
            for _ in range(n)])

        if self.has_bias:
            self.bais_members = nn.ParameterList(
                [nn.Parameter(module.bias.new_zeros(self.out_channels)) for _ in range(n)])

        self.main_module.reset_parameters()

        for lora_a in self.lora_a_members:
            nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        for lora_b in self.lora_b_members:
            nn.init.zeros_(lora_b)
        if self.has_bias:
            for i, bias in enumerate(self.bais_members):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lora_a_members[i])
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(bias, -bound, bound)

    def get_weight(self):
        # print(self.lora_alpha)
        # self.gamma = torch.clamp(self.gamma, 0.5, 1)
        #  (1/self.n + torch.sigmoid(self.gamma)*(self.alpha[i]-1/self.n))
        w = self.main_module.weight + self.lora_alpha * sum(
            [(self.lora_b_members[i] @ self.lora_a_members[i]).view(self.main_module.weight.shape) *
             self.alpha[i] for i in range(self.n)])
        if self.has_bias:
            b = self.main_module.bias + self.lora_alpha * sum(
                [self.bais_members[i] * self.alpha[i] for i in range(self.n)])
        else:
            b = None
        return w, b

    def get_othogonal_loss(self):
        index_set = range(self.n)
        if self.n > 3:
            index_set = random.sample(range(self.n), 3)
        loss = 0
        eye = torch.eye(len(index_set)).to(self.lora_a_members[0].device)
        # one_matrix = torch.ones((len(index_set), len(index_set))).to(self.lora_a_members[0].device)
        eye.requires_grad = False
        # one_matrix.requires_grad = False
        W = torch.stack([(self.lora_b_members[i] @ self.lora_a_members[i]).view(-1) for i in index_set])
        W_norm = F.normalize(W, dim=1)
        # W_norm = W
        loss += torch.sum(torch.square(W_norm.mm(W_norm.transpose(0, 1)) - eye))
        # if self.has_bias:
        #     B = torch.stack([(self.bais_members[i]).view(-1) for i in range(self.n)])
        #     B_norm = F.normalize(B, dim=1)
        #     loss += torch.sum(torch.square(B_norm.mm(B_norm.transpose(0, 1)) - eye))
        return loss

    def get_similarity_loss(self):
        # loss = 0
        # for i in range(self.n):
        #     loss += torch.square(torch.cosine_similarity((self.lora_b_members[i]@self.lora_a_members[i]).view(-1), self.main_module.weight.view(-1), dim=0)-0)
        #     if self.has_bias:
        #         loss += torch.square(torch.cosine_similarity((self.bais_members[i]).view(-1), self.main_module.bias.view(-1), dim=0)-0)
        index_set = range(self.n)
        if self.n >= 5:
            index_set = random.sample(range(self.n), 5)
        loss = 0
        eye = torch.eye(len(index_set)).to(self.lora_a_members[0].device)
        eye.requires_grad = False
        W = torch.stack([(self.lora_b_members[i] @ self.lora_a_members[i]).view(-1) for i in index_set])
        W_norm = F.normalize(W, dim=1)
        loss += torch.mean((W_norm.mm(W_norm.transpose(0, 1)) - eye))
        return loss

    def __repr__(self):
        return "LoRAConv(n={}, {}, {}, kernel_size={}, stride={})".format(
            self.n, self.in_channels, self.out_channels, self.kernel_size, self.stride
        )

    # def retrieve_member(self, index):
    #     w = self.members[index].weight
    #     b = self.members[index].bias
    #     return w, b

    # def retrieve_member_weight(self, index):
    #     return self.members[index].weight

    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w, b = self.get_weight()
        x = F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
        return x


class LoRALinear(LoRAModule):
    def __init__(self, module: nn.Linear, n: int, reinit: bool, r: int, lora_alpha: float, gamma_init: float):
        super().__init__(n, reinit)
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.r = min(r, self.in_features, self.out_features)
        self.lora_alpha = lora_alpha
        self.main_module = module
        self.lora_a_members = nn.ParameterList(
            [nn.Parameter(module.weight.new_zeros((self.r, self.in_features))) for _ in range(n)])
        self.lora_b_members = nn.ParameterList(
            [nn.Parameter(module.weight.new_zeros((self.out_features, self.r))) for _ in range(n)])
        self.bais_members = nn.ParameterList([nn.Parameter(module.bias.new_zeros(self.out_features)) for _ in range(n)])
        self.main_module.reset_parameters()

        for lora_a in self.lora_a_members:
            nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        for lora_b in self.lora_b_members:
            nn.init.zeros_(lora_b)
        for i, bias in enumerate(self.bais_members):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lora_a_members[i])
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(bias, -bound, bound)
        # else:
        #     self.members = nn.ModuleList([self._create_new_(module) for _ in range(n)])

    def get_weight(self):
        # print(self.lora_alpha)
        # if self.use_lora:
        # self.gamma = torch.clamp(self.gamma, 0.5, 1)
        w = self.main_module.weight + self.lora_alpha * sum([(self.lora_b_members[i] @ self.lora_a_members[i]) *
                                                             self.alpha[i] for i in range(self.n)])
        b = self.main_module.bias + self.lora_alpha * sum([self.bais_members[i] *
                                                           self.alpha[i] for i in range(self.n)])
        # else:
        #     w = sum([self.members[i].weight * self.alpha[i] for i in range(self.n)])
        #     b = sum([self.members[i].bias * self.alpha[i] for i in range(self.n)])
        return w, b

    def get_othogonal_loss(self):
        index_set = range(self.n)
        if self.n > 3:
            index_set = random.sample(range(self.n), 3)
        loss = 0
        eye = torch.eye(len(index_set)).to(self.lora_a_members[0].device)
        eye.requires_grad = False
        W = torch.stack([(self.lora_b_members[i] @ self.lora_a_members[i]).view(-1) for i in index_set])
        W_norm = F.normalize(W, dim=1)
        loss += torch.sum(torch.square(W_norm.mm(W_norm.transpose(0, 1)) - eye))
        # B = torch.stack([(self.bais_members[i]).view(-1) for i in range(self.n)])
        # B_norm = F.normalize(B, dim=1)
        # loss += torch.sum(torch.square(B_norm.mm(B_norm.transpose(0, 1)) - eye))
        return loss

    def get_similarity_loss(self):
        # loss = 0
        # for i in range(self.n):
        #     loss += torch.square(torch.cosine_similarity((self.lora_b_members[i]@self.lora_a_members[i]).view(-1), self.main_module.weight.view(-1), dim=0)-0)
        #     loss += torch.square(torch.cosine_similarity((self.bais_members[i]).view(-1), self.main_module.bias.view(-1), dim=0)-0)
        index_set = range(self.n)
        if self.n >= 5:
            index_set = random.sample(range(self.n), 5)
        loss = 0
        eye = torch.eye(len(index_set)).to(self.lora_a_members[0].device)
        eye.requires_grad = False
        W = torch.stack([(self.lora_b_members[i] @ self.lora_a_members[i]).view(-1) for i in index_set])
        W_norm = F.normalize(W, dim=1)
        loss += torch.mean(W_norm.mm(W_norm.transpose(0, 1)) - eye)
        return loss

    def __repr__(self) -> str:
        return "LoRALinear(n={}, in_features={}, out_features={})".format(
            self.n, self.in_features, self.out_features is not None
        )

    # def retrieve_member(self, index):
    #     w = self.members[index].weight
    #     b = self.members[index].bias

    #     return w, b

    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w, b = self.get_weight()
        x = F.linear(x, w, b)
        return x


class LoRABatchNorm2d(LoRAModule):
    "pretty much copy pasted from official pytorch source code"

    def __init__(self, module: nn.BatchNorm2d, n: int, reinit: bool, lora_alpha: float, gamma_init: float):
        super().__init__(n, reinit)
        self.main_module = module
        self.members = nn.ModuleList([self._create_new_(module) for _ in range(n)])
        self.lora_alpha = lora_alpha
        # self.lora_alpha = nn.Parameter(torch.FloatTensor([1]))
        # copy the attributes of the bn layer
        self.num_features = module.num_features
        self.num_features = module.num_features
        self.eps = module.eps
        self.momentum = module.momentum
        self.affine = module.affine
        self.track_running_stats = module.track_running_stats
        # self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))
        # self.gamma.requires_grad = False

        factory_kwargs = {"device": None, "dtype": None}

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(self.num_features, **factory_kwargs))
            self.register_buffer("running_var", torch.ones(self.num_features, **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(0, dtype=torch.long, **{k: v for k, v in factory_kwargs.items() if k != "dtype"}),
            )
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def get_weight(self):
        # print(self.lora_alpha)
        # self.gamma = torch.clamp(self.gamma, 0.5, 1)
        w = self.main_module.weight + self.lora_alpha * sum(
            [self.members[i].weight * self.alpha[i] for i in range(self.n)])
        b = self.main_module.bias + self.lora_alpha * sum([self.members[i].bias * self.alpha[i] for i in range(self.n)])
        return w, b


    def get_othogonal_loss(self):
        loss = 0
        # eye = torch.eye(self.n).to(self.main_module.weight.device)
        # eye.requires_grad = False
        # W = torch.stack([(self.members[i].weight).view(-1) for i in range(self.n)])
        # W_norm = F.normalize(W, dim=1)
        # loss += torch.sum(torch.square(W_norm.mm(W_norm.transpose(0, 1)) - eye))
        # B = torch.stack([(self.members[i].bias).view(-1) for i in range(self.n)])
        # B_norm = F.normalize(B, dim=1)
        # loss += torch.sum(torch.square(B_norm.mm(B_norm.transpose(0, 1)) - eye))
        return loss

    def __repr__(self):
        return "LoRABatchNorm2d(n={}, {num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}".format(
            self.n, **self.__dict__
        )

    def retrieve_member(self, index):
        w = self.members[index].weight
        b = self.members[index].bias
        return w, b

    def retrieve_member_weight(self, index):
        return self.members[index].weight

    def forward(self, input):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w, b = self.get_weight()
        # The rest is code in the PyTorch source forward pass for batchnorm.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            w,
            b,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
