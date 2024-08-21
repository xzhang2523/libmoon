import torch
from torch import nn




class PFLModel(nn.Module):
    def __init__(self, n_obj):
        super(PFLModel, self).__init__()


        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, n_obj)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
