from torch import nn


class FullyConnected(nn.Module):
    def __init__(self, dim, architecture='M1'):
        super().__init__()

        if not architecture in ['M4', 'M1', 'M2', 'M3']:
            self.f = nn.Sequential(
                nn.Linear(dim[0], 60),
                nn.ReLU(),
                nn.Linear(60, 25),
                nn.ReLU(),
                nn.Linear(25, 1),
            )
        elif architecture == 'M1':
            self.f = nn.Sequential(
                nn.Linear(dim[0], 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        elif architecture == 'M2':
            self.f = nn.Sequential(
                nn.Linear(dim[0], 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        elif architecture == 'M3':
            self.f = nn.Sequential(
                nn.Linear(dim[0], 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        else:
            self.f = nn.Sequential(
                nn.Linear(dim[0], 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

    def forward(self, data):
        # x = batch['data']
        return dict(logits=self.f(data))
