from torch import nn


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

        # print()
        return dict(logits=self.f(x))