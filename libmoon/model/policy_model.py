import torch
from torch import nn
import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, state_space, action_space):
        super(Policy, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.action_range = action_space[-1]

        # Define a simple neural network for policy estimation
        self.base_model = nn.Sequential(
            nn.Linear(self.state_space, 60),  # Input layer
            nn.ReLU(),  # Activation function
            nn.Linear(60, 25),  # Hidden layer
            nn.ReLU(),  # Activation function
            nn.Linear(25, self.action_space)  # Output layer (number of actions)
        )

    def forward(self, state, stochastic=True):
        # Pass the state through the network
        logits = self.base_model(state)

        # Convert logits to probabilities using a softmax function
        probabilities = torch.softmax(logits, dim=-1)
        if stochastic:
            # Sample an action according to the action probabilities
            action = torch.multinomial(probabilities, num_samples=1)
        else:
            # Select the action with the highest probability
            action = torch.argmax(probabilities, dim=-1)

        return action

