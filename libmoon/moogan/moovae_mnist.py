import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F


# Define autoencoder model
class autoencoder(nn.Module):
    def __init__(
            self,
            x_dim,
            hidden_dim,
            z_dim=10
    ):
        super(autoencoder, self).__init__()

        # Define autoencoding layers
        self.enc_layer1 = nn.Linear(x_dim, hidden_dim)
        self.enc_layer2 = nn.Linear(hidden_dim, z_dim)

        # Define autoencoding layers
        self.dec_layer1 = nn.Linear(z_dim, hidden_dim)
        self.dec_layer2 = nn.Linear(hidden_dim, x_dim)

    def encoder(self, x):
        # Define encoder network
        x = F.relu(self.enc_layer1(x))
        z = F.relu(self.enc_layer2(x))
        return z

    def decoder(self, z):
        # Define decoder network
        output = F.relu(self.dec_layer1(z))
        output = F.relu(self.dec_layer2(output))
        return output

    def forward(self, x):
        # Define the full network
        z = self.encoder(x)
        output = self.decoder(z)
        return output


def train_model(X, learning_rate=1e-3, batch_size=128, num_epochs=15):
    # Create DataLoader object to generate minibatches
    X = torch.tensor(X).float()
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate model and optimizer
    model = autoencoder(x_dim=X.shape[1], hidden_dim=256, z_dim=50)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define the loss function
    def loss_function(output, x):
        recon_loss = F.mse_loss(output, x, reduction='sum')
        return recon_loss

    # Train the model
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Get batch
            x = batch[0]

            # Forward pass
            output = model(x)

            # Calculate loss
            loss = loss_function(output, x)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Add batch loss to epoch loss
            epoch_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(X)}")

