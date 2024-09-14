import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Define the Encoder network
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# Define the Decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))
        return x_recon


# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


# Loss function: VAE loss = Reconstruction loss + KL divergence
def loss_function(recon_x, x, mu, logvar):
    # Binary Cross-Entropy reconstruction loss
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL Divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# Training the VAE
def train_vae(model, dataloader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.view(-1, 28 * 28)  # Flatten the image (for MNIST)
            optimizer.zero_grad()
            # Forward pass
            recon_data, mu, logvar = model(data)
            # Compute loss
            loss = loss_function(recon_data, data, mu, logvar)
            # Backprop and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')



# Main code: Load MNIST dataset and train the VAE
if __name__ == '__main__':
    # Load MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    # Define model, optimizer
    input_dim = 28 * 28  # For MNIST
    hidden_dim = 400
    latent_dim = 20
    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    train_vae(model, train_loader, optimizer, epochs=10)
