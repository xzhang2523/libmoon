import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, z_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# Define the decoder part of the VAE
class Decoder(nn.Module):
    def __init__(self, z_dim=20, hidden_dim=400, output_dim=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))


# Define the overall VAE model
class VAE(nn.Module):
    def __init__(self, z_dim=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim=z_dim)
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def generate_img(self, z):
        return self.decoder(z)


# Define the VAE loss function
def vae_loss(recon_x, x, mu, logvar):
    '''
        :param recon_x: reconstructed x.
    '''
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def evaluate_vae(mu, logvar, vae_model):
    hidden = vae_model.reparameterize(mu, logvar)
    predict = vae_model.generate_img(hidden)
    return predict


# Training the VAE
def train_vae(model, train_loader_1, train_loader_2, optimizer, epochs=10):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader_1):
            data = data.view(-1, 784)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss_history.append(loss.item())
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader_1.dataset)}')

    return mu, logvar, loss_history




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE example')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset_1 = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_dataset_2 = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)

    train_loader_1 = DataLoader(train_dataset_1, batch_size=args.batch_size, shuffle=True)
    train_loader_2 = DataLoader(train_dataset_2, batch_size=args.batch_size, shuffle=True)

    # Instantiate the VAE model and optimizer
    vae = VAE()
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)

    mu, logvar, loss_history = train_vae(vae, train_loader_1, train_loader_2, optimizer, epochs=args.epochs)
    res = evaluate_vae(mu, logvar, vae)
    # plot 8 images
    fig, axes = plt.subplots(2, 4, figsize=(8, 8))  # 4 rows, 2 columns

    # Plot the first 8 images
    for i in range(8):
        img = res[i].view(28, 28).detach().numpy()  # Assuming each image is 28x28
        ax = axes[i // 4, i % 4]  # Determine position in grid
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # Turn off the axis labels for clarity

    plt.tight_layout()
    plt.show()