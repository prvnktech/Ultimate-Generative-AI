import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Device configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Hyperparameters
# -----------------------------
input_dim = 28 * 28   # MNIST images
hidden_dim = 128
latent_dim = 20
batch_size = 128
epochs = 10
learning_rate = 1e-3

# -----------------------------
# Dataset (MNIST)
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# -----------------------------
# VAE Model
# -----------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

# -----------------------------
# Loss Function
# -----------------------------
def vae_loss(recon_x, x, mu, logvar):
    reconstruction_loss = nn.functional.binary_cross_entropy(
        recon_x, x, reduction='sum'
    )
    kl_divergence = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    return reconstruction_loss + kl_divergence

# -----------------------------
# Initialize Model & Optimizer
# -----------------------------
model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# Training Loop
# -----------------------------
model.train()
for epoch in range(epochs):
    total_loss = 0
    for images, _ in train_loader:
        images = images.view(-1, input_dim).to(device)

        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)
        loss = vae_loss(recon_images, images, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# -----------------------------
# Visualization
# -----------------------------
model.eval()
with torch.no_grad():
    # Take one batch
    images, _ = next(iter(train_loader))
    images = images.to(device)

    # Original image
    original_image = images[0].view(28, 28).cpu().numpy()

    # Reconstructed image
    recon_images, _, _ = model(images.view(-1, input_dim))
    reconstructed_image = recon_images[0].view(28, 28).cpu().numpy()

    # Generated image
    z = torch.randn(1, latent_dim).to(device)
    generated_image = model.decode(z).view(28, 28).cpu().numpy()

# -----------------------------
# Plot Results
# -----------------------------
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(generated_image, cmap='gray')
plt.title('Generated')
plt.axis('off')

plt.show()
