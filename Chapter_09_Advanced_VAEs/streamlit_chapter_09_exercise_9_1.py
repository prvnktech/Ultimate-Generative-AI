import streamlit as st
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("β-VAE Demo with MNIST")
beta = st.slider("Beta (Disentanglement Strength)", 1, 10, 4)

class Encoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, 256)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc(x))
        return self.mu(h), self.logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256)
        self.out = nn.Linear(256, 784)

    def forward(self, z):
        h = torch.relu(self.fc(z))
        return torch.sigmoid(self.out(h))

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def loss_fn(x, x_hat, mu, logvar):
    recon = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return recon + beta * kl

@st.cache_resource
def load_model():
    model = nn.Module()
    model.encoder = Encoder()
    model.decoder = Decoder()
    return model.to(device)

model = load_model()

transform = transforms.ToTensor()
data = datasets.MNIST("./data", train=True, download=True, transform=transform)
loader = DataLoader(data, batch_size=64, shuffle=True)

if st.button("Train One Epoch"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for x, _ in loader:
        x = x.view(x.size(0), -1).to(device)
        mu, logvar = model.encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = model.decoder(z)

        loss = loss_fn(x, x_hat, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    st.success("Training complete")

model.eval()
x, _ = next(iter(loader))
x = x.view(x.size(0), -1).to(device)
with torch.no_grad():
    mu, logvar = model.encoder(x)
    z = reparameterize(mu, logvar)
    x_hat = model.decoder(z)

fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for i in range(8):
    axes[0, i].imshow(x[i].view(28, 28).cpu(), cmap="gray")
    axes[1, i].imshow(x_hat[i].view(28, 28).cpu(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].axis("off")

st.pyplot(fig)
