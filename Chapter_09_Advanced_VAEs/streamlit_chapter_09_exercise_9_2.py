import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="CVAE Demo", layout="centered")
st.title("Conditional Variational Autoencoder (CVAE)")
st.write("Generate MNIST digits conditioned on labels")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Running on: {device}")

# -----------------------------
# Hyperparameters
# -----------------------------
latent_dim = 10
beta = st.slider("KL Weight (beta)", 1.0, 5.0, 1.0)
epochs = st.slider("Epochs per training run", 1, 5, 1)

# -----------------------------
# CVAE Model
# -----------------------------
class CVAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=10, label_dim=10):
        super().__init__()
        self.encoder = nn.Linear(input_dim + label_dim, 256)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Linear(latent_dim + label_dim, 256)
        self.out = nn.Linear(256, input_dim)

    def encode(self, x, y):
        xy = torch.cat([x, y], dim=1)
        h = torch.relu(self.encoder(xy))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        h = torch.relu(self.decoder(zy))
        return torch.sigmoid(self.out(h))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y)
        return x_hat, mu, logvar

def cvae_loss(x, x_hat, mu, logvar, beta):
    recon = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl

# -----------------------------
# Load data
# -----------------------------
@st.cache_resource
def load_data():
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    return DataLoader(dataset, batch_size=128, shuffle=True)

loader = load_data()

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    model = CVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer

model, optimizer = load_model()

# -----------------------------
# Training
# -----------------------------
if st.button("Train CVAE"):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device)
            y_onehot = torch.nn.functional.one_hot(y, num_classes=10).float().to(device)

            x_hat, mu, logvar = model(x, y_onehot)
            loss = cvae_loss(x, x_hat, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        st.write(f"Epoch {epoch + 1} loss: {total_loss / len(loader.dataset):.4f}")

    st.success("Training complete")

# -----------------------------
# Conditional Generation
# -----------------------------
st.subheader("Generate Digits")
digit = st.selectbox("Choose digit (0–9)", list(range(10)))

model.eval()
with torch.no_grad():
    y = torch.zeros(10, 10).to(device)
    y[:, digit] = 1
    z = torch.randn(10, latent_dim).to(device)
    samples = model.decode(z, y)

fig, axes = plt.subplots(1, 10, figsize=(12, 2))
for i in range(10):
    axes[i].imshow(samples[i].view(28, 28).cpu(), cmap="gray")
    axes[i].axis("off")

st.pyplot(fig)
