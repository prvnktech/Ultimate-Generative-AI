import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
LATENT_DIM = 512
BATCH_SIZE = 16
IMAGE_SIZE = 32
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Mapping Network
# =========================
def mapping_network(latent_dim, layers=8):
    layers_list = []
    for _ in range(layers):
        layers_list.append(nn.Linear(latent_dim, latent_dim))
        layers_list.append(nn.ReLU())
    return nn.Sequential(*layers_list)

# =========================
# Progressive Block
# =========================
class ProgressiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.upsample(x)
        x = torch.relu(self.conv2(x))
        return x

# =========================
# Generator
# =========================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = mapping_network(LATENT_DIM)
        self.fc = nn.Linear(LATENT_DIM, 128 * 4 * 4)

        self.block1 = ProgressiveBlock(128, 128)  # 4 → 8
        self.block2 = ProgressiveBlock(128, 64)   # 8 → 16
        self.block3 = ProgressiveBlock(64, 32)    # 16 → 32

        self.to_rgb = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, z):
        w = self.mapping(z)
        x = self.fc(w)
        x = x.view(-1, 128, 4, 4)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.sigmoid(self.to_rgb(x))
        return x

# =========================
# Discriminator (FIXED)
# =========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),   # 32 → 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 16 → 8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))                # force 4×4
        )

        self.classifier = nn.Linear(128 * 4 * 4, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# =========================
# Initialize Models
# =========================
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
g_optimizer = optim.Adam(G.parameters(), lr=1e-4)
d_optimizer = optim.Adam(D.parameters(), lr=1e-4)

# =========================
# Training Loop
# =========================
for epoch in range(EPOCHS):
    for _ in range(50):
        real_images = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        z = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)

        fake_images = G(z)

        # ---- Train Discriminator ----
        real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)
        fake_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE)

        d_real_loss = criterion(D(real_images), real_labels)
        d_fake_loss = criterion(D(fake_images.detach()), fake_labels)
        d_loss = d_real_loss + d_fake_loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ---- Train Generator ----
        g_loss = criterion(D(fake_images), real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}]  G Loss: {g_loss.item():.4f}  D Loss: {d_loss.item():.4f}")

# =========================
# Visualization
# =========================
def show_images(images, rows, cols):
    fig, axs = plt.subplots(rows, cols, figsize=(6, 6))
    for i, ax in enumerate(axs.flatten()):
        img = images[i].detach().cpu().permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis("off")
    plt.show()

z = torch.randn(9, LATENT_DIM).to(DEVICE)
generated_images = G(z)
show_images(generated_images, 3, 3)
