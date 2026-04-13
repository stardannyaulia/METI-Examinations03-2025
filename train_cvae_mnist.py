"""
Conditional VAE (CVAE) for MNIST Handwritten Digit Generation
============================================================
Train from scratch on MNIST - optimized for T4 GPU (Google Colab)
Architecture: CNN Encoder + CNN Decoder with digit conditioning (0-9)

Usage:
    python train_cvae_mnist.py

Output:
    - cvae_mnist.pth  (trained model weights)
    - samples/        (generated sample images per epoch)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM  = 128
NUM_CLASSES = 10
EPOCHS      = 60
BATCH_SIZE  = 256
LR          = 2e-4
BETA        = 4.0        # KL weight (β-VAE: higher = more disentangled)
IMG_SIZE    = 28
SAVE_DIR    = "samples"
MODEL_PATH  = "cvae_mnist.pth"

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Training on: {DEVICE}")

# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True
)

# ─────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────

class ConvEncoder(nn.Module):
    """CNN Encoder: image + one-hot label → mu, log_var"""
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        # Label embedding projected to spatial map
        self.label_embed = nn.Embedding(num_classes, 16)
        self.label_proj  = nn.Linear(16, IMG_SIZE * IMG_SIZE)

        # Input: 1 (image) + 1 (label map) = 2 channels
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),   # 14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.flatten_dim = 128 * 4 * 4  # 2048
        self.fc_mu      = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x, labels):
        B = x.size(0)
        lbl = self.label_embed(labels)          # B x 16
        lbl = self.label_proj(lbl)              # B x 784
        lbl = lbl.view(B, 1, IMG_SIZE, IMG_SIZE)
        x   = torch.cat([x, lbl], dim=1)        # B x 2 x 28 x 28
        h   = self.conv(x).view(B, -1)
        return self.fc_mu(h), self.fc_log_var(h)


class ConvDecoder(nn.Module):
    """CNN Decoder: latent z + one-hot label → reconstructed image"""
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 32)

        in_dim = latent_dim + 32
        self.fc = nn.Linear(in_dim, 128 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),   # 32x32
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 5, padding=0),                        # 28x28 (32-5+1)
            nn.Sigmoid(),
        )

    def forward(self, z, labels):
        lbl = self.label_embed(labels)    # B x 32
        x   = torch.cat([z, lbl], dim=1)  # B x (latent+32)
        x   = self.fc(x).view(-1, 128, 4, 4)
        return self.deconv(x)


class CVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, num_classes)
        self.decoder = ConvDecoder(latent_dim, num_classes)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        mu, log_var = self.encoder(x, labels)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z, labels)
        return recon, mu, log_var

    @torch.no_grad()
    def generate(self, labels, num_samples=1, temperature=1.0):
        """Generate images given digit labels with controllable diversity"""
        self.eval()
        B = len(labels)
        z = torch.randn(B, self.latent_dim, device=DEVICE) * temperature
        return self.decoder(z, labels)


# ─────────────────────────────────────────
# Loss Function
# ─────────────────────────────────────────

def cvae_loss(recon, x, mu, log_var, beta=BETA):
    # Reconstruction loss (BCE pixel-wise)
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ─────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────

def train():
    model     = CVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Fixed test samples: 10 digits × 5 samples each
    fixed_labels = torch.arange(NUM_CLASSES, device=DEVICE).repeat(5)  # 50
    fixed_labels = fixed_labels.view(5, 10).T.reshape(-1)              # interleaved

    print(f"\n{'='*60}")
    print(f"  CVAE MNIST Training  |  Latent: {LATENT_DIM}  |  β={BETA}")
    print(f"{'='*60}\n")

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = total_recon = total_kl = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            recon, mu, log_var = model(images, labels)
            loss, recon_l, kl_l = cvae_loss(recon, images, mu, log_var)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss  += loss.item()
            total_recon += recon_l.item()
            total_kl    += kl_l.item()

        scheduler.step()

        avg_loss  = total_loss  / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kl    = total_kl    / len(train_loader)

        print(f"Epoch [{epoch:3d}/{EPOCHS}] "
              f"Loss: {avg_loss:8.2f} | "
              f"Recon: {avg_recon:7.2f} | "
              f"KL: {avg_kl:6.2f} | "
              f"LR: {scheduler.get_last_lr()[0]:.5f}")

        # Save sample grid every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                samples = model.generate(fixed_labels, temperature=0.8)
                save_image(
                    samples,
                    f"{SAVE_DIR}/epoch_{epoch:03d}.png",
                    nrow=10,
                    normalize=False,
                    padding=2,
                )
            print(f"  → Saved sample grid: {SAVE_DIR}/epoch_{epoch:03d}.png")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\n✓ Training complete! Best loss: {best_loss:.2f}")
    print(f"✓ Model saved to: {MODEL_PATH}")
    return model


# ─────────────────────────────────────────
# Quick inference test
# ─────────────────────────────────────────

def test_generation(model_path=MODEL_PATH):
    model = CVAE().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print("\n─── Generation Test ───")
    for digit in range(10):
        labels = torch.full((5,), digit, dtype=torch.long, device=DEVICE)
        imgs   = model.generate(labels, temperature=0.9)
        save_image(imgs, f"{SAVE_DIR}/test_digit_{digit}.png", nrow=5)
        print(f"  Digit {digit}: 5 samples → {SAVE_DIR}/test_digit_{digit}.png")

    # Full grid: all digits
    all_labels = torch.arange(10, device=DEVICE).repeat(5)
    all_imgs   = model.generate(all_labels, temperature=0.9)
    save_image(all_imgs, f"{SAVE_DIR}/all_digits_grid.png", nrow=10, padding=3)
    print(f"\n✓ Full grid: {SAVE_DIR}/all_digits_grid.png")


if __name__ == "__main__":
    model = train()
    test_generation()
