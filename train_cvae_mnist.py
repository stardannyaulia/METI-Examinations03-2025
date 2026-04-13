"""
╔══════════════════════════════════════════════════════════════════════╗
║          Conditional VAE (CVAE) — MNIST Handwritten Digit            ║
║          Train from scratch · PyTorch · T4 GPU (Google Colab)        ║
╚══════════════════════════════════════════════════════════════════════╝

Architecture  : CNN Encoder + CNN Decoder with digit class conditioning
Conditioning  : Label embedding injected in both encoder and decoder
Loss          : beta-VAE  =  BCE_recon  +  beta x KL_divergence
Epochs        : 60  (~20-25 min on T4)
Output        : cvae_mnist.pth  +  samples/  folder

Usage
-----
    python train_cvae_mnist.py
    python train_cvae_mnist.py --epochs 30 --beta 3.0
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

# ── CLI args ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",     type=int,   default=60)
parser.add_argument("--batch_size", type=int,   default=256)
parser.add_argument("--latent_dim", type=int,   default=128)
parser.add_argument("--lr",         type=float, default=2e-4)
parser.add_argument("--beta",       type=float, default=4.0)
parser.add_argument("--save_dir",   type=str,   default="samples")
parser.add_argument("--model_path", type=str,   default="cvae_mnist.pth")
args = parser.parse_args()

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = 28
NUM_CLASSES = 10
LATENT_DIM  = args.latent_dim
EPOCHS      = args.epochs
BATCH_SIZE  = args.batch_size
LR          = args.lr
BETA        = args.beta
SAVE_DIR    = args.save_dir
MODEL_PATH  = args.model_path

os.makedirs(SAVE_DIR, exist_ok=True)

print("\n" + "=" * 62)
print("  Conditional VAE  |  MNIST  |  Training Config")
print("=" * 62)
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"  Device     : {DEVICE}  ({gpu_name})")
print(f"  Latent dim : {LATENT_DIM}")
print(f"  Epochs     : {EPOCHS}   Batch: {BATCH_SIZE}")
print(f"  LR         : {LR}   beta (KL): {BETA}")
print(f"  Model out  : {MODEL_PATH}")
print("=" * 62 + "\n")

# ── Dataset ───────────────────────────────────────────────────────────
transform = transforms.ToTensor()
train_ds  = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=(DEVICE.type == "cuda"), drop_last=True
)
print(f"  Dataset: {len(train_ds):,} training images loaded.\n")


# ── Model blocks ──────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual conv block for stable gradients."""
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class ConvEncoder(nn.Module):
    """Image + label -> (mu, log_var)"""
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 16)
        self.label_proj  = nn.Linear(16, IMG_SIZE * IMG_SIZE)

        self.net = nn.Sequential(
            # 2ch in (image + label map) -> 32   @ 28x28
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 64  @ 14x14
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64),
            # 64 -> 128 @ 7x7
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            ResBlock(128),
            # 128 -> 256 @ 4x4
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        flat = 256 * 4 * 4
        self.fc_mu      = nn.Linear(flat, latent_dim)
        self.fc_log_var = nn.Linear(flat, latent_dim)

    def forward(self, x, labels):
        B = x.size(0)
        lbl = self.label_embed(labels)
        lbl = self.label_proj(lbl).view(B, 1, IMG_SIZE, IMG_SIZE)
        h   = torch.cat([x, lbl], dim=1)
        h   = self.net(h).view(B, -1)
        return self.fc_mu(h), self.fc_log_var(h)


class ConvDecoder(nn.Module):
    """z + label -> image [0,1]"""
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 32)
        self.fc = nn.Linear(latent_dim + 32, 256 * 4 * 4)

        self.net = nn.Sequential(
            # 256 @ 4x4 -> 128 @ 8x8
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            ResBlock(128),
            # 128 @ 8x8 -> 64 @ 16x16
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64),
            # 64 @ 16x16 -> 32 @ 32x32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            # 32 @ 32x32 -> 16 @ 28x28  (stride-1 conv crops 4px)
            nn.Conv2d(32, 16, 5, padding=0, bias=False),
            nn.BatchNorm2d(16), nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 1 (grayscale output)
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z, labels):
        lbl = self.label_embed(labels)
        x   = torch.cat([z, lbl], dim=1)
        x   = self.fc(x).view(-1, 256, 4, 4)
        return self.net(x)


class CVAE(nn.Module):
    """
    Conditional VAE for class-conditional MNIST digit generation.

    .generate(labels, temperature) - main inference method
    """
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder    = ConvEncoder(latent_dim, num_classes)
        self.decoder    = ConvDecoder(latent_dim, num_classes)
        self.latent_dim = latent_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, x, labels):
        mu, log_var = self.encoder(x, labels)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decoder(z, labels)
        return recon, mu, log_var

    @torch.no_grad()
    def generate(self, labels, temperature=0.85):
        """
        Generate images for digit labels.

        Args:
            labels      : LongTensor shape (N,) with values 0-9
            temperature : float - controls diversity
                          0.5-0.7  sharp/uniform
                          0.8-1.0  balanced (recommended)
                          1.0-1.3  more varied / expressive

        Returns:
            Tensor (N, 1, 28, 28) in [0, 1]
        """
        self.eval()
        z = torch.randn(len(labels), self.latent_dim,
                        device=labels.device) * temperature
        return self.decoder(z, labels)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Loss ──────────────────────────────────────────────────────────────

def cvae_loss(recon, x, mu, log_var, beta=BETA):
    """beta-VAE loss = recon_BCE + beta * KL"""
    B          = x.size(0)
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / B
    kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / B
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()


# ── Helpers ───────────────────────────────────────────────────────────

def sample_grid(model, device, temperature=0.85, cols=10):
    """10 digits x 5 samples arranged in a grid."""
    model.eval()
    with torch.no_grad():
        labels = torch.arange(NUM_CLASSES, device=device).repeat(5)
        labels = labels.view(5, NUM_CLASSES).T.reshape(-1)
        imgs   = model.generate(labels, temperature=temperature)
        return make_grid(imgs, nrow=cols, normalize=False,
                         padding=2, pad_value=0.15)


def fmt_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ── Training ──────────────────────────────────────────────────────────

def train():
    model     = CVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.05)

    print(f"  Model parameters : {model.n_params():,}\n")

    best_loss  = float("inf")
    t0_total   = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        sum_loss = sum_recon = sum_kl = 0.0
        t0_ep = time.time()

        for imgs, labels in train_loader:
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            recon, mu, lv = model(imgs, labels)
            loss, rl, kl  = cvae_loss(recon, imgs, mu, lv)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            sum_loss  += loss.item()
            sum_recon += rl
            sum_kl    += kl

        scheduler.step()

        n   = len(train_loader)
        al  = sum_loss  / n
        ar  = sum_recon / n
        ak  = sum_kl    / n
        ep_t = time.time() - t0_ep
        eta  = (time.time() - t0_total) / epoch * (EPOCHS - epoch)
        lrnow = scheduler.get_last_lr()[0]
        star = " *" if al < best_loss else ""

        print(f"  [{epoch:3d}/{EPOCHS}]  "
              f"loss {al:8.2f}  recon {ar:7.2f}  kl {ak:5.2f}  "
              f"lr {lrnow:.5f}  {fmt_time(ep_t)}/ep  eta {fmt_time(eta)}{star}")

        # Save grid
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            g = sample_grid(model, DEVICE)
            save_image(g, os.path.join(SAVE_DIR, f"epoch_{epoch:03d}.png"))

        # Save checkpoint
        if al < best_loss:
            best_loss = al
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "loss":       best_loss,
                "config":     {"latent_dim": LATENT_DIM, "num_classes": NUM_CLASSES},
            }, MODEL_PATH)

    print(f"\n  Done!  time={fmt_time(time.time()-t0_total)}  "
          f"best_loss={best_loss:.2f}  saved={MODEL_PATH}\n")
    return model


# ── Post-training test ────────────────────────────────────────────────

def test_generation(model_path=MODEL_PATH):
    print("  Running post-training generation test …")
    model = CVAE().to(DEVICE)
    ckpt  = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    for digit in range(NUM_CLASSES):
        labels = torch.full((5,), digit, dtype=torch.long, device=DEVICE)
        imgs   = model.generate(labels, temperature=0.9)
        save_image(imgs, os.path.join(SAVE_DIR, f"test_digit_{digit}.png"),
                   nrow=5, padding=3)

    for temp, tag in [(0.5, "sharp"), (0.85, "balanced"), (1.2, "diverse")]:
        g = sample_grid(model, DEVICE, temperature=temp)
        save_image(g, os.path.join(SAVE_DIR, f"final_{tag}.png"))

    print(f"  Test images saved to {SAVE_DIR}/\n")


if __name__ == "__main__":
    train()
    test_generation()
