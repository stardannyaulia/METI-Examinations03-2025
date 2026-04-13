"""
Handwritten Digit Generator - Gradio App
=========================================
Deploy on HuggingFace Spaces (SDK: gradio)

File structure for HF Spaces:
  app.py          ← this file
  cvae_mnist.pth  ← trained model weights
  requirements.txt

requirements.txt contents:
  torch==2.1.0
  torchvision==0.16.0
  gradio==4.44.0
  Pillow>=9.0
  numpy>=1.24
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import io
import os

# ─────────────────────────────────────────
# Model (must match training architecture)
# ─────────────────────────────────────────
LATENT_DIM  = 128
NUM_CLASSES = 10
IMG_SIZE    = 28
DEVICE      = torch.device("cpu")   # HF Spaces free tier: CPU
MODEL_PATH  = "cvae_mnist.pth"

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 16)
        self.label_proj  = nn.Linear(16, IMG_SIZE * IMG_SIZE)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
        )
        self.flatten_dim = 128 * 4 * 4
        self.fc_mu      = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x, labels):
        B = x.size(0)
        lbl = self.label_embed(labels)
        lbl = self.label_proj(lbl).view(B, 1, IMG_SIZE, IMG_SIZE)
        x   = torch.cat([x, lbl], dim=1)
        h   = self.conv(x).view(B, -1)
        return self.fc_mu(h), self.fc_log_var(h)

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 32)
        self.fc = nn.Linear(latent_dim + 32, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 5, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, z, labels):
        lbl = self.label_embed(labels)
        x   = torch.cat([z, lbl], dim=1)
        x   = self.fc(x).view(-1, 128, 4, 4)
        return self.deconv(x)

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder(LATENT_DIM, NUM_CLASSES)
        self.decoder = ConvDecoder(LATENT_DIM, NUM_CLASSES)
        self.latent_dim = LATENT_DIM

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, x, labels):
        mu, log_var = self.encoder(x, labels)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z, labels), mu, log_var

    @torch.no_grad()
    def generate(self, labels, temperature=0.85):
        self.eval()
        z = torch.randn(len(labels), self.latent_dim) * temperature
        return self.decoder(z, labels)

# ─────────────────────────────────────────
# Load model once at startup
# ─────────────────────────────────────────
model = CVAE()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print(f"✓ Model loaded from {MODEL_PATH}")
else:
    print(f"⚠ Model not found at {MODEL_PATH} – using untrained weights for demo")
model.eval()

# ─────────────────────────────────────────
# Image generation + grid assembly
# ─────────────────────────────────────────
UPSCALE   = 8          # 28px → 224px per digit
PADDING   = 12
BG_COLOR  = (15, 15, 20)
GRID_COLS = 5

def make_grid_image(digit: int, temperature: float = 0.85) -> Image.Image:
    """Generate 5 unique samples for a digit and return a PIL grid image."""
    labels = torch.full((GRID_COLS,), digit, dtype=torch.long)
    imgs   = model.generate(labels, temperature=temperature)   # (5,1,28,28)

    cell_px = IMG_SIZE * UPSCALE
    grid_w  = GRID_COLS * cell_px + (GRID_COLS + 1) * PADDING
    grid_h  = cell_px + 2 * PADDING + 48   # 48px for label bar

    canvas  = Image.new("RGB", (grid_w, grid_h), color=BG_COLOR)

    for i, img_tensor in enumerate(imgs):
        arr = img_tensor.squeeze().numpy()          # 28×28 float32 [0,1]
        arr = (arr * 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="L")
        pil = pil.resize((cell_px, cell_px), Image.NEAREST)
        pil_rgb = Image.new("RGB", pil.size)
        # Colorize: white strokes on dark bg
        pil_rgb.paste(Image.merge("RGB", [pil, pil, pil]))
        x_off = PADDING + i * (cell_px + PADDING)
        canvas.paste(pil_rgb, (x_off, PADDING))

    # Label bar at bottom
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except:
        font = ImageFont.load_default()

    label_text = f"Digit: {digit}  ·  5 unique samples"
    draw.text(
        (grid_w // 2, grid_h - 26),
        label_text,
        fill=(200, 200, 210),
        font=font,
        anchor="mm",
    )
    return canvas

# ─────────────────────────────────────────
# Gradio interface
# ─────────────────────────────────────────
DIGIT_CHOICES = [str(d) for d in range(10)]

def generate(digit_str: str, temperature: float) -> Image.Image:
    digit = int(digit_str)
    return make_grid_image(digit, temperature)

css = """
body { font-family: 'Space Grotesk', sans-serif; background: #0d0d12; }
.gradio-container { max-width: 760px !important; margin: auto; }
#title { text-align: center; margin-bottom: 0.2em; }
#subtitle { text-align: center; color: #888; margin-bottom: 1.5em; font-size: 0.9em; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="✍ Handwritten Digit Generator",
    theme=gr.themes.Base(
        primary_hue="violet",
        neutral_hue="slate",
    ),
    css=css,
) as demo:
    gr.HTML("""
        <h1 id='title' style='font-size:2em; color:#e8e8f0;'>
            ✍ Handwritten Digit Generator
        </h1>
        <p id='subtitle'>
            Conditional VAE trained from scratch on MNIST &nbsp;·&nbsp;
            Each generation is unique via latent noise
        </p>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            digit_input = gr.Dropdown(
                choices=DIGIT_CHOICES,
                value="7",
                label="Select Digit (0–9)",
                interactive=True,
            )
            temp_slider = gr.Slider(
                minimum=0.3,
                maximum=1.5,
                value=0.85,
                step=0.05,
                label="Diversity (temperature)",
                info="Lower = sharper/uniform, Higher = more varied",
            )
            gen_btn = gr.Button("⚡ Generate 5 Samples", variant="primary", size="lg")

        with gr.Column(scale=2):
            output_img = gr.Image(
                label="Generated Digits",
                type="pil",
                height=320,
                show_download_button=True,
            )

    gr.Examples(
        examples=[["0", 0.8], ["3", 0.9], ["7", 0.85], ["9", 1.1]],
        inputs=[digit_input, temp_slider],
        outputs=output_img,
        fn=generate,
        cache_examples=True,
        label="Quick Examples",
    )

    gr.HTML("""
        <div style='text-align:center; margin-top:1.5em; color:#555; font-size:0.8em;'>
            Model: Conditional VAE · Latent dim: 128 · β=4.0 · Trained 60 epochs on MNIST
        </div>
    """)

    # Wire up
    gen_btn.click(fn=generate, inputs=[digit_input, temp_slider], outputs=output_img)
    digit_input.change(fn=generate, inputs=[digit_input, temp_slider], outputs=output_img)

if __name__ == "__main__":
    demo.launch(share=True)   # share=True gives a public ngrok link from Colab
