"""
Handwritten Digit Generator — Gradio App
==========================================
Deploy on HuggingFace Spaces (SDK: gradio)

Folder structure required on HF Spaces:
    app.py
    cvae_mnist.pth
    requirements.txt

Local run:
    pip install gradio torch torchvision Pillow
    python app.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────
LATENT_DIM  = 128
NUM_CLASSES = 10
IMG_SIZE    = 28
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = "cvae_mnist.pth"

# Display config
CELL_PX   = 196    # each digit rendered at 196x196px
PADDING   = 10
NUM_COLS  = 5
BG        = (14, 14, 20)
CELL_BG   = (24, 24, 34)


# ── Model definition (must match train_cvae_mnist.py exactly) ─────────

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 16)
        self.label_proj  = nn.Linear(16, IMG_SIZE * IMG_SIZE)
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            ResBlock(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu      = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x, labels):
        B = x.size(0)
        lbl = self.label_proj(self.label_embed(labels)).view(B, 1, IMG_SIZE, IMG_SIZE)
        h   = self.net(torch.cat([x, lbl], dim=1)).view(B, -1)
        return self.fc_mu(h), self.fc_log_var(h)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 32)
        self.fc = nn.Linear(latent_dim + 32, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 5, padding=0, bias=False),
            nn.BatchNorm2d(16), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z, labels):
        lbl = self.label_embed(labels)
        x   = self.fc(torch.cat([z, lbl], dim=1)).view(-1, 256, 4, 4)
        return self.net(x)


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = ConvEncoder(LATENT_DIM, NUM_CLASSES)
        self.decoder    = ConvDecoder(LATENT_DIM, NUM_CLASSES)
        self.latent_dim = LATENT_DIM

    def reparameterize(self, mu, log_var):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def forward(self, x, labels):
        mu, lv = self.encoder(x, labels)
        return self.decoder(self.reparameterize(mu, lv), labels), mu, lv

    @torch.no_grad()
    def generate(self, labels, temperature=0.85):
        self.eval()
        z = torch.randn(len(labels), self.latent_dim, device=DEVICE) * temperature
        return self.decoder(z, labels)


# ── Load model ────────────────────────────────────────────────────────
model = CVAE().to(DEVICE)

if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    # Support both plain state_dict and our checkpoint format
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd)
    print(f"[app] Model loaded from {MODEL_PATH}")
else:
    print(f"[app] WARNING: {MODEL_PATH} not found — using random weights.")

model.eval()


# ── Image grid builder ────────────────────────────────────────────────

def _try_font(size=22):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def build_grid(digit: int, temperature: float) -> Image.Image:
    """
    Generate 5 unique samples for `digit` and return a PIL image grid.
    Each sample is upscaled from 28x28 → CELL_PX x CELL_PX (NEAREST).
    """
    labels = torch.full((NUM_COLS,), digit, dtype=torch.long, device=DEVICE)
    imgs   = model.generate(labels, temperature=temperature)   # (5,1,28,28)

    label_bar = 44
    grid_w = NUM_COLS * CELL_PX + (NUM_COLS + 1) * PADDING
    grid_h = CELL_PX + 2 * PADDING + label_bar

    canvas = Image.new("RGB", (grid_w, grid_h), BG)
    draw   = ImageDraw.Draw(canvas)

    for i, t in enumerate(imgs):
        arr = (t.squeeze().cpu().numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="L").resize(
            (CELL_PX, CELL_PX), Image.NEAREST
        )
        # Dark cell background
        x0 = PADDING + i * (CELL_PX + PADDING)
        y0 = PADDING
        cell_img = Image.new("RGB", (CELL_PX, CELL_PX), CELL_BG)
        # Paste grayscale digit (white on dark)
        cell_img.paste(Image.merge("RGB", [pil, pil, pil]))
        canvas.paste(cell_img, (x0, y0))

        # Sample index
        draw.text((x0 + CELL_PX - 6, y0 + CELL_PX - 4),
                  f"#{i+1}", fill=(80, 80, 100),
                  font=_try_font(14), anchor="rb")

    # Bottom label bar
    font = _try_font(22)
    cy   = grid_h - label_bar // 2
    draw.text((grid_w // 2, cy),
              f"digit  {digit}   ·   5 unique samples   ·   temp {temperature:.2f}",
              fill=(180, 180, 200), font=font, anchor="mm")

    return canvas


# ── Gradio interface ──────────────────────────────────────────────────

CHOICES = [str(d) for d in range(10)]

def generate(digit_str: str, temperature: float) -> Image.Image:
    return build_grid(int(digit_str), float(temperature))


# Warm up (avoids slow first request)
with torch.no_grad():
    _ = model.generate(torch.zeros(1, dtype=torch.long, device=DEVICE))

css = """
.gradio-container { max-width: 700px !important; margin: 0 auto; }
#gen-btn { font-size: 1rem; font-weight: 600; }
footer { display: none !important; }
"""

with gr.Blocks(title="Handwritten Digit Generator", css=css,
               theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate")) as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 1rem 0 .5rem;">
      <h1 style="font-size:1.8rem; font-weight:700; margin:0;">
        ✍&nbsp; Handwritten Digit Generator
      </h1>
      <p style="color:#888; font-size:.88rem; margin:.4rem 0 0;">
        Conditional VAE trained from scratch on MNIST &nbsp;·&nbsp;
        Unique latent noise per generation
      </p>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=200):
            digit_dd = gr.Dropdown(
                choices=CHOICES, value="7",
                label="Digit  (0 – 9)",
            )
            temp_sl = gr.Slider(
                minimum=0.30, maximum=1.40, value=0.85, step=0.05,
                label="Diversity  (temperature)",
                info="Low = precise · High = expressive",
            )
            gen_btn = gr.Button("⚡  Generate 5 Samples",
                                variant="primary", elem_id="gen-btn")

        with gr.Column(scale=2):
            out_img = gr.Image(
                label="Generated samples",
                type="pil",
                height=CELL_PX + 2 * PADDING + 44 + 20,
                show_download_button=True,
            )

    gr.Examples(
        examples=[["0", 0.80], ["2", 0.90], ["3", 0.85],
                  ["7", 0.85], ["8", 0.75], ["9", 1.10]],
        inputs=[digit_dd, temp_sl],
        outputs=out_img,
        fn=generate,
        cache_examples=True,
        label="Quick examples",
    )

    gr.HTML("""
    <div style="text-align:center; margin-top:1.2rem;
                color:#555; font-size:.75rem; line-height:1.8;">
      Architecture: Conditional β-VAE &nbsp;·&nbsp;
      Latent dim: 128 &nbsp;·&nbsp;
      β = 4.0 &nbsp;·&nbsp;
      Trained 60 epochs · MNIST 60k images
    </div>
    """)

    gen_btn.click(fn=generate, inputs=[digit_dd, temp_sl], outputs=out_img)
    digit_dd.change(fn=generate, inputs=[digit_dd, temp_sl], outputs=out_img)
    temp_sl.release(fn=generate, inputs=[digit_dd, temp_sl], outputs=out_img)

    # Initial render
    demo.load(fn=generate,
              inputs=[digit_dd, temp_sl],
              outputs=out_img)


if __name__ == "__main__":
    # share=True gives a public https://xxxxx.gradio.live link
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,          # remove for HF Spaces deploy
        show_error=True,
    )
