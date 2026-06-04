# ✍️ StarLive Conditional Variational Autographic Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.44%2B-F97316?logo=gradio&logoColor=white)](https://gradio.app)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/spaces)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-4DABCF?logo=numpy&logoColor=white)](https://numpy.org)

---

## Directory

- [Overview](#overview)
- [Features & Tech Stack](#features--tech-stack)
- [System Workflow](#system-workflow)
- [User Guide](#user-guide)
  - [Equipment](#equipment)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)
  - [Arsitektur Model](#arsitektur-model)
  - [Design Rationale](#design-rationale)
  - [Hasil & Evaluasi](#hasil--evaluasi)
  - [Limitations](#limitations)
  - [Future Development](#future-development)
  - [Referensi](#referensi)
- [Author](#author)

---

## Overview

Proyek ini dikembangkan sebagai jawaban pada tes seleksi **METI Government of Japan AI and Tech Internship 2025**. Sistem mengimplementasikan **Conditional β-VAE** yang dilatih dari awal pada dataset MNIST — menghasilkan angka tulisan tangan unik dan dapat dikenali, dikondisikan berdasarkan label kelas (0–9).

Seluruh 50 sampel yang dihasilkan (5 per angka × 10 angka, temperature=0.85) diuji terhadap **GPT-4o dan berhasil dikenali dengan akurasi 100%**. Model dapat diakses langsung melalui antarmuka web Gradio yang di-deploy ke HuggingFace Spaces.

```
Digit: 7    Temperature: 0.85
┌──────┬──────┬──────┬──────┬──────┐
│  7   │  7   │  7   │  7   │  7   │  ← 5 sampel unik
│(bold │(slant│(thin │(serif│(print│     dari ruang laten
│ hand)│  ed) │  )   │ like)│  )   │
└──────┴──────┴──────┴──────┴──────┘
```

---

## Features & Tech Stack

### Features

- **Generasi dikondisi label** — pilih angka 0–9, hasilkan 5 sampel unik setiap saat
- **Kontrol temperature** — atur keragaman visual dari presisi (0.3) hingga ekspresif (1.4)
- **Antarmuka web Gradio** — siap di-deploy ke HuggingFace Spaces tanpa server
- **Tiga mode jalankan** — Google Colab (GPU T4), lokal GPU, atau lokal CPU (pengujian)
- **Dapat digunakan sebagai library** — semua modul dapat diimpor langsung di Python

### Tech Stack

| Komponen | Teknologi |
|---|---|
| Bahasa | Python 3.10+ |
| Framework Deep Learning | PyTorch 2.1 + torchvision 0.16 |
| Antarmuka Web | Gradio 4.44 |
| Deployment | HuggingFace Spaces (CPU Basic, free tier) |
| Dataset | MNIST (60.000 gambar training) |
| Perangkat Keras Training | Google Colab T4 GPU (~20–25 menit) |

---

## System Workflow

### Flowchart

```
┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Input               │     │  ConvEncoder         │     │  Reparameterize      │
│  Image (28×28)      │────▶│  Stem → Down×3       │────▶│  μ, log σ²  →  z     │
│  + Label embed      │     │  + ResBlocks          │     │  z ∈ ℝ¹²⁸           │
└─────────────────────┘     └──────────────────────┘     └──────────┬───────────┘
                                                                      │
                                                                      ▼
┌─────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Output             │     │  ConvDecoder         │     │  z + noise ×         │
│  Image (28×28)      │◀────│  FC → Up×3           │◀────│  temperature         │
│  nilai [0, 1]       │     │  + ResBlocks          │     │  + Label embed       │
└─────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

### Penjelasan

| Langkah | Proses | Keterangan |
|---|---|---|
| 1 | Input encoding | Gambar 28×28 + label embedding digabung sebagai 2-kanal input |
| 2 | ConvEncoder | 4 lapisan conv + 2 ResBlocks menghasilkan μ dan log σ² |
| 3 | Reparameterize | Sampel `z ∈ ℝ¹²⁸` dari distribusi laten |
| 4 | Noise × temperature | Skala keragaman dikontrol sebelum decoding |
| 5 | ConvDecoder | FC → 3 ConvTranspose + 2 ResBlocks → gambar output 28×28 |

---

## User Guide

### Equipment

Pastikan hal berikut tersedia sebelum memulai:

- Python **3.10** atau lebih baru
- GPU CUDA *(sangat direkomendasikan — CPU sangat lambat untuk training)*
- Atau akun Google untuk menjalankan via **Google Colab (T4 GPU, gratis)**
- Akun HuggingFace *(opsional — hanya untuk deployment)*

---

### Installation

#### Opsi 1: Google Colab *(Direkomendasikan)*

1. Buka `CVAE_MNIST_Train_and_Deploy.ipynb` di Colab
2. Atur **Runtime → Change runtime type → T4 GPU**
3. Jalankan semua sel — training berlangsung ~20–25 menit
4. Sel ke-6 menjalankan Gradio dengan tautan publik `gradio.live`
5. Sel ke-7 melakukan deployment permanen ke HuggingFace Spaces

#### Opsi 2: Lokal dengan GPU

```bash
# 1. Clone repositori
git clone https://github.com/YOUR_USERNAME/digit-generator.git
cd digit-generator

# 2. Instal dependensi
pip install -r requirements.txt

# 3. Latih model
python train_cvae_mnist.py

# 4. Jalankan aplikasi
python app.py
# → http://localhost:7860
```

#### Opsi 3: Lokal dengan CPU *(hanya untuk pengujian)*

```bash
python train_cvae_mnist.py --epochs 5 --batch_size 64
```

#### Deployment ke HuggingFace Spaces

1. Buat akun di [huggingface.co](https://huggingface.co)
2. Buat Space baru di [huggingface.co/new-space](https://huggingface.co/new-space) — SDK: **Gradio**, Hardware: **CPU Basic**
3. Dapatkan write token di [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Upload file via CLI:

```bash
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi(token="hf_YOUR_TOKEN")
for f in ["app.py", "cvae_mnist.pth", "requirements.txt"]:
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f,
        repo_id="YOUR_USERNAME/digit-generator",
        repo_type="space",
    )
    print(f"Uploaded {f}")
EOF
```

5. Space aktif di: `https://huggingface.co/spaces/YOUR_USERNAME/digit-generator`

> Free tier HF Spaces mendukung hingga 5 GB. Total ukuran file proyek ini ~15–20 MB.

---

### Configuration

**Argumen skrip pelatihan:**

```bash
python train_cvae_mnist.py [OPTIONS]
```

| Argumen | Tipe | Default | Keterangan |
|---|---|---|---|
| `--epochs` | INT | `60` | Jumlah epoch pelatihan |
| `--batch_size` | INT | `256` | Ukuran batch |
| `--latent_dim` | INT | `128` | Dimensionalitas ruang laten |
| `--lr` | FLOAT | `2e-4` | Learning rate |
| `--beta` | FLOAT | `4.0` | Bobot divergensi KL (β-VAE) |
| `--save_dir` | STR | `samples` | Direktori output gambar sampel |
| `--model_path` | STR | `cvae_mnist.pth` | Path file output model |

**Contoh penggunaan:**

```bash
python train_cvae_mnist.py                                  # default penuh
python train_cvae_mnist.py --beta 2.0 --latent_dim 256     # lebih ekspresif
python train_cvae_mnist.py --epochs 5 --batch_size 128     # uji cepat
python train_cvae_mnist.py --model_path models/my_cvae.pth # path kustom
```

**Panduan Temperature:**

| Temperature | Efek | Gunakan Saat |
|---|---|---|
| 0.3 – 0.6 | Sangat tajam dan seragam | Pengujian presisi dan keterbacaan |
| 0.7 – 0.9 | Variasi alami, semua dapat dikenali | **Default yang direkomendasikan** |
| 1.0 – 1.2 | Variasi gaya lebih tinggi | Eksplorasi artistik |
| 1.3 – 1.4 | Ekspresivitas tinggi, ada sedikit noise | Augmentasi data kreatif |

**Generate gambar via Python:**

```python
import torch
from train_cvae_mnist import CVAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CVAE().to(DEVICE)
ckpt  = torch.load("cvae_mnist.pth", map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Hasilkan 5 sampel angka "7"
labels = torch.full((5,), 7, dtype=torch.long, device=DEVICE)
imgs   = model.generate(labels, temperature=0.85)
# imgs.shape → (5, 1, 28, 28), nilai dalam [0, 1]

from torchvision.utils import save_image
save_image(imgs, "my_sevens.png", nrow=5)
```

---

### Troubleshooting

| Masalah | Kemungkinan Penyebab | Solusi |
|---|---|---|
| `ModuleNotFoundError` | Dependensi belum terinstal | Jalankan `pip install -r requirements.txt` |
| Training sangat lambat | Tidak ada GPU | Gunakan Google Colab T4 atau aktifkan CUDA |
| `FileNotFoundError: cvae_mnist.pth` | Model belum dilatih | Jalankan `train_cvae_mnist.py` terlebih dahulu |
| Space HF tidak mau build | Token tidak punya izin write | Buat ulang token dengan scope `write` di HF settings |
| Gambar output berbentuk kotak | Pembesaran NEAREST pada resolusi besar | Wajar — resolusi native MNIST memang 28×28 |

---

## Development Notes

### Arsitektur Model

**Conditional β-VAE** dengan dimensi laten 128 dan ~3.8 juta parameter.

| Komponen | Detail |
|---|---|
| Encoder | Input 2 kanal (gambar + label map) → 4 conv + 2 ResBlocks → FC → μ, log σ² |
| Decoder | FC → 3 ConvTranspose + 2 ResBlocks → conv output 28×28, aktivasi Sigmoid |
| Conditioning | Label embedding diproyeksikan ke spatial map (encoder) dan digabung dengan z (decoder) |
| Loss | `L = BCE(recon, x) + β × KL(q(z｜x,y) ‖ p(z))` dengan β=4.0 |

**Detail pelatihan:**

| Hiperparameter | Nilai |
|---|---|
| Epoch | 60 |
| Batch size | 256 |
| Optimizer | Adam (lr=2e-4, β₁=0.9, β₂=0.999) |
| LR schedule | CosineAnnealing (min lr=1e-5) |
| Gradient clipping | max_norm=1.0 |
| Weight decay | 1e-5 |
| Dataset | MNIST train split — 60.000 gambar |

**Kurva loss yang diharapkan:**

| Epoch | Total Loss | Recon Loss | KL Loss |
|---|---|---|---|
| 1 | ~380 | ~340 | ~10 |
| 10 | ~175 | ~135 | ~10 |
| 30 | ~120 | ~82 | ~10 |
| 60 | ~95 | ~62 | ~8–12 |

---

### Design Rationale

**Mengapa VAE dan bukan GAN?**
Stabilitas pelatihan. GAN rentan terhadap mode collapse, vanishing gradient, dan divergensi — memerlukan penyeimbangan generator dan diskriminator yang cermat. VAE dengan regularisasi β dilatih secara andal dalam satu pass dan menghasilkan ruang laten yang halus dan dapat dikontrol.

**Mengapa β=4.0?**
Nilai β yang lebih tinggi memaksa encoder menggunakan ruang laten yang lebih terstruktur dan terdisentanglasi. Sampel acak dari prior `N(0,I)` menghasilkan angka yang lebih koheren dengan variasi alami — ideal untuk generasi gambar yang dapat dikenali.

**Mengapa ResBlocks?**
Koneksi residual menstabilkan aliran gradien pada jaringan yang lebih dalam. Bahkan pada model kecil ini, ResBlocks mencegah stagnasi pelatihan dan memungkinkan decoder mempelajari detail sapuan yang lebih halus.

**Mengapa temperature pada z?**
Sebagai pengganti truncation (umum pada GAN), noise diskalakan dengan `z ~ N(0, T²I)`. Pendekatan ini dapat diturunkan secara diferensial, mempertahankan bentuk distribusi, dan memberikan kontrol kontinu antara presisi (T rendah) dan kreativitas (T tinggi).

---

### Hasil & Evaluasi

**Uji pengenalan GPT-4o** (50 sampel, temperature=0.85):

| Angka | Dikenali | Kepercayaan |
|:---:|---|---|
| 0–9 (semua) | 5/5 masing-masing | Tinggi |
| **Total** | **50/50** | **100%** |

**Metrik model pada epoch 60:**

```
Total loss  :  ~95
Recon loss  :  ~62   (BCE, sum/batch)
KL loss     :  ~10
```

---

### Limitations

| Komponen | Batasan |
|---|---|
| Resolusi output | 28×28 piksel native — pembesaran NEAREST menghasilkan tampilan kotak |
| Ambiguitas angka | Sampel di dekat batas keputusan (1 vs 7, 4 vs 9) dapat menyerupai angka lain pada temperature tinggi |
| Generalisasi dataset | Hanya dilatih pada MNIST — tidak dapat digeneralisasi ke alfabet, aksara lain, atau simbol non-angka |

### Future Development

Beberapa pengembangan yang dapat dilakukan ke depan:

- [ ] **Resolusi lebih tinggi** — upscaling dengan model super-resolution (ESRGAN) untuk output yang lebih tajam
- [ ] **Dukungan aksara lain** — pelatihan pada dataset huruf latin, Hiragana, atau Hangul
- [ ] **Interpolasi laten** — antarmuka untuk menginterpolasi antara dua angka di ruang laten
- [ ] **Conditional diffusion model** — eksplorasi arsitektur alternatif untuk kualitas gambar lebih tinggi
- [ ] **Evaluasi FID score** — metrik kuantitatif standar untuk kualitas generasi gambar

---

### Referensi

- [MNIST Database](http://yann.lecun.com/exdb/mnist/) — LeCun, Cortes, Burges (1998)
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fchgxl) — Higgins et al. (2017)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Kingma & Welling (2013)
- [HuggingFace Spaces](https://huggingface.co/spaces) — Platform hosting model gratis

---

<p align="center">
  <b>Pengembangan dari tim StarLive SAINT</b>
</p>

<p align="center"><i>Danny Aulia · Said Hasan Hanafiah · Noah Von Nobelius · Arvian Raveindra Pradana</i></p>
