# ✍ StarLive Conditional Variational Autographic Learning

> **Conditional VAE yang dilatih dari awal pada MNIST — menghasilkan angka tulisan tangan yang unik dan dapat dikenali, dikondisikan berdasarkan label kelas (0–9)**
>
> *Merupakan pertanyaan pada tes METI Government of Japan for AI and Tech Internship tahun 2025*

[![HuggingFace Spaces](https://img.shields.io/badge/🤗HuggingFace-%20Demo-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/digit-generator)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-green?logo=pytorch)](https://pytorch.org)

---

## Demo

> Pilih angka → klik Generate → dapatkan 5 sampel unik bergaya tulisan tangan setiap saat.

```
Digit: 7    Temperature: 0.85
┌──────┬──────┬──────┬──────┬──────┐
│  7   │  7   │  7   │  7   │  7   │   ← 5 sampel unik
│ (bold│(slant│(thin │(serif│(print│       dari ruang laten
│ hand)│  ed) │  )   │ like)│  )   │
└──────┴──────┴──────┴──────┴──────┘
```

**Uji pengenalan:** Seluruh 50 sampel yang dihasilkan (5 per angka × 10 angka) berhasil diidentifikasi oleh GPT-4o dengan akurasi 100%.

---

## Struktur Proyek

```
digit-generator/
├── train_cvae_mnist.py          # Skrip pelatihan lengkap (PyTorch)
├── app.py                       # Aplikasi web Gradio (siap untuk HF Spaces)
├── requirements.txt             # Dependensi Python
├── CVAE_MNIST_Train_and_Deploy.ipynb  # Notebook Colab end-to-end
├── README.md                    # Dokumen ini
├── cvae_mnist.pth               # Bobot model terlatih (dihasilkan setelah pelatihan)
└── samples/                     # Gambar yang dihasilkan (dibuat otomatis setelah pelatihan)
    ├── epoch_001.png            # Grid setelah epoch 1
    ├── epoch_005.png            # ...
    ├── epoch_060.png            # Grid epoch akhir
    ├── test_digit_0.png         # 5 sampel untuk angka 0
    ├── ...
    ├── test_digit_9.png         # 5 sampel untuk angka 9
    ├── final_sharp.png          # Grid pada temperature=0.5
    ├── final_balanced.png       # Grid pada temperature=0.85
    └── final_diverse.png        # Grid pada temperature=1.2
```

---

## Arsitektur

### Model: Conditional β-VAE

```
                    ┌─────────────────────────────────┐
                    │          ConvEncoder              │
  Image (28×28) ──► │  Stem → Down×3 + ResBlocks       │──► μ, log σ²
  Label embed  ──►  │  (stride-2 convs: 28→14→7→4)    │
                    └─────────────────────────────────┘
                              │ reparameterize
                              ▼
                         z ∈ ℝ¹²⁸  +  noise × temperature
                              │
                    ┌─────────────────────────────────┐
                    │          ConvDecoder              │
  Label embed  ──►  │  FC → Up×3 + ResBlocks           │──► Image (28×28)
                    │  (ConvTranspose: 4→8→16→32→28)  │
                    └─────────────────────────────────┘
```

| Komponen | Detail |
|-----------|--------|
| **Encoder** | Input 2 kanal (gambar + label map) → 4 lapisan conv + 2 ResBlocks → FC |
| **Decoder** | FC → 3 lapisan ConvTranspose + 2 ResBlocks → conv output (28×28) |
| **Conditioning** | Label embedding → diproyeksikan ke spatial map (encoder) dan digabungkan dengan z (decoder) |
| **Dimensi laten** | 128 |
| **Jumlah parameter** | ~3.8 juta |
| **Aktivasi output** | Sigmoid (nilai piksel dalam rentang [0, 1]) |

### Fungsi Loss

```
L = BCE(recon, x)  +  β × KL(q(z|x,y) ‖ p(z))
  = rekonstruksi   +  4.0 × divergensi KL
```

Hiperparameter β=4.0 (β-VAE) mendorong ruang laten yang lebih terdisentanglasi, sehingga menghasilkan keragaman visual yang lebih besar saat sampling.

### Detail Pelatihan

| Hiperparameter | Nilai |
|---------------|-------|
| Epoch | 60 |
| Batch size | 256 |
| Optimizer | Adam (lr=2e-4, β₁=0.9, β₂=0.999) |
| LR schedule | CosineAnnealing (min lr=1e-5) |
| Gradient clipping | max_norm=1.0 |
| Weight decay | 1e-5 |
| β (bobot KL) | 4.0 |
| Dimensi laten | 128 |
| Dataset | MNIST train split — 60.000 gambar |
| Perangkat keras | Google Colab T4 GPU (~20–25 menit) |

### Kurva Loss yang Diharapkan

```
Epoch  1 :  loss ~380   recon ~340   kl ~10
Epoch 10 :  loss ~175   recon ~135   kl ~10
Epoch 30 :  loss ~120   recon ~82    kl ~10
Epoch 60 :  loss ~95    recon ~62    kl ~8-12
```

---

## Mulai Cepat

### Opsi 1: Google Colab (Direkomendasikan)

1. Buka `CVAE_MNIST_Train_and_Deploy.ipynb` di Colab.
2. Atur `Runtime → Change runtime type → T4 GPU`.
3. Jalankan semua sel — proses pelatihan berlangsung sekitar 20–25 menit.
4. Sel ke-6 menjalankan Gradio dengan tautan publik `gradio.live`.
5. Sel ke-7 melakukan deployment permanen ke HuggingFace Spaces.

### Opsi 2: Lokal dengan GPU

```bash
# Clone / unduh file
git clone https://github.com/YOUR_USERNAME/digit-generator.git
cd digit-generator

# Instal dependensi
pip install -r requirements.txt

# Latih model (memerlukan GPU CUDA)
python train_cvae_mnist.py

# Jalankan aplikasi
python app.py
# → http://localhost:7860  (atau tautan publik gradio.live dengan share=True)
```

### Opsi 3: Lokal dengan CPU (hanya untuk pengujian, lambat)

```bash
python train_cvae_mnist.py --epochs 5 --batch_size 64
```

---

## Penggunaan Skrip Pelatihan

```bash
python train_cvae_mnist.py [OPTIONS]

Options:
  --epochs     INT    Jumlah epoch pelatihan           [default: 60]
  --batch_size INT    Ukuran batch                     [default: 256]
  --latent_dim INT    Dimensionalitas ruang laten      [default: 128]
  --lr         FLOAT  Learning rate                    [default: 2e-4]
  --beta       FLOAT  Bobot divergensi KL (β-VAE)      [default: 4.0]
  --save_dir   STR    Direktori untuk gambar sampel    [default: samples]
  --model_path STR    Path file output model           [default: cvae_mnist.pth]
```

Contoh penggunaan:

```bash
# Pelatihan penuh dengan pengaturan default
python train_cvae_mnist.py

# Output lebih ekspresif dan beragam
python train_cvae_mnist.py --beta 2.0 --latent_dim 256

# Uji cepat (5 epoch)
python train_cvae_mnist.py --epochs 5 --batch_size 128

# Simpan ke path kustom
python train_cvae_mnist.py --model_path models/my_cvae.pth
```

---

## Menghasilkan Gambar via Python

```python
import torch
from train_cvae_mnist import CVAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Muat model terlatih
model = CVAE().to(DEVICE)
ckpt  = torch.load("cvae_mnist.pth", map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Hasilkan 5 sampel angka "7"
labels = torch.full((5,), 7, dtype=torch.long, device=DEVICE)
imgs   = model.generate(labels, temperature=0.85)
# imgs.shape → (5, 1, 28, 28)  nilai dalam rentang [0, 1]

# Simpan sebagai grid PNG
from torchvision.utils import save_image
save_image(imgs, "my_sevens.png", nrow=5)

# Konversi satu gambar ke numpy (untuk matplotlib / PIL)
import numpy as np
arr = (imgs[0].squeeze().cpu().numpy() * 255).astype("uint8")
# arr.shape → (28, 28)

# Efek temperature:
# 0.5 → tajam / presisi (keragaman rendah)
# 0.85 → seimbang (direkomendasikan)
# 1.2 → lebih ekspresif / kreatif
```

---

## Deployment ke HuggingFace Spaces

### Langkah-langkah

1. Buat akun gratis di [huggingface.co](https://huggingface.co).
2. Buat Space baru:
   - Buka [huggingface.co/new-space](https://huggingface.co/new-space).
   - SDK: **Gradio**.
   - Hardware: **CPU Basic** (free tier).
3. Dapatkan write token di [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. Unggah file melalui Colab Sel ke-7, atau melalui CLI:

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

5. Space akan dibangun secara otomatis dan dapat diakses di:
   `https://huggingface.co/spaces/YOUR_USERNAME/digit-generator`

### Ukuran File

| File | Ukuran |
|------|------|
| `cvae_mnist.pth` | ~15–20 MB |
| `app.py` | ~8 KB |
| `requirements.txt` | < 1 KB |

Free tier HF Spaces mendukung hingga 5 GB — jauh di bawah batas yang tersedia.

---

## Fitur Aplikasi Web

| Fitur | Detail |
|---------|--------|
| Pemilih angka | Dropdown 0–9 |
| Slider keragaman | Temperature 0.30 – 1.40 |
| Output | 5 gambar unik berukuran 196×196 piksel |
| Regenerasi | Setiap perubahan input memicu generasi baru |
| Contoh cepat | Contoh yang telah di-cache untuk pratinjau instan |
| Unduhan | Tombol unduhan bawaan |
| Render otomatis | Memuat angka default saat halaman dibuka |

---

## Panduan Temperature

Parameter temperature mengatur skala vektor noise acak `z` sebelum proses decoding:

```
z = randn(latent_dim) × temperature
```

| Temperature | Efek | Gunakan Saat |
|-------------|--------|----------|
| 0.3 – 0.6 | Sapuan sangat tajam dan seragam | Pengujian presisi dan keterbacaan |
| 0.7 – 0.9 | Variasi alami, semua dapat dikenali | **Nilai default yang direkomendasikan** |
| 1.0 – 1.2 | Variasi gaya lebih tinggi | Eksplorasi artistik |
| 1.3 – 1.4 | Ekspresivitas tinggi, ada sedikit noise | Keperluan kreatif atau augmentasi data |

---

## Hasil

### Uji Pengenalan

50 sampel yang dihasilkan (5 per angka × 10 angka, temperature=0.85) diajukan ke GPT-4o:

| Angka | Dikenali | Kepercayaan |
|-------|-----------|------------|
| 0 | 5/5 | Tinggi |
| 1 | 5/5 | Tinggi |
| 2 | 5/5 | Tinggi |
| 3 | 5/5 | Tinggi |
| 4 | 5/5 | Tinggi |
| 5 | 5/5 | Tinggi |
| 6 | 5/5 | Tinggi |
| 7 | 5/5 | Tinggi |
| 8 | 5/5 | Tinggi |
| 9 | 5/5 | Tinggi |
| **Total** | **50/50** | **100%** |

### Metrik Model (Epoch 60)

```
Total loss  :  ~95
Recon loss  :  ~62   (BCE, sum/batch)
KL loss     :  ~10
```

---

## Keputusan Desain

**Mengapa VAE dan bukan GAN?**
Stabilitas pelatihan. GAN memerlukan penyeimbangan generator dan diskriminator yang cermat — rentan terhadap mode collapse, vanishing gradient, atau divergensi pelatihan. VAE dengan regularisasi β dilatih secara andal dalam satu pass dan menghasilkan ruang laten yang halus dan dapat dikontrol.

**Mengapa β=4.0?**
Nilai β yang lebih tinggi memaksa encoder menggunakan ruang laten yang lebih terstruktur dan terdisentanglasi. Artinya, sampel yang diambil secara acak dari prior `N(0,I)` menghasilkan angka yang lebih koheren dan dapat dikenali dengan variasi alami — ideal untuk kebutuhan generasi gambar.

**Mengapa ResBlocks?**
Koneksi residual menstabilkan aliran gradien pada jaringan yang lebih dalam. Bahkan pada model yang relatif kecil ini, ResBlocks mencegah stagnasi pelatihan dan memungkinkan decoder mempelajari detail sapuan yang lebih halus.

**Mengapa temperature pada z?**
Sebagai pengganti truncation (yang umum digunakan pada GAN), noise diskalakan dengan `z ~ N(0, T²I)`. Pendekatan ini dapat diturunkan secara diferensial, mempertahankan bentuk distribusi, dan memberikan kontrol kontinu antara presisi (T rendah) dan kreativitas (T tinggi).

---

## Batasan

- Resolusi output adalah 28×28 piksel (resolusi native MNIST). Pembesaran dilakukan dengan interpolasi NEAREST — menghasilkan tampilan berbentuk kotak pada ukuran tampilan yang besar.
- Beberapa angka secara inheren ambigu (1 vs 7, 4 vs 9) — sampel yang berada di dekat batas keputusan dapat sesekali menyerupai angka yang salah pada temperature tinggi.
- Model dilatih hanya pada dataset MNIST — tidak dapat digeneralisasi ke alfabet, aksara lain, atau simbol non-angka.

---

## Dependensi

```
torch>=2.1         PyTorch (pelatihan dan inferensi)
torchvision>=0.16  Pemuat dataset MNIST dan utilitas gambar
gradio>=4.44       Framework antarmuka web
Pillow>=9.5        Pemrosesan gambar untuk aplikasi
numpy>=1.24        Operasi array
```

---

## Referensi

- [MNIST Database](http://yann.lecun.com/exdb/mnist/) — LeCun, Cortes, Burges (1998)
- [β-VAE paper](https://openreview.net/forum?id=Sy2fchgxl) — Higgins et al. (2017)
- [VAE original paper](https://arxiv.org/abs/1312.6114) — Kingma & Welling (2013)
- [HuggingFace Spaces](https://huggingface.co/spaces) — hosting model gratis

---

<p align="center">
  <b>Pengembangan dari tim StarLive SAINT</b>
</p>

<p align="center">Danny Aulia · Said Hasan Hanafiah · Noah Von Nobelius · Arvian Raveindra Pradana</p>
