# Ouroboros3D — Implementasi Paper (Tahap 1 + 2)

Implementasi dari paper **"Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion"** (CVPR 2025), versi ringan yang bisa jalan di Colab free T4 atau GPU lokal kecil.

## Struktur File

| File | Isi |
|---|---|
| `model.py` | **Arsitektur lengkap Ouroboros3D**: `MultiViewGenerator` + `FeedbackEncoder` + `GaussianPredictor` + `GaussianRenderer` |
| `dataset.py` | Data loader dengan preprocessing (resize, normalize [-1,1], augmentasi), split train/val 85/15 |
| `train.py` | Joint training loop dengan 3D-aware feedback, self-conditioning, 3 komponen loss (mv, render, ccm), dan ablation flags |
| `generate_synthetic_dataset.py` | Generator dataset sintetis 60 objek × 8 view (RGB + CCM + camera poses). Langsung jalan tanpa download. |
| `prepare_objaverse.py` | Download subset 50 objek Objaverse-Rand6View dari HuggingFace (data real, opsional) |
| `requirements.txt` | Dependencies: torch, torchvision, numpy, pillow, tqdm, matplotlib |
| `Ouroboros3D_Colab.ipynb` | Notebook all-in-one untuk Colab (self-contained, pakai `%%writefile`) |

## Cara Jalankan

### Opsi A — Lokal

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Generate dataset sintetis (atau pakai prepare_objaverse.py untuk data real)
python generate_synthetic_dataset.py

# 3. Training full Ouroboros3D (joint + feedback)
python train.py --epochs 2 --batch-size 4 --img-size 64 --use-feedback --joint --tag full
```

### Opsi B — Google Colab

1. Upload `Ouroboros3D_Colab.ipynb` ke Colab (File → Upload notebook)
2. Runtime → Change runtime type → **T4 GPU** (gratis)
3. Runtime → Run all
4. Notebook akan otomatis tulis semua file `.py`, generate dataset, dan training

### Opsi C — Pakai Dataset Objaverse Real

```bash
pip install -r requirements.txt
pip install huggingface_hub
python prepare_objaverse.py        # download ~100-200 MB ke data/objaverse/
python train.py --data-root data/objaverse --epochs 2 --batch-size 4 --img-size 64 --use-feedback --joint --tag full
```

## Tahap yang Dicover

**Tahap 1 — Baseline Pipeline:**
- Data pipeline dengan preprocessing & augmentasi
- Baseline model (CNN encoder-decoder multi-view)
- Training loop dengan loss turun terbukti

**Tahap 2 — 3D-aware Recursive Diffusion:**
- `GaussianPredictor G` — rekonstruksi 3D Gaussian Splatting dari multi-view
- `GaussianRenderer` — differentiable renderer (RGB + CCM maps)
- `FeedbackEncoder` — RGB & CCM feedback encoders (meniru T2I-Adapter di paper)
- Recursive loop dengan 3D-aware feedback
- Joint training dengan probabilistic self-conditioning (p=0.5 sesuai paper)
- Ablation flags (`--use-feedback`, `--joint`) untuk replikasi Tabel 2 paper

## Contoh Command (Ablation Study)

```bash
# Full Ouroboros3D (joint + feedback) — matches "CCM dan RGB "
python train.py --epochs 2 --batch-size 4 --img-size 64 --use-feedback --joint --tag full

# Ablation: baseline (no joint, no feedback)
python train.py --epochs 2 --batch-size 4 --img-size 64 --tag baseline

# Ablation: joint tanpa feedback 
python train.py --epochs 2 --batch-size 4 --img-size 64 --joint --tag joint_only
```

## Dataset Alternatif (Real bukan generate python)

Jika ingin pakai data real selain Objaverse-Rand6View:
- **Google Scanned Objects (GSO)** — dataset evaluasi di paper: https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research
- **Objaverse LVIS subset** — https://huggingface.co/datasets/allenai/objaverse

Letakkan di `data/<nama>/object_XXX/views/view_YY.png` + siapkan CCM + camera poses sesuai format dataset sintetis (lihat `generate_synthetic_dataset.py` untuk referensi format).
