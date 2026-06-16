# Ouroboros3D — Implementasi Paper

Implementasi dari paper **"Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion"** (CVPR 2025), versi ringan yang bisa jalan di Colab free T4 atau GPU lokal kecil.

Konsep inti tetap dipertahankan (recursive **3D-aware feedback**: generate multi-view →
rekonstruksi 3D Gaussian → render → feedback ke step berikutnya). Karena tidak terhubung
ke engine 3D/Blender, **output yang ditampilkan = kumpulan multi-view image** + plot 3D
Gaussian sebagai bukti konsep (bukan ekspor mesh 3D).

## Struktur File

| File | Isi |
|---|---|
| `model.py` | **Arsitektur lengkap Ouroboros3D**: `MultiViewGenerator` + `FeedbackEncoder` + `GaussianPredictor` + `GaussianRenderer` |
| `dataset.py` | Data loader (resize, normalize [-1,1], augmentasi, mask siluet), split train/val 85/15 |
| `train.py` | Joint training loop dengan 3D-aware feedback, self-conditioning, dan 4 komponen loss (mv, render, ccm, mask) + ablation flags |
| `prepare_shrec.py` | Konversi dataset **SHREC 2026** → format loader (pose COLMAP nyata + mask siluet). *Self-contained.* |
| `prepare_objaverse.py` | (opsional/legacy) Download subset Objaverse-Rand6View dari HuggingFace |
| `requirements.txt` | Dependencies: torch, torchvision, numpy, pillow, tqdm, matplotlib |
| `Ouroboros3D_Colab.ipynb` | Notebook all-in-one untuk Colab (clone repo → SHREC → training → visualisasi) |

## Dataset — SHREC 2026

[SHREC 2026: Reconstruction of High-Frequency Geometry](https://shapevision.dcc.uchile.cl/cllull-shrec2026/):
938 objek heritage, **90 render multi-view per objek**, dengan **pose kamera COLMAP nyata**
(intrinsics + extrinsics). Pose nyata membuat generator bisa belajar geometri yang benar.

`prepare_shrec.py` menemukan tiap objek secara terpisah (gambar + pose **milik objek itu
sendiri**, anti-campur antar-objek), lalu menulis per objek:

```
data/real/object_XXX/
  views/view_YY.png        # multi-view RGB (objek di-komposit ke background putih)
  ccm/view_YY.npy          # placeholder nol (loss CCM dimatikan: --w-ccm 0)
  mask/view_YY.npy         # mask siluet (supervisi bentuk via --w-mask)
  camera_poses.npy         # pose c2w OpenGL [N,4,4]
```

## Cara Jalankan

### Opsi A — Google Colab (utama)

1. Upload `Ouroboros3D_Colab.ipynb` ke Colab (File → Upload notebook)
2. Runtime → Change runtime type → **T4 GPU** (gratis)
3. Jalankan cell `[0]` (clone repo) → setup → `[S1]`/`[S2]` (download + konversi SHREC) → `[3]`–`[8]`

Notebook menarik semua file `.py` langsung dari repo via `git clone` (tanpa upload zip).

### Opsi B — Lokal

```bash
# 1. Setup
pip install -r requirements.txt
pip install "rembg[cpu]" kaggle

# 2. Download SHREC 2026 dari Kaggle (butuh ~/.kaggle/kaggle.json)
kaggle datasets download -d cristianllull/shrec-2026-retrieval-of-high-frequency-geometry \
    -p data/shrec --unzip

# 3. Konversi -> data/real (500 objek, 16 view/objek)
python prepare_shrec.py --input data/shrec --output data/real \
    --num-views 16 --max-objects 500 --mask-method rembg

# 4. Training full Ouroboros3D (joint + feedback)
python train.py --data-root data/real --num-views 16 --epochs 10 --batch-size 2 \
    --img-size 64 --num-gaussians 128 --use-feedback --joint --w-ccm 0 --w-mask 0.5 --tag real
```

## Komponen Konsep

- `MultiViewGenerator F_θ` — dari conditioning image → prediksi N-1 target view
- `GaussianPredictor G` — rekonstruksi 3D Gaussian Splatting dari multi-view
- `GaussianRenderer` — differentiable renderer (RGB + CCM + alpha/siluet)
- `FeedbackEncoder` — RGB & CCM feedback encoders (meniru T2I-Adapter di paper)
- Recursive loop dengan 3D-aware feedback + probabilistic self-conditioning (p=0.5 sesuai paper)
- Loss siluet (`--w-mask`) menggantikan supervisi CCM untuk data real tanpa geometri GT

## Contoh Command (Ablation Study)

```bash
# Full Ouroboros3D (joint + feedback)
python train.py --data-root data/real --num-views 16 --img-size 64 --use-feedback --joint --w-ccm 0 --w-mask 0.5 --tag full

# Ablation: baseline (no joint, no feedback)
python train.py --data-root data/real --num-views 16 --img-size 64 --w-ccm 0 --w-mask 0.5 --tag baseline

# Ablation: joint tanpa feedback
python train.py --data-root data/real --num-views 16 --img-size 64 --joint --w-ccm 0 --w-mask 0.5 --tag joint_only
```
