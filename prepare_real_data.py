"""
Konversi dataset multi-view REAL (PNG saja) -> format yang dipahami dataset.py.

Dataset tiap objek punya beberapa gambar multi-view (PNG/JPG), TANPA CCM

ASUMSI POSE: view diurutkan keliling objek (turntable) dengan azimuth merata
0..360. Kalau urutan view-mu beda, sesuaikan --total-azimuth / urutan file.

Contoh:
    python prepare_real_data.py --input data/drive_download --output data/real
    python prepare_real_data.py --input data/drive_download --output data/real --num-views 6
"""
import argparse
import glob
import os
import shutil

import numpy as np
from PIL import Image

# reuse fungsi pose dari generator sintetis (kamera menghadap origin)
from generate_synthetic_dataset import build_camera_pose

IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def list_images(d):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(d, f"*{ext}")))
    # unik + terurut (nama file menentukan urutan view)
    return sorted(set(files))


def find_object_dirs(input_dir):
    """
    Cari folder objek. Dukung dua layout:
      A) input/<obj>/views/*.png
      B) input/<obj>/*.png   (gambar langsung di folder objek)
    """
    objects = []
    for sub in sorted(glob.glob(os.path.join(input_dir, "*"))):
        if not os.path.isdir(sub):
            continue
        views_sub = os.path.join(sub, "views")
        if os.path.isdir(views_sub) and list_images(views_sub):
            objects.append((sub, list_images(views_sub)))      # layout A
        elif list_images(sub):
            objects.append((sub, list_images(sub)))            # layout B
    return objects


def estimate_foreground(im_rgba, bg_threshold=25):
    """
    Estimasi mask foreground (objek) + komposit ke background putih.

    Strategi:
      - Kalau gambar punya alpha channel -> pakai alpha (paling akurat).
      - Kalau tidak -> tebak warna background dari piksel tepi (median border),
        lalu mask = piksel yang cukup beda dari warna background.

    Return: (rgb_white_uint8 [H,W,3], mask_float [H,W] in {0,1})
    """
    arr = np.array(im_rgba)
    if arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[..., :3].astype(np.float32)
        mask = (arr[..., 3] > 127).astype(np.float32)
    else:
        rgb = np.array(im_rgba.convert("RGB")).astype(np.float32)
        h, w, _ = rgb.shape
        border = np.concatenate([
            rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]
        ], axis=0)
        bg = np.median(border, axis=0)                       # warna background
        dist = np.linalg.norm(rgb - bg[None, None, :], axis=-1)
        mask = (dist > bg_threshold).astype(np.float32)

    # komposit objek ke putih (renderer asumsikan background putih)
    m3 = mask[..., None]
    rgb_white = (rgb * m3 + 255.0 * (1.0 - m3)).clip(0, 255).astype(np.uint8)
    return rgb_white, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder hasil download dari Drive")
    ap.add_argument("--output", default="data/real", help="Folder output format-standar")
    ap.add_argument("--num-views", type=int, default=0,
                    help="Jumlah view per objek. 0 = pakai minimum yang ada di semua objek.")
    ap.add_argument("--total-azimuth", type=float, default=360.0,
                    help="Rentang azimuth total view (derajat). 360 = keliling penuh.")
    ap.add_argument("--elevation", type=float, default=0.0, help="Elevasi kamera (derajat).")
    ap.add_argument("--no-mask", action="store_true",
                    help="Jangan generate mask foreground (skip deteksi background).")
    ap.add_argument("--bg-threshold", type=float, default=25.0,
                    help="Ambang jarak warna untuk pisahkan objek dari background (0-255).")
    args = ap.parse_args()

    objs = find_object_dirs(args.input)
    if not objs:
        raise SystemExit(
            f"Tidak ada objek/gambar ditemukan di '{args.input}'.\n"
            f"Pastikan strukturnya: {args.input}/<nama_objek>/<gambar>.png"
        )

    counts = [len(imgs) for _, imgs in objs]
    print(f"Ditemukan {len(objs)} objek. Jumlah view per objek: "
          f"min={min(counts)}, max={max(counts)}")

    num_views = args.num_views or min(counts)
    if num_views < 2:
        raise SystemExit(f"num_views={num_views} terlalu kecil (butuh >= 2 view per objek).")

    usable = [(d, imgs) for d, imgs in objs if len(imgs) >= num_views]
    skipped = len(objs) - len(usable)
    print(f"Pakai num_views={num_views}. Objek terpakai={len(usable)}, dilewati(<{num_views} view)={skipped}")

    # poses sama untuk semua objek (azimuth merata)
    azimuths = np.linspace(0, args.total_azimuth, num_views, endpoint=(args.total_azimuth >= 360.0) is False)
    poses = np.stack(
        [build_camera_pose(az, elevation_deg=args.elevation) for az in azimuths], axis=0
    ).astype(np.float32)  # [N,4,4]

    os.makedirs(args.output, exist_ok=True)
    for new_id, (src_dir, imgs) in enumerate(usable):
        obj_out = os.path.join(args.output, f"object_{new_id:03d}")
        views_out = os.path.join(obj_out, "views")
        ccm_out = os.path.join(obj_out, "ccm")
        os.makedirs(views_out, exist_ok=True)
        os.makedirs(ccm_out, exist_ok=True)
        if not args.no_mask:
            mask_out = os.path.join(obj_out, "mask")
            os.makedirs(mask_out, exist_ok=True)

        for v in range(num_views):
            src_img = imgs[v]
            dst_img = os.path.join(views_out, f"view_{v:02d}.png")
            with Image.open(src_img) as im:
                if args.no_mask:
                    rgb = np.array(im.convert("RGB"))
                    mask = None
                else:
                    rgb, mask = estimate_foreground(im, args.bg_threshold)
            Image.fromarray(rgb).save(dst_img)

            h, w = rgb.shape[0], rgb.shape[1]
            # CCM placeholder (nol) -> loss CCM tetap non-aktif (tak ada geometri GT)
            np.save(os.path.join(ccm_out, f"view_{v:02d}.npy"),
                    np.zeros((h, w, 3), dtype=np.float32))
            # mask foreground (siluet) -> dipakai loss --w-mask untuk supervisi bentuk 3D
            if mask is not None:
                np.save(os.path.join(mask_out, f"view_{v:02d}.npy"),
                        mask.astype(np.float32))

        np.save(os.path.join(obj_out, "camera_poses.npy"), poses)

    print(f"\nSelesai. Dataset format-standar di: {args.output}/")
    mask_flag = "" if args.no_mask else " --w-mask 0.5"
    print(f"  -> latih dengan:  --data-root {args.output} --num-views {num_views} --w-ccm 0{mask_flag}")


if __name__ == "__main__":
    main()
