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
from tqdm import tqdm

IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def build_camera_pose(azimuth_deg, elevation_deg=0.0, radius=2.0):
    """4x4 camera-to-world matrix (kamera menghadap origin)."""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    cam_pos = np.array([
        radius * np.cos(el) * np.sin(az),
        radius * np.sin(el),
        radius * np.cos(el) * np.cos(az),
    ], dtype=np.float32)
    forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
    up = np.array([0, 1, 0], dtype=np.float32)
    right = np.cross(up, -forward)
    right = right / (np.linalg.norm(right) + 1e-8)
    up_new = np.cross(-forward, right)
    R = np.stack([right, up_new, -forward], axis=0)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.T
    pose[:3, 3] = cam_pos
    return pose


def list_images(d):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(d, f"*{ext}")))
    # unik + terurut (nama file menentukan urutan view)
    return sorted(set(files))


def find_object_dirs(input_dir, exclude=()):
    """
    Cari folder objek. Dukung dua layout:
      A) input/<obj>/views/*.png
      B) input/<obj>/*.png   (gambar langsung di folder objek)
    exclude: nama folder yang dilewati (mis. {"images"}).
    """
    exclude = {e.lower() for e in exclude}
    objects = []
    for sub in sorted(glob.glob(os.path.join(input_dir, "*"))):
        if not os.path.isdir(sub):
            continue
        if os.path.basename(sub).lower() in exclude:
            print(f"  [exclude] {os.path.basename(sub)}")
            continue
        views_sub = os.path.join(sub, "views")
        if os.path.isdir(views_sub) and list_images(views_sub):
            objects.append((sub, list_images(views_sub)))      # layout A
        elif list_images(sub):
            objects.append((sub, list_images(sub)))            # layout B
    return objects


def pick_indices(m, n, mode):
    """Pilih n indeks dari m gambar. 'even' = tersebar merata, 'first' = n pertama."""
    if mode == "even" and n < m:
        return [int(round(i * (m - 1) / (n - 1))) for i in range(n)]
    return list(range(n))


def resize_letterbox(im, size):
    """Resize jaga proporsi (aspect ratio) lalu pad ke kotak size x size.
    RGB -> pad putih; RGBA -> pad transparan (jadi background di mask)."""
    w, h = im.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    im_r = im.resize((nw, nh))
    off = ((size - nw) // 2, (size - nh) // 2)
    if im.mode == "RGBA":
        canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        canvas.paste(im_r, off)
    else:
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(im_r.convert("RGB"), off)
    return canvas


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


def segment_rembg(im, session):
    """Segmentasi objek pakai rembg (U2Net) -> mask jauh lebih akurat untuk
    foto background alami/bertekstur. Return (rgb_white_uint8, mask_float)."""
    from rembg import remove
    out = remove(im.convert("RGB"), session=session)   # RGBA, alpha = matte objek
    arr = np.array(out)
    rgb = arr[..., :3].astype(np.float32)
    mask = (arr[..., 3] > 127).astype(np.float32)
    m3 = mask[..., None]
    rgb_white = (rgb * m3 + 255.0 * (1.0 - m3)).clip(0, 255).astype(np.uint8)
    return rgb_white, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder hasil download dari Drive")
    ap.add_argument("--output", default="data/real", help="Folder output format-standar")
    ap.add_argument("--out-size", type=int, default=128,
                    help="Resize gambar ke NxN saat simpan (training cuma 64-128px; "
                         "full-res bikin file CCM raksasa & sangat lambat).")
    ap.add_argument("--num-views", type=int, default=0,
                    help="Jumlah view per objek. 0 = pakai minimum yang ada di semua objek.")
    ap.add_argument("--total-azimuth", type=float, default=360.0,
                    help="Rentang azimuth total view (derajat). 360 = keliling penuh.")
    ap.add_argument("--elevation", type=float, default=0.0, help="Elevasi kamera (derajat).")
    ap.add_argument("--no-mask", action="store_true",
                    help="Jangan generate mask foreground (skip segmentasi).")
    ap.add_argument("--mask-method", choices=["rembg", "border"], default="rembg",
                    help="rembg=segmentasi U2Net (akurat utk background alami). "
                         "border=tebak warna tepi (cepat, hanya utk background polos).")
    ap.add_argument("--bg-threshold", type=float, default=25.0,
                    help="[border] Ambang jarak warna pisahkan objek dari background (0-255).")
    ap.add_argument("--exclude", nargs="+", default=[],
                    help="Nama folder yang dilewati, mis: --exclude images")
    ap.add_argument("--stretch", action="store_true",
                    help="Resize paksa kotak (gepeng). Default: jaga proporsi + padding putih.")
    ap.add_argument("--subsample", choices=["even", "first"], default="even",
                    help="Cara ambil N view dari foto yang tersedia. even=tersebar merata.")
    args = ap.parse_args()

    objs = find_object_dirs(args.input, exclude=args.exclude)
    if not objs:
        raise SystemExit(
            f"Tidak ada objek/gambar ditemukan di '{args.input}'.\n"
            f"Pastikan strukturnya: {args.input}/<nama_objek>/<gambar>.png"
        )

    counts = [len(imgs) for _, imgs in objs]
    print(f"Ditemukan {len(objs)} objek:")
    for d, imgs in objs:
        print(f"  {os.path.basename(d):20s} {len(imgs)} foto")
    print(f"Jumlah view per objek: min={min(counts)}, max={max(counts)}")

    num_views = args.num_views or min(counts)
    if num_views < 2:
        raise SystemExit(f"num_views={num_views} terlalu kecil (butuh >= 2 view per objek).")

    usable = [(d, imgs) for d, imgs in objs if len(imgs) >= num_views]
    skipped = len(objs) - len(usable)
    print(f"Pakai num_views={num_views}. Objek terpakai={len(usable)}, dilewati(<{num_views} view)={skipped}")
    total_reads = num_views * len(usable)
    print(f"Total foto yang dibaca dari Drive: {total_reads}"
          + ("  (besar -> lambat; pertimbangkan --num-views lebih kecil, mis 12)"
             if total_reads > 120 else ""))

    # poses sama untuk semua objek (azimuth merata)
    azimuths = np.linspace(0, args.total_azimuth, num_views, endpoint=(args.total_azimuth >= 360.0) is False)
    poses = np.stack(
        [build_camera_pose(az, elevation_deg=args.elevation) for az in azimuths], axis=0
    ).astype(np.float32)  # [N,4,4]

    # Siapkan session rembg sekali (lazy + fallback ke border kalau gagal)
    session = None
    if not args.no_mask and args.mask_method == "rembg":
        try:
            from rembg import new_session
            print("Menyiapkan rembg (U2Net)... (download model ~170MB saat pertama kali)")
            session = new_session("u2net")
        except (Exception, SystemExit) as e:
            print(f"  [warn] rembg tidak tersedia ({e!r}). "
                  f"Fallback ke --mask-method border. "
                  f"Saran: pip install \"rembg[cpu]\".")
            args.mask_method = "border"
            session = None

    os.makedirs(args.output, exist_ok=True)
    pbar = tqdm(list(enumerate(usable)), desc="Konversi", unit="objek")
    for new_id, (src_dir, imgs) in pbar:
        pbar.set_postfix_str(os.path.basename(src_dir))
        obj_out = os.path.join(args.output, f"object_{new_id:03d}")
        views_out = os.path.join(obj_out, "views")
        ccm_out = os.path.join(obj_out, "ccm")
        os.makedirs(views_out, exist_ok=True)
        os.makedirs(ccm_out, exist_ok=True)
        if not args.no_mask:
            mask_out = os.path.join(obj_out, "mask")
            os.makedirs(mask_out, exist_ok=True)

        sel = pick_indices(len(imgs), num_views, args.subsample)
        for v, src_idx in enumerate(sel):
            src_img = imgs[src_idx]
            dst_img = os.path.join(views_out, f"view_{v:02d}.png")
            with Image.open(src_img) as im:
                # resize dulu -> hemat I/O drastis (CCM/mask ikut kecil)
                im = (im.resize((args.out_size, args.out_size)) if args.stretch
                      else resize_letterbox(im, args.out_size))
                if args.no_mask:
                    rgb = np.array(im.convert("RGB"))
                    mask = None
                elif args.mask_method == "rembg":
                    rgb, mask = segment_rembg(im, session)
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
