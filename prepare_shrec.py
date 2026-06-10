"""
Konversi dataset SHREC 2026 (High-Frequency Geometry) -> format loader dataset.py.

SHREC menyediakan, per objek: banyak gambar multi-view (PNG) + pose kamera
format COLMAP (intrinsics+extrinsics). Berbeda dengan foto Drive biasa, di sini
POSE KAMERA NYATA tersedia -> generator bisa belajar (tidak blur).

Yang dilakukan:
  - Cari objek (folder berisi gambar + file pose).
  - Parse pose: COLMAP images.txt  ATAU  transforms.json (NeRF-style).
  - Konversi extrinsics -> matriks 4x4 camera-to-world (konvensi OpenGL: y-up,
    kamera melihat -Z) supaya cocok dengan GaussianRenderer.
  - Normalisasi skala supaya kamera berjarak ~target-radius dari origin
    (renderer ortografik mengasumsikan objek di origin, skala kanonik [-1,1]).
  - Subsample N view, generate mask (rembg/border), tulis:
        object_XXX/views/view_YY.png
        object_XXX/ccm/view_YY.npy        (placeholder nol -> loss CCM mati)
        object_XXX/mask/view_YY.npy       (siluet)
        object_XXX/camera_poses.npy       (pose NYATA per objek)

Contoh:
    python prepare_shrec.py --input data/shrec_download --output data/real \
        --num-views 16 --max-objects 60 --mask-method rembg

Catatan: pakai loader & training yang sama:
    python train.py --data-root data/real --num-views 16 --w-ccm 0 --w-mask 0.5 ...
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

# reuse helper dari converter Drive (satu sumber, hindari duplikasi)
from prepare_real_data import (
    list_images, pick_indices, resize_letterbox,
    estimate_foreground, segment_rembg,
)


# ----------------------------- pose parsing -----------------------------

def qvec2rotmat(q):
    """Quaternion COLMAP (w,x,y,z) -> rotasi 3x3 (world->camera)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def colmap_w2c_to_c2w_opengl(qvec, tvec):
    """COLMAP (world->cam, kamera lihat +Z, y-down) -> c2w OpenGL (lihat -Z, y-up)."""
    R_w2c = qvec2rotmat(qvec)              # world->cam
    R_c2w = R_w2c.T
    C = -R_c2w @ np.asarray(tvec, dtype=np.float64)   # pusat kamera (world)
    # flip sumbu y,z kamera: OpenCV -> OpenGL
    R_c2w_gl = R_c2w @ np.diag([1.0, -1.0, -1.0])
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R_c2w_gl
    pose[:3, 3] = C
    return pose


def parse_colmap_images_txt(path):
    """Parse COLMAP images.txt -> dict {basename_gambar: pose_c2w_opengl [4x4]}."""
    poses = {}
    with open(path, "r") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    # format: 2 baris per gambar; baris-1 = pose, baris-2 = points2D (diabaikan)
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if len(parts) < 10:
            continue
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        qvec = list(map(float, parts[1:5]))
        tvec = list(map(float, parts[5:8]))
        name = parts[9]
        poses[os.path.basename(name)] = colmap_w2c_to_c2w_opengl(qvec, tvec)
        i += 1  # lewati baris points2D
    return poses


def parse_transforms_json(path):
    """Parse transforms.json (NeRF) -> dict {basename: c2w [4x4]} (sudah OpenGL)."""
    poses = {}
    with open(path, "r") as f:
        data = json.load(f)
    for fr in data.get("frames", []):
        name = os.path.basename(fr["file_path"])
        if "." not in name:           # NeRF kadang tanpa ekstensi
            name += ".png"
        poses[name] = np.array(fr["transform_matrix"], dtype=np.float64)
    return poses


def find_pose_file(obj_dir):
    """Cari sumber pose di dalam obj_dir (cek beberapa lokasi umum COLMAP)."""
    cands = [
        os.path.join(obj_dir, "images.txt"),
        os.path.join(obj_dir, "sparse", "images.txt"),
        os.path.join(obj_dir, "sparse", "0", "images.txt"),
        os.path.join(obj_dir, "colmap", "images.txt"),
        os.path.join(obj_dir, "transforms.json"),
        os.path.join(obj_dir, "transforms_train.json"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    # fallback: cari rekursif
    for pat in ("images.txt", "transforms*.json"):
        hit = glob.glob(os.path.join(obj_dir, "**", pat), recursive=True)
        if hit:
            return hit[0]
    return None


def load_poses(pose_path):
    if pose_path.endswith(".json"):
        return parse_transforms_json(pose_path)
    return parse_colmap_images_txt(pose_path)


# ----------------------------- object discovery -----------------------------

def find_images_for_object(obj_dir):
    """Gambar bisa langsung di obj_dir atau di subfolder images/."""
    for sub in (os.path.join(obj_dir, "images"), obj_dir):
        imgs = list_images(sub)
        if imgs:
            return imgs
    # rekursif terakhir
    imgs = []
    for ext in (".png", ".jpg", ".jpeg"):
        imgs += glob.glob(os.path.join(obj_dir, "**", f"*{ext}"), recursive=True)
    return sorted(set(imgs))


def find_objects(input_dir, exclude):
    exclude = {e.lower() for e in exclude}
    objs = []
    for sub in sorted(glob.glob(os.path.join(input_dir, "*"))):
        if not os.path.isdir(sub) or os.path.basename(sub).lower() in exclude:
            continue
        imgs = find_images_for_object(sub)
        pose_path = find_pose_file(sub)
        if imgs and pose_path:
            objs.append((sub, imgs, pose_path))
    return objs


def normalize_poses(c2w_list, target_radius):
    """Skala translasi kamera supaya rata-rata jarak ke origin = target_radius.
    (asumsi objek di origin, sesuai data object-centric SHREC.)"""
    centers = np.stack([p[:3, 3] for p in c2w_list], axis=0)
    radii = np.linalg.norm(centers, axis=1)
    mean_r = float(np.mean(radii)) + 1e-8
    s = target_radius / mean_r
    out = []
    for p in c2w_list:
        q = p.copy()
        q[:3, 3] = q[:3, 3] * s
        out.append(q.astype(np.float32))
    return out


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder hasil download SHREC")
    ap.add_argument("--output", default="data/real", help="Folder output format loader")
    ap.add_argument("--num-views", type=int, default=16, help="View per objek (subsample dari 90).")
    ap.add_argument("--max-objects", type=int, default=0, help="Batasi jumlah objek (0=semua).")
    ap.add_argument("--out-size", type=int, default=128, help="Resize gambar ke NxN.")
    ap.add_argument("--target-radius", type=float, default=2.0,
                    help="Jarak kamera ke origin setelah normalisasi (samakan dgn renderer).")
    ap.add_argument("--exclude", nargs="+", default=[], help="Folder yang dilewati.")
    ap.add_argument("--subsample", choices=["even", "first"], default="even")
    ap.add_argument("--stretch", action="store_true", help="Resize paksa kotak (gepeng).")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--mask-method", choices=["rembg", "border"], default="rembg")
    ap.add_argument("--bg-threshold", type=float, default=25.0)
    args = ap.parse_args()

    objs = find_objects(args.input, args.exclude)
    if not objs:
        raise SystemExit(
            f"Tidak ada objek (gambar + pose) ditemukan di '{args.input}'.\n"
            "Struktur diharap: <input>/<objek>/ berisi gambar + images.txt "
            "(COLMAP) atau transforms.json.\n"
            "Kalau pose-mu berupa images.bin (COLMAP biner), konversi dulu:\n"
            "  colmap model_converter --input_path <sparse> --output_path <sparse> --output_type TXT"
        )

    # cocokkan gambar <-> pose per objek, simpan yang punya keduanya
    usable = []
    for obj_dir, imgs, pose_path in objs:
        try:
            poses = load_poses(pose_path)
        except Exception as e:
            print(f"  [skip] {os.path.basename(obj_dir)}: gagal parse pose ({e})")
            continue
        pairs = [(p, poses[os.path.basename(p)]) for p in imgs
                 if os.path.basename(p) in poses]
        if len(pairs) >= args.num_views:
            usable.append((obj_dir, pairs))
        else:
            print(f"  [skip] {os.path.basename(obj_dir)}: hanya {len(pairs)} gambar berpose "
                  f"(< {args.num_views})")

    if not usable:
        raise SystemExit("Tidak ada objek dengan cukup gambar-berpose. "
                         "Cek apakah NAME di pose cocok dengan nama file gambar.")

    if args.max_objects and len(usable) > args.max_objects:
        usable = usable[:args.max_objects]
    print(f"Objek terpakai: {len(usable)} (num_views={args.num_views})")

    # session rembg sekali
    session = None
    if not args.no_mask and args.mask_method == "rembg":
        try:
            from rembg import new_session
            print("Menyiapkan rembg (U2Net)...")
            session = new_session("u2net")
        except (Exception, SystemExit) as e:
            print(f"  [warn] rembg tidak tersedia ({e!r}). Fallback ke border.")
            args.mask_method = "border"

    try:
        from tqdm import tqdm
        iterator = tqdm(list(enumerate(usable)), desc="Konversi SHREC", unit="objek")
    except Exception:
        iterator = enumerate(usable)

    os.makedirs(args.output, exist_ok=True)
    for new_id, (obj_dir, pairs) in iterator:
        sel = pick_indices(len(pairs), args.num_views, args.subsample)
        obj_out = os.path.join(args.output, f"object_{new_id:03d}")
        views_out = os.path.join(obj_out, "views")
        ccm_out = os.path.join(obj_out, "ccm")
        os.makedirs(views_out, exist_ok=True)
        os.makedirs(ccm_out, exist_ok=True)
        if not args.no_mask:
            mask_out = os.path.join(obj_out, "mask")
            os.makedirs(mask_out, exist_ok=True)

        c2w_sel = [pairs[idx][1] for idx in sel]
        poses_norm = normalize_poses(c2w_sel, args.target_radius)

        for v, idx in enumerate(sel):
            src_img = pairs[idx][0]
            with Image.open(src_img) as im:
                im = (im.resize((args.out_size, args.out_size)) if args.stretch
                      else resize_letterbox(im, args.out_size))
                if args.no_mask:
                    rgb = np.array(im.convert("RGB")); mask = None
                elif args.mask_method == "rembg":
                    rgb, mask = segment_rembg(im, session)
                else:
                    rgb, mask = estimate_foreground(im, args.bg_threshold)
            Image.fromarray(rgb).save(os.path.join(views_out, f"view_{v:02d}.png"))
            h, w = rgb.shape[0], rgb.shape[1]
            np.save(os.path.join(ccm_out, f"view_{v:02d}.npy"),
                    np.zeros((h, w, 3), dtype=np.float32))
            if mask is not None:
                np.save(os.path.join(mask_out, f"view_{v:02d}.npy"), mask.astype(np.float32))

        np.save(os.path.join(obj_out, "camera_poses.npy"),
                np.stack(poses_norm, axis=0).astype(np.float32))

    print(f"\nSelesai. Dataset di: {args.output}/")
    print(f"  -> latih: python train.py --data-root {args.output} "
          f"--num-views {args.num_views} --w-ccm 0 --w-mask 0.5 --use-feedback --joint")


if __name__ == "__main__":
    main()
