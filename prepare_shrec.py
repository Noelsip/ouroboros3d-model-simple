"""
Konversi dataset SHREC 2026 (Reconstruction of High-Frequency Geometry)
-> format loader dataset.py.

SHREC menyediakan, PER OBJEK: 90 gambar multi-view (PNG) + pose kamera COLMAP
(intrinsics + extrinsics). Karena POSE KAMERA NYATA tersedia, generator bisa
belajar geometri yang benar (hasil tidak blur seperti foto turntable).

Yang dilakukan:
  - Temukan objek secara REKURSIF: sebuah folder dianggap "1 objek" hanya jika
    dia punya gambar SENDIRI (langsung / di subfolder images/) DAN file pose
    SENDIRI (images.txt COLMAP / transforms.json). Setelah ketemu, TIDAK menelusur
    lebih dalam -> mencegah gambar antar-objek tercampur jadi satu.
  - Parse pose: COLMAP images.txt ATAU transforms.json (NeRF-style).
  - Konversi extrinsics -> matriks 4x4 camera-to-world (konvensi OpenGL: y-up,
    kamera melihat -Z) agar cocok dengan GaussianRenderer.
  - Normalisasi skala supaya kamera berjarak ~target-radius dari origin.
  - Subsample N view, generate mask (rembg/border), tulis per objek:
        object_XXX/views/view_YY.png
        object_XXX/ccm/view_YY.npy        (placeholder nol -> loss CCM mati)
        object_XXX/mask/view_YY.npy       (siluet)
        object_XXX/camera_poses.npy       (pose NYATA per objek)

Contoh:
    python prepare_shrec.py --input data/shrec --output data/real \
        --num-views 16 --max-objects 500 --mask-method rembg

Catatan: pakai loader & training yang sama:
    python train.py --data-root data/real --num-views 16 --w-ccm 0 --w-mask 0.5 ...
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


# ----------------------------- helper gambar (self-contained) -----------------------------

def list_images(d):
    """Semua gambar di folder d (non-rekursif), unik + terurut nama."""
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(d, f"*{ext}")))
    return sorted(set(files))


def pick_indices(m, n, mode):
    """Pilih n indeks dari m gambar. 'even' = tersebar merata, 'first' = n pertama."""
    if mode == "even" and n < m:
        return [int(round(i * (m - 1) / (n - 1))) for i in range(n)]
    return list(range(n))


def resize_letterbox(im, size):
    """Resize jaga proporsi lalu pad ke kotak size x size.
    RGB -> pad putih; RGBA -> pad transparan."""
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
    """Estimasi mask foreground tanpa rembg: pakai alpha kalau ada, kalau tidak
    tebak warna background dari piksel tepi. Return (rgb_white_uint8, mask_float)."""
    arr = np.array(im_rgba)
    if arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[..., :3].astype(np.float32)
        mask = (arr[..., 3] > 127).astype(np.float32)
    else:
        rgb = np.array(im_rgba.convert("RGB")).astype(np.float32)
        border = np.concatenate([
            rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]
        ], axis=0)
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(rgb - bg[None, None, :], axis=-1)
        mask = (dist > bg_threshold).astype(np.float32)
    m3 = mask[..., None]
    rgb_white = (rgb * m3 + 255.0 * (1.0 - m3)).clip(0, 255).astype(np.uint8)
    return rgb_white, mask


def segment_rembg(im, session):
    """Segmentasi objek pakai rembg (U2Net). Return (rgb_white_uint8, mask_float)."""
    from rembg import remove
    out = remove(im.convert("RGB"), session=session)
    arr = np.array(out)
    rgb = arr[..., :3].astype(np.float32)
    mask = (arr[..., 3] > 127).astype(np.float32)
    m3 = mask[..., None]
    rgb_white = (rgb * m3 + 255.0 * (1.0 - m3)).clip(0, 255).astype(np.uint8)
    return rgb_white, mask


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
    R_w2c = qvec2rotmat(qvec)
    R_c2w = R_w2c.T
    C = -R_c2w @ np.asarray(tvec, dtype=np.float64)
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
        if "." not in name:
            name += ".png"
        poses[name] = np.array(fr["transform_matrix"], dtype=np.float64)
    return poses


def load_poses(pose_path):
    if pose_path.endswith(".json"):
        return parse_transforms_json(pose_path)
    return parse_colmap_images_txt(pose_path)


# ----------------------------- object discovery (anti cross-object) -----------------------------

# Lokasi DANGKAL (tidak rekursif) tempat pose 1 objek biasanya berada.
_POSE_SUBDIRS = ["", "sparse", os.path.join("sparse", "0"), "colmap"]
_POSE_NAMES = ["images.txt", "transforms.json", "transforms_train.json"]


def find_pose_file(obj_dir):
    """Cari file pose MILIK obj_dir saja (dangkal) -> tidak mengambil pose objek lain."""
    for sub in _POSE_SUBDIRS:
        base = os.path.join(obj_dir, sub) if sub else obj_dir
        for nm in _POSE_NAMES:
            c = os.path.join(base, nm)
            if os.path.isfile(c):
                return c
    return None


def find_images_for_object(obj_dir):
    """Gambar MILIK obj_dir: langsung di obj_dir atau di subfolder images/ (dangkal).
    TIDAK rekursif -> tidak menyedot gambar dari folder objek lain."""
    for sub in (obj_dir, os.path.join(obj_dir, "images")):
        imgs = list_images(sub)
        if imgs:
            return imgs
    return []


def is_object_root(d):
    """True jika d adalah folder 1 objek: punya gambar sendiri DAN pose sendiri."""
    return bool(find_images_for_object(d)) and (find_pose_file(d) is not None)


def find_objects(input_dir, exclude, max_depth=6):
    """
    Telusuri pohon folder secara rekursif. Begitu sebuah folder dikenali sebagai
    objek (punya gambar + pose sendiri), folder itu dicatat dan TIDAK ditelusur
    lebih dalam. Ini yang mencegah satu 'objek' menelan gambar objek-objek lain.
    """
    exclude = {e.lower() for e in exclude}
    found = []

    def walk(d, depth):
        if depth > max_depth:
            return
        if is_object_root(d):
            found.append(d)
            return  # jangan masuk ke dalam objek
        for sub in sorted(glob.glob(os.path.join(d, "*"))):
            if not os.path.isdir(sub):
                continue
            if os.path.basename(sub).lower() in exclude:
                continue
            walk(sub, depth + 1)

    walk(input_dir, 0)
    return sorted(found)


def normalize_poses(c2w_list, target_radius):
    """Skala translasi kamera supaya rata-rata jarak ke origin = target_radius."""
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
    ap.add_argument("--max-objects", type=int, default=500,
                    help="Batasi jumlah objek (0=semua 938). Default 500 -> cukup untuk "
                         "model belajar tapi hemat ruang/waktu.")
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

    obj_dirs = find_objects(args.input, args.exclude)
    if not obj_dirs:
        raise SystemExit(
            f"Tidak ada objek (gambar + pose) ditemukan di '{args.input}'.\n"
            "Struktur diharap: <input>/.../<objek>/ berisi gambar (langsung atau di "
            "subfolder images/) + file pose (images.txt COLMAP di ./, sparse/, "
            "sparse/0/, atau transforms.json).\n"
            "Kalau pose-mu berupa images.bin (COLMAP biner), konversi dulu ke TXT:\n"
            "  colmap model_converter --input_path <sparse> --output_path <sparse> --output_type TXT"
        )
    print(f"Ditemukan {len(obj_dirs)} folder objek (tiap objek terpisah).")

    # cocokkan gambar <-> pose DALAM tiap objek (isolasi -> tidak campur antar objek)
    usable = []
    for obj_dir in obj_dirs:
        imgs = find_images_for_object(obj_dir)
        pose_path = find_pose_file(obj_dir)
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

        # VERIFIKASI anti-campur: 3 objek pertama, cetak sumber + contoh nama file.
        if new_id < 3:
            sample = [os.path.basename(pairs[i][0]) for i in sel[:3]]
            print(f"  object_{new_id:03d}  <-  {obj_dir}  (contoh view: {sample})")

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

    print(f"\nSelesai. Dataset di: {args.output}/ ({len(usable)} objek)")
    print(f"  -> latih: python train.py --data-root {args.output} "
          f"--num-views {args.num_views} --w-ccm 0 --w-mask 0.5 --use-feedback --joint")


if __name__ == "__main__":
    main()
