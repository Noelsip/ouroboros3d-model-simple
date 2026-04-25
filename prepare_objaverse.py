import os
import json
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed

# Config
REPO_ID = "huanngzh/Objaverse-Rand6View"
NUM_OBJECTS = 50
OUTPUT_DIR = "data/objaverse"
IMG_SIZE = 64  # resize semua gambar
NUM_VIEWS_OUT = 8  # kode pakai 8 view
MAX_PARALLEL = 4   # parallel download thread

# 50 object ID pertama
OBJECT_IDS = [
    "019ce6e03b90427fbbebdfb9af9cd761", "029389da502d41a0aeada600137ae98b",
    "036467d0751642bf81c3c21eb1e64f66", "052cb069eb39472aa7b197e93263b93c",
    "050300ca39594a7383762d3c78b86fe6", "0575fb8cbf5f4ea8af7d3a8aa606c20a",
    "056c701100584c9abbd89015c49d9e9b", "0602f8824e70413387800a9350627c15",
    "07c2576f160e4de99aeb340ed342525f", "08093e5f9df54f7bb8a125bcb548988b",
    "0a52988e43ef406f8a073d7d89366406", "0a6b0e23465f4f2898194619edafb523",
    "0b69dd280793421ab2bda8ffb8988b30", "0c05826750d841cbbc568329596d2311",
    "0c3ae8ebd82f4dc19adff0b3237dea8a", "0ca74c49418e493096956fb433958352",
    "0e0cb794fa3e47bd978f63a26a18f6ee", "0f07ab4882064affb26f5940d666d0dd",
    "0f77705ee3654143a38dca9647a8b3a7", "0fe3b4476c1b45b08965bf4acad7fa94",
    "10214c65aaed47f0b84d5dd24b527247", "109850030d3943768f2c0af603343418",
    "1117f2b814ca457295c5624fd2fa8647", "1191d7b9f5f94de99549c2cc894525b9",
    "120616b52caa469597f89bd0b90e0536", "11f1db0a58a3486f89ceb4aed7fda32f",
    "12400246807548a2b678a44e429af010", "131e8e1408e040d3aa4d2c8130829481",
    "12a5f372d7ac4fd39660ae2a8079fb5b", "131d2477e612491e8825beb7a0104bec",
    "13d34af0aabf442c93b111472d0df7b6", "13e46f30d1db4161afc6ad9afba794de",
    "142dc666e43045dc82e4ddbf58b0d381", "14395daaca32448c980b65e512ce3a76",
    "145ffc45e5f5422bb652e52b836119c0", "1464e06b191041208b52d0bd038b3349",
    "152371aa3cea4b61bb3e0e7b1577ae76", "154a912dd3ad427ca568067be8fdc26f",
    "15a6077806104e4d852bf5f6b11c9af2", "1632deaa5d4b403cae954f95647375d9",
    "170648050dab4eb6b81312820d362897", "1779f886209f433db69ca6afa8af055e",
    "18900a5f1bb64547879985abbcd6b6b4", "19967bfcaaa04375b07f01c530bd188e",
    "1a3a7c3e05a04836bf458e523c9db9d8", "1a6fcef7887c4d7f8e3097dc51401c24",
    "1a809d4b75f740a4ae0d79147878c7f2", "1bccd001e2794970b6a244825c5889e0",
    "1c61f5dcc60e4f559b9bd4b31faa4052", "1d0d772f895c437da1f5e10d63e33060",
]


def get_shard(obj_id):
    """Shard dalam dataset ditentukan oleh 2 char pertama dari UUID."""
    return obj_id[:2]


def build_default_camera_pose(azimuth_deg, elevation_deg=0.0, radius=2.0):
    """Fallback camera pose jika meta.json tidak ada atau gagal parse."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cam_pos = np.array([
        radius * math.cos(el) * math.sin(az),
        radius * math.sin(el),
        radius * math.cos(el) * math.cos(az),
    ], dtype=np.float32)
    forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
    up = np.array([0, 1, 0], dtype=np.float32)
    right = np.cross(up, -forward); right = right / (np.linalg.norm(right) + 1e-8)
    up_new = np.cross(-forward, right)
    R = np.stack([right, up_new, -forward], axis=0)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.T
    pose[:3, 3] = cam_pos
    return pose


def generate_pseudo_ccm(rgb_img_np, pose_matrix):
    """
    Karena dataset asli tidak provide CCM, kita generate pseudo-CCM dari:
    - Foreground mask (pixel non-background)
    - Gradient pixel position (sebagai proxy XYZ canonical)

    BUKAN CCM asli, hanya placeholder supaya CCM loss tetap jalan.
    """
    H, W = rgb_img_np.shape[:2]
    # Foreground mask: pixel yang bukan background putih/hitam murni
    gray = rgb_img_np.mean(axis=-1)
    mask = ((gray > 10) & (gray < 245)).astype(np.float32)

    # CCM dummy: X dari kolom pixel, Y dari baris, Z dari brightness
    xs = np.linspace(-1, 1, W)[None, :].repeat(H, axis=0)
    ys = np.linspace(1, -1, H)[:, None].repeat(W, axis=1)
    zs = (gray / 255.0 - 0.5) * 2.0  # brightness sebagai proxy depth

    ccm = np.stack([xs, ys, zs], axis=-1).astype(np.float32)
    ccm = ccm * mask[..., None]  # mask out background
    return ccm


def parse_camera_from_meta(meta_json, view_idx):
    """
    Mencoba ekstrak camera extrinsic 4x4 dari meta.json.
    Format meta bisa bervariasi, jadi try beberapa key umum.
    """
    try:
        views = meta_json.get("locations", meta_json.get("views", None))
        if views and view_idx < len(views):
            v = views[view_idx]
            # Common formats: "transform_matrix", "extrinsic", "c2w"
            for key in ["transform_matrix", "extrinsic", "c2w", "RT"]:
                if key in v:
                    return np.array(v[key], dtype=np.float32).reshape(4, 4)
    except Exception:
        pass
    return None


def download_file_safe(repo_id, filename):
    """Download 1 file dari HF. Return local path atau None kalau gagal."""
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
        )
    except Exception as e:
        return None


def process_one_object(obj_id, out_dir):
    """Download semua file satu objek dan convert ke format kita."""
    shard = get_shard(obj_id)
    base_path = f"data/{shard}/{obj_id}"

    # Coba download 6 RGB + meta
    rgb_paths = []
    for i in range(6):
        p = download_file_safe(REPO_ID, f"{base_path}/color_{i:04d}.webp")
        if p is None:
            return False, f"Gagal download view {i}"
        rgb_paths.append(p)

    meta_path = download_file_safe(REPO_ID, f"{base_path}/meta.json")
    meta = {}
    if meta_path:
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    # Setup folder output
    obj_out = os.path.join(out_dir, f"object_{obj_id[:8]}")
    os.makedirs(os.path.join(obj_out, "views"), exist_ok=True)
    os.makedirs(os.path.join(obj_out, "ccm"), exist_ok=True)
    os.makedirs(os.path.join(obj_out, "depth"), exist_ok=True)

    poses = []
    # Convert 6 view asli, lalu duplikasi 2 terakhir -> total 8 view
    for out_idx in range(NUM_VIEWS_OUT):
        src_idx = out_idx if out_idx < 6 else (out_idx - 6)  # 6, 7 -> duplikat view 0, 1
        src_path = rgb_paths[src_idx]

        # Load, resize, save sebagai PNG
        img = Image.open(src_path).convert("RGB")
        img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_resized.save(os.path.join(obj_out, "views", f"view_{out_idx:02d}.png"))

        img_np = np.array(img_resized)

        # Camera pose
        pose = parse_camera_from_meta(meta, src_idx)
        if pose is None:
            # Fallback: asumsikan 8 view orbit equally-spaced
            pose = build_default_camera_pose(out_idx * (360.0 / NUM_VIEWS_OUT))
        poses.append(pose)

        # Pseudo-CCM (placeholder)
        ccm = generate_pseudo_ccm(img_np, pose)
        np.save(os.path.join(obj_out, "ccm", f"view_{out_idx:02d}.npy"), ccm)

        # Depth placeholder (kosong, kode tidak pakai)
        depth = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        np.save(os.path.join(obj_out, "depth", f"view_{out_idx:02d}.npy"), depth)

    np.save(os.path.join(obj_out, "camera_poses.npy"), np.stack(poses, axis=0))
    return True, None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"   Downloading {NUM_OBJECTS} objek dari {REPO_ID}")
    print(f"   Output: {OUTPUT_DIR}/")
    print(f"   Resolusi: {IMG_SIZE}×{IMG_SIZE}")
    print(f"   Parallel: {MAX_PARALLEL} threads\n")

    object_ids = OBJECT_IDS[:NUM_OBJECTS]
    success, failed = 0, 0

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {executor.submit(process_one_object, oid, OUTPUT_DIR): oid
                   for oid in object_ids}
        with tqdm(total=len(futures), desc="Processing") as pbar:
            for fut in as_completed(futures):
                oid = futures[fut]
                try:
                    ok, err = fut.result()
                    if ok:
                        success += 1
                    else:
                        failed += 1
                        pbar.write(f"{oid[:8]}: {err}")
                except Exception as e:
                    failed += 1
                    pbar.write(f"{oid[:8]}: exception {e}")
                pbar.update(1)

    print(f"\nSelesai!")
    print(f"   Sukses: {success} objek")
    print(f"   Gagal : {failed} objek")

    # Verify output
    import glob
    result_dirs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "object_*")))
    print(f"\n Total objek ter-prepare: {len(result_dirs)}")
    if result_dirs:
        print(f"   Contoh folder: {result_dirs[0]}")
        print(f"   Isi: {os.listdir(result_dirs[0])}")


if __name__ == "__main__":
    main()
