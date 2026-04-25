import os
import numpy as np
from PIL import Image
from tqdm import tqdm

NUM_OBJECTS = 60
NUM_VIEWS = 8
IMG_SIZE = 128
OUTPUT_DIR = "data/synthetic"
SEED = 42

np.random.seed(SEED)


def make_sphere_volume(radius: float, size: int = 64) -> np.ndarray:
    grid = np.zeros((size, size, size), dtype=np.float32)
    c = size // 2
    xs, ys, zs = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing='ij')
    d = np.sqrt((xs - c) ** 2 + (ys - c) ** 2 + (zs - c) ** 2)
    grid[d <= radius * size / 2] = 1.0
    return grid


def make_cube_volume(side: float, size: int = 64) -> np.ndarray:
    grid = np.zeros((size, size, size), dtype=np.float32)
    half = int(side * size / 2)
    c = size // 2
    grid[c - half:c + half, c - half:c + half, c - half:c + half] = 1.0
    return grid


def make_cylinder_volume(radius: float, height: float, size: int = 64) -> np.ndarray:
    grid = np.zeros((size, size, size), dtype=np.float32)
    c = size // 2
    half_h = int(height * size / 2)
    xs, ys = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    d = np.sqrt((xs - c) ** 2 + (ys - c) ** 2)
    mask2d = d <= radius * size / 2
    for z in range(c - half_h, c + half_h):
        grid[:, :, z][mask2d] = 1.0
    return grid


def rotate_volume_y(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    size = volume.shape[0]
    c = size // 2
    xs, ys, zs = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing='ij')
    xs_c = xs - c
    zs_c = zs - c
    src_x = cos_a * xs_c + sin_a * zs_c + c
    src_z = -sin_a * xs_c + cos_a * zs_c + c
    src_x = np.clip(np.round(src_x).astype(int), 0, size - 1)
    src_y = np.clip(ys, 0, size - 1)
    src_z = np.clip(np.round(src_z).astype(int), 0, size - 1)
    return volume[src_x, src_y, src_z]


def render_with_ccm(volume, color, angle_deg, img_size=128):
    """Render RGB + depth + CCM (canonical XYZ) + mask."""
    rot_vol = rotate_volume_y(volume, angle_deg)
    size = rot_vol.shape[0]

    depth_map = np.zeros((size, size), dtype=np.float32)
    ccm_map = np.zeros((size, size, 3), dtype=np.float32)
    mask = np.zeros((size, size), dtype=bool)

    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    c = size // 2

    for z in range(size - 1, -1, -1):
        layer_mask = (rot_vol[:, :, z] > 0.5) & ~mask
        if not layer_mask.any():
            continue
        depth_map[layer_mask] = z / size
        ys_idx, xs_idx = np.where(layer_mask)
        x_cam = xs_idx - c
        z_cam = z - c
        # Inverse rotation -> canonical frame
        x_can = cos_a * x_cam - sin_a * z_cam
        z_can = sin_a * x_cam + cos_a * z_cam
        y_can = ys_idx - c
        ccm_map[ys_idx, xs_idx, 0] = x_can / (size / 2)
        ccm_map[ys_idx, xs_idx, 1] = y_can / (size / 2)
        ccm_map[ys_idx, xs_idx, 2] = z_can / (size / 2)
        mask |= layer_mask

    # Layout image-friendly
    depth_map = np.flipud(depth_map.T)
    ccm_map = np.flipud(ccm_map.transpose(1, 0, 2))
    mask = np.flipud(mask.T)

    shading = 0.4 + 0.6 * depth_map
    rgb = np.ones((size, size, 3), dtype=np.float32)
    for c_idx in range(3):
        rgb[..., c_idx] = np.where(mask, color[c_idx] * shading, 1.0)
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

    # Resize
    rgb_img = np.array(Image.fromarray(rgb).resize((img_size, img_size), Image.BILINEAR))
    depth_img = np.array(Image.fromarray(depth_map).resize((img_size, img_size), Image.BILINEAR))
    ccm_resized = np.zeros((img_size, img_size, 3), dtype=np.float32)
    for ch in range(3):
        ccm_resized[..., ch] = np.array(
            Image.fromarray(ccm_map[..., ch]).resize((img_size, img_size), Image.BILINEAR)
        )
    mask_img = np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((img_size, img_size), Image.NEAREST)
    ) > 127
    return rgb_img, depth_img, ccm_resized, mask_img


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


def generate_object(obj_id, output_dir):
    shape_type = np.random.choice(['sphere', 'cube', 'cylinder'])
    color = np.random.rand(3) * 0.8 + 0.2
    voxel_size = 48

    if shape_type == 'sphere':
        vol = make_sphere_volume(radius=np.random.uniform(0.5, 0.85), size=voxel_size)
    elif shape_type == 'cube':
        vol = make_cube_volume(side=np.random.uniform(0.5, 0.85), size=voxel_size)
    else:
        vol = make_cylinder_volume(
            radius=np.random.uniform(0.3, 0.6),
            height=np.random.uniform(0.5, 0.9),
            size=voxel_size,
        )

    obj_dir = os.path.join(output_dir, f"object_{obj_id:03d}")
    os.makedirs(os.path.join(obj_dir, "views"), exist_ok=True)
    os.makedirs(os.path.join(obj_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(obj_dir, "ccm"), exist_ok=True)

    azimuths = np.linspace(0, 360, NUM_VIEWS, endpoint=False)
    poses = []
    for view_idx, az in enumerate(azimuths):
        rgb, depth, ccm, mask = render_with_ccm(vol, color, az, img_size=IMG_SIZE)
        Image.fromarray(rgb).save(os.path.join(obj_dir, "views", f"view_{view_idx:02d}.png"))
        np.save(os.path.join(obj_dir, "depth", f"view_{view_idx:02d}.npy"), depth.astype(np.float32))
        np.save(os.path.join(obj_dir, "ccm", f"view_{view_idx:02d}.npy"), ccm.astype(np.float32))
        poses.append(build_camera_pose(az))

    np.save(os.path.join(obj_dir, "camera_poses.npy"), np.stack(poses, axis=0))


def main():
    print(f"Generating {NUM_OBJECTS} objects x {NUM_VIEWS} views @ {IMG_SIZE}x{IMG_SIZE}")
    print("Each view: RGB + depth + CCM + camera pose")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for obj_id in tqdm(range(NUM_OBJECTS), desc="Rendering"):
        generate_object(obj_id, OUTPUT_DIR)
    print(f"\nDataset (v2 dengan CCM + poses) tersimpan di: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
