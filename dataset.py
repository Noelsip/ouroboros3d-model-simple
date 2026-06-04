import os
import glob
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class MultiViewDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        img_size: int = 128,
        num_views: int = 8,
        augment: bool = True,
    ):
        self.root_dir = root_dir
        self.img_size = img_size
        self.num_views = num_views
        self.augment = augment

        self.object_dirs: List[str] = sorted(
            [d for d in glob.glob(os.path.join(root_dir, "object_*"))
             if os.path.isdir(os.path.join(d, "views"))]
        )
        if len(self.object_dirs) == 0:
            raise RuntimeError(
                f"No objects in '{root_dir}'. Run generate_synthetic_dataset.py first."
            )

        # resizw
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),  # -> [-1, 1]
        ])
        self.cond_augment = transforms.ColorJitter(0.1, 0.1, 0.1)

    def __len__(self):
        return len(self.object_dirs)

    def _load_rgb(self, path, apply_flip, apply_jitter):
        img = Image.open(path).convert("RGB")
        if apply_jitter:
            img = self.cond_augment(img)
        if apply_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return self.img_transform(img)

    def _load_ccm(self, path, apply_flip):
        """Load CCM [H, W, 3] -> tensor [3, H, W] di range [-1, 1], resized to img_size."""
        ccm = np.load(path).astype(np.float32)  # [H, W, 3]
        if apply_flip:
            ccm = np.ascontiguousarray(ccm[:, ::-1, :])
            # X component flip sign (mirror di sumbu X canonical)
            ccm[..., 0] *= -1.0
        ccm_t = torch.from_numpy(ccm).permute(2, 0, 1)  # [3, H, W]
        # Resize ke img_size kalau berbeda
        if ccm_t.shape[-1] != self.img_size:
            ccm_t = torch.nn.functional.interpolate(
                ccm_t.unsqueeze(0), size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return ccm_t

    def _load_mask(self, path, apply_flip):
        """Load mask [H, W] -> tensor [1, H, W] di {0,1}, resized ke img_size."""
        m = np.load(path).astype(np.float32)
        if m.ndim == 3:
            m = m[..., 0]
        if apply_flip:
            m = np.ascontiguousarray(m[:, ::-1])
        m_t = torch.from_numpy(m).unsqueeze(0)  # [1, H, W]
        if m_t.shape[-1] != self.img_size:
            m_t = torch.nn.functional.interpolate(
                m_t.unsqueeze(0), size=(self.img_size, self.img_size),
                mode="nearest",
            ).squeeze(0)
        return m_t

    def __getitem__(self, idx):
        obj_dir = self.object_dirs[idx]
        view_paths = sorted(glob.glob(os.path.join(obj_dir, "views", "view_*.png")))[:self.num_views]
        ccm_paths = sorted(glob.glob(os.path.join(obj_dir, "ccm", "view_*.npy")))[:self.num_views]
        mask_paths = sorted(glob.glob(os.path.join(obj_dir, "mask", "view_*.npy")))[:self.num_views]
        has_mask = len(mask_paths) >= self.num_views

        apply_flip = self.augment and (torch.rand(1).item() < 0.5)

        rgbs, ccms, masks = [], [], []
        for i in range(self.num_views):
            jitter = self.augment and (i == 0) and (torch.rand(1).item() < 0.5)
            rgbs.append(self._load_rgb(view_paths[i], apply_flip, jitter))
            ccms.append(self._load_ccm(ccm_paths[i], apply_flip))
            if has_mask:
                masks.append(self._load_mask(mask_paths[i], apply_flip))

        rgb_t = torch.stack(rgbs, dim=0)   # [N, 3, H, W]
        ccm_t = torch.stack(ccms, dim=0)   # [N, 3, H, W]

        if has_mask:
            # Mask siluet eksplisit (data real tanpa CCM)
            mask = torch.stack(masks, dim=0)                             # [N, 1, H, W]
        else:
            # Foreground mask dari ccm: pixel dengan norm > 0 = foreground
            mask = (ccm_t.abs().sum(dim=1, keepdim=True) > 1e-5).float()  # [N, 1, H, W]

        # Camera poses
        poses = np.load(os.path.join(obj_dir, "camera_poses.npy"))[:self.num_views]
        poses_t = torch.from_numpy(poses).float()  # [N, 4, 4]

        return {
            "cond_image": rgb_t[0],         # [3, H, W]
            "target_rgb": rgb_t[1:],        # [N-1, 3, H, W]
            "target_ccm": ccm_t[1:],        # [N-1, 3, H, W]
            "target_mask": mask[1:],        # [N-1, 1, H, W]
            "all_rgb": rgb_t,               # [N, 3, H, W]
            "all_ccm": ccm_t,               # [N, 3, H, W]
            "all_mask": mask,               # [N, 1, H, W]
            "poses": poses_t,               # [N, 4, 4]
        }


def build_dataloaders(
    root_dir: str = "data/synthetic",
    img_size: int = 128,
    num_views: int = 8,
    batch_size: int = 4,
    val_split: float = 0.15,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    full_train = MultiViewDataset(root_dir, img_size, num_views, augment=True)
    full_val = MultiViewDataset(root_dir, img_size, num_views, augment=False)

    n_total = len(full_train)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )

    train_ds = torch.utils.data.Subset(full_train, list(train_indices))
    val_ds = torch.utils.data.Subset(full_val, list(val_indices))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    print("=" * 60)
    print("SANITY CHECK: MultiViewDataset v2 (dengan CCM + poses)")
    print("=" * 60)

    train_loader, val_loader = build_dataloaders(
        root_dir="data/synthetic", batch_size=4, num_workers=0
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")

    batch = next(iter(train_loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            rng = f"[{v.min():.3f}, {v.max():.3f}]"
            print(f"  {k:15s}: {tuple(v.shape)}  range={rng}")

    print("\nData loader v2 OK")
