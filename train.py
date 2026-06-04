import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from dataset import build_dataloaders
from model import Ouroboros3D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/synthetic")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=64,
                   help="Resolusi image. 64 untuk demo cepat, 128 untuk kualitas lebih baik.")
    p.add_argument("--num-views", type=int, default=8)
    p.add_argument("--num-gaussians", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Ablation toggles
    p.add_argument("--use-feedback", action="store_true",
                   help="Aktifkan 3D-aware feedback (full Ouroboros3D).")
    p.add_argument("--joint", action="store_true",
                   help="Joint train G + F_θ. Default: only train F_θ.")
    p.add_argument("--self-cond-prob", type=float, default=0.5,
                   help="Prob. self-conditioning saat training (paper: 0.5).")
    # Loss weights
    p.add_argument("--w-mv", type=float, default=1.0, help="Multi-view gen loss weight")
    p.add_argument("--w-render", type=float, default=0.5, help="Rendered RGB loss weight")
    p.add_argument("--w-ccm", type=float, default=0.3, help="CCM geometry loss weight")
    p.add_argument("--w-mask", type=float, default=0.0,
                   help="Silhouette/mask loss weight (alpha vs mask). Pakai untuk data real tanpa CCM.")
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--tag", type=str, default="run",
                   help="Tag untuk checkpoint (misal: 'full', 'no_feedback')")
    return p.parse_args()


def compute_losses(outputs, batch, args):
    """
    outputs: list dict hasil model (tiap elemen = 1 recursive step)
    batch:   dict dari dataloader
    """
    target_rgb = batch["target_rgb"]   # [B, N-1, 3, H, W]
    all_rgb = batch["all_rgb"]         # [B, N, 3, H, W]
    all_ccm = batch["all_ccm"]         # [B, N, 3, H, W]
    all_mask = batch["all_mask"]       # [B, N, 1, H, W]

    # supervise step terakhir (biar bisa multi-step di inference)
    o = outputs[-1]
    pred_target = o["pred_target"]     # [B, N-1, 3, H, W]
    rendered_rgb = o["rendered_rgb"]   # [B, N, 3, H, W]
    rendered_ccm = o["rendered_ccm"]   # [B, N, 3, H, W]
    alpha = o["alpha"]                 # [B, N, 1, H, W]

    # Multi-view generator loss (L1 + 0.5 * L2)
    loss_mv = F.l1_loss(pred_target, target_rgb) + 0.5 * F.mse_loss(pred_target, target_rgb)

    losses = {"mv": loss_mv.item()}
    total = args.w_mv * loss_mv

    # Rendered RGB loss (hanya kalau joint training dan w_render > 0)
    if args.joint and args.w_render > 0:
        loss_render = F.l1_loss(rendered_rgb, all_rgb)
        losses["render"] = loss_render.item()
        total = total + args.w_render * loss_render
    else:
        losses["render"] = 0.0

    # CCM geometry loss (hanya di daerah foreground -- mask)
    if args.joint and args.w_ccm > 0:
        mask_3ch = all_mask.expand_as(all_ccm)
        # Clamp ground truth CCM ke [-1, 1] karena range dataset bisa sedikit lebih (interpolasi)
        gt_ccm_clamped = all_ccm.clamp(-1.0, 1.0)
        loss_ccm = (F.l1_loss(rendered_ccm * mask_3ch,
                              gt_ccm_clamped * mask_3ch, reduction="sum")
                    / (mask_3ch.sum().clamp(min=1.0)))
        losses["ccm"] = loss_ccm.item()
        total = total + args.w_ccm * loss_ccm
    else:
        losses["ccm"] = 0.0

    # Silhouette loss: rendered alpha vs ground-truth mask (geometri untuk data real)
    if args.joint and args.w_mask > 0:
        loss_mask = F.binary_cross_entropy(alpha.clamp(1e-6, 1 - 1e-6), all_mask)
        losses["mask"] = loss_mask.item()
        total = total + args.w_mask * loss_mask
    else:
        losses["mask"] = 0.0

    losses["total"] = total.item()
    return total, losses


def train_one_epoch(model, loader, optimizer, device, args, epoch):
    model.train()
    running = {"mv": 0.0, "render": 0.0, "ccm": 0.0, "mask": 0.0, "total": 0.0}
    n = 0
    pbar = tqdm(loader, desc=f"[Ep {epoch}] train", leave=False)
    for batch in pbar:
        # Move batch to device
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        cond_image = batch["cond_image"]
        poses = batch["poses"]

        optimizer.zero_grad(set_to_none=True)

        # Self-conditioning with prob p
        prev_feedback = None
        if args.use_feedback and (torch.rand(1).item() < args.self_cond_prob):
            with torch.no_grad():
                dry = model(cond_image, poses, num_recursive_steps=1)
                prev_feedback = {
                    "rgb": dry[0]["rendered_rgb"][:, 0].detach(),
                    "ccm": dry[0]["rendered_ccm"][:, 0].detach(),
                }

        outputs = model(cond_image, poses, num_recursive_steps=1,
                        prev_feedback=prev_feedback)
        loss, parts = compute_losses(outputs, batch, args)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        for k, v in parts.items():
            running[k] += v
        n += 1
        pbar.set_postfix(
            mv=f"{parts['mv']:.3f}",
            rnd=f"{parts['render']:.3f}",
            ccm=f"{parts['ccm']:.3f}",
            msk=f"{parts['mask']:.3f}",
            tot=f"{parts['total']:.3f}",
        )
    return {k: v / max(1, n) for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, device, args, epoch):
    model.eval()
    running = {"mv": 0.0, "render": 0.0, "ccm": 0.0, "mask": 0.0, "total": 0.0}
    n = 0
    pbar = tqdm(loader, desc=f"[Ep {epoch}] val  ", leave=False)
    for batch in pbar:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        outputs = model(batch["cond_image"], batch["poses"], num_recursive_steps=1)
        _, parts = compute_losses(outputs, batch, args)
        for k, v in parts.items():
            running[k] += v
        n += 1
    return {k: v / max(1, n) for k, v in running.items()}


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: use_feedback={args.use_feedback}, joint={args.joint}, tag={args.tag}")

    print("\n[1/4] Dataloaders...")
    train_loader, val_loader = build_dataloaders(
        root_dir=args.data_root,
        img_size=args.img_size,
        num_views=args.num_views,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"  train={len(train_loader)} batch, val={len(val_loader)} batch")

    print("\n[2/4] Building Ouroboros3D model...")
    model = Ouroboros3D(
        num_views=args.num_views,
        base_ch=32,
        num_gaussians=args.num_gaussians,
        img_size=args.img_size,
        use_feedback=args.use_feedback,
    ).to(device)
    n = model.count_parameters()
    print(f"  params: {n:,} ({n/1e6:.2f}M)")
    print(f"    - mv_generator : {sum(p.numel() for p in model.mv_generator.parameters())/1e6:.2f}M")
    print(f"    - reconstructor: {sum(p.numel() for p in model.reconstructor.parameters())/1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n[3/4] Training for {args.epochs} epoch(s)...")
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        va = validate(model, val_loader, device, args, epoch)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} "
              f"| train: mv={tr['mv']:.3f} rnd={tr['render']:.3f} ccm={tr['ccm']:.3f} msk={tr['mask']:.3f} tot={tr['total']:.3f} "
              f"| val: mv={va['mv']:.3f} rnd={va['render']:.3f} ccm={va['ccm']:.3f} msk={va['mask']:.3f} tot={va['total']:.3f} "
              f"| {elapsed:.1f}s")
        history.append({"epoch": epoch, "train": tr, "val": va})

    print("\n[4/4] Saving...")
    ckpt_path = os.path.join(args.save_dir, f"ouroboros_{args.tag}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
        "history": history,
    }, ckpt_path)
    print(f"  Saved: {ckpt_path}")

    print("\n" + "=" * 60)
    print("TRAINING SELESAI")
    print("=" * 60)
    first = history[0]["train"]["total"]
    last = history[-1]["train"]["total"]
    print(f"Train total loss: {first:.4f} -> {last:.4f} (-{(first - last)/first*100:.1f}%)")
    print(f"Waktu total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
