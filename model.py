from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F



# CNN blocks
def conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


def upconv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )



# Encoder, ViewHead, Decoder (dari tahap 1 + feedback hook)
class Encoder(nn.Module):
    """128 -> 8 spatial, 3 -> base*8 channel. Returns features per stage untuk feedback injection."""

    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.s1 = nn.Sequential(conv_block(in_ch, base_ch), conv_block(base_ch, base_ch, stride=2))
        self.s2 = nn.Sequential(conv_block(base_ch, base_ch * 2), conv_block(base_ch * 2, base_ch * 2, stride=2))
        self.s3 = nn.Sequential(conv_block(base_ch * 2, base_ch * 4), conv_block(base_ch * 4, base_ch * 4, stride=2))
        self.s4 = nn.Sequential(conv_block(base_ch * 4, base_ch * 8), conv_block(base_ch * 8, base_ch * 8, stride=2))

    def forward(self, x, feedback_feats=None):
        """
        feedback_feats (opsional): tuple (f1, f2, f3, f4) — fitur injection tiap stage.
            f1: [B, base, 64, 64]
            f2: [B, base*2, 32, 32]
            f3: [B, base*4, 16, 16]
            f4: [B, base*8, 8, 8]
        """
        h1 = self.s1(x)
        if feedback_feats is not None: h1 = h1 + feedback_feats[0]
        h2 = self.s2(h1)
        if feedback_feats is not None: h2 = h2 + feedback_feats[1]
        h3 = self.s3(h2)
        if feedback_feats is not None: h3 = h3 + feedback_feats[2]
        h4 = self.s4(h3)
        if feedback_feats is not None: h4 = h4 + feedback_feats[3]
        return h4, (h1, h2, h3, h4)


class ViewHead(nn.Module):
    def __init__(self, num_target_views, latent_ch=256, spatial=8):
        super().__init__()
        self.num_views = num_target_views
        self.view_embed = nn.Parameter(
            torch.randn(num_target_views, latent_ch, spatial, spatial) * 0.02
        )
        self.mix = nn.Sequential(
            conv_block(latent_ch * 2, latent_ch),
            conv_block(latent_ch, latent_ch),
        )

    def forward(self, shared, pose_feat=None):
        """pose_feat (opsional): [B, V, C] embedding pose target per view -> ditambahkan
        ke view embedding supaya tiap slot view "tahu" sudut kamera targetnya."""
        B, C, H, W = shared.shape
        V = self.num_views
        shared_exp = shared.unsqueeze(1).expand(B, V, C, H, W)
        embed_exp = self.view_embed.unsqueeze(0).expand(B, V, C, H, W)
        if pose_feat is not None:
            embed_exp = embed_exp + pose_feat.reshape(B, V, C, 1, 1)
        cat = torch.cat([shared_exp, embed_exp], dim=2).reshape(B * V, 2 * C, H, W)
        out = self.mix(cat)
        return out.reshape(B, V, C, H, W)


class Decoder(nn.Module):
    def __init__(self, out_ch=3, base_ch=32):
        super().__init__()
        self.up1 = upconv_block(base_ch * 8, base_ch * 4)
        self.up2 = upconv_block(base_ch * 4, base_ch * 2)
        self.up3 = upconv_block(base_ch * 2, base_ch)
        self.up4 = upconv_block(base_ch, base_ch)
        self.out = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = self.up1(x); x = self.up2(x); x = self.up3(x); x = self.up4(x)
        return torch.tanh(self.out(x))



# Feedback encoder
class FeedbackEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.stem = conv_block(in_ch, base_ch)
        # 128 -> 64
        self.down1 = conv_block(base_ch, base_ch, stride=2)
        # 64 -> 32
        self.down2 = conv_block(base_ch, base_ch * 2, stride=2)
        # 32 -> 16
        self.down3 = conv_block(base_ch * 2, base_ch * 4, stride=2)
        # 16 -> 8
        self.down4 = conv_block(base_ch * 4, base_ch * 8, stride=2)

    def forward(self, x):
        """
        x: [B, 3, 128, 128]
        returns 4 feature maps matching encoder stages:
            f1 [B, base,   64, 64]
            f2 [B, base*2, 32, 32]
            f3 [B, base*4, 16, 16]
            f4 [B, base*8, 8, 8]
        """
        x = self.stem(x)
        f1 = self.down1(x)        # 64
        f2 = self.down2(f1)       # 32
        f3 = self.down3(f2)       # 16
        f4 = self.down4(f3)       # 8
        return (f1, f2, f3, f4)



# MultiViewGenerator F_θ (bungkus encoder+view head+decoder)
class MultiViewGenerator(nn.Module):
    """
    cond_image [+ optional feedback] -> prediksi V target view.
    """

    def __init__(self, num_target_views=7, base_ch=32, use_feedback=True,
                 use_rgb_feedback=None, use_ccm_feedback=None, img_size=128,
                 use_pose_cond=True):
        super().__init__()
        # use_rgb_feedback/use_ccm_feedback: kontrol terpisah utk ablasi (Tab. 2 paper:
        # Joint Training x CCM Feedback x RGB Feedback). Default None -> ikut use_feedback
        # (perilaku lama: keduanya nyala/mati bersamaan).
        if use_rgb_feedback is None:
            use_rgb_feedback = use_feedback
        if use_ccm_feedback is None:
            use_ccm_feedback = use_feedback
        self.num_target_views = num_target_views
        self.use_rgb_feedback = use_rgb_feedback
        self.use_ccm_feedback = use_ccm_feedback
        self.use_feedback = use_rgb_feedback or use_ccm_feedback
        self.use_pose_cond = use_pose_cond
        latent_ch = base_ch * 8
        # Encoder downsample 4x (16x spatial reduction: stride=2 x 4 stages)
        spatial = max(1, img_size // 16)
        self.encoder = Encoder(in_ch=3, base_ch=base_ch)
        self.view_head = ViewHead(num_target_views, latent_ch=latent_ch, spatial=spatial)
        self.decoder = Decoder(out_ch=3, base_ch=base_ch)

        if use_rgb_feedback:
            self.rgb_fb_encoder = FeedbackEncoder(in_ch=3, base_ch=base_ch)
        if use_ccm_feedback:
            self.ccm_fb_encoder = FeedbackEncoder(in_ch=3, base_ch=base_ch)

        if use_pose_cond:
            # Encode pose target RELATIF terhadap conditioning view (9 rot + 3 trans = 12)
            # -> embedding per view. Bikin generator pose-aware (tidak averaging jadi blob).
            self.pose_mlp = nn.Sequential(
                nn.Linear(12, latent_ch), nn.SiLU(),
                nn.Linear(latent_ch, latent_ch),
            )

    def _pose_feat(self, poses):
        """poses [B, N, 4, 4] (view 0 = conditioning) -> pose_feat [B, N-1, latent_ch]
        memakai pose target relatif ke kamera conditioning (invarian orientasi global)."""
        B, N = poses.shape[:2]
        cond_inv = torch.inverse(poses[:, 0:1])              # [B, 1, 4, 4]
        rel = torch.matmul(cond_inv.expand(B, N - 1, 4, 4), poses[:, 1:])  # [B, V, 4, 4]
        R = rel[..., :3, :3].reshape(B, N - 1, 9)
        t = rel[..., :3, 3]                                  # [B, V, 3]
        return self.pose_mlp(torch.cat([R, t], dim=-1))      # [B, V, latent_ch]

    def forward(self, cond_image, poses=None, rgb_feedback=None, ccm_feedback=None):
        """
        cond_image:   [B, 3, H, W]
        poses:        [B, N, 4, 4] semua pose kamera (view 0 = conditioning) -> pose-cond
        rgb_feedback: [B, 3, H, W] atau None (rendered rgb dari reconstruction step sebelumnya)
        ccm_feedback: [B, 3, H, W] atau None
        """
        # Tiap modalitas feedback (RGB, CCM) bisa dipakai sendiri-sendiri atau gabungan
        # (dijumlahkan per stage) -> dipakai utk ablasi terpisah seperti Tab. 2 paper.
        feedback_parts = []
        if self.use_rgb_feedback and (rgb_feedback is not None):
            feedback_parts.append(self.rgb_fb_encoder(rgb_feedback))
        if self.use_ccm_feedback and (ccm_feedback is not None):
            feedback_parts.append(self.ccm_fb_encoder(ccm_feedback))
        feedback_feats = None
        if feedback_parts:
            feedback_feats = feedback_parts[0]
            for extra in feedback_parts[1:]:
                feedback_feats = tuple(a + b for a, b in zip(feedback_feats, extra))

        pose_feat = None
        if self.use_pose_cond and (poses is not None):
            pose_feat = self._pose_feat(poses)

        shared, _ = self.encoder(cond_image, feedback_feats=feedback_feats)
        view_latents = self.view_head(shared, pose_feat=pose_feat)   # [B, V, C, 8, 8]
        B, V, C, H, W = view_latents.shape
        flat = view_latents.reshape(B * V, C, H, W)
        out = self.decoder(flat)                        # [B*V, 3, 128, 128]
        return out.reshape(B, V, *out.shape[1:])        # [B, V, 3, 128, 128]



# 5. Reconstruction Model G: multi-view -> 3D Gaussians

class GaussianPredictor(nn.Module):
    """
    Predict K Gaussian primitives dari semua view.

    Setiap Gaussian:
        - pos [3]     dalam canonical frame, range ~ [-1, 1]
        - rgb [3]     range [0, 1]
        - opacity [1] range [0, 1]
        - scale [1]   skalar radius (log-scale)

    Arsitektur: encode semua view -> pool ke global feature -> MLP -> K Gaussians
    """

    def __init__(self, num_gaussians=512, in_views=8, base_ch=32):
        super().__init__()
        self.K = num_gaussians
        self.in_views = in_views

        # Per-view encoder (shared weight)
        self.per_view_enc = nn.Sequential(
            conv_block(3, base_ch),
            conv_block(base_ch, base_ch, stride=2),
            conv_block(base_ch, base_ch * 2, stride=2),
            conv_block(base_ch * 2, base_ch * 4, stride=2),
            conv_block(base_ch * 4, base_ch * 8, stride=2),
        )
        # Adaptive pool ke 4x4 agar tidak tergantung img_size input
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        feat_dim = base_ch * 8 * 4 * 4

        self.merge = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
        )
        # Output: K * 8 parameter (3 pos + 3 rgb + 1 opacity + 1 scale)
        self.heads = nn.Linear(512, self.K * 8)

    def forward(self, multi_view):
        """
        multi_view: [B, N, 3, H, W]
        returns dict of Gaussians (B, K, ...)
        """
        B, N, C, H, W = multi_view.shape
        x = multi_view.reshape(B * N, C, H, W)
        feat = self.per_view_enc(x)                            # [B*N, base*8, H', W']
        feat = self.pool(feat)                                 # [B*N, base*8, 4, 4]
        feat = feat.reshape(B, N, -1)                          # [B, N, feat]
        feat = feat.mean(dim=1)                                # [B, feat]  pool over views
        feat = self.merge(feat)
        raw = self.heads(feat).reshape(B, self.K, 8)

        pos = torch.tanh(raw[..., 0:3])                        # [-1, 1]
        rgb = torch.sigmoid(raw[..., 3:6])                     # [0, 1]
        opa = torch.sigmoid(raw[..., 6:7])                     # [0, 1]
        # Scale di log-space, clamp agar tidak meledak
        scale = torch.sigmoid(raw[..., 7:8]) * 0.15 + 0.01     # [0.01, 0.16]

        return {"pos": pos, "rgb": rgb, "opacity": opa, "scale": scale}



# Differentiable renderer untuk Gaussian Splatting
class GaussianRenderer(nn.Module):
    """
    Implementasi sederhana (bukan CUDA asli gsplat) dari Gaussian splatting
    differentiable:
      - Transform Gaussian positions ke camera frame via pose inverse
      - Proyeksi ortografik ke image plane
      - Alpha-compositing berdasarkan depth

    Menghasilkan:
      - rgb  [B, 3, H, W] di range [-1, 1] (match normalisasi data)
      - ccm  [B, 3, H, W] canonical xyz (= posisi Gaussian di canonical frame)
    """

    def __init__(self, img_size=128):
        super().__init__()
        self.H = img_size
        self.W = img_size

    def forward(self, gaussians, poses):
        """
        gaussians: dict with pos [B,K,3], rgb [B,K,3], opacity [B,K,1], scale [B,K,1]
        poses:     [B, N_render, 4, 4] camera-to-world matrices

        returns:
          rgb_out [B, N_render, 3, H, W]  range [-1, 1]
          ccm_out [B, N_render, 3, H, W]  range [-1, 1]
        """
        B, K, _ = gaussians["pos"].shape
        N = poses.shape[1]
        H, W = self.H, self.W
        device = gaussians["pos"].device

        pos = gaussians["pos"]            # [B, K, 3]
        rgb = gaussians["rgb"]            # [B, K, 3]
        opa = gaussians["opacity"]        # [B, K, 1]
        scale = gaussians["scale"]        # [B, K, 1]

        # Expand pos ke [B, N, K, 3]
        pos_exp = pos.unsqueeze(1).expand(B, N, K, 3)
        ones = torch.ones(B, N, K, 1, device=device)
        pos_h = torch.cat([pos_exp, ones], dim=-1)   # [B, N, K, 4]

        # World -> camera: inverse of camera-to-world
        poses_inv = torch.inverse(poses)              # [B, N, 4, 4]
        # pos_cam = poses_inv @ pos_h (batch matmul)
        pos_cam = torch.einsum("bnij,bnkj->bnki", poses_inv, pos_h)[..., :3]  # [B,N,K,3]

        # Proyeksi ortografik: (x, y) langsung koordinat image, z = depth
        x_img = pos_cam[..., 0]   # [B, N, K]
        y_img = pos_cam[..., 1]   # [B, N, K]
        z_depth = -pos_cam[..., 2]  # kedalaman (>= 0 jika di depan kamera)

        # Convert (x_img, y_img) di range [-1, 1] -> pixel [0, W-1]
        px = (x_img * 0.5 + 0.5) * (W - 1)    # [B, N, K]
        py = (1.0 - (y_img * 0.5 + 0.5)) * (H - 1)

        # Grid pixel coords
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )  # [H, W]

        # Menghitung pengaruh tiap Gaussian di tiap pixel dengan Gaussian kernel
        sigma_bk = scale.squeeze(-1)                              # [B, K]
        sigma_px = sigma_bk.unsqueeze(1).expand(B, N, K) * (W / 2.0)   # [B, N, K]
        sigma_px = sigma_px.clamp(min=0.5, max=W / 4.0)

        rgb_out = torch.zeros(B, N, 3, H, W, device=device)
        ccm_out = torch.zeros(B, N, 3, H, W, device=device)
        alpha_out = torch.zeros(B, N, 1, H, W, device=device)

        CHUNK = 32  # proses 32 Gaussians sekaligus

        for b in range(B):
            for n in range(N):
                # Akumulator per view
                sum_a = torch.zeros(H, W, device=device)
                sum_a_rgb = torch.zeros(3, H, W, device=device)
                sum_a_pos = torch.zeros(3, H, W, device=device)
                raw_sum = torch.zeros(H, W, device=device)  # untuk alpha total

                for k_start in range(0, K, CHUNK):
                    k_end = min(K, k_start + CHUNK)
                    k_slice = slice(k_start, k_end)
                    Kc = k_end - k_start

                    px_ck = px[b, n, k_slice].reshape(Kc, 1, 1)
                    py_ck = py[b, n, k_slice].reshape(Kc, 1, 1)
                    sig_ck = sigma_px[b, n, k_slice].reshape(Kc, 1, 1)
                    opa_ck = opa[b, k_slice].reshape(Kc, 1, 1)
                    rgb_ck = rgb[b, k_slice].reshape(Kc, 3, 1, 1)
                    pos_ck = pos[b, k_slice].reshape(Kc, 3, 1, 1)
                    depth_ck = (z_depth[b, n, k_slice] > 0).float().reshape(Kc, 1, 1)

                    dx = xx.unsqueeze(0) - px_ck
                    dy = yy.unsqueeze(0) - py_ck
                    g = torch.exp(-0.5 * (dx * dx + dy * dy) / (sig_ck * sig_ck + 1e-6))
                    a = opa_ck * g * depth_ck   # [Kc, H, W]

                    sum_a = sum_a + a.sum(dim=0)
                    sum_a_rgb = sum_a_rgb + (a.unsqueeze(1) * rgb_ck).sum(dim=0)
                    sum_a_pos = sum_a_pos + (a.unsqueeze(1) * pos_ck).sum(dim=0)
                    raw_sum = raw_sum + a.sum(dim=0)

                # Normalize weights -> expected color & canonical pos
                denom = sum_a.clamp(min=1e-4)
                rgb_pix = sum_a_rgb / denom.unsqueeze(0)   # [3, H, W]
                ccm_pix = sum_a_pos / denom.unsqueeze(0)

                alpha = 1.0 - torch.exp(-raw_sum)          # [H, W]

                rgb_final = alpha.unsqueeze(0) * rgb_pix + (1 - alpha).unsqueeze(0) * 1.0
                ccm_final = alpha.unsqueeze(0) * ccm_pix

                rgb_out[b, n] = rgb_final
                ccm_out[b, n] = ccm_final
                alpha_out[b, n, 0] = alpha

        # Scale RGB dari [0, 1] ke [-1, 1] agar konsisten dengan data
        rgb_out = rgb_out * 2.0 - 1.0

        return rgb_out, ccm_out, alpha_out



# Full Ouroboros3D pipeline

class Ouroboros3D(nn.Module):
    """
    Pipeline lengkap dengan recursive 3D-aware feedback.
    """

    def __init__(
        self,
        num_views=8,                    # total view termasuk conditioning
        base_ch=32,
        num_gaussians=512,
        img_size=128,
        use_feedback=True,
        use_rgb_feedback=None,
        use_ccm_feedback=None,
        use_pose_cond=True,
    ):
        super().__init__()
        if use_rgb_feedback is None:
            use_rgb_feedback = use_feedback
        if use_ccm_feedback is None:
            use_ccm_feedback = use_feedback
        self.num_views = num_views
        self.num_target_views = num_views - 1
        self.img_size = img_size
        self.use_rgb_feedback = use_rgb_feedback
        self.use_ccm_feedback = use_ccm_feedback
        self.use_feedback = use_rgb_feedback or use_ccm_feedback
        self.use_pose_cond = use_pose_cond

        self.mv_generator = MultiViewGenerator(
            num_target_views=self.num_target_views,
            base_ch=base_ch,
            use_rgb_feedback=use_rgb_feedback,
            use_ccm_feedback=use_ccm_feedback,
            img_size=img_size,
            use_pose_cond=use_pose_cond,
        )
        self.reconstructor = GaussianPredictor(
            num_gaussians=num_gaussians,
            in_views=num_views,
            base_ch=base_ch,
        )
        self.renderer = GaussianRenderer(img_size=img_size)

    def forward(self, cond_image, poses, num_recursive_steps=1, prev_feedback=None):
        """
        cond_image:  [B, 3, H, W]
        poses:       [B, N, 4, 4]  all camera poses (view 0 = conditioning)
        num_recursive_steps: berapa kali loop multi-view→recon→render→feedback

        Returns list[dict] — output tiap step rekursif.
        """
        B = cond_image.shape[0]
        outputs = []

        rgb_feedback = prev_feedback["rgb"] if prev_feedback else None
        ccm_feedback = prev_feedback["ccm"] if prev_feedback else None

        for step in range(num_recursive_steps):
            # 1) Generate multi-view
            pred_target = self.mv_generator(
                cond_image,
                poses=poses,
                rgb_feedback=rgb_feedback,
                ccm_feedback=ccm_feedback,
            )   # [B, V=N-1, 3, H, W]

            # 2) Concat conditioning image dan predicted targets -> full multi-view
            all_views = torch.cat([cond_image.unsqueeze(1), pred_target], dim=1)  # [B, N, 3, H, W]

            # 3) Reconstruct 3D Gaussians
            gaussians = self.reconstructor(all_views)

            # 4) Render Gaussians dari semua poses -> rendered rgb + ccm
            rendered_rgb, rendered_ccm, alpha = self.renderer(gaussians, poses)
            # rendered_rgb/ccm: [B, N, 3, H, W]

            outputs.append({
                "pred_target": pred_target,      # [B, V, 3, H, W]
                "gaussians": gaussians,
                "rendered_rgb": rendered_rgb,    # [B, N, 3, H, W]
                "rendered_ccm": rendered_ccm,    # [B, N, 3, H, W]
                "alpha": alpha,
            })

            # 5) Siapkan feedback untuk step berikutnya:
            rgb_feedback = rendered_rgb[:, 0]    # [B, 3, H, W]
            ccm_feedback = rendered_ccm[:, 0]    # [B, 3, H, W]

        return outputs

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# 8. Sanity checks
if __name__ == "__main__":
    print("=" * 60)
    print("MODEL V2 SANITY CHECK")
    print("=" * 60)

    model = Ouroboros3D(num_views=8, base_ch=32, num_gaussians=256, img_size=128)
    n = model.count_parameters()
    print(f"Total parameter: {n:,} ({n/1e6:.2f}M)")
    breakdown = {
        "mv_generator": sum(p.numel() for p in model.mv_generator.parameters()),
        "reconstructor": sum(p.numel() for p in model.reconstructor.parameters()),
        "renderer": sum(p.numel() for p in model.renderer.parameters()),
    }
    for k, v in breakdown.items():
        print(f"  {k:15s}: {v:,} ({v/1e6:.2f}M)")

    # Dummy data
    B, N = 1, 8
    cond = torch.randn(B, 3, 128, 128)
    poses = torch.eye(4).reshape(1, 1, 4, 4).repeat(B, N, 1, 1)
    # Beri sedikit variasi pada translasi z supaya bukan singular
    for n in range(N):
        az = n * (360 / N)
        a = torch.deg2rad(torch.tensor(az))
        poses[:, n, 0, 3] = 2.0 * torch.sin(a)
        poses[:, n, 2, 3] = 2.0 * torch.cos(a)

    print("\n--- Forward pass (1 recursive step) ---")
    with torch.no_grad():
        outs = model(cond, poses, num_recursive_steps=1)
    o = outs[0]
    print(f"  pred_target  : {tuple(o['pred_target'].shape)}  range=[{o['pred_target'].min():.3f}, {o['pred_target'].max():.3f}]")
    print(f"  rendered_rgb : {tuple(o['rendered_rgb'].shape)} range=[{o['rendered_rgb'].min():.3f}, {o['rendered_rgb'].max():.3f}]")
    print(f"  rendered_ccm : {tuple(o['rendered_ccm'].shape)} range=[{o['rendered_ccm'].min():.3f}, {o['rendered_ccm'].max():.3f}]")
    print(f"  gaussians.pos: {tuple(o['gaussians']['pos'].shape)}")

    print("\n--- Forward pass (2 recursive steps) ---")
    with torch.no_grad():
        outs = model(cond, poses, num_recursive_steps=2)
    for i, o in enumerate(outs):
        print(f"  step {i}: pred_target range=[{o['pred_target'].min():.3f}, {o['pred_target'].max():.3f}]"
              f"  rendered_rgb mean={o['rendered_rgb'].mean():.3f}")

    print("\nModel v2 OK")
