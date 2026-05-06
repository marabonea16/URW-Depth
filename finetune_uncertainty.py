"""
Fine-tune uncertainty head (and dispconv) with proper Laplacian NLL loss.

Problem: sigma collapsed to 0 because training used sigma.detach() giving sigma
         no upward gradient. raw_uncert trained to ≈-35, sigma≈0 everywhere.

Fix: reinitialize uncertconv, jointly fine-tune uncertconv+dispconv with:
     L = depth_loss + nll_loss
  where:
     depth_loss = (1-sigma.detach()) * photo * mask  (unchanged)
     nll_loss = photo.detach() * exp(-raw.clamp(-5,5)) + raw  (Laplacian NLL)

This gives sigma proper gradients: sigma increases when photo > 1 (large error).
After fine-tuning, sigma is calibrated → TTA works.

Usage:
    python finetune_uncertainty.py \
        --load_weights_folder models/Tiny-Depth-Weather-Robust-Feature-Supression/models/weights_49 \
        --eval_mono --height 192 --width 640 --scales 0 \
        --data_path /home/ubuntu/TinyDepth --png \
        --use_feature_suppression \
        --finetune_epochs 5 --finetune_lr 1e-5 \
        --model_name Tiny-Depth-Weather-Robust-Feature-Supression
"""

from __future__ import absolute_import, division, print_function

import os
import copy
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from networks.configuration import get_config
from layer import (disp_to_depth, BackprojectDepth, Project3D,
                   SSIM, transformation_from_parameters)
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

SEVERITY = 0.45


def apply_fog(img, severity=SEVERITY):
    arr = np.array(img, dtype=np.float32)
    fog_color = np.array([220, 220, 220], dtype=np.float32)
    h = arr.shape[0]
    g = np.linspace(severity, severity * 0.3, h, dtype=np.float32)[:, None, None]
    arr = arr * (1 - g) + fog_color * g
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_rain(img, severity=SEVERITY):
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    rng = np.random.RandomState(7)
    for _ in range(int(severity * 600)):
        x = rng.randint(0, w); y = rng.randint(0, h - 20)
        length = rng.randint(10, 25); alpha = rng.uniform(0.3, 0.6)
        for k in range(length):
            arr[min(y+k, h-1), min(x+k//3, w-1)] *= (1 - alpha)
            arr[min(y+k, h-1), min(x+k//3, w-1)] += 200 * alpha
    arr = arr * (1 - severity * 0.15) + 128 * severity * 0.15
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_snow(img, severity=SEVERITY):
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    gray = arr.mean(axis=2, keepdims=True)
    arr = arr * (1 - severity * 0.4) + gray * severity * 0.4
    rng = np.random.RandomState(13)
    num_flakes = int(severity * 800)
    ys = rng.randint(0, h, num_flakes); xs = rng.randint(0, w, num_flakes)
    alphas = rng.uniform(0.5, 1.0, num_flakes)
    for y, x, a in zip(ys, xs, alphas):
        arr[y, x] = arr[y, x] * (1 - a) + 255 * a
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


WEATHER_FNS = [apply_fog, apply_rain, apply_snow]


class WeatherDataset(datasets.KITTIRAWDataset):
    def __init__(self, *args, weather_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weather_fn = weather_fn

    def get_color(self, folder, frame_index, side, do_flip):
        img = super().get_color(folder, frame_index, side, do_flip)
        if self.weather_fn is not None:
            img = self.weather_fn(img)
        return img


def compute_reprojection_loss(pred, target, ssim_fn):
    abs_diff = torch.abs(target - pred)
    l1 = abs_diff.mean(1, True)
    ssim_loss = ssim_fn(pred, target).mean(1, True)
    return 0.85 * ssim_loss + 0.15 * l1


def finetune(opt):
    device = torch.device("cuda")
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")

    filenames = readlines(os.path.join(splits_dir, "eigen_zhou", "train_files.txt"))
    # Use a subset for speed
    subset_size = getattr(opt, "finetune_subset", 10000)
    filenames = filenames[:subset_size]

    config = get_config(opt)
    num_ch_enc = [64, 64, 128, 160, 320]

    # --- Build models ---
    encoder = networks.build_model(config, img_width=opt.width, img_height=opt.height)
    depth_decoder = networks.FusionDecoder(
        num_ch_enc,
        use_feature_suppression=getattr(opt, "use_feature_suppression", False))
    pose_encoder = networks.ResnetEncoder(18, False, num_input_images=2)
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc,
                                        num_input_features=1,
                                        num_frames_to_predict_for=2)

    # --- Load weights ---
    wf = opt.load_weights_folder
    enc_dict = torch.load(os.path.join(wf, "encoder.pth"), map_location=device)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in enc_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(
        torch.load(os.path.join(wf, "depth.pth"), map_location=device), strict=False)
    pose_encoder.load_state_dict(
        torch.load(os.path.join(wf, "pose_encoder.pth"), map_location=device))
    pose_decoder.load_state_dict(
        torch.load(os.path.join(wf, "pose.pth"), map_location=device))

    encoder.to(device).eval()
    pose_encoder.to(device).eval()
    pose_decoder.to(device).eval()
    depth_decoder.to(device).train()

    # Freeze: encoder, pose
    for p in encoder.parameters(): p.requires_grad_(False)
    for p in pose_encoder.parameters(): p.requires_grad_(False)
    for p in pose_decoder.parameters(): p.requires_grad_(False)
    # Freeze depth decoder backbone (all except dispconv and uncertconv)
    for name, p in depth_decoder.named_parameters():
        if 'dispconv' not in name and 'uncertconv' not in name:
            p.requires_grad_(False)

    # Re-initialize uncertconv to bias=-2 (sigmoid(-2)≈0.12)
    uncertconv = depth_decoder.convs[("uncertconv", 0)]
    inner = uncertconv.conv if hasattr(uncertconv, 'conv') else uncertconv
    nn.init.normal_(inner.weight, mean=0.0, std=0.01)
    if inner.bias is not None:
        nn.init.constant_(inner.bias, -2.0)  # sigmoid(-2)≈0.12 → moderate uncertainty

    trainable_params = (list(depth_decoder.convs[("dispconv", 0)].parameters()) +
                        list(depth_decoder.convs[("uncertconv", 0)].parameters()))
    optimizer = torch.optim.Adam(trainable_params, lr=getattr(opt, "finetune_lr", 1e-5))

    # --- Geometry ---
    ssim_fn = SSIM().to(device)
    backproject = BackprojectDepth(opt.batch_size if hasattr(opt,'batch_size') else 4,
                                   opt.height, opt.width).to(device)
    project = Project3D(opt.batch_size if hasattr(opt,'batch_size') else 4,
                        opt.height, opt.width).to(device)
    dxy = torch.zeros(opt.batch_size if hasattr(opt,'batch_size') else 4, 2, device=device)

    n_epochs = getattr(opt, "finetune_epochs", 5)
    bs = getattr(opt, "batch_size", 4)

    print("-> Fine-tuning uncertconv+dispconv ({} samples, {} epochs, lr={})".format(
        len(filenames), n_epochs, getattr(opt, 'finetune_lr', 1e-5)))
    print("-> Loss: photo_loss + Laplacian NLL (calibrates sigma)")

    # Resizer for full-res color
    resize_color = transforms.Resize((opt.height, opt.width),
                                     interpolation=transforms.InterpolationMode.BILINEAR)

    for epoch in range(n_epochs):
        weather_fn = WEATHER_FNS[epoch % len(WEATHER_FNS)]
        dataset = WeatherDataset(
            opt.data_path, filenames, opt.height, opt.width,
            [-1, 0, 1], 4, is_train=True, img_ext='.png',
            weather_fn=weather_fn)
        dataloader = DataLoader(dataset, bs, shuffle=True,
                                num_workers=4, pin_memory=True, drop_last=True)

        total_photo = 0.0; total_nll = 0.0; total_sigma = 0.0; n_batches = 0

        for batch in dataloader:
            color = batch[("color_MiS", 0, 0)].to(device)
            color_prev = batch[("color_MiS", -1, 0)].to(device)
            color_next = batch[("color_MiS",  1, 0)].to(device)
            K     = batch[("K_MiS", 0)].to(device).float()
            inv_K = batch[("inv_K_MiS", 0)].to(device).float()

            B = color.shape[0]
            if B != bs:
                continue  # skip incomplete batch

            # --- Depth prediction ---
            # Encoder is frozen but we need grad for dispconv/uncertconv.
            # Restart the computation graph at feats so backbone output has grad_fn.
            with torch.no_grad():
                feats = encoder(color)
            # Detach from encoder, then re-enable grad so the backbone (even frozen)
            # becomes part of the autograd graph and gradients reach dispconv/uncertconv.
            feats = [f.detach().requires_grad_(True) for f in feats]
            output = depth_decoder(feats)

            raw_uncert = output[("uncert", 0)]  # [B,1,H,W] raw log-variance
            sigma = torch.sigmoid(raw_uncert)
            disp = output[("disp", 0)][:, 0:1]
            _, depth = disp_to_depth(disp, 0.1, 100.0)

            # --- Pose estimation ---
            with torch.no_grad():
                pose_in_prev = torch.cat([color_prev, color], dim=1)
                axisangle_prev, trans_prev = pose_decoder([pose_encoder(pose_in_prev)])
                T_prev = transformation_from_parameters(
                    axisangle_prev[:, 0], trans_prev[:, 0], invert=True)

                pose_in_next = torch.cat([color, color_next], dim=1)
                axisangle_next, trans_next = pose_decoder([pose_encoder(pose_in_next)])
                T_next = transformation_from_parameters(
                    axisangle_next[:, 0], trans_next[:, 0], invert=False)

            # --- Photometric loss ---
            photo_losses = []
            dxy_b = dxy[:B]
            for T, src in [(T_prev, color_prev), (T_next, color_next)]:
                cam_pts = backproject(depth, inv_K, dxy_b)
                pix_coords = project(cam_pts, K, T, dxy_b)
                reconstructed = F.grid_sample(
                    src, pix_coords,
                    mode="bilinear", padding_mode="border", align_corners=False)
                photo_losses.append(compute_reprojection_loss(reconstructed, color, ssim_fn))

            photo = torch.cat(photo_losses, dim=1).min(dim=1, keepdim=True)[0]  # [B,1,H,W]

            # auto-mask: ignore pixels where photo > identity reprojection
            with torch.no_grad():
                ident_photo = torch.cat([
                    compute_reprojection_loss(color_prev, color, ssim_fn),
                    compute_reprojection_loss(color_next, color, ssim_fn)
                ], dim=1).min(dim=1, keepdim=True)[0]
            mask = (photo < ident_photo).float()

            # --- Depth loss (unchanged from training) ---
            depth_loss = ((1.0 - sigma.detach()) * photo * mask).mean()

            # --- NLL uncertainty loss (calibrates sigma properly) ---
            # Laplacian NLL: L = photo * exp(-raw) + raw
            # ∂L/∂raw = -photo * exp(-raw) + 1 = 0 → raw* = log(photo)
            # sigmoid(log(photo)) = photo/(1+photo) → sigma tracks photo ✓
            raw_clamped = raw_uncert.clamp(-5.0, 5.0)
            nll_loss = (photo.detach() * torch.exp(-raw_clamped) + raw_clamped).mean()

            # --- Smoothness ---
            norm_disp = disp / (disp.mean(2, True).mean(3, True) + 1e-7)
            smooth_loss = (
                (norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:]).abs().mean() +
                (norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :]).abs().mean()
            )

            total_loss = depth_loss + nll_loss + 0.001 * smooth_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_photo += depth_loss.item()
            total_nll += nll_loss.item()
            total_sigma += sigma.mean().item()
            n_batches += 1

        print("  Epoch {}: photo={:.4f}, nll={:.4f}, sigma_mean={:.4f}".format(
            epoch + 1,
            total_photo / max(n_batches, 1),
            total_nll / max(n_batches, 1),
            total_sigma / max(n_batches, 1)))

    # --- Save fine-tuned weights ---
    out_dir = os.path.join("models", opt.save_model_name, "models", "weights_49_unc")
    os.makedirs(out_dir, exist_ok=True)
    # Copy all files from weights_49
    import shutil
    for fname in os.listdir(wf):
        shutil.copy2(os.path.join(wf, fname), os.path.join(out_dir, fname))
    # Overwrite depth.pth with fine-tuned version
    torch.save(depth_decoder.state_dict(), os.path.join(out_dir, "depth.pth"))
    print("\n-> Saved fine-tuned model to: {}".format(out_dir))

    # Quick sigma check
    depth_decoder.eval()
    x = torch.randn(1, 3, opt.height, opt.width).to(device) * 0.5 + 0.5
    with torch.no_grad():
        out = depth_decoder(encoder(x))
        s = torch.sigmoid(out[("uncert", 0)])
        print("-> Sigma range (random img): [{:.3f}, {:.3f}]".format(
            s.min().item(), s.max().item()))


if __name__ == "__main__":
    options = MonodepthOptions()
    options.parser.add_argument("--finetune_epochs", type=int, default=5)
    options.parser.add_argument("--finetune_lr", type=float, default=1e-5)
    options.parser.add_argument("--finetune_subset", type=int, default=10000)
    options.parser.add_argument("--save_model_name", type=str,
                                default="Tiny-Depth-Weather-Robust-Feature-Supression")
    finetune(options.parse())
