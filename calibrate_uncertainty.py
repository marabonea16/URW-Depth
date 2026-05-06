"""
Calibrate uncertainty head (uncertconv) so sigma is meaningful for TTA.

Problem: sigma collapsed to ~0 during training because the loss (1-sigma.detach())*photo + w*sigma
         gives sigma only one gradient direction: minimize sigma → collapse to 0.

Fix: calibrate sigma to track depth change caused by weather:
     sigma_target = |depth_weather - depth_clean| / depth_clean  (normalized to [0,1])

This gives:
  - sigma ≈ 0 in clear regions (weather doesn't affect depth)
  - sigma ≈ 1 in weather-affected regions (fog/rain/snow changes depth)

After calibration, TTA via uncertainty minimization makes sense:
  - High sigma on test image → weather artifacts → minimize sigma → adapt to be confident
"""

from __future__ import absolute_import, division, print_function

import os
import copy
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from networks.configuration import get_config
from layer import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

SEVERITY = 0.45


def apply_fog(img, severity=SEVERITY):
    arr = np.array(img, dtype=np.float32)
    fog_color = np.array([220, 220, 220], dtype=np.float32)
    h = arr.shape[0]
    gradient = np.linspace(severity, severity * 0.3, h, dtype=np.float32)[:, None, None]
    arr = arr * (1 - gradient) + fog_color * gradient
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_rain(img, severity=SEVERITY):
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    rng = np.random.RandomState(7)
    for _ in range(int(severity * 600)):
        x = rng.randint(0, w); y = rng.randint(0, h - 20)
        length = rng.randint(10, 25); alpha = rng.uniform(0.3, 0.6)
        for k in range(length):
            arr[min(y+k,h-1), min(x+k//3,w-1)] *= (1 - alpha)
            arr[min(y+k,h-1), min(x+k//3,w-1)] += 200 * alpha
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


def calibrate(opt):
    device = torch.device("cuda")
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")

    # Use 5000 training samples (fast calibration)
    filenames = readlines(os.path.join(splits_dir, "eigen_zhou", "train_files.txt"))
    filenames = filenames[:5000]

    config = get_config(opt)
    num_ch_enc = [64, 64, 128, 160, 320]

    encoder = networks.build_model(config, img_width=opt.width, img_height=opt.height)
    depth_decoder = networks.FusionDecoder(
        num_ch_enc, use_feature_suppression=getattr(opt, "use_feature_suppression", False))

    encoder_dict = torch.load(
        os.path.join(opt.load_weights_folder, "encoder.pth"), map_location=device)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(
        torch.load(os.path.join(opt.load_weights_folder, "depth.pth"), map_location=device),
        strict=False)

    encoder.to(device).eval()
    depth_decoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in depth_decoder.parameters():
        p.requires_grad_(False)

    # Only uncertconv is trainable
    uncertconv = depth_decoder.convs[("uncertconv", 0)]
    for p in uncertconv.parameters():
        p.requires_grad_(True)

    # Re-initialize uncertconv weights to small values so sigmoid outputs ~0.5 initially
    # Conv3x3 wraps nn.Conv2d in .conv
    inner_conv = uncertconv.conv if hasattr(uncertconv, 'conv') else uncertconv
    torch.nn.init.normal_(inner_conv.weight, mean=0.0, std=0.01)
    if inner_conv.bias is not None:
        torch.nn.init.zeros_(inner_conv.bias)

    optimizer = torch.optim.Adam(uncertconv.parameters(), lr=1e-4)

    # Clean dataset (no weather)
    dataset_clean = datasets.KITTIRAWDataset(
        opt.data_path, filenames, opt.height, opt.width,
        [0], 4, is_train=False, img_ext='.png')

    print("-> Calibrating uncertainty head ({} samples, {} epochs)".format(
        len(filenames), opt.calib_epochs))
    print("-> Target: sigma tracks depth change caused by weather augmentation")

    for epoch in range(opt.calib_epochs):
        total_loss = 0.0
        n_batches = 0

        # Randomly pick a weather function for this epoch
        weather_fn = WEATHER_FNS[epoch % len(WEATHER_FNS)]
        dataset_weather = WeatherDataset(
            opt.data_path, filenames, opt.height, opt.width,
            [0], 4, is_train=False, img_ext='.png',
            weather_fn=weather_fn)

        dataloader_clean = DataLoader(dataset_clean, 8, shuffle=False,
                                      num_workers=4, pin_memory=True, drop_last=True)
        dataloader_weather = DataLoader(dataset_weather, 8, shuffle=False,
                                        num_workers=4, pin_memory=True, drop_last=True)

        for batch_c, batch_w in zip(dataloader_clean, dataloader_weather):
            img_clean = batch_c[("color_MiS", 0, 0)].to(device)
            img_weather = batch_w[("color_MiS", 0, 0)].to(device)

            # Clean depth (frozen)
            with torch.no_grad():
                feats_c = encoder(img_clean)
                out_c = depth_decoder(feats_c)
                disp_c = out_c[("disp", 0)][:, 0:1].detach()

            # Weather: get features, compute sigma, get weather depth
            # depth_decoder is in eval but uncertconv has grad
            with torch.no_grad():
                feats_w = encoder(img_weather)

            # Forward through decoder (partially with grad for uncertconv only)
            # Need to get the feature map `d` that feeds uncertconv
            # We'll call depth_decoder manually to intercept `d`
            depth_decoder.train()  # allow uncertconv grad
            out_w = depth_decoder(feats_w)
            disp_w = out_w[("disp", 0)][:, 0:1].detach()

            # sigma from weather image
            sigma_w = torch.sigmoid(out_w[("uncert", 0)])

            # Target: normalized relative depth change due to weather
            with torch.no_grad():
                delta = (disp_w - disp_c).abs() / (disp_c.abs() + 1e-7)
                sigma_target = (delta / (delta.mean() + 1e-7)).clamp(0, 1)

            # Calibration loss: sigma should track depth change
            calib_loss = F.mse_loss(sigma_w, sigma_target)
            # Small regularization to prevent sigma going to 1 everywhere
            reg_loss = 0.1 * sigma_w.mean()
            loss = calib_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uncertconv.parameters(), max_norm=1.0)
            optimizer.step()

            depth_decoder.eval()

            total_loss += loss.item()
            n_batches += 1

        avg_sigma = sigma_w.detach().mean().item()
        print("  Epoch {}: loss={:.4f}, sigma_mean={:.4f}".format(
            epoch + 1, total_loss / n_batches, avg_sigma))

    # Save calibrated depth decoder
    out_path = os.path.join(opt.load_weights_folder, "depth_calibrated.pth")
    torch.save(depth_decoder.state_dict(), out_path)
    print("\n-> Saved calibrated depth decoder to: {}".format(out_path))

    # Quick sanity check
    depth_decoder.eval()
    x_clean = torch.randn(1, 3, opt.height, opt.width).to(device) * 0.5 + 0.5
    x_fog = apply_fog(
        Image.fromarray((x_clean[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)))
    from torchvision import transforms
    x_fog_t = transforms.ToTensor()(x_fog).unsqueeze(0).to(device)

    with torch.no_grad():
        feats_clean = encoder(x_clean)
        out_clean = depth_decoder(feats_clean)
        sigma_clean = torch.sigmoid(out_clean[("uncert", 0)]).mean().item()

        feats_fog = encoder(x_fog_t)
        out_fog = depth_decoder(feats_fog)
        sigma_fog = torch.sigmoid(out_fog[("uncert", 0)]).mean().item()

    print("\nSanity check:")
    print("  sigma (clean input): {:.4f}".format(sigma_clean))
    print("  sigma (fog input):   {:.4f}".format(sigma_fog))
    print("  -> Fog sigma > clean sigma: {}".format(sigma_fog > sigma_clean))


if __name__ == "__main__":
    options = MonodepthOptions()
    options.parser.add_argument("--calib_epochs", type=int, default=3,
                                help="number of epochs for uncertainty calibration")
    calibrate(options.parse())
