"""
Evaluare pe imagini cu vreme adversa simulata (fog, rain, snow).

Aplica weather augmentation pe imaginile de test KITTI si evalueaza modelul.
Suporta atat inferenta normala cat si TTA (Test-Time Adaptation).

Folosire:
    # evaluare normala cu fog:
    python evaluate_weather.py \
        --load_weights_folder models/Tiny-Depth-Weather-Robust-Feature-Supression/models/weights_49 \
        --eval_mono --height 192 --width 640 --scales 0 \
        --data_path /home/ubuntu/TinyDepth --png \
        --weather_type fog --weather_severity moderate \
        --use_feature_suppression --use_wandb --wandb_project tinydepth

    # cu TTA:
    python evaluate_weather.py ... --use_tta --tta_steps 3 --tta_lr 1e-5
"""

from __future__ import absolute_import, division, print_function

import os
import copy
import cv2
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from networks.configuration import get_config
from layer import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)

try:
    import wandb
except ImportError:
    wandb = None

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# ── weather functions ──────────────────────────────────────────────────────────

SEVERITY_MAP = {
    "mild":     0.25,
    "moderate": 0.45,
    "severe":   0.65,
}

def apply_fog(img, severity=0.45):
    arr = np.array(img, dtype=np.float32)
    fog_color = np.array([220, 220, 220], dtype=np.float32)
    h = arr.shape[0]
    gradient = np.linspace(severity, severity * 0.3, h, dtype=np.float32)[:, None, None]
    arr = arr * (1 - gradient) + fog_color * gradient
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_rain(img, severity=0.45):
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    num_streaks = int(severity * 600)
    rng = np.random.RandomState(42)
    for _ in range(num_streaks):
        x = rng.randint(0, w)
        y = rng.randint(0, h - 20)
        length = rng.randint(10, 25)
        alpha = rng.uniform(0.3, 0.6)
        for k in range(length):
            yi = min(y + k, h - 1)
            xi = min(x + k // 3, w - 1)
            arr[yi, xi] = arr[yi, xi] * (1 - alpha) + 200 * alpha
    arr = arr * (1 - severity * 0.15) + 128 * severity * 0.15
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_snow(img, severity=0.45):
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    gray = arr.mean(axis=2, keepdims=True)
    arr = arr * (1 - severity * 0.4) + gray * severity * 0.4
    rng = np.random.RandomState(42)
    num_flakes = int(severity * 800)
    ys = rng.randint(0, h, num_flakes)
    xs = rng.randint(0, w, num_flakes)
    alphas = rng.uniform(0.5, 1.0, num_flakes)
    for y, x, a in zip(ys, xs, alphas):
        arr[y, x] = arr[y, x] * (1 - a) + 255 * a
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

WEATHER_FNS = {"fog": apply_fog, "rain": apply_rain, "snow": apply_snow}

# ── metrics ────────────────────────────────────────────────────────────────────

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse     = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel  = np.mean(np.abs(gt - pred) / gt)
    sq_rel   = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

# ── TTA step ───────────────────────────────────────────────────────────────────

def tta_step(encoder, depth_decoder,
             frame_target,
             tta_lr, n_steps, device, consistency_weight=1.0):
    """
    Uncertainty Minimization TTA:
    Minimizeaza media sigma (incertitudinea) + consistency fata de predictia initiala.
    Nu necesita frame-uri vecine sau retea de pose.
    Pixelii cu sigma mare = afectati de weather -> forteaza predictii mai sigure.
    """
    original_state = copy.deepcopy(depth_decoder.state_dict())
    # actualizeaza ambele capete de output
    tta_params = (list(depth_decoder.convs[("dispconv", 0)].parameters()) +
                  list(depth_decoder.convs[("uncertconv", 0)].parameters()))
    optimizer = torch.optim.Adam(tta_params, lr=tta_lr)

    with torch.no_grad():
        feats_init = encoder(frame_target)
        output_init = depth_decoder(feats_init)
        disp_init = output_init[("disp", 0)][:, 0:1].detach()

    for _ in range(n_steps):
        optimizer.zero_grad()
        with torch.no_grad():
            feats = encoder(frame_target)
        output = depth_decoder(feats)

        disp  = output[("disp", 0)][:, 0:1]
        sigma = torch.sigmoid(output[("uncert", 0)])

        # uncertainty minimization: forteaza predictii sigure pe regiunile cu weather
        uncert_loss = sigma.mean()
        # consistency: disp nu deriva prea mult de la predictia initiala
        consist_loss = F.l1_loss(disp, disp_init)

        total_loss = uncert_loss + consistency_weight * consist_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(tta_params, max_norm=1.0)
        optimizer.step()

    depth_decoder.eval()
    with torch.no_grad():
        out_final = depth_decoder(encoder(frame_target))
        disp_final, _ = disp_to_depth(out_final[("disp", 0)][:, 0:1], 0.1, 100.0)

    depth_decoder.load_state_dict(original_state)
    depth_decoder.train()
    return disp_final

# ── dataset cu weather aplicat pe imagini ─────────────────────────────────────

class WeatherKITTIDataset(datasets.KITTIRAWDataset):
    """Wrapper care aplica weather augmentation pe imaginile incarcate."""
    def __init__(self, *args, weather_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weather_fn = weather_fn

    def get_color(self, folder, frame_index, side, do_flip):
        img = super().get_color(folder, frame_index, side, do_flip)
        if self.weather_fn is not None:
            img = self.weather_fn(img)
        return img

# ── main evaluate ──────────────────────────────────────────────────────────────

def evaluate(opt):
    MIN_DEPTH, MAX_DEPTH = 1e-3, 80
    device = torch.device("cuda")

    assert opt.eval_mono, "Necesita --eval_mono"

    weather_type = getattr(opt, "weather_type", "fog")
    severity_str = getattr(opt, "weather_severity", "moderate")
    severity = SEVERITY_MAP.get(severity_str, 0.45)
    use_tta  = getattr(opt, "use_tta", False)

    weather_fn = WEATHER_FNS.get(weather_type)
    if weather_fn is None:
        raise ValueError(f"weather_type necunoscut: {weather_type}. Alege: fog, rain, snow")

    # fix severity
    _weather_fn = lambda img: weather_fn(img, severity=severity)

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))
    print("-> Weather: {} | severity: {} ({:.2f}) | TTA: {}".format(
        weather_type, severity_str, severity, use_tta))

    filenames  = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    config     = get_config(opt)
    num_ch_enc = [64, 64, 128, 160, 320]

    encoder = networks.build_model(config, img_width=opt.width, img_height=opt.height)
    depth_decoder = networks.FusionDecoder(
        num_ch_enc,
        use_feature_suppression=getattr(opt, "use_feature_suppression", False))

    encoder_dict = torch.load(os.path.join(opt.load_weights_folder, "encoder.pth"),
                              map_location=device)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(
        torch.load(os.path.join(opt.load_weights_folder, "depth.pth"), map_location=device),
        strict=False)

    encoder.to(device).eval()
    for p in encoder.parameters(): p.requires_grad_(False)

    # --- dataset cu weather (doar frame 0, vecinii se incarc separat pt TTA) ---
    dataset = WeatherKITTIDataset(
        opt.data_path, filenames, opt.height, opt.width,
        [0], 4, is_train=False, img_ext='.png',
        weather_fn=_weather_fn)
    dataloader = DataLoader(dataset, 1 if use_tta else 16,
                            shuffle=False, num_workers=2,
                            pin_memory=True, drop_last=False)

    # --- TTA setup (uncertainty minimization, nu necesita pose/vecini) ---
    if use_tta:
        depth_decoder.to(device).train()
    else:
        depth_decoder.to(device).eval()

    # --- GT depth ---
    gt_path   = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1',
                        allow_pickle=True)["data"]

    pred_disps       = []
    sample_uncert_maps = []
    sample_input_imgs  = []

    print("-> Computing predictions ({} images)...".format(len(filenames)))

    if use_tta:
        for idx, data in enumerate(dataloader):
            frame_target = data[("color_MiS", 0, 0)].to(device)

            disp_out = tta_step(
                encoder, depth_decoder, frame_target,
                tta_lr=opt.tta_lr, n_steps=opt.tta_steps, device=device,
                consistency_weight=getattr(opt, "tta_consistency_weight", 1.0))

            pred_disps.append(disp_out.cpu().numpy()[:, 0])

            if len(sample_uncert_maps) < 8:
                depth_decoder.eval()
                with torch.no_grad():
                    out_u = depth_decoder(encoder(frame_target))
                if ("uncert", 0) in out_u:
                    sample_uncert_maps.append(out_u[("uncert", 0)].cpu().numpy()[0, 0])
                    sample_input_imgs.append(data[("color_MiS", 0, 0)].numpy()[0])
                depth_decoder.train()

            if (idx + 1) % 50 == 0:
                print("  [{}/{}]".format(idx + 1, len(filenames)))
    else:
        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color_MiS", 0, 0)].to(device)
                output = depth_decoder(encoder(input_color))
                pred_disp, _ = disp_to_depth(
                    output[("disp", 0)][:, 0, :, :].unsqueeze(1),
                    opt.min_depth, opt.max_depth)
                pred_disps.append(pred_disp.cpu()[:, 0].numpy())

                if len(sample_uncert_maps) < 8 and ("uncert", 0) in output:
                    u    = output[("uncert", 0)].cpu().numpy()
                    imgs = data[("color_MiS", 0, 0)].numpy()
                    n = min(8 - len(sample_uncert_maps), u.shape[0])
                    sample_uncert_maps.extend(u[:n, 0])
                    sample_input_imgs.extend(imgs[:n])

    pred_disps = np.concatenate(pred_disps)

    # --- metrici ---
    errors, ratios = [], []
    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_h, gt_w = gt_depth.shape
        pred_disp  = cv2.resize(pred_disps[i], (gt_w, gt_h))
        pred_depth = 1.0 / pred_disp

        mask = gt_depth > 0
        pred_depth = pred_depth[mask].clip(MIN_DEPTH, MAX_DEPTH)
        gt_d       = gt_depth[mask].clip(MIN_DEPTH, MAX_DEPTH)

        ratio = np.median(gt_d) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth = (pred_depth * ratio).clip(MIN_DEPTH, MAX_DEPTH)
        errors.append(compute_errors(gt_d, pred_depth))

    ratios = np.array(ratios)
    mean_errors = np.array(errors).mean(0)

    print("\nScaling ratios | med: {:.3f} | std: {:.3f}".format(
        np.median(ratios), ratios.std()))
    print("\n   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |")
    print(("&   {:.3f}  " * 7).format(*mean_errors))

    # --- wandb ---
    if getattr(opt, "use_wandb", False) and wandb is not None:
        mode = "tta" if use_tta else "no_tta"
        run_name = (getattr(opt, "wandb_run_name", None) or
                    f"weather-{weather_type}-{severity_str}-{mode}")
        wandb.init(project=opt.wandb_project,
                   entity=getattr(opt, "wandb_entity", None),
                   name=run_name)
        log_dict = {
            f"weather_{weather_type}/abs_rel":  float(mean_errors[0]),
            f"weather_{weather_type}/sq_rel":   float(mean_errors[1]),
            f"weather_{weather_type}/rmse":     float(mean_errors[2]),
            f"weather_{weather_type}/rmse_log": float(mean_errors[3]),
            f"weather_{weather_type}/a1":       float(mean_errors[4]),
            f"weather_{weather_type}/a2":       float(mean_errors[5]),
            f"weather_{weather_type}/a3":       float(mean_errors[6]),
            "weather/type":     weather_type,
            "weather/severity": severity_str,
            "tta/enabled":      use_tta,
        }
        if use_tta:
            log_dict["tta/steps"] = opt.tta_steps
            log_dict["tta/lr"]    = opt.tta_lr

        if sample_uncert_maps:
            imgs_wandb = []
            for u, img in zip(sample_uncert_maps, sample_input_imgs):
                u_norm  = (u - u.min()) / (u.max() - u.min() + 1e-8)
                u_color = cv2.applyColorMap((u_norm * 255).astype(np.uint8),
                                            cv2.COLORMAP_VIRIDIS)
                u_color = cv2.cvtColor(u_color, cv2.COLOR_BGR2RGB)
                rgb = np.transpose(img, (1, 2, 0))
                rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255).astype(np.uint8)
                combined = np.concatenate([rgb, u_color], axis=1)
                imgs_wandb.append(wandb.Image(combined,
                    caption=f"input ({weather_type}) | uncertainty"))
            log_dict[f"weather_{weather_type}/uncertainty_maps"] = imgs_wandb

        wandb.log(log_dict)
        wandb.finish()

    print("\n-> Done!")
    return mean_errors


if __name__ == "__main__":
    options = MonodepthOptions()
    options.parser.add_argument("--weather_type", type=str, default="fog",
                                choices=["fog", "rain", "snow"],
                                help="tipul de vreme adversa")
    options.parser.add_argument("--weather_severity", type=str, default="moderate",
                                choices=["mild", "moderate", "severe"],
                                help="intensitatea vremii")
    options.parser.add_argument("--use_tta", action="store_true",
                                help="foloseste TTA la inferenta")
    options.parser.add_argument("--tta_steps", type=int, default=5)
    options.parser.add_argument("--tta_lr",    type=float, default=1e-5)
    options.parser.add_argument("--tta_consistency_weight", type=float, default=1.0)
    evaluate(options.parse())
