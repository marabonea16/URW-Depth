"""
Evaluare pe KITTI-C (RoboDepth benchmark, NeurIPS 2023).
18 tipuri de coruptii x 5 severitati, 697 imagini per subset.

Usage:
  python evaluate_kitti_c.py \
    --load_weights_folder models/URW-Depth-S2/models/weights_14 \
    --kitti_c_path kitti_c/kitti_c \
    --data_path /home/ubuntu/TinyDepth \
    --corruptions fog snow frost brightness dark all \
    --eval_mono
"""

from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

from networks.configuration import get_config
from layer import disp_to_depth
from utils import readlines
import datasets
import networks

try:
    import wandb
except ImportError:
    wandb = None

cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

ALL_CORRUPTIONS = [
    "brightness", "color_quant", "contrast", "dark",
    "defocus_blur", "elastic_transform", "fog", "frost",
    "gaussian_noise", "glass_blur", "impulse_noise", "iso_noise",
    "jpeg_compression", "motion_blur", "pixelate", "shot_noise",
    "snow", "zoom_blur",
]

WEATHER_CORRUPTIONS = ["fog", "frost", "snow", "dark", "brightness", "contrast"]


def compute_errors(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def load_model(weights_folder, device, height=192, width=640, use_feature_suppression=False):
    encoder_path = os.path.join(weights_folder, "encoder.pth")
    decoder_path = os.path.join(weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path, map_location=device)

    class _Opt:
        img_height = height
        img_width = width
        encoder = "tiny_vit_5m_22k_distill"
        scales = [0]

    config = get_config(_Opt())
    encoder = networks.build_model(config, img_width=width, img_height=height)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    num_ch_enc = [64, 64, 128, 160, 320]
    decoder = networks.FusionDecoder(num_ch_enc, use_feature_suppression=use_feature_suppression)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device), strict=False)

    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder


def evaluate_one(encoder, decoder, data_path, device, height=192, width=640, batch_size=None):
    """Ruleaza modelul pe un subset KITTI-C si returneaza erorile medii."""
    MIN_DEPTH, MAX_DEPTH = 1e-3, 80.0

    if batch_size is None:
        batch_size = 4 if (height > 256 or width > 768) else 16

    filenames = readlines(os.path.join(splits_dir, "eigen", "test_files.txt"))
    dataset = datasets.KITTIRAWDataset(
        data_path, filenames, height, width, [0], 4, is_train=False, img_ext='.png'
    )
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=4,
                        pin_memory=True, drop_last=False)

    gt_path = os.path.join(splits_dir, "eigen", "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1',
                        allow_pickle=True)["data"]

    pred_disps = []
    with torch.no_grad():
        for data in loader:
            inp = data[("color_MiS", 0, 0)].to(device)
            out = decoder(encoder(inp))
            pred_disp, _ = disp_to_depth(
                out[("disp", 0)][:, 0, :, :].unsqueeze(1), 0.1, 100.0
            )
            pred_disps.append(pred_disp.cpu()[:, 0].numpy())

    pred_disps = np.concatenate(pred_disps)

    errors, ratios = [], []
    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        if gt_depth is None:
            continue
        gt_h, gt_w = gt_depth.shape
        pred_depth = 1.0 / cv2.resize(pred_disps[i], (gt_w, gt_h))

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop = np.array([0.40810811 * gt_h, 0.99189189 * gt_h,
                         0.03594771 * gt_w, 0.96405229 * gt_w]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        if mask.sum() == 0:
            continue

        pd, gd = pred_depth[mask], gt_depth[mask]
        ratio = np.median(gd) / np.median(pd)
        ratios.append(ratio)
        pd = np.clip(pd * ratio, MIN_DEPTH, MAX_DEPTH)
        errors.append(compute_errors(gd, pd))

    return np.array(errors).mean(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights_folder", required=True)
    parser.add_argument("--kitti_c_path", default="kitti_c/kitti_c",
                        help="Path to kitti_c root (contains fog/, snow/, ...)")
    parser.add_argument("--corruptions", nargs="+", default=["fog", "snow", "frost", "dark", "brightness"],
                        help="Corruption types. Use 'all' for all 18, 'weather' for weather subset.")
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--eval_mono", action="store_true", required=True)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--use_feature_suppression", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="tinydepth")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    opt = parser.parse_args()

    corruptions = opt.corruptions
    if "all" in corruptions:
        corruptions = ALL_CORRUPTIONS
    elif "weather" in corruptions:
        corruptions = WEATHER_CORRUPTIONS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Loading weights from {opt.load_weights_folder}")
    encoder, decoder = load_model(
        opt.load_weights_folder, device,
        height=opt.height, width=opt.width,
        use_feature_suppression=opt.use_feature_suppression
    )

    print(f"\n{'Corruption':>20} | " + " | ".join([f"  sev{s}  " for s in opt.severities]) + " | mean")
    print("-" * (24 + 12 * len(opt.severities) + 10))

    all_errors = []
    results_per_corruption = {}
    for corruption in corruptions:
        sev_errors = []
        for sev in opt.severities:
            data_path = os.path.join(opt.kitti_c_path, corruption, str(sev), "kitti_data")
            if not os.path.isdir(data_path):
                print(f"   [SKIP] {corruption}/sev{sev} not found")
                continue
            errs = evaluate_one(encoder, decoder, data_path, device, height=opt.height, width=opt.width, batch_size=getattr(opt, 'batch_size', None))
            sev_errors.append(errs)
            all_errors.append(errs)

        if sev_errors:
            results_per_corruption[corruption] = np.mean([e[0] for e in sev_errors])
            mean_abs = results_per_corruption[corruption]
            sev_str = " | ".join([f"  {e[0]:.3f}  " for e in sev_errors])
            print(f"  {corruption:>20} | {sev_str} | {mean_abs:.3f}")

    overall = np.array(all_errors).mean(0) if all_errors else None
    if overall is not None:
        print(f"\n{'OVERALL MEAN':>20}   abs_rel={overall[0]:.3f} | sq_rel={overall[1]:.3f} | rmse={overall[2]:.3f} | a1={overall[4]:.3f}")

    if opt.use_wandb and wandb is not None and overall is not None:
        run_name = opt.wandb_run_name or f"kitti-c-{os.path.basename(opt.load_weights_folder)}"
        wandb.init(project=opt.wandb_project, name=run_name)
        log_dict = {
            "kitti_c/abs_rel":  float(overall[0]),
            "kitti_c/sq_rel":   float(overall[1]),
            "kitti_c/rmse":     float(overall[2]),
            "kitti_c/rmse_log": float(overall[3]),
            "kitti_c/a1":       float(overall[4]),
            "kitti_c/a2":       float(overall[5]),
            "kitti_c/a3":       float(overall[6]),
        }
        for corr, abs_rel in results_per_corruption.items():
            log_dict[f"kitti_c_per/{corr}"] = float(abs_rel)
        wandb.log(log_dict)
        wandb.finish()

    print("\n-> Done!")


if __name__ == "__main__":
    main()
