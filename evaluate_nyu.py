"""
Evaluare cross-domain (zero-shot, fara fine-tuning) pe NYU Depth v2,
pentru un model antrenat exclusiv pe KITTI -- protocol identic cu cel
folosit in paper-ul original TinyDepth (Tabel 5) si in literatura
monodepth (BTS, AdaBins, Monodepth2).

Protocol:
  - 654 imagini din split-ul oficial de test (Silberman/Eigen)
  - Eigen crop NYU: [45:471, 41:601]
  - max_depth = 10 m, min_depth = 1e-3 m
  - Scalare mediana (modelul e self-supervised, fara scara metrica)

Usage:
  python evaluate_nyu.py --load_weights_folder models/URW-Depth-S2/models/weights_14 \
    --nyu_path nyu_data/test --use_feature_suppression --use_wandb
"""
from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from networks.configuration import get_config
from layer import disp_to_depth
import networks

try:
    import wandb
except ImportError:
    wandb = None

cv2.setNumThreads(0)

NYU_CROP = (45, 471, 41, 601)  # y0, y1, x0, x1


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


class NYUTestDataset(Dataset):
    def __init__(self, nyu_path, height, width):
        self.img_dir = os.path.join(nyu_path, "images")
        self.depth_dir = os.path.join(nyu_path, "depths")
        self.filenames = sorted(os.listdir(self.img_dir))
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = cv2.imread(os.path.join(self.img_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (self.width, self.height))
        img_t = torch.from_numpy(img_resized / 255.0).permute(2, 0, 1).float()
        return img_t, fname


def load_model(weights_folder, device, height, width, use_feature_suppression):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights_folder", required=True)
    parser.add_argument("--nyu_path", default="nyu_data/test")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--use_feature_suppression", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="tinydepth")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    opt = parser.parse_args()

    MIN_DEPTH, MAX_DEPTH = 1e-3, 10.0  # NYU: max 10m (vs 80m la KITTI)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Loading weights from {opt.load_weights_folder}")
    encoder, decoder = load_model(
        opt.load_weights_folder, device, opt.height, opt.width, opt.use_feature_suppression
    )

    dataset = NYUTestDataset(opt.nyu_path, opt.height, opt.width)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    pred_disps = []
    filenames_all = []
    with torch.no_grad():
        for imgs, fnames in loader:
            imgs = imgs.to(device)
            out = decoder(encoder(imgs))
            pred_disp, _ = disp_to_depth(out[("disp", 0)][:, 0, :, :].unsqueeze(1), 0.1, 100.0)
            pred_disps.append(pred_disp.cpu()[:, 0].numpy())
            filenames_all.extend(fnames)

    pred_disps = np.concatenate(pred_disps)

    y0, y1, x0, x1 = NYU_CROP
    errors, ratios = [], []
    depth_dir = os.path.join(opt.nyu_path, "depths")

    for i, fname in enumerate(filenames_all):
        gt_depth_mm = cv2.imread(os.path.join(depth_dir, fname), cv2.IMREAD_UNCHANGED)
        gt_depth = gt_depth_mm.astype(np.float32) / 1000.0  # metri
        gt_h, gt_w = gt_depth.shape

        pred_disp = cv2.resize(pred_disps[i], (gt_w, gt_h))
        pred_depth = 1.0 / pred_disp

        mask = np.zeros_like(gt_depth, dtype=bool)
        mask[y0:y1, x0:x1] = True
        mask = np.logical_and(mask, np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH))

        if mask.sum() == 0:
            continue

        pd, gd = pred_depth[mask], gt_depth[mask]
        ratio = np.median(gd) / np.median(pd)
        ratios.append(ratio)
        pd = np.clip(pd * ratio, MIN_DEPTH, MAX_DEPTH)
        errors.append(compute_errors(gd, pd))

    mean_errors = np.array(errors).mean(0)
    ratios = np.array(ratios)
    print(f" Scaling ratios | med: {np.median(ratios):.3f} | std: {np.std(ratios/np.median(ratios)):.3f}")
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

    if opt.use_wandb and wandb is not None:
        run_name = opt.wandb_run_name or f"eval-nyu-{os.path.basename(opt.load_weights_folder)}"
        wandb.init(project=opt.wandb_project, name=run_name)
        wandb.log({
            "nyu/abs_rel": float(mean_errors[0]),
            "nyu/sq_rel": float(mean_errors[1]),
            "nyu/rmse": float(mean_errors[2]),
            "nyu/rmse_log": float(mean_errors[3]),
            "nyu/a1": float(mean_errors[4]),
            "nyu/a2": float(mean_errors[5]),
            "nyu/a3": float(mean_errors[6]),
        })
        wandb.finish()


if __name__ == "__main__":
    main()
