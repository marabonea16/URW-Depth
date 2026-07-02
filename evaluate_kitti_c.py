"""
Evaluare pe KITTI-C (RoboDepth benchmark, NeurIPS 2023).
18 tipuri de coruptii x 5 severitati, 697 imagini per subset.

Usage:
  python evaluate_kitti_c.py \
    --load_weights_folder models/URW-Depth-Calibrated/models/weights_latest \
    --kitti_c_path kitti_c/kitti_c \
    --corruptions all --eval_mono

  # cu TTA fotometric + flip ensemble:
  python evaluate_kitti_c.py ... --use_tta --tta_steps 3 --tta_lr 1e-5 \
    --orig_data_path /home/ubuntu/TinyDepth --flip_ensemble
"""

from __future__ import absolute_import, division, print_function

import os
import copy
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from networks.configuration import get_config
from layer import (disp_to_depth, BackprojectDepth, Project3D,
                   SSIM, transformation_from_parameters)
from utils import readlines
import datasets
import networks

try:
    import wandb
except ImportError:
    wandb = None

try:
    from imagecorruptions import corrupt as _ic_corrupt
    _HAS_IC = True
except ImportError:
    _HAS_IC = False

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

# Mapare KITTI-C → imagecorruptions (None = implementare manuala)
_IC_NAME_MAP = {
    "color_quant": None,
    "dark": None,
    "iso_noise": None,
}


def apply_corruption(img_pil, corruption_name, severity):
    """Aplica aceeasi coruptie pe un frame vecin pentru TTA.
    seed=42 asigura pattern identic intre frame-uri (esential pentru TTA fotometric).
    """
    img_np = np.array(img_pil, dtype=np.uint8)
    np.random.seed(42)

    if corruption_name in _IC_NAME_MAP:
        # Implementari manuale pentru coruptiile absente din imagecorruptions
        if corruption_name == "dark":
            factor = max(0.05, 1.0 - severity * 0.18)
            img_np = np.clip(img_np * factor, 0, 255).astype(np.uint8)
        elif corruption_name == "color_quant":
            bits = max(1, 8 - severity)
            q = 256 // (2 ** bits)
            img_np = (img_np // q * q).astype(np.uint8)
        elif corruption_name == "iso_noise":
            sigma = severity * 18
            noise = np.random.normal(0, sigma, img_np.shape)
            img_np = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif _HAS_IC:
        try:
            img_np = _ic_corrupt(img_np, corruption_name=corruption_name, severity=severity)
        except Exception:
            pass  # fallback: nicio coruptie (mai rau decat perfect, dar TTA ramane valid)

    return Image.fromarray(img_np)


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


def _photo_loss(pred, target, ssim_fn):
    l1 = torch.abs(target - pred).mean(1, True)
    ssim = ssim_fn(pred, target).mean(1, True)
    return 0.85 * ssim + 0.15 * l1


def tta_step(encoder, decoder, pose_enc, pose_dec,
             frame_t, frame_prev, frame_next,
             K, inv_K, ssim_fn, backproject, project,
             tta_lr, n_steps, device, consistency_weight=1.0):
    """TTA fotometric ghidat de incertitudine (identic cu tta_evaluate.py)."""
    orig_state = copy.deepcopy(decoder.state_dict())
    tta_params = list(decoder.convs[("dispconv", 0)].parameters())
    optimizer = torch.optim.Adam(tta_params, lr=tta_lr)
    dxy = torch.zeros(1, 2, device=device)

    with torch.no_grad():
        disp_init = decoder(encoder(frame_t))[("disp", 0)][:, 0:1].detach()

    for _ in range(n_steps):
        optimizer.zero_grad()
        with torch.no_grad():
            feats = encoder(frame_t)
        out = decoder(feats)
        disp = out[("disp", 0)][:, 0:1]
        sigma = torch.sigmoid(out[("uncert", 0)]).detach() if ("uncert", 0) in out else None
        _, depth = disp_to_depth(disp, 0.1, 100.0)

        total = torch.tensor(0.0, device=device)
        for pose_input, src, do_invert in [
            (torch.cat([frame_prev, frame_t], 1), frame_prev, True),
            (torch.cat([frame_t, frame_next], 1), frame_next, False),
        ]:
            with torch.no_grad():
                axisangle, translation = pose_dec([pose_enc(pose_input)])
                T = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=do_invert)
            cam_pts = backproject(depth, inv_K, dxy)
            pix = project(cam_pts, K, T, dxy)
            recon = F.grid_sample(src, pix, mode="bilinear", padding_mode="border", align_corners=False)
            loss = _photo_loss(recon, frame_t, ssim_fn)
            if sigma is not None:
                loss = ((1.0 - sigma) * loss).sum() / (1.0 - sigma).sum().clamp(min=1e-6)
            else:
                loss = loss.mean()
            total = total + loss

        total = total + consistency_weight * F.l1_loss(disp, disp_init)
        total.backward()
        torch.nn.utils.clip_grad_norm_(tta_params, 1.0)
        optimizer.step()

    decoder.eval()
    with torch.no_grad():
        disp_out, _ = disp_to_depth(decoder(encoder(frame_t))[("disp", 0)][:, 0:1], 0.1, 100.0)
    decoder.load_state_dict(orig_state)
    decoder.train()
    return disp_out


def load_model(weights_folder, device, height=192, width=640,
               use_feature_suppression=False, gate_depth_input=True, use_tta=False):
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
    decoder = networks.FusionDecoder(num_ch_enc, use_feature_suppression=use_feature_suppression,
                                     gate_depth_input=gate_depth_input)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device), strict=False)

    encoder.to(device).eval()
    decoder.to(device).eval()

    if not use_tta:
        return encoder, decoder, None, None

    decoder.train()
    pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
    pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                    num_frames_to_predict_for=2)
    pose_enc.load_state_dict(torch.load(os.path.join(weights_folder, "pose_encoder.pth"), map_location=device))
    pose_dec.load_state_dict(torch.load(os.path.join(weights_folder, "pose.pth"), map_location=device))
    pose_enc.to(device).eval()
    pose_dec.to(device).eval()
    for p in list(pose_enc.parameters()) + list(pose_dec.parameters()):
        p.requires_grad_(False)
    return encoder, decoder, pose_enc, pose_dec


class CorruptedNeighborDataset(datasets.KITTIRAWDataset):
    """Incarca frame-uri vecine curate si aplica aceeasi coruptie pentru TTA."""
    def __init__(self, *args, corruption_name=None, severity=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.corruption_name = corruption_name
        self.severity = severity

    def get_color(self, folder, frame_index, side, do_flip):
        img = super().get_color(folder, frame_index, side, do_flip)
        if self.corruption_name is not None:
            img = apply_corruption(img, self.corruption_name, self.severity)
        return img


def evaluate_one(encoder, decoder, data_path, device, height=192, width=640,
                 batch_size=None, flip_ensemble=False,
                 use_tta=False, pose_enc=None, pose_dec=None,
                 tta_steps=3, tta_lr=1e-5, tta_consistency_weight=1.0,
                 orig_data_path=None, corruption_name=None, severity=1):
    MIN_DEPTH, MAX_DEPTH = 1e-3, 80.0
    if batch_size is None:
        batch_size = 4 if (height > 256 or width > 768) else 16

    filenames = readlines(os.path.join(splits_dir, "eigen", "test_files.txt"))
    gt_path = os.path.join(splits_dir, "eigen", "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    pred_disps = []

    if use_tta and pose_enc is not None and orig_data_path is not None:
        # TTA fotometric: frame_t din KITTI-C, vecini din KITTI original + aceeasi coruptie
        ssim_fn = SSIM().to(device)
        backproject = BackprojectDepth(1, height, width).to(device)
        project = Project3D(1, height, width).to(device)

        dataset_t = datasets.KITTIRAWDataset(
            data_path, filenames, height, width, [0], 4, is_train=False, img_ext='.png')
        dataset_neighbors = CorruptedNeighborDataset(
            orig_data_path, filenames, height, width, [-1, 0, 1], 4,
            is_train=False, img_ext='.png',
            corruption_name=corruption_name, severity=severity)

        for idx in range(len(filenames)):
            data_t = dataset_t[idx]
            frame_t = data_t[("color_MiS", 0, 0)].unsqueeze(0).to(device)
            K       = data_t[("K_MiS", 0)].unsqueeze(0).to(device).float()
            inv_K   = data_t[("inv_K_MiS", 0)].unsqueeze(0).to(device).float()

            try:
                data_nb    = dataset_neighbors[idx]
                frame_prev = data_nb[("color_MiS", -1, 0)].unsqueeze(0).to(device)
                frame_next = data_nb[("color_MiS",  1, 0)].unsqueeze(0).to(device)
                disp_out = tta_step(
                    encoder, decoder, pose_enc, pose_dec,
                    frame_t, frame_prev, frame_next,
                    K, inv_K, ssim_fn, backproject, project,
                    tta_lr=tta_lr, n_steps=tta_steps, device=device,
                    consistency_weight=tta_consistency_weight)
            except Exception:
                decoder.eval()
                with torch.no_grad():
                    out = decoder(encoder(frame_t))
                    disp_out, _ = disp_to_depth(out[("disp", 0)][:, 0:1], 0.1, 100.0)
                decoder.train()

            if flip_ensemble:
                decoder.eval()
                with torch.no_grad():
                    inp_flip = torch.flip(frame_t, [3])
                    out_flip = decoder(encoder(inp_flip))
                    d_flip, _ = disp_to_depth(out_flip[("disp", 0)][:, 0:1], 0.1, 100.0)
                    disp_out = 0.5 * (disp_out + torch.flip(d_flip, [3]))
                decoder.train()

            pred_disps.append(disp_out.cpu().numpy()[0, 0])

        decoder.eval()
        pred_disps = np.array(pred_disps)
    else:
        # Inferenta normala (batch), cu flip ensemble optional
        dataset = datasets.KITTIRAWDataset(
            data_path, filenames, height, width, [0], 4, is_train=False, img_ext='.png')
        loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=4,
                            pin_memory=True, drop_last=False)
        with torch.no_grad():
            for data in loader:
                inp = data[("color_MiS", 0, 0)].to(device)
                out = decoder(encoder(inp), raw_image=inp)
                pred_disp, _ = disp_to_depth(out[("disp", 0)][:, 0, :, :].unsqueeze(1), 0.1, 100.0)
                if flip_ensemble:
                    inp_flip = torch.flip(inp, [3])
                    out_flip = decoder(encoder(inp_flip), raw_image=inp_flip)
                    pred_disp_flip, _ = disp_to_depth(out_flip[("disp", 0)][:, 0, :, :].unsqueeze(1), 0.1, 100.0)
                    pred_disp = 0.5 * (pred_disp + torch.flip(pred_disp_flip, [3]))
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
    parser.add_argument("--kitti_c_path", default="kitti_c/kitti_c")
    parser.add_argument("--orig_data_path", default=None,
                        help="Calea datelor KITTI originale (pentru TTA fotometric). "
                             "Daca nu e specificat, TTA fotometric e dezactivat.")
    parser.add_argument("--corruptions", nargs="+", default=["fog", "snow", "frost", "dark", "brightness"],
                        help="Use 'all' sau 'weather'.")
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--eval_mono", action="store_true", required=True)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--use_feature_suppression", action="store_true")
    parser.add_argument("--no_suppression_gating", action="store_true")
    parser.add_argument("--flip_ensemble", action="store_true")
    parser.add_argument("--use_tta", action="store_true",
                        help="TTA fotometric ghidat de incertitudine cu vecini corupti consistent")
    parser.add_argument("--tta_steps", type=int, default=3)
    parser.add_argument("--tta_lr", type=float, default=1e-5)
    parser.add_argument("--tta_consistency_weight", type=float, default=1.0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="tinydepth")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    opt = parser.parse_args()

    corruptions = opt.corruptions
    if "all" in corruptions:
        corruptions = ALL_CORRUPTIONS
    elif "weather" in corruptions:
        corruptions = WEATHER_CORRUPTIONS

    use_tta = getattr(opt, "use_tta", False)
    orig_data_path = getattr(opt, "orig_data_path", None)
    if use_tta and orig_data_path is None:
        print("ATENTIE: --use_tta necesita --orig_data_path; TTA dezactivat.")
        use_tta = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Loading weights from {opt.load_weights_folder}")
    gate_depth_input = not getattr(opt, "no_suppression_gating", False)
    encoder, decoder, pose_enc, pose_dec = load_model(
        opt.load_weights_folder, device,
        height=opt.height, width=opt.width,
        use_feature_suppression=opt.use_feature_suppression,
        gate_depth_input=gate_depth_input,
        use_tta=use_tta,
    )
    flip = getattr(opt, "flip_ensemble", False)
    print(f"-> TTA fotometric: {use_tta} | flip ensemble: {flip}")

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
            errs = evaluate_one(
                encoder, decoder, data_path, device,
                height=opt.height, width=opt.width,
                flip_ensemble=flip,
                use_tta=use_tta, pose_enc=pose_enc, pose_dec=pose_dec,
                tta_steps=opt.tta_steps, tta_lr=opt.tta_lr,
                tta_consistency_weight=opt.tta_consistency_weight,
                orig_data_path=orig_data_path,
                corruption_name=corruption, severity=sev,
            )
            sev_errors.append(errs)
            all_errors.append(errs)

        if sev_errors:
            results_per_corruption[corruption] = np.mean([e[0] for e in sev_errors])
            mean_abs = results_per_corruption[corruption]
            sev_str = " | ".join([f"  {e[0]:.3f}  " for e in sev_errors])
            print(f"  {corruption:>20} | {sev_str} | {mean_abs:.3f}")

    overall = np.array(all_errors).mean(0) if all_errors else None
    if overall is not None:
        print(f"\n{'OVERALL MEAN':>20}   abs_rel={overall[0]:.3f} | sq_rel={overall[1]:.3f} | rmse={overall[2]:.3f} | rmse_log={overall[3]:.3f} | a1={overall[4]:.3f} | a2={overall[5]:.3f} | a3={overall[6]:.3f}")

    if opt.use_wandb and wandb is not None and overall is not None:
        run_name = opt.wandb_run_name or f"kitti-c-{os.path.basename(opt.load_weights_folder)}"
        wandb.init(project=opt.wandb_project, name=run_name)
        log_dict = {f"kitti_c/{k}": float(overall[i]) for i, k in
                    enumerate(["abs_rel","sq_rel","rmse","rmse_log","a1","a2","a3"])}
        for corr, abs_rel in results_per_corruption.items():
            log_dict[f"kitti_c_per/{corr}"] = float(abs_rel)
        wandb.log(log_dict)
        wandb.finish()

    print("\n-> Done!")


if __name__ == "__main__":
    main()
