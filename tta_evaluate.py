"""
Test-Time Adaptation (TTA) evaluation for URW-Depth.

Pentru fiecare imagine de test:
1. Incarca frame-urile vecine (t-1, t, t+1)
2. Ruleaza N pasi de gradient cu loss fotometric weighted de uncertainty (1-sigma)
3. Actualizeaza doar parametrii depth decoder-ului
4. Produce predictia finala adaptata

Folosire:
    python tta_evaluate.py \
        --load_weights_folder models/Tiny-Depth-Weather-Robust-Feature-Supression/models/weights_49 \
        --eval_mono --height 192 --width 640 --scales 0 \
        --data_path /home/ubuntu/TinyDepth --png \
        --tta_steps 5 --tta_lr 1e-4 \
        --use_feature_suppression --use_wandb \
        --wandb_project tinydepth --wandb_run_name tta-weather-robust
"""

from __future__ import absolute_import, division, print_function

import os
import copy
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from networks.configuration import get_config
from layer import (disp_to_depth, BackprojectDepth, Project3D,
                   SSIM, transformation_from_parameters)
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


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_photometric_loss(pred, target, ssim_fn):
    """SSIM + L1 photometric loss, identic cu trainer.py"""
    abs_diff = torch.abs(target - pred)
    l1 = abs_diff.mean(1, True)
    ssim_loss = ssim_fn(pred, target).mean(1, True)
    loss = 0.85 * ssim_loss + 0.15 * l1
    return loss


def tta_step(encoder, depth_decoder, pose_encoder, pose_decoder,
             frame_target, frame_prev, frame_next,
             K, inv_K, ssim_fn, backproject, project,
             tta_lr, n_steps, device):
    """
    Ruleaza N pasi TTA pe un singur sample.
    Actualizeaza doar depth_decoder (encoder si pose raman frozen).
    Returneaza disparitatea adaptata.
    """
    # salveaza starea initiala pentru restaurare dupa
    original_state = copy.deepcopy(depth_decoder.state_dict())

    optimizer = torch.optim.Adam(depth_decoder.parameters(), lr=tta_lr)

    H, W = frame_target.shape[2], frame_target.shape[3]

    for step in range(n_steps):
        optimizer.zero_grad()

        # --- depth prediction pe frame-ul target ---
        with torch.no_grad():
            feats_target = encoder(frame_target)
        output = depth_decoder(feats_target)

        disp = output[("disp", 0)][:, 0:1]  # [1,1,H,W]
        uncert_raw = output[("uncert", 0)]   # [1,1,H,W] log-variance
        sigma = torch.sigmoid(uncert_raw)    # [0,1]: 0=sigur, 1=nesigur

        _, depth = disp_to_depth(disp, 0.1, 100.0)

        # --- pose: t-1 -> t ---
        pose_input_prev = torch.cat([frame_prev, frame_target], dim=1)
        pose_feats_prev = [pose_encoder(pose_input_prev)]
        axisangle_prev, translation_prev = pose_decoder(pose_feats_prev)
        T_prev = transformation_from_parameters(
            axisangle_prev[:, 0], translation_prev[:, 0], invert=True)

        # --- pose: t+1 -> t ---
        pose_input_next = torch.cat([frame_target, frame_next], dim=1)
        pose_feats_next = [pose_encoder(pose_input_next)]
        axisangle_next, translation_next = pose_decoder(pose_feats_next)
        T_next = transformation_from_parameters(
            axisangle_next[:, 0], translation_next[:, 0], invert=False)

        # --- reprojection loss ---
        total_loss = torch.tensor(0.0, device=device)

        for T, frame_src in [(T_prev, frame_prev), (T_next, frame_next)]:
            cam_points = backproject(depth, inv_K)
            pix_coords = project(cam_points, K, T)
            reconstructed = F.grid_sample(
                frame_src, pix_coords,
                mode="bilinear", padding_mode="border", align_corners=False)

            photo_loss = compute_photometric_loss(reconstructed, frame_target, ssim_fn)  # [1,1,H,W]

            # uncertainty-weighted: pixelii cu sigma mare contribuie mai putin
            weighted_loss = ((1.0 - sigma.detach()) * photo_loss).mean()
            # regularizare sigma ca sa nu cada spre 0
            reg_loss = (0.1 * sigma).mean()
            total_loss = total_loss + weighted_loss + reg_loss

        total_loss.backward()
        optimizer.step()

    # --- predictie finala cu parametrii adaptati ---
    depth_decoder.eval()
    with torch.no_grad():
        feats_final = encoder(frame_target)
        output_final = depth_decoder(feats_final)
        disp_final = output_final[("disp", 0)][:, 0:1]

    # restaureaza parametrii originali pentru urmatorul sample
    depth_decoder.load_state_dict(original_state)
    depth_decoder.train()

    return disp_final


def evaluate(opt):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    device = torch.device("cuda" if not opt.no_cuda else "cpu")

    assert opt.eval_mono, "TTA evaluation requires --eval_mono"

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

    config = get_config(opt)
    num_ch_enc = [64, 64, 128, 160, 320]

    # --- build models ---
    encoder = networks.build_model(config, img_width=opt.width, img_height=opt.height)
    depth_decoder = networks.FusionDecoder(
        num_ch_enc,
        use_feature_suppression=getattr(opt, "use_feature_suppression", False))

    pose_encoder = networks.ResnetEncoder(18, False, num_input_images=2)
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, num_input_features=1,
                                        num_frames_to_predict_for=2)

    # --- load weights ---
    encoder_dict = torch.load(os.path.join(opt.load_weights_folder, "encoder.pth"),
                              map_location=device)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(
        torch.load(os.path.join(opt.load_weights_folder, "depth.pth"), map_location=device),
        strict=False)
    pose_encoder.load_state_dict(
        torch.load(os.path.join(opt.load_weights_folder, "pose_encoder.pth"), map_location=device))
    pose_decoder.load_state_dict(
        torch.load(os.path.join(opt.load_weights_folder, "pose.pth"), map_location=device))

    encoder.to(device).eval()
    depth_decoder.to(device).train()   # train mode pt TTA (BN stats update)
    pose_encoder.to(device).eval()
    pose_decoder.to(device).eval()

    # frozen encoder si pose
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in pose_encoder.parameters():
        p.requires_grad_(False)
    for p in pose_decoder.parameters():
        p.requires_grad_(False)

    # --- dataset cu frame-uri vecine [-1, 0, 1] ---
    dataset = datasets.KITTIRAWDataset(
        opt.data_path, filenames,
        opt.height, opt.width,
        [-1, 0, 1], 4, is_train=False, img_ext='.png')
    dataloader = DataLoader(dataset, 1, shuffle=False,
                            num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    # --- geometrie ---
    ssim_fn = SSIM().to(device)
    backproject = BackprojectDepth(1, opt.height, opt.width).to(device)
    project = Project3D(1, opt.height, opt.width).to(device)

    # --- GT depth ---
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    pred_disps = []
    sample_uncert_maps = []
    sample_input_imgs = []

    print("-> Running TTA ({} steps, lr={}) on {} images".format(
        opt.tta_steps, opt.tta_lr, len(filenames)))

    for idx, data in enumerate(dataloader):
        frame_target = data[("color_MiS", 0, 0)].to(device)
        frame_prev   = data[("color_MiS", -1, 0)].to(device)
        frame_next   = data[("color_MiS", 1, 0)].to(device)

        K    = data[("K_MiS", 0)].to(device).float()
        inv_K = data[("inv_K_MiS", 0)].to(device).float()

        disp_adapted = tta_step(
            encoder, depth_decoder, pose_encoder, pose_decoder,
            frame_target, frame_prev, frame_next,
            K, inv_K, ssim_fn, backproject, project,
            tta_lr=opt.tta_lr, n_steps=opt.tta_steps, device=device)

        pred_disp = disp_adapted.cpu().numpy()[:, 0]  # [1,H,W]
        pred_disps.append(pred_disp)

        # colecteaza uncertainty maps pentru vizualizare
        if len(sample_uncert_maps) < 8:
            with torch.no_grad():
                out_tmp = depth_decoder(encoder(frame_target))
            if ("uncert", 0) in out_tmp:
                u = out_tmp[("uncert", 0)].cpu().numpy()[0, 0]
                sample_uncert_maps.append(u)
                sample_input_imgs.append(data[("color_MiS", 0, 0)].numpy()[0])

        if (idx + 1) % 50 == 0:
            print("  [{}/{}]".format(idx + 1, len(filenames)))

    pred_disps = np.concatenate(pred_disps)

    # --- metrici ---
    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_h, gt_w = gt_depth.shape

        pred_disp = cv2.resize(pred_disps[i], (gt_w, gt_h))
        pred_depth = 1.0 / pred_disp

        mask = gt_depth > 0
        pred_depth = pred_depth[mask]
        gt_depth_m = gt_depth[mask]

        pred_depth = pred_depth.clip(MIN_DEPTH, MAX_DEPTH)
        gt_depth_m = gt_depth_m.clip(MIN_DEPTH, MAX_DEPTH)

        ratio = np.median(gt_depth_m) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio

        pred_depth = pred_depth.clip(MIN_DEPTH, MAX_DEPTH)
        errors.append(compute_errors(gt_depth_m, pred_depth))

    ratios = np.array(ratios)
    print("\nScaling ratios | med: {:.3f} | std: {:.3f}".format(
        np.median(ratios), ratios.std()))

    mean_errors = np.array(errors).mean(0)
    print("\n   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |")
    print(("&   {:.3f}  " * 7).format(*mean_errors))

    # --- wandb logging ---
    if getattr(opt, "use_wandb", False) and wandb is not None:
        run_name = getattr(opt, "wandb_run_name", None) or "tta-eval"
        wandb.init(project=opt.wandb_project,
                   entity=getattr(opt, "wandb_entity", None),
                   name=run_name)
        log_dict = {
            "eval/abs_rel":  float(mean_errors[0]),
            "eval/sq_rel":   float(mean_errors[1]),
            "eval/rmse":     float(mean_errors[2]),
            "eval/rmse_log": float(mean_errors[3]),
            "eval/a1":       float(mean_errors[4]),
            "eval/a2":       float(mean_errors[5]),
            "eval/a3":       float(mean_errors[6]),
            "tta/steps":     opt.tta_steps,
            "tta/lr":        opt.tta_lr,
        }

        if sample_uncert_maps:
            uncert_images = []
            for u, img in zip(sample_uncert_maps, sample_input_imgs):
                u_norm = (u - u.min()) / (u.max() - u.min() + 1e-8)
                u_color = cv2.applyColorMap((u_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                u_color = cv2.cvtColor(u_color, cv2.COLOR_BGR2RGB)
                rgb = np.transpose(img, (1, 2, 0))
                rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255).astype(np.uint8)
                combined = np.concatenate([rgb, u_color], axis=1)
                uncert_images.append(wandb.Image(combined, caption="input | uncertainty (TTA)"))
            log_dict["eval/uncertainty_maps"] = uncert_images

        wandb.log(log_dict)
        wandb.finish()

    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    options.parser.add_argument("--tta_steps", type=int, default=5,
                                help="numar de pasi gradient la TTA")
    options.parser.add_argument("--tta_lr", type=float, default=1e-4,
                                help="learning rate pentru TTA")
    options.parser.add_argument("--no_cuda", action="store_true",
                                help="daca e setat, foloseste CPU")
    evaluate(options.parse())
