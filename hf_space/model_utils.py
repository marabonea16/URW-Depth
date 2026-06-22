"""
Incarcare modele URW-Depth / TinyDepth direct de pe HuggingFace Hub si
rulare inferenta pentru demo-ul interactiv.
"""
import os
import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from networks.configuration import get_config
import networks
from layer import disp_to_depth

REPO_ID = "mara-bonea-16/tinydepth-experiments"

# Progresia cronologica a ablatiei (ordinea contributiilor adaugate)
ABLATION_MODELS = [
    ("TinyDepth (baseline)", "Tiny-Depth-b6", "weights_49", False),
    ("+ Cap de incertitudine", "Tiny-Depth-Basic-Uncertainty-Head-2", "weights_49", False),
    ("+ Automascare ghidata", "Tiny-Depth-Uncertainty-Guided-Automasking", "weights_49", False),
    ("+ Suprimare caracteristici", "Tiny-Depth-Weather-Robust-Feature-Supression", "weights_49", True),
    ("URW-Depth-S2 (final, 640x192)", "URW-Depth-S2", "weights_14", True),
    ("URW-Depth-HiRes (1280x384)", "URW-Depth-HiRes-S2", "weights_7", True),
]

MODEL_HEIGHT_WIDTH = {
    "URW-Depth-HiRes-S2": (384, 1280),
}

NUM_CH_ENC = [64, 64, 128, 160, 320]

_model_cache = {}


def _download_weights(model_dir, weights_dir):
    local_dir = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{model_dir}/models/{weights_dir}/encoder.pth",
    )
    base = os.path.dirname(local_dir)
    hf_hub_download(repo_id=REPO_ID, filename=f"{model_dir}/models/{weights_dir}/depth.pth")
    return base


def load_model(model_dir, weights_dir, use_feature_suppression, device="cpu"):
    cache_key = (model_dir, weights_dir)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    height, width = MODEL_HEIGHT_WIDTH.get(model_dir, (192, 640))
    weights_folder = _download_weights(model_dir, weights_dir)

    class _Opt:
        img_height = height
        img_width = width
        encoder = "tiny_vit_5m_22k_distill"
        scales = [0]

    config = get_config(_Opt())
    encoder = networks.build_model(config, img_width=width, img_height=height)
    encoder_dict = torch.load(os.path.join(weights_folder, "encoder.pth"), map_location=device)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    decoder = networks.FusionDecoder(NUM_CH_ENC, use_feature_suppression=use_feature_suppression)
    decoder.load_state_dict(
        torch.load(os.path.join(weights_folder, "depth.pth"), map_location=device), strict=False)

    encoder.to(device).eval()
    decoder.to(device).eval()

    _model_cache[cache_key] = (encoder, decoder, height, width)
    return encoder, decoder, height, width


def run_inference(pil_image, model_dir, weights_dir, use_feature_suppression, device="cpu"):
    """Ruleaza modelul pe o imagine PIL si returneaza (depth_colormap, uncert_colormap, disp_raw)."""
    encoder, decoder, height, width = load_model(model_dir, weights_dir, use_feature_suppression, device)

    img = np.array(pil_image.convert("RGB"))
    img_resized = cv2.resize(img, (width, height))
    inp = torch.from_numpy(img_resized / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = decoder(encoder(inp))
        disp, _ = disp_to_depth(out[("disp", 0)][:, 0:1], 0.1, 100.0)
        disp_np = disp[0, 0].cpu().numpy()

        uncert_np = None
        if ("uncert", 0) in out:
            uncert_np = torch.sigmoid(out[("uncert", 0)])[0, 0].cpu().numpy()

    disp_norm = (disp_np - disp_np.min()) / (disp_np.max() - disp_np.min() + 1e-8)
    depth_color = cv2.applyColorMap((disp_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

    uncert_color = None
    if uncert_np is not None:
        u_norm = (uncert_np - uncert_np.min()) / (uncert_np.max() - uncert_np.min() + 1e-8)
        uncert_color = cv2.applyColorMap((u_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        uncert_color = cv2.cvtColor(uncert_color, cv2.COLOR_BGR2RGB)

    return depth_color, uncert_color, disp_np
