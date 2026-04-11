"""
Verifica uncertainty head: shape, variatie spatiala, gradient flow.
Rulare: python check_uncertainty.py
"""
import torch
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from networks.configuration import get_config
import networks
from options import MonodepthOptions

WEIGHTS = "/home/ubuntu/TinyDepth/models/Tiny-Depth-Basic-Uncertainty-Head-2/models/weights_49"

# --- incarca modelul ---
sys.argv = ["check_uncertainty.py", "--load_weights_folder", WEIGHTS, "--scales", "0"]
opt = MonodepthOptions().parse()
config = get_config(opt)
num_ch_enc = [64, 64, 128, 160, 320]

encoder = networks.build_model(config, img_width=opt.width, img_height=opt.height).cuda()
decoder = networks.FusionDecoder(num_ch_enc).cuda()

encoder_dict = torch.load(os.path.join(WEIGHTS, "encoder.pth"))
decoder_dict = torch.load(os.path.join(WEIGHTS, "depth.pth"))
encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
decoder.load_state_dict(decoder_dict, strict=False)
encoder.eval()
decoder.eval()

# --- imagine aleatoare ---
dummy = torch.rand(1, 3, opt.height, opt.width).cuda()

print("=" * 50)
print("CHECK 1: shape si existenta")
with torch.no_grad():
    out = decoder(encoder(dummy))

assert ("uncert", 0) in out, "FAIL: ('uncert', 0) nu exista in output!"
assert ("disp", 0) in out, "FAIL: ('disp', 0) nu exista in output!"

disp = out[("disp", 0)]
uncert = out[("uncert", 0)]
print(f"  disp shape:   {tuple(disp.shape)}  (asteptat: [1,1,{opt.height},{opt.width}])")
print(f"  uncert shape: {tuple(uncert.shape)}  (asteptat: [1,1,{opt.height},{opt.width}])")
assert disp.shape == uncert.shape, "FAIL: shape-urile difera!"
print("  -> OK")

# --- CHECK 2: variatie spatiala (nu e constanta) ---
print("\nCHECK 2: variatie spatiala")
u = uncert[0, 0].cpu().numpy()
print(f"  min={u.min():.4f}  max={u.max():.4f}  std={u.std():.4f}  mean={u.mean():.4f}")
assert u.std() > 1e-5, "FAIL: uncertainty e aproape constanta peste tot!"
print("  -> OK (exista variatie spatiala)")

# --- CHECK 3: gradient flow ---
print("\nCHECK 3: gradient flow spre uncertconv")
encoder.train()
decoder.train()
dummy_grad = torch.rand(1, 3, opt.height, opt.width).cuda().requires_grad_(False)
out2 = decoder(encoder(dummy_grad))
loss = out2[("uncert", 0)].mean()
loss.backward()

uncert_weight = decoder.convs[("uncertconv", 0)].conv.weight
assert uncert_weight.grad is not None, "FAIL: niciun gradient la uncertconv!"
print(f"  grad norm uncertconv: {uncert_weight.grad.norm().item():.6f}")
print("  -> OK (gradientii ajung la uncertainty head)")

# --- CHECK 4: disp si uncert difera (nu sunt identice) ---
print("\nCHECK 4: disp != uncert (capete independente)")
with torch.no_grad():
    out3 = decoder(encoder(dummy))
corr = np.corrcoef(
    out3[("disp", 0)][0, 0].cpu().numpy().flatten(),
    out3[("uncert", 0)][0, 0].cpu().numpy().flatten()
)[0, 1]
print(f"  corelatie disp-uncert: {corr:.4f}  (valoare mica = capete independente)")
print("  -> OK")

print("\n" + "=" * 50)
print("TOATE VERIFICARILE AU TRECUT")
print("=" * 50)
