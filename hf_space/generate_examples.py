"""
Genereaza exemplele precalculate pentru cele doua galerii statice din Space:
  1. Progresie cronologica a ablatiei (6 modele, pe 2 imagini de exemplu)
  2. Grid de severitate meteo (fog/rain/snow x severitati crescatoare, pe URW-Depth-S2)

Ruleaza o singura data, local, pe CPU. Salveaza PNG-uri in examples/.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image
from model_utils import run_inference, ABLATION_MODELS
from weather_aug import apply_weather

SAMPLE_IMAGES = [
    ("scene1", "../kitti_c/kitti_c/clean/kitti_data/2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000000.png"),
    ("scene2", "../kitti_c/kitti_c/clean/kitti_data/2011_09_26/2011_09_26_drive_0027_sync/image_02/data/0000000000.png"),
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "examples")
os.makedirs(OUT_DIR, exist_ok=True)


def safe_name(s):
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")


def generate_ablation_progression():
    for scene_name, img_path in SAMPLE_IMAGES:
        img = Image.open(img_path).convert("RGB")
        rgb_path = os.path.join(OUT_DIR, f"{scene_name}_input.png")
        img.save(rgb_path)
        for idx, (name, model_dir, weights_dir, use_fs) in enumerate(ABLATION_MODELS):
            depth_color, _, _ = run_inference(img, model_dir, weights_dir, use_fs, device="cpu")
            out_path = os.path.join(OUT_DIR, f"{scene_name}_ablation_{idx}_{safe_name(name)}.png")
            Image.fromarray(depth_color).save(out_path)
            print(f"  saved {out_path}")


def generate_weather_grid():
    scene_name, img_path = SAMPLE_IMAGES[0]
    img = Image.open(img_path).convert("RGB")
    model_name, model_dir, weights_dir, use_fs = ABLATION_MODELS[-2]  # URW-Depth-S2
    severities = [0.0, 0.25, 0.45, 0.65, 0.85]
    for weather_type in ["fog", "rain", "snow"]:
        for sev in severities:
            wimg = img if sev == 0 else apply_weather(img, weather_type, sev)
            depth_color, _, _ = run_inference(wimg, model_dir, weights_dir, use_fs, device="cpu")
            sev_tag = "clean" if sev == 0 else f"sev{int(sev*100):02d}"
            out_path = os.path.join(OUT_DIR, f"weather_{weather_type}_{sev_tag}_rgb.png")
            wimg.save(out_path)
            out_path2 = os.path.join(OUT_DIR, f"weather_{weather_type}_{sev_tag}_depth.png")
            Image.fromarray(depth_color).save(out_path2)
            print(f"  saved {out_path2}")


if __name__ == "__main__":
    print("-> Generating ablation progression examples...")
    generate_ablation_progression()
    print("-> Generating weather severity grid...")
    generate_weather_grid()
    print("-> Done.")
