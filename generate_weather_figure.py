"""
Genereaza imagini cu augmentarea meteo (ceata/ploaie/zapada) la severitati
crescatoare, pentru figurile din lucrare (Capitolul 3, augmentare de date).

Output: weather_figures/{weather}_{severity_tag}.png (individual)
        weather_figures/grid_{weather}.png (grid orizontal, o singura imagine)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hf_space"))

import numpy as np
from PIL import Image
from weather_aug import apply_fog, apply_rain, apply_snow

SAMPLE_IMG = "kitti_c/kitti_c/clean/kitti_data/2011_09_26/2011_09_26_drive_0009_sync/image_02/data/0000000000.png"
OUT_DIR = "weather_figures"
os.makedirs(OUT_DIR, exist_ok=True)

SEVERITIES = [0.0, 0.25, 0.45, 0.65, 0.85]
WEATHER_FNS = {"fog": apply_fog, "rain": apply_rain, "snow": apply_snow}


def main():
    img = Image.open(SAMPLE_IMG).convert("RGB")

    for weather_name, fn in WEATHER_FNS.items():
        row_imgs = []
        for sev in SEVERITIES:
            out_img = img if sev == 0.0 else fn(img, sev)
            tag = "clean" if sev == 0.0 else f"sev{int(sev*100):02d}"
            path = os.path.join(OUT_DIR, f"{weather_name}_{tag}.png")
            out_img.save(path)
            print(f"  saved {path}")
            row_imgs.append(np.array(out_img))

        # grid orizontal: concateneaza toate severitatile intr-o singura imagine
        grid = np.concatenate(row_imgs, axis=1)
        grid_path = os.path.join(OUT_DIR, f"grid_{weather_name}.png")
        Image.fromarray(grid).save(grid_path)
        print(f"-> grid salvat: {grid_path}")

    # grid combinat: 3 randuri (fog/rain/snow) x 5 coloane (severitati)
    all_rows = []
    for weather_name, fn in WEATHER_FNS.items():
        row_imgs = []
        for sev in SEVERITIES:
            out_img = img if sev == 0.0 else fn(img, sev)
            row_imgs.append(np.array(out_img))
        all_rows.append(np.concatenate(row_imgs, axis=1))
    full_grid = np.concatenate(all_rows, axis=0)
    full_path = os.path.join(OUT_DIR, "grid_all_weather.png")
    Image.fromarray(full_grid).save(full_path)
    print(f"-> grid complet salvat: {full_path}")


if __name__ == "__main__":
    main()
