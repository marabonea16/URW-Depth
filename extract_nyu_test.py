"""
Extrage cele 654 de imagini de test oficiale din NYU Depth v2 (labeled subset),
folosind split-ul oficial Eigen/Silberman (testNdxs din splits.mat).

Protocol identic cu cel folosit in literatura monodepth (BTS, AdaBins, etc.)
pentru evaluarea generalizarii cross-domain a modelelor antrenate pe KITTI.

Usage:
  python extract_nyu_test.py --mat_path nyu_data/nyu_depth_v2_labeled.mat \
      --splits_path nyu_data/splits.mat --out_dir nyu_data/test
"""
import argparse
import os
import h5py
import numpy as np
import scipy.io
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_path", default="nyu_data/nyu_depth_v2_labeled.mat")
    parser.add_argument("--splits_path", default="nyu_data/splits.mat")
    parser.add_argument("--out_dir", default="nyu_data/test")
    opt = parser.parse_args()

    os.makedirs(os.path.join(opt.out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(opt.out_dir, "depths"), exist_ok=True)

    splits = scipy.io.loadmat(opt.splits_path)
    test_idxs = sorted(int(x) for x in splits["testNdxs"].flatten())
    print(f"-> {len(test_idxs)} imagini de test (split oficial)")

    f = h5py.File(opt.mat_path, "r")
    images = f["images"]   # [N, 3, H, W]
    depths = f["depths"]   # [N, H, W], in metri

    for out_i, idx in enumerate(test_idxs):
        mat_i = idx - 1  # testNdxs e 1-indexat (MATLAB)

        img = images[mat_i].transpose(2, 1, 0)  # -> [H, W, 3], RGB
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        depth = depths[mat_i].transpose(1, 0)  # -> [H, W], metri
        depth_mm = (depth * 1000.0).astype(np.uint16)

        cv2.imwrite(os.path.join(opt.out_dir, "images", f"{out_i:04d}.png"), img_bgr)
        cv2.imwrite(os.path.join(opt.out_dir, "depths", f"{out_i:04d}.png"), depth_mm)

        if out_i % 100 == 0:
            print(f"   {out_i}/{len(test_idxs)}")

    f.close()
    print(f"-> Salvat {len(test_idxs)} imagini in {opt.out_dir}")


if __name__ == "__main__":
    main()
