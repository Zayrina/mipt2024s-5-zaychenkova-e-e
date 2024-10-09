from pytorch_msssim import ms_ssim
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from haar_psi import compute_HaarPSI_similarity
from image_similarity_measures.quality_metrics import uiq, rmse, fsim, ssim
from piqa import ssim
from skimage.metrics import structural_similarity
from tqdm import tqdm


def arg_parse():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--gt_images_dir', default="transformed_data/", type=Path, help='Path to gt images')
    parser.add_argument('--test_images_dir', default="tested_sim/", type=Path, help='Path to test images')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    gt_image_paths = os.listdir(args.gt_images_dir)
    test_image_paths = os.listdir(args.test_images_dir)

    ssim_values = []
    haar_values = []
    fsim_values = []

    for test_image_path in tqdm(test_image_paths):
        gt_image_path = test_image_path[17:test_image_path.find(".png") + 4]

        if gt_image_path in gt_image_paths:
            gt_image = cv2.imread(args.gt_images_dir / gt_image_path)
            test_image = cv2.imread(args.test_images_dir / test_image_path)

            new_size = (min(gt_image.shape[0], test_image.shape[0]), min(gt_image.shape[1], test_image.shape[1]))
            gt_image = cv2.resize(gt_image, new_size, interpolation=cv2.INTER_AREA) 
            test_image = cv2.resize(test_image, new_size, interpolation=cv2.INTER_AREA) 

            haar_values.append(compute_HaarPSI_similarity(gt_image, test_image))
            fsim_values.append(fsim(gt_image, test_image))

            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            ssim_values.append(structural_similarity(gt_image, test_image))

    metrics = {"MS-SSIM": ssim_values, "HaarPSI": haar_values, "FSIM": fsim_values}
    print(metrics)
    for metric_name, metric_list in metrics.items():
        med_metric = np.mean(metric_list)
        std_metric = np.std(metric_list)
        print(f"{metric_name}: {med_metric} +- {std_metric}")