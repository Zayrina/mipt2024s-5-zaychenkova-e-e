import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import shutil


def arg_parse():
    parser = argparse.ArgumentParser(description='Split dataset')
    parser.add_argument('--test_images_dir', default="Source/dataset/pertubed_barcodes_7/test", type=Path, help='Path to test images')
    parser.add_argument('--train_images_dir', default="Source/dataset/pertubed_barcodes_7/color", type=Path, help='Path to train images')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    test_image_paths = os.listdir(args.test_images_dir)
    train_image_paths = os.listdir(args.train_images_dir)

    for path in test_image_paths: 
        path = Path(path)   
        if path.suffix == ".png":
            if os.path.exists(f"{args.train_images_dir}/{path.stem}.gw"):
                os.remove(f"{args.train_images_dir}/{path.stem}.gw")
            else:
                print("The file does not exist")
            