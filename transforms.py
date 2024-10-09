import cv2
import numpy as np
from typing import List, Tuple, Mapping
from augraphy import (AugraphyPipeline, BookBinding, Folding, BadPhotoCopy, 
                      BadPhotoCopy, LightingGradient, DirtyScreen, 
                      LCDScreenPattern, Jpeg, DirtyRollers)
import albumentations as A

import argparse
import os
from pathlib import Path
import cv2
import numpy as np


def book_bending(image: np.ndarray, curve_value:int = None, p_flip=.5, backdrop_color=None, p=0.5, keypoints=None):
    curve_value = np.random.randint(0, int(max(image.shape) / 1.5)) if curve_value is None else curve_value
    backdrop_color = np.max(image) if backdrop_color is None else backdrop_color
    flipped = np.random.binomial(1, p_flip)
    if flipped:
        image = image[::-1]
    p = np.random.binomial(1, p)
    if np.random.binomial(1, p):
        image, _, keypoints, _ = BookBinding().curve_page(img=image, curve_value=curve_value, backdrop_color=backdrop_color, keypoints=keypoints)
    return image[::-1] if flipped else image, keypoints, p # for consistency

pp = 0.5

albums_compose_first = A.Compose([
    A.HorizontalFlip(p=pp),
    A.VerticalFlip(p=pp),
    A.Blur(blur_limit=3, p=pp),
    A.RandomBrightnessContrast(p=pp),
    A.GaussNoise(var_limit=(30, 120), p=pp),
    A.CLAHE(clip_limit=2.0, p=pp),
    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=pp),
])

albums_affine = A.Compose(
        [A.Affine(fit_output=True, cval=255, p=pp, shear=(-20, 20))]
    )

def apply_augms(image: np.ndarray):
    
    out = albums_compose_first(image=image)

    image = out['image']

    augraphy_pixel_wise = AugraphyPipeline(
        ink_phase=[
            BadPhotoCopy(p=pp), 
            LightingGradient(p=pp),
            DirtyScreen(value_range=(202, 255), p=pp), 
            LCDScreenPattern(pattern_value_range=(48, 255), p=pp),
            Jpeg(quality_range=(8, 10)),
            DirtyRollers(p=pp)
            ]
        )
    
    out = augraphy_pixel_wise.augment(image)

    image = out['output']
    return image
    


def arg_parse():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--images_dir', default="init_data/", type=Path, help='Path to images')
    parser.add_argument('--dir2save', default="transformed_data/", type=Path, help='Path to save images')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    image_paths = os.listdir(args.images_dir)

    if not os.path.exists(args.dir2save):
        os.makedirs(args.dir2save)

    for image_path in image_paths:
        image = cv2.imread(str(args.images_dir / image_path))

        try: 
            image = apply_augms(image)
        except:
            continue
        
        cv2.imwrite(f"{args.dir2save}/{image_path}", image)