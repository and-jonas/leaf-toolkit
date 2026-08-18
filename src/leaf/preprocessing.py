import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys


class Preprocessor:
    """Minimal preprocessor.
    """

    def __init__(self, crop_sz: Optional[Tuple[int, int]] = None, crop_offsets: Optional[Tuple[int, int]] = None):

        self.crop_sz = crop_sz
        self.crop_offsets = crop_offsets

    pass

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, object]]]:
        """Try rotate+crop using module-level or instance config and find_marker.

        Returns (image_out, info) where image_out is the cropped/rotated image or
        the original image when no crop/transform succeeded.
        """

        # gather sizes/offsets from instance or globals
        csx1, csx2 = self.crop_sz
        cox1, cox2 = self.crop_offsets

        img_rotate = image
        if image.shape == (5464, 8192, 3):
            img_rotate = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # perform cropping
        y1 = int(cox1)
        x1 = int(cox2)
        y2 = y1 + int(csx1)
        x2 = x1 + int(csx2)
        if 0 <= y1 < y2 <= img_rotate.shape[0] and 0 <= x1 < x2 <= img_rotate.shape[1]:
            img_crop = img_rotate[y1:y2, x1:x2, :]
            if img_crop.shape == (int(csx1), int(csx2), 3):
                return img_crop
            else:
                logging.warning(f"crop produced unexpected shape: {img_crop.shape}")



