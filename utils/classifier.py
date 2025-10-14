import cv2
import numpy as np
from skimage.feature import hog
from constants import SIZE


def fill_holes(mask):
    mask = mask.astype(np.uint8)

    h, w = mask.shape[:2]
    flood = mask.copy()
    mask_flood = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, mask_flood, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)

    filled = flood_inv | mask
    return filled


def preprocess_hand(masked_hand, size=SIZE):
    h, w = masked_hand.shape[:2]

    scale = size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(masked_hand, (new_w, new_h))

    pad_w = size - new_w
    pad_h = size - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    resized = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    return resized


def compute_hog_features(img_gray):
    features = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return features
