import numpy as np


MIN_CONTOUR_AREA = 10000
CLASSIFY_AREA_THRESHOLD = 13000
HISTORY_LENGTH = 30
SIZE = 64

FRAME_COUNT = 0
SAVE_INTERVAL = 2

IOU_THRESHOLD = 0.5

MEAN = np.array([110.0, 155.0])  # [Cb, Cr]
COV = np.array([[841.0, 59.0], [59.0, 654.0]])

INV_COV = np.linalg.inv(COV)
DET_COV = np.linalg.det(COV)
NORM_FACTOR = 1.0 / (2 * np.pi * np.sqrt(DET_COV))
