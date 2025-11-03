import cv2
import numpy as np


def create_trackbars():
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L - Y", "Trackbars", 38, 255, lambda x: None)
    cv2.createTrackbar("L - Cr", "Trackbars", 161, 255, lambda x: None)
    cv2.createTrackbar("L - Cb", "Trackbars", 74, 255, lambda x: None)
    cv2.createTrackbar("U - Y", "Trackbars", 255, 255, lambda x: None)
    cv2.createTrackbar("U - Cr", "Trackbars", 193, 255, lambda x: None)
    cv2.createTrackbar("U - Cb", "Trackbars", 195, 255, lambda x: None)


def get_values_from_trackbars():
    l_y = cv2.getTrackbarPos("L - Y", "Trackbars")
    l_cr = cv2.getTrackbarPos("L - Cr", "Trackbars")
    l_cb = cv2.getTrackbarPos("L - Cb", "Trackbars")

    u_y = cv2.getTrackbarPos("U - Y", "Trackbars")
    u_cr = cv2.getTrackbarPos("U - Cr", "Trackbars")
    u_cb = cv2.getTrackbarPos("U - Cb", "Trackbars")

    lower = np.array([l_y, l_cr, l_cb])
    upper = np.array([u_y, u_cr, u_cb])

    return lower, upper
